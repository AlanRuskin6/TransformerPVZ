
import os
import sys
import time
import random
import subprocess
from datetime import datetime
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

# 减少 TensorFlow 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
import numpy as np

# === 极致优化: PyTorch 硬件加速 ===
if torch.cuda.is_available():
    # 1. 启用 cuDNN Benchmark (针对固定输入尺寸加速)
    torch.backends.cudnn.benchmark = True
    # 2. 启用 Tensor Cores (针对 30系显卡)
    torch.set_float32_matmul_precision('high')

# 设置 CPU 线程数 (防止抢占)
torch.set_num_threads(8)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack, VecNormalize
from models.attention_extractor import PVZAttentionExtractor

# 导入 PVZ 环境
from envs import PVZEnv
from hook_client import inject_dll
from hook_client.injector import find_pvz_process
from config import load_config

# 加载全局配置
cfg = load_config()

# =============================================================================
# 配置区域 - 建议通过 config.py 或命令行参数修改
# =============================================================================
DEFAULT_GAME_PATH = cfg.game_path
MODEL_PATH = cfg.model_save_path
LOAD_PATH = cfg.model_load_path

# ⚠️ 风险提示: 
# 由于涉及内存读写与汇编注入，程序运行过程中可能存在崩溃风险。
# 建议用户根据需求进一步完善代码的健壮性（如增加异常捕获、自动重启机制等）。
# =============================================================================

# 检测 GPU
if torch.cuda.is_available():
    device = "cuda"
    print(f"[设备] {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("[设备] CPU")


# =============================================================================
# 自动启动游戏和注入 DLL
# =============================================================================
def launch_game_and_inject(game_path: str = None, wait_time: float = 3.0, port: int = 12345) -> bool:
    """
    自动启动游戏并注入 Hook DLL
    
    Args:
        game_path: 游戏可执行文件路径
                                 wait_time: 启动后等待时间（秒）
        port: Hook 服务端口
    
    Returns:
        是否成功
    """
    if game_path is None:
        game_path = DEFAULT_GAME_PATH
    
    # 检查游戏是否已运行
    pid = find_pvz_process()
    if not pid:
        # 检查游戏文件是否存在
        if not os.path.exists(game_path):
            print(f"[错误] 游戏文件不存在: {game_path}")
            return False
        try:
            # 启动游戏（不等待，在后台运行）
            subprocess.Popen(
                [game_path],
                cwd=os.path.dirname(game_path),
                creationflags=subprocess.DETACHED_PROCESS
            )
        except Exception as e:
            print(f"[错误] 启动游戏失败: {e}")
            return False
        
        time.sleep(wait_time)
        
        pid = find_pvz_process()
        if not pid:
            print("[错误] 游戏启动失败")
            return False
    
    # 注入 DLL
    if inject_dll():
        print(f"[Hook] DLL注入成功 (port {port})")
        # 等待 Hook 初始化
        time.sleep(1.0)
        return True
    else:
        print("[错误] DLL注入失败")
        return False


def mask_fn(env):
    """获取动作掩码的回调函数"""
    return env.unwrapped._get_action_mask(
        env.unwrapped.pvz.get_game_state() if env.unwrapped.pvz else None
    )


def linear_schedule(initial_value: float):
    """
    线性学习率衰减函数
    :param initial_value: 初始学习率
    :return: schedule function
    """
    def func(progress_remaining: float):
        """
        Progress remaining decreases from 1 (beginning) to 0.
        """
        return progress_remaining * initial_value
    return func


def cosine_schedule(initial_value: float, warmup_steps: int = 0, total_steps: int = 1):
    """
    余弦退火学习率调度 (含热身)
    :param initial_value: 初始学习率
    :param warmup_steps: 热身步数 (占总步数的比例，0-1)
    :param total_steps: 总步数 (用于计算进度)
    :return: schedule function
    """
    def func(progress_remaining: float):
        """
        Progress remaining decreases from 1 (beginning) to 0.
        Current progress = 1 - progress_remaining
        """
        current_progress = 1.0 - progress_remaining
        
        # 热身阶段
        if current_progress < warmup_steps:
            return initial_value * (current_progress / warmup_steps)
        
        # 余弦退火阶段
        decay_progress = (current_progress - warmup_steps) / (1.0 - warmup_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_progress))
        return initial_value * cosine_decay
        
    return func


# =============================================================================
# 优化1: 动态探索系数回调
# =============================================================================
class DynamicEntropyCallback(BaseCallback):
    """
    动态调整探索系数 (ent_coef)
    
    策略:
    - 训练初期: 高探索 (ent_coef 大) - 大胆尝试各种策略
    - 训练后期: 低探索 (ent_coef 小) - 稳定收敛到最优策略
    
    衰减方式:
    - 线性衰减: ent_coef = start - (start - end) * progress
    - 指数衰减: ent_coef = end + (start - end) * exp(-decay * progress)
    - 余弦衰减: ent_coef = end + (start - end) * 0.5 * (1 + cos(pi * progress))
    """
    
    def __init__(
        self,
        start_ent_coef: float = 0.1,
        end_ent_coef: float = 0.01,
        decay_type: str = "cosine",  # "linear", "exponential", "cosine"
        total_timesteps: int = 500000,
        warmup_steps: int = 10000,  # 热身期保持高探索
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.start_ent_coef = start_ent_coef
        self.end_ent_coef = end_ent_coef
        self.decay_type = decay_type
        self.total_timesteps = total_timesteps
        self.warmup_steps = warmup_steps
        self.last_print_step = 0
        
    def _on_step(self) -> bool:
        # 计算进度
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        
        # 热身期保持高探索
        if self.num_timesteps < self.warmup_steps:
            new_ent_coef = self.start_ent_coef
        else:
            # 调整进度 (排除热身期)
            adjusted_progress = (self.num_timesteps - self.warmup_steps) / (
                self.total_timesteps - self.warmup_steps
            )
            adjusted_progress = min(1.0, max(0.0, adjusted_progress))
            
            if self.decay_type == "linear":
                new_ent_coef = self.start_ent_coef - (
                    self.start_ent_coef - self.end_ent_coef
                ) * adjusted_progress
                
            elif self.decay_type == "exponential":
                decay_rate = 5.0  # 控制衰减速度
                new_ent_coef = self.end_ent_coef + (
                    self.start_ent_coef - self.end_ent_coef
                ) * np.exp(-decay_rate * adjusted_progress)
                
            elif self.decay_type == "cosine":
                # 余弦退火 - 平滑衰减
                new_ent_coef = self.end_ent_coef + (
                    self.start_ent_coef - self.end_ent_coef
                ) * 0.5 * (1 + np.cos(np.pi * adjusted_progress))
            else:
                new_ent_coef = self.start_ent_coef
        
        # 更新模型的 ent_coef
        self.model.ent_coef = new_ent_coef
        
        # 每 10000 步打印一次
        if self.verbose and self.num_timesteps - self.last_print_step >= 10000:
            print(f"  📊 动态探索: ent_coef = {new_ent_coef:.4f} (进度: {progress*100:.1f}%)")
            self.last_print_step = self.num_timesteps
            
        return True


# =============================================================================
# 优化2: 多样化训练环境包装器
# =============================================================================
class DiversifiedPVZEnv(PVZEnv):
    """
    多样化训练环境
    
    策略:
    - 随机跳过开局若干帧，让 AI 学会应对各种中途状态
    - 随机初始阳光 (模拟不同经济状况)
    - 随机选择是否预先放置一些植物
    """
    
    def __init__(
        self,
        diversify_prob: float = 0.3,  # 30% 概率使用多样化
        skip_frames_range: Tuple[int, int] = (0, 500),  # 跳过帧数范围
        random_sun_range: Tuple[int, int] = (50, 500),  # 随机阳光范围
        **kwargs
    ):
        super().__init__(**kwargs)
        self.diversify_prob = diversify_prob
        self.skip_frames_range = skip_frames_range
        self.random_sun_range = random_sun_range
        self._diversify_enabled = True
        
    def set_diversify(self, enabled: bool):
        """动态开关多样化训练"""
        self._diversify_enabled = enabled
        
    def reset(self, seed=None, options=None):
        """重置环境，有概率进行多样化"""
        obs, info = super().reset(seed=seed, options=options)
        
        # 决定是否多样化
        if self._diversify_enabled and random.random() < self.diversify_prob:
            self._apply_diversification()
            # 重新获取观测
            obs = self._get_observation()
        
        # 标记需要重置记忆
        info['needs_memory_reset'] = True
            
        return obs, info
    
    def _apply_diversification(self):
        """应用多样化策略"""
        if not self.hook_client or not self.hook_client.connected:
            return
            
        # 策略1: 随机跳过若干帧 (模拟中途状态)
        skip_frames = random.randint(*self.skip_frames_range)
        if skip_frames > 0:
            print(f"  🎲 多样化: 跳过 {skip_frames} 帧，模拟中途状态")
            # 执行若干次等待，让游戏自然推进
            wait_iterations = min(skip_frames // max(1, self.frame_skip), 50)  # 最多50次
            for i in range(wait_iterations):
                # 自动收集阳光
                if i % 5 == 0:  # 每5次收集一次
                    self.hook_client.collect()
                time.sleep(0.002)  # 2ms延迟，配合高速游戏
            
            # 再次收集阳光确保资源
            self.hook_client.collect()
            
            # 更新缓存的游戏状态
            if self.pvz:
                self._cached_game_state = self.pvz.get_game_state()


# =============================================================================
# 优化2.5: 记忆重置回调
# =============================================================================
class MemoryResetCallback(BaseCallback):
    """
    记忆重置回调 - 在每个episode开始时重置模型记忆
    
    功能:
    - 检测episode重置信号
    - 调用特征提取器的reset_memory方法
    - 支持多环境并行
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_starts = None
        
    def _on_step(self) -> bool:
        # 获取当前的dones标志
        if hasattr(self.locals, 'dones'):
            dones = self.locals['dones']
        elif 'done' in self.locals:
            dones = self.locals['done']
        else:
            return True
        
        # 检测哪些环境需要重置记忆
        if isinstance(dones, (list, np.ndarray)):
            dones = np.array(dones)
        else:
            dones = np.array([dones])
        
        # 对于刚完成的环境,下一步将是新episode的开始
        # 我们在这里标记,在下一次rollout开始时重置
        if hasattr(self, 'model') and hasattr(self.model.policy, 'features_extractor'):
            extractor = self.model.policy.features_extractor
            if hasattr(extractor, 'reset_memory') and np.any(dones):
                # 记录哪些环境需要重置
                if self.episode_starts is None:
                    self.episode_starts = dones
                else:
                    self.episode_starts = np.logical_or(self.episode_starts, dones)
                
        return True
    
    def _on_rollout_start(self) -> None:
        """在rollout开始时重置标记的环境的记忆"""
        if self.episode_starts is not None and hasattr(self.model.policy, 'features_extractor'):
            extractor = self.model.policy.features_extractor
            if hasattr(extractor, 'reset_memory'):
                # 获取环境数量
                n_envs = self.training_env.num_envs if hasattr(self.training_env, 'num_envs') else 1
                
                # 重置所有标记的环境的记忆
                if np.any(self.episode_starts):
                    try:
                        extractor.reset_memory(
                            batch_size=n_envs,
                            device=extractor.lstm_h.device
                        )
                        if self.verbose > 0:
                            print(f"已重置 {np.sum(self.episode_starts)} 个环境的记忆")
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"记忆重置失败: {e}")
                
                # 清空标记
                self.episode_starts = None


# =============================================================================
# 优化3: 失败优先学习回调
# =============================================================================
class FailurePrioritizedCallback(BaseCallback):
    """
    失败优先学习
    
    策略:
    - 记录失败的 episode
    - 失败后进行额外的经验回放/重训练
    - 增加失败 episode 的权重
    
    实现方式:
    - 监控 episode 结束
    - 失败时记录并触发额外训练
    """
    
    def __init__(
        self,
        failure_buffer_size: int = 100,  # 保存最近 100 次失败
        extra_train_on_failure: bool = True,  # 失败时是否额外训练
        failure_penalty_multiplier: float = 2.0,  # 失败惩罚倍数
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.failure_buffer_size = failure_buffer_size
        self.extra_train_on_failure = extra_train_on_failure
        self.failure_penalty_multiplier = failure_penalty_multiplier
        
        # 统计
        self.total_episodes = 0
        self.failure_episodes = 0
        self.success_episodes = 0
        self.recent_results = deque(maxlen=100)  # 最近 100 局结果
        
        # 失败经验缓冲 (可用于后续分析)
        self.failure_buffer: List[Dict[str, Any]] = []
        
    def _on_step(self) -> bool:
        # 检查是否有 episode 结束
        infos = self.locals.get("infos", [])
        
        for info in infos:
            if "episode" in info:
                self.total_episodes += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                
                # 判断是否失败 (奖励为负或长度很短)
                is_failure = ep_reward < 0 or ep_length < 100
                
                if is_failure:
                    self.failure_episodes += 1
                    self.recent_results.append(0)
                    self._on_failure(info, ep_reward, ep_length)
                else:
                    self.success_episodes += 1
                    self.recent_results.append(1)
                    
                # 打印统计
                if self.verbose and self.total_episodes % 10 == 0:
                    win_rate = sum(self.recent_results) / len(self.recent_results) * 100
                    print(f"Episode {self.total_episodes}: "
                          f"胜率 {win_rate:.1f}% | "
                          f"成功 {self.success_episodes} | "
                          f"失败 {self.failure_episodes}")
                    
        return True
    
    def _on_failure(self, info: Dict, reward: float, length: int):
        """处理失败 episode"""
        # 记录失败信息
        failure_info = {
            "episode": self.total_episodes,
            "reward": reward,
            "length": length,
            "timesteps": self.num_timesteps,
        }
        
        self.failure_buffer.append(failure_info)
        if len(self.failure_buffer) > self.failure_buffer_size:
            self.failure_buffer.pop(0)
            
        if self.verbose:
            print(f"  ❌ 失败 #{self.failure_episodes}: "
                  f"奖励={reward:.1f}, 长度={length}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_episodes": self.total_episodes,
            "success_episodes": self.success_episodes,
            "failure_episodes": self.failure_episodes,
            "win_rate": self.success_episodes / max(1, self.total_episodes),
            "recent_win_rate": sum(self.recent_results) / max(1, len(self.recent_results)),
        }


class DetailedLogCallback(BaseCallback):
    """
    详细日志记录器 - 记录动作、注意力和游戏状态的关系到 log 文件
    """
    def __init__(self, verbose=0, log_freq=100):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.last_log_step = 0
        self.episode_num = 0
        self.step_in_episode = 0
        self.reward_breakdown = {}

    def _on_step(self) -> bool:
        from utils.logger import log_attention_debug, log_game_state_debug
        
        self.step_in_episode += 1
        
        # 每 log_freq 步记录一次详细信息
        if self.num_timesteps - self.last_log_step >= self.log_freq:
            self._log_attention_coherence(self.locals.get('actions')[0])
            self._log_game_state()
            self.last_log_step = self.num_timesteps
            
        # 检测 episode 结束
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_num += 1
                ep_reward = info['episode']['r']
                ep_length = info['episode']['l']
                is_win = ep_reward > 0
                
                # 记录 episode 摘要
                from utils.logger import log_episode_summary
                log_episode_summary(
                    episode=self.episode_num,
                    steps=int(ep_length),
                    reward=ep_reward,
                    reward_breakdown=self.reward_breakdown,
                    win=is_win
                )
                
                # 重置
                self.step_in_episode = 0
                self.reward_breakdown = {}
                
                # 如果失败，记录失败分析
                if not is_win:
                    self._log_failure_analysis(info)
        
        return True

    def _log_attention_coherence(self, action):
        from utils.logger import log_attention_debug
        
        # 获取注意力权重
        attn_weights = None
        if hasattr(self.model.policy.features_extractor, 'last_attn_weights'):
            weights_tensor = self.model.policy.features_extractor.last_attn_weights
            if weights_tensor is not None:
                # 动态计算形状
                flat_size = weights_tensor.shape[1]
                cols = 9
                rows = flat_size // cols
                try:
                    attn_weights = weights_tensor[0].detach().cpu().numpy().reshape(rows, cols)
                except ValueError:
                    # 如果形状不匹配 (例如包含 CLS token)，尝试忽略额外的 token
                    if flat_size > rows * cols:
                        # 假设最后的是 grid tokens
                        grid_tokens = flat_size - (rows * cols)
                        # 这里简化处理，如果无法 reshape 就不记录
                        pass
        
        if attn_weights is None:
            return

        # 获取观测
        obs = self.locals.get('new_obs')
        
        # 计算统计量
        max_attn_pos = np.unravel_index(attn_weights.argmax(), attn_weights.shape)
        
        # 获取最大威胁位置 (从 grid 的 threat channel)
        threat_pos = None
        if obs and 'grid' in obs:
            grid = obs['grid'][0]  # (rows, 9, channels)
            if grid.shape[2] > 9:
                threat_map = grid[:, :, 9]
                threat_pos = np.unravel_index(threat_map.argmax(), threat_map.shape)
        
        # 记录到文件
        log_attention_debug(
            episode=self.episode_num,
            step=self.step_in_episode,
            attn_weights=attn_weights,
            action=action,
            max_pos=max_attn_pos,
            threat_pos=threat_pos
        )
    
    def _log_game_state(self):
        from utils.logger import log_game_state_debug
        
        # 从 info 获取游戏状态
        infos = self.locals.get('infos', [])
        if not infos or len(infos) == 0:
            return
        
        info = infos[0]
        log_game_state_debug(
            step=self.step_in_episode,
            sun=info.get('sun', 0),
            wave=info.get('wave', 0),
            zombies=info.get('zombie_count', 0),
            plants=info.get('plant_count', 0),
            lawnmowers=info.get('lawnmowers', []),
            is_paused=info.get('is_paused', False)
        )

    def _log_failure_analysis(self, info):
        from utils.logger import log_failure_analysis
        
        # 获取注意力分布
        attn_distribution = None
        if hasattr(self.model.policy.features_extractor, 'last_attn_weights'):
            weights_tensor = self.model.policy.features_extractor.last_attn_weights
            if weights_tensor is not None:
                # 动态计算形状
                flat_size = weights_tensor.shape[1]
                cols = 9
                rows = flat_size // cols
                try:
                    attn_weights = weights_tensor[0].detach().cpu().numpy().reshape(rows, cols)
                    # 按行求和得到每行的注意力分配
                    attn_distribution = attn_weights.sum(axis=1).tolist()
                except ValueError:
                    pass
        
        log_failure_analysis(
            episode=self.episode_num,
            step=self.step_in_episode,
            reason="Zombie reached house" if info['episode']['r'] < 0 else "Unknown",
            attn_distribution=attn_distribution
        )


# =============================================================================
# 自动收集回调 (使用代码 Patch 方式)
# =============================================================================
class AutoCollectCallback(BaseCallback):
    """
    使用代码 Patch 方式启用自动收集（类似 PvZ Tools）
    
    在训练开始时一次性启用 Patch，无需每步手动收集
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.patch_enabled = False

    def _on_training_start(self) -> bool:
        """训练开始时启用 Patch"""
        env = self._get_base_env()
        
        if env and hasattr(env, 'pvz') and env.pvz:
            try:
                success = env.pvz.enable_auto_collect_patch(enable=True)
                if success:
                    self.patch_enabled = True
                    print("[AutoCollect] ✓ 自动收集 Patch 已启用 (代码修改方式)")
                else:
                    print("[AutoCollect] ✗ 启用自动收集 Patch 失败")
            except Exception as e:
                print(f"[AutoCollect] ✗ 启用 Patch 时出错: {e}")
        
        return True
    
    def _on_training_end(self) -> bool:
        """训练结束时恢复 Patch"""
        if self.patch_enabled:
            env = self._get_base_env()
            
            if env and hasattr(env, 'pvz') and env.pvz:
                try:
                    env.pvz.enable_auto_collect_patch(enable=False)
                    print("[AutoCollect] 自动收集 Patch 已恢复")
                except Exception as e:
                    print(f"[AutoCollect] 恢复 Patch 时出错: {e}")
        
        return True
    
    def _get_base_env(self):
        """获取底层环境"""
        env = self.training_env
        
        # 处理 VecEnv 包装器
        if hasattr(env, 'venv'):
            env = env.venv
        elif hasattr(env, 'env'):
            env = env.env
            
        # 处理 DummyVecEnv / SubprocVecEnv
        if hasattr(env, 'envs'):
            env = env.envs[0]  # 获取第一个环境
        
        # 解包到最底层
        if hasattr(env, 'unwrapped'):
            env = env.unwrapped
        
        return env
    
    def _on_step(self) -> bool:
        """每步不需要额外操作，Patch 会自动生效"""
        return True


# =============================================================================
# 简洁监控回调 - 关键信息 + 调试
# =============================================================================
class SimpleMonitorCallback(BaseCallback):
    """简洁监控 - 失败次数 + 生存时长"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None
        self.episode_count = 0
        self.fail_count = 0  # 失败次数（僵尸进屋）
        self.total_len = 0   # 累计生存步数
        self.recent_lens = deque(maxlen=20)  # 最近20局生存时长
        self.best_len = 0    # 最佳生存时长
        self.last_print_step = 0
        
    def _on_training_start(self):
        self.start_time = time.time()
        print("\n" + "=" * 60)
        print("训练开始")
        print("=" * 60)
        
    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_count += 1
                ep_reward = info['episode']['r']
                ep_len = int(info['episode']['l'])
                
                # 更新统计
                self.total_len += ep_len
                self.recent_lens.append(ep_len)
                self.best_len = max(self.best_len, ep_len)
                
                # 检查是否失败（僵尸进屋）
                # 优先使用 info 中的 win 标记 (最准确)
                is_win = info.get('win', False)
                game_ended = info.get('game_ended', False)
                
                # 失败定义: 游戏结束且未胜利 (排除超时截断)
                # 注意: info['game_ended'] 在超时(TimeLimit)时通常为 False，只有真正胜负分出时为 True
                is_fail = game_ended and not is_win
                
                if is_fail:
                    self.fail_count += 1
                
                # 计算统计
                avg_len = self.total_len / self.episode_count
                recent_avg = sum(self.recent_lens) / len(self.recent_lens) if self.recent_lens else 0
                fail_rate = self.fail_count / self.episode_count * 100
                
                # 状态标记
                if is_win:
                    status = "🏆"  # 胜利
                elif is_fail:
                    status = "💀"  # 失败
                else:
                    status = "✓"  # 存活/超时
                
                # 简洁输出
                print(f"\n[EP{self.episode_count:>4}] {status} len={ep_len:<5} rew={ep_reward:>+6.1f} | "
                      f"近20局均:{recent_avg:>5.0f} 最佳:{self.best_len:<5} 失败率:{fail_rate:>4.1f}%")
                
                # 打印奖励详情 (如果有)
                if 'reward_details' in info:
                    print(f"[EP 奖励详情] 总分: {ep_reward:.1f}")
                    sorted_stats = sorted(info['reward_details'].items(), key=lambda x: abs(x[1]), reverse=True)
                    for k, v in sorted_stats:
                        if abs(v) > 0.1:
                            print(f"   - {k}: {v:.1f}")
        return True
    
    def _on_rollout_end(self):
        """每个 rollout 结束时输出所有关键调试信息"""
        # 每 5000 步输出一次调试信息
        if self.num_timesteps - self.last_print_step >= 5000:
            self.last_print_step = self.num_timesteps
            
            # 获取训练指标
            logger = self.model.logger
            if hasattr(logger, 'name_to_value'):
                metrics = logger.name_to_value
                
                # PPO 核心指标
                approx_kl = metrics.get('train/approx_kl', 0)
                entropy = metrics.get('train/entropy_loss', 0)
                clip_frac = metrics.get('train/clip_fraction', 0)
                explained_var = metrics.get('train/explained_variance', 0)
                
                # 损失指标
                policy_loss = metrics.get('train/policy_gradient_loss', 0)
                value_loss = metrics.get('train/value_loss', 0)
                loss = metrics.get('train/loss', 0)
                
                # 学习率
                learning_rate = metrics.get('train/learning_rate', 0)
                
                # 判断是否健康
                kl_ok = 0.005 < approx_kl < 0.1
                ent_ok = entropy < -0.5
                clip_ok = clip_frac > 0.01
                ev_ok = explained_var > 0.1
                
                elapsed = time.time() - self.start_time
                fps = self.num_timesteps / elapsed if elapsed > 0 else 0
                
                # 健康状态汇总
                health = sum([kl_ok, ent_ok, clip_ok, ev_ok])
                health_icon = "🟢" if health >= 3 else "🟡" if health >= 2 else "🔴"
                
                print(f"\n  {health_icon} [{self.num_timesteps:,}步 | {fps:.0f}it/s | lr={learning_rate:.2e}]")
                print(f"     kl={approx_kl:.2e}{'✓' if kl_ok else '⚠️'} "
                      f"ent={entropy:.3f}{'✓' if ent_ok else '⚠️'} "
                      f"clip={clip_frac:.3f}{'✓' if clip_ok else '⚠️'} "
                      f"ev={explained_var:.3f}{'✓' if ev_ok else '⚠️'}")
                print(f"     policy_loss={policy_loss:.4f} value_loss={value_loss:.2f} total_loss={loss:.2f}")
    
    def _on_training_end(self):
        avg_len = self.total_len / max(1, self.episode_count)
        fail_rate = self.fail_count / max(1, self.episode_count) * 100
        print(f"\n训练结束: {self.episode_count}局 | 平均生存:{avg_len:.0f}步 | 最佳:{self.best_len}步 | 失败率:{fail_rate:.1f}%")


# =============================================================================
# 综合速度监控回调
# =============================================================================
class AdvancedSpeedCallback(BaseCallback):
    """高级速度和状态监控 - 实时单行更新"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = None
        self.last_time = None
        self.last_steps = 0
        self.speed_history = deque(maxlen=10)
        self.episode_rewards = deque(maxlen=100)
        self.episode_count = 0
        self.win_count = 0
        
    def _on_training_start(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_steps = 0
        print("\n" + "=" * 70)
        print("训练开始 (实时监控)")
        print("=" * 70)
        
    def _on_step(self):
        # 检测 episode 结束
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_count += 1
                ep_reward = info['episode']['r']
                self.episode_rewards.append(ep_reward)
                # 简单判断胜利 (奖励 > 0)
                if ep_reward > 0:
                    self.win_count += 1
        return True
    
    def _on_rollout_end(self):
        current_time = time.time()
        current_steps = self.num_timesteps
        
        # 计算速度
        elapsed = current_time - self.last_time
        steps = current_steps - self.last_steps
        if elapsed > 0:
            speed = steps / elapsed
            self.speed_history.append(speed)
            avg_speed = sum(self.speed_history) / len(self.speed_history)
            total_elapsed = current_time - self.start_time
            
            # 估算剩余时间
            total_ts = self.model.num_timesteps if hasattr(self.model, '_total_timesteps') else 1000000
            if hasattr(self.model, '_total_timesteps'):
                total_ts = self.model._total_timesteps
            remaining = total_ts - current_steps
            eta_seconds = remaining / avg_speed if avg_speed > 0 else 0
            
            # 格式化时间
            if eta_seconds > 3600:
                eta_str = f"{int(eta_seconds//3600)}h{int((eta_seconds%3600)//60)}m"
            else:
                eta_str = f"{int(eta_seconds//60)}m{int(eta_seconds%60)}s"
            
            # 计算胜率
            win_rate = (self.win_count / self.episode_count * 100) if self.episode_count > 0 else 0
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0
            
            # 获取当前探索系数
            ent_coef = getattr(self.model, 'ent_coef', 0.01)
            
            # 实时单行输出 (使用 \r 覆盖)
            progress = current_steps / total_ts * 100
            status = (f"\r🎮 {current_steps:>8,}/{total_ts:,} ({progress:5.1f}%) | "
                     f"⚡{speed:>4.0f}it/s | "
                     f"🏆{win_rate:>5.1f}% | "
                     f"📊{avg_reward:>+6.1f} | "
                     f"🔍{ent_coef:.3f} | "
                     f"⏳{eta_str:>8}")
            print(status, end='', flush=True)
        
        self.last_time = current_time
        self.last_steps = current_steps
        
    def _on_training_end(self):
        total_time = time.time() - self.start_time
        avg_speed = self.num_timesteps / total_time if total_time > 0 else 0
        win_rate = (self.win_count / self.episode_count * 100) if self.episode_count > 0 else 0
        print()  # 换行
        print("\n" + "=" * 70)
        print(f"✅ 训练完成!")
        print(f"  📈 总步数: {self.num_timesteps:,}")
        print(f"  ⏱️ 总时间: {total_time/60:.1f} 分钟")
        print(f"  ⚡ 平均速度: {avg_speed:.0f} it/s")
        print(f"  🎯 总局数: {self.episode_count} | 胜率: {win_rate:.1f}%")
        print("=" * 70)


# =============================================================================
# 异步模型保存回调 (不阻塞主线程)
# =============================================================================
import threading
import copy

class AsyncSingleModelCallback(BaseCallback):
    """每隔一定步数异步保存模型，只保留一个文件 (覆盖)，避免阻塞训练"""
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_thread = None

    def _save_model_thread(self, model, save_path):
        """后台保存线程"""
        try:
            model.save(save_path)
            
            # 保存 VecNormalize 统计信息
            path_no_ext = os.path.splitext(save_path)[0]
            vec_norm_path = path_no_ext + "_vecnormalize.pkl"
            if hasattr(self.training_env, "save"):
                try:
                    self.training_env.save(vec_norm_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"\n[警告] 异步保存模型失败: {e}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # 如果上一次保存还没结束，跳过本次保存
            if self.save_thread and self.save_thread.is_alive():
                if self.verbose > 0:
                    print(f"\r⚠️ 上次保存尚未完成，跳过本次保存 (Step {self.num_timesteps})", end="", flush=True)
                return True

            # 启动新线程保存
            # 注意：SB3 的 save 方法通常是线程安全的，但为了绝对安全，
            # 最好是保存参数的副本，不过对于大模型这会消耗双倍内存。
            # 考虑到 PPO 训练时参数变化，直接 save 可能会有极小概率导致保存的模型参数不一致，
            # 但通常是可以接受的，且比阻塞好得多。
            self.save_thread = threading.Thread(
                target=self._save_model_thread,
                args=(self.model, self.save_path)
            )
            self.save_thread.start()

            if self.verbose > 0:
                print(f"\r💾 正在后台保存模型: {self.save_path} (Step {self.num_timesteps})", end="", flush=True)
        return True


# =============================================================================
# 热力图可视化回调
# =============================================================================
class HeatmapCallback(BaseCallback):
    """
    生成实时热力图 HTML
    """
    def __init__(self, save_path="heatmap.html", refresh_rate=10, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.refresh_rate = refresh_rate

    def _on_step(self) -> bool:
        if self.n_calls % self.refresh_rate == 0:
            try:
                # 获取最新的观测 (VecEnv -> Dict -> grid)
                obs = self.locals.get('new_obs')
                if obs and 'grid' in obs:
                    # 触发一次预测以更新注意力权重 (predict 会自动切换到 eval 模式)
                    # 这对于我们在 attention_extractor.py 中添加的钩子是必须的
                    self.model.predict(obs, deterministic=True)
                    
                    # 获取注意力权重
                    attn_weights = None
                    if hasattr(self.model.policy.features_extractor, 'last_attn_weights'):
                        # last_attn_weights 是 (B, 45)
                        weights_tensor = self.model.policy.features_extractor.last_attn_weights
                        if weights_tensor is not None:
                            # 动态计算形状
                            flat_size = weights_tensor.shape[1]
                            cols = 9
                            rows = flat_size // cols
                            try:
                                attn_weights = weights_tensor[0].detach().cpu().numpy().reshape(rows, cols)
                            except ValueError:
                                pass
                            
                            # Attention weights extracted (silent mode)

                    # 取第一个环境的观测
                    grid = obs['grid'][0]  # (rows, 9, 11)
                    self.generate_html(grid, attn_weights)
            except Exception as e:
                pass # 忽略错误，不影响训练
        return True

    def generate_html(self, grid, attn_map=None):
        # grid shape: (rows, 9, 11)
        rows, cols, channels = grid.shape
        
        # 通道 8: DPS (Blue)
        # 通道 9: Threat (Red)
        
        # 检查通道数，兼容旧配置
        if channels > 9:
            dps_map = grid[:, :, 8]
            threat_map = grid[:, :, 9]
        else:
            # 尝试从通道 2 和 4 提取 (旧版混合通道)
            # 但这很难分离，这里假设是新版配置
            dps_map = np.zeros((rows, cols))
            threat_map = np.zeros((rows, cols))

        # 准备 Attention HTML
        attn_html = ""
        if attn_map is not None:
            # 归一化以便显示 (0-1)
            # 注意力权重通常和为1，但单个值可能很小，或者如果用了 sigmoid 可能会不同
            # 这里我们做一个简单的最大值归一化来增强对比度
            max_val = attn_map.max()
            if max_val > 0:
                display_map = attn_map / max_val
            else:
                display_map = attn_map
            
            attn_html = f"""
            <div class="container">
                <h2 style="color: #ffff00;">Attention (AI关注点)</h2>
                <table>
                    {self._generate_table_rows(display_map, 'yellow')}
                </table>
            </div>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="refresh" content="1"> <!-- 每1秒自动刷新 -->
            <title>PVZ AI Heatmap</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: #222; color: #fff; display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }}
                .container {{ text-align: center; margin: 10px; }}
                table {{ border-collapse: collapse; margin: 10px auto; }}
                td {{ width: 50px; height: 50px; border: 1px solid #444; text-align: center; font-size: 10px; color: rgba(255,255,255,0.8); }}
                h2 {{ margin-bottom: 5px; font-size: 18px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2 style="color: #4da6ff;">DPS Heatmap (火力覆盖)</h2>
                <table>
                    {self._generate_table_rows(dps_map, 'blue')}
                </table>
            </div>
            <div class="container">
                <h2 style="color: #ff4d4d;">Threat Heatmap (僵尸威胁)</h2>
                <table>
                    {self._generate_table_rows(threat_map, 'red')}
                </table>
            </div>
            {attn_html}
        </body>
        </html>
        """
        
        with open(self.save_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _generate_table_rows(self, matrix, color_theme):
        rows_html = ""
        for r in range(matrix.shape[0]):
            rows_html += "<tr>"
            for c in range(matrix.shape[1]):
                val = matrix[r, c]
                # 颜色计算
                if color_theme == 'blue':
                    # 蓝色: rgba(0, 100, 255, alpha)
                    bg_color = f"rgba(0, 120, 255, {val:.2f})"
                elif color_theme == 'red':
                    # 红色: rgba(255, 50, 50, alpha)
                    bg_color = f"rgba(255, 50, 50, {val:.2f})"
                elif color_theme == 'yellow':
                    # 黄色: rgba(255, 255, 0, alpha)
                    bg_color = f"rgba(255, 255, 0, {val:.2f})"
                else:
                    bg_color = "rgba(0,0,0,0)"
                
                fmt = ".3f" if color_theme == 'yellow' else ".2f"
                rows_html += f'<td style="background-color: {bg_color}">{val:{fmt}}</td>'
            rows_html += "</tr>"
        return rows_html


# =============================================================================
# 主函数
# =============================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="高级 PVZ 训练 (含三大优化)")
    
    # 基础参数 - 稳定配置，适合初期训练
    parser.add_argument("--timesteps", "-t", type=int, default=500000, help="训练步数")
    parser.add_argument("--speed", "-s", type=float, default=cfg.game_speed, help=f"游戏速度 (默认: {cfg.game_speed}x, 最高10x)")
    parser.add_argument("--frameskip", "-f", type=int, default=4, help="帧跳过 (4=适中，视野更远)")
    parser.add_argument("--batch", "-b", type=int, default=1024, help="Batch size (GPU空闲，增大Batch)")
    parser.add_argument("--n_steps", "-n", type=int, default=4096, help="N steps (针对 RTX 3050 优化，约 80 秒/更新)")
    parser.add_argument("--n_epochs", type=int, default=20, help="训练轮数 (数据珍贵，多练几轮)")
    parser.add_argument("--net", type=str, default="large", 
                       choices=["small", "medium", "large", "xlarge", "huge"], help="网络大小")
    parser.add_argument("--port", "-p", type=int, default=12345, help="Hook 端口")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    
    # 优化1: 动态探索系数 - 降低初始值，加快收敛
    parser.add_argument("--start_ent", type=float, default=0.15, help="初始探索系数 (0.15=较少随机)")
    parser.add_argument("--end_ent", type=float, default=0.01, help="最终探索系数")
    parser.add_argument("--ent_decay", type=str, default="linear",
                       choices=["linear", "exponential", "cosine"], help="探索衰减方式")
    
    # 优化2: 多样化训练
    parser.add_argument("--diversify", type=float, default=0.0, help="多样化概率 (0-1)")
    parser.add_argument("--no_diversify", action="store_true", default=True, help="禁用多样化")
    
    # 优化3: 失败优先学习
    parser.add_argument("--no_failure_priority", action="store_true", help="禁用失败优先学习")
    
    # 加载与保存模型
    parser.add_argument("--load", type=str, default=LOAD_PATH, help=f"加载已有模型继续训练 (默认: {LOAD_PATH})")
    parser.add_argument("--save_path", type=str, default=MODEL_PATH, help=f"模型保存路径 (默认: {MODEL_PATH})")
    parser.add_argument("--auto_resume", action="store_true", default=True, help="自动加载最新模型继续训练 (默认启用)")
    parser.add_argument("--no_auto_resume", action="store_true", help="禁用自动加载，从零开始训练")
    parser.add_argument("--save_freq", type=int, default=10000, help="自动保存频率 (步)")

    # 注意力特征抽取器（默认开启，可用 --no_attn 关闭回退 MLP）
    parser.add_argument("--no_attn", action="store_true", help="禁用注意力特征抽取器，使用默认MLP")
    
    # 自动启动和注入
    parser.add_argument("--auto_start", action="store_true", default=True, help="自动启动游戏和注入DLL (默认启用)")
    parser.add_argument("--no_auto_start", action="store_true", help="禁用自动启动 (手动启动游戏和注入)")
    parser.add_argument("--game_path", type=str, default=DEFAULT_GAME_PATH, help=f"游戏路径 (默认: {DEFAULT_GAME_PATH})")
    parser.add_argument("--wait_time", type=float, default=3.0, help="游戏启动等待时间 (秒)")
    
    args = parser.parse_args()
    
    # 初始化日志系统 - 记录到 log 目录
    from utils.logger import get_logger, LogLevel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    logger = get_logger(level=LogLevel.DEBUG, file_path=log_file)
    print(f"\n[日志] 调试信息将保存到: {log_file}")
    
    print("\n" + "=" * 60)
    print("高级 PVZ 训练 - 三大优化")
    print("=" * 60)
    
    # 网络配置 - 激进版本
    net_configs = {
        "small": dict(pi=[256, 256], vf=[256, 256]),
        "medium": dict(pi=[512, 512, 256], vf=[512, 512, 256]),
        "large": dict(pi=[1024, 512, 256], vf=[1024, 512, 256]),
        "xlarge": dict(pi=[2048, 1024, 512, 256], vf=[2048, 1024, 512, 256]),
        "huge": dict(pi=[4096, 2048, 1024, 512], vf=[4096, 2048, 1024, 512]),  # 巨型网络
    }
    net_arch = net_configs[args.net]
    
    # 计算实际速度
    actual_game_speed = min(args.speed, 10.0)  # tick_ms 最小为1，最高10x
    effective_speed = actual_game_speed * args.frameskip
    
    print(f"\n配置:")
    print(f"  速度: {actual_game_speed}x | 帧跳过: {args.frameskip} | 网络: {args.net}")
    print(f"  Batch: {args.batch} | Steps: {args.n_steps} | LR: {args.lr}")
    print(f"  探索: {args.start_ent} → {args.end_ent}")
    
    # 自动启动游戏和注入 DLL
    if not args.no_auto_start:
        if not launch_game_and_inject(
            game_path=args.game_path,
            wait_time=args.wait_time,
            port=args.port
        ):
            print("无法启动游戏或注入 DLL，训练终止")
            print("  请使用 --no_auto_start 选项手动启动游戏和注入")
            return
    else:
        print("自动启动已禁用，请确保游戏已启动并注入 DLL")
    
    # 查找模型
    def find_latest_model():
        """查找最新的模型文件"""
        # 优先查找统一路径
        if os.path.exists(MODEL_PATH):
            return MODEL_PATH
        
        # 兼容旧版本：搜索其他可能的模型位置
        import glob
        patterns = [
            "models/advanced_*/final_model.zip",
            "models/*/final_model.zip",
            "models/*.zip",
        ]
        
        all_models = []
        for pattern in patterns:
            all_models.extend(glob.glob(pattern))
        
        if not all_models:
            return None
        
        # 按修改时间排序，返回最新的
        latest = max(all_models, key=os.path.getmtime)
        return latest
    
    # 决定是否加载模型
    load_path = args.load
    if not args.no_auto_resume and load_path is None:
        load_path = find_latest_model()
        if load_path:
            print(f"自动恢复: 找到最新模型 {load_path}")
        else:
            print(f"未找到已有模型，从零开始训练")

    # 创建环境
    print(f"建环境...")
    
    def make_env():
        if args.no_diversify:
            env = PVZEnv(
                hook_port=args.port,
                game_speed=args.speed,
                frame_skip=args.frameskip,
            )
        else:
            env = DiversifiedPVZEnv(
                hook_port=args.port,
                game_speed=args.speed,
                frame_skip=args.frameskip,
                diversify_prob=args.diversify,
            )
        env = ActionMasker(env, mask_fn)
        return env
    
    env = DummyVecEnv([make_env])
    
    # 1. 监控 (记录原始奖励)
    env = VecMonitor(env)
    
    # 2. 归一化 (关键优化: 稳定 PPO 训练)
    # 归一化观测值和奖励，防止梯度爆炸/消失
    if load_path:
        path_no_ext = os.path.splitext(load_path)[0]
        vec_norm_path = path_no_ext + "_vecnormalize.pkl"
        if os.path.exists(vec_norm_path):
            print(f"加载归一化统计: {vec_norm_path}")
            env = VecNormalize.load(vec_norm_path, env)
            env.training = True
            env.norm_reward = True
            env.norm_obs = True
        else:
            print(f"未找到归一化统计文件，创建新的归一化层")
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 3. 帧堆叠 (赋予时间感知)
    # 堆叠最近4帧，让AI能感知僵尸的移动速度和波次节奏
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    
    # 创建/加载模型
    print(f"创建模型...")
    
    policy_kwargs = dict(net_arch=net_arch)
    if not args.no_attn:
        policy_kwargs.update(
            features_extractor_class=PVZAttentionExtractor,
            features_extractor_kwargs=dict(
                hidden_size=128,  # 精简维度：GPU 不是瓶颈，优先减少计算量
                attn_heads=4,     # 精简注意力头数
                ff_dim=256,       # 精简前馈维度
                dropout=0.0,
                num_layers=2,     # 精简层数：2层足够，推理快 2 倍
            ),
        )
    
    if load_path:
        print(f"加载模型: {load_path}")
        try:
            model = MaskablePPO.load(load_path, env=env, device=device)
            # 更新超参数 - 使用线性衰减学习率
            model.learning_rate = linear_schedule(args.lr)
            
            # 关键修复: 如果加载的模型 n_steps 与当前参数不一致，需要调整 buffer 大小
            if model.n_steps != args.n_steps:
                print(f"模型 n_steps ({model.n_steps}) 与参数 ({args.n_steps}) 不一致，正在调整...")
                model.n_steps = args.n_steps
                model.rollout_buffer.buffer_size = args.n_steps
                model.rollout_buffer.reset()
                
            model.batch_size = args.batch
            model.n_epochs = args.n_epochs
        except ValueError as exc:
            # 观测空间不匹配（例如通道数从8改为11）时，自动从头训练
            if "Observation spaces do not match" in str(exc):
                print("观测空间已变更（例如网格通道数从8 -> 11），将从零开始重新训练")
                load_path = None
            else:
                raise

    if not load_path:
        # 使用余弦退火学习率 (10% 热身)
        lr_schedule = cosine_schedule(args.lr, warmup_steps=0.1)
        
        # 增强策略网络配置
        policy_kwargs.update(dict(
            activation_fn=torch.nn.GELU,  # 使用 GELU 激活函数 (比 ReLU 更平滑)
            optimizer_class=torch.optim.AdamW,  # 使用 AdamW 优化器 (更好的权重衰减)
            optimizer_kwargs=dict(weight_decay=1e-5),
        ))

        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            learning_rate=lr_schedule,
            n_steps=args.n_steps,
            batch_size=args.batch,
            n_epochs=args.n_epochs,
            gamma=0.995,  # 提高 Gamma (0.99 -> 0.995) 以关注更长远的未来
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.start_ent,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.03,  # 新增: 目标 KL 散度 (防止策略更新过猛)
            policy_kwargs=policy_kwargs,
            verbose=1,  # 关闭SB3日志，用自己的输出
            device=device,
        )
    
    # GPU 显存
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU 显存: {allocated:.2f} GB")
    
    # 设置回调
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/advanced_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    callbacks = [
        MemoryResetCallback(verbose=0),  # 🆕 记忆重置 (在episode开始时)
        AutoCollectCallback(),    # 强制自动收集
        SimpleMonitorCallback(),  # 简洁监控：连胜/连败
        AsyncSingleModelCallback(save_freq=args.save_freq, save_path=args.save_path, verbose=1), # 异步保存
        HeatmapCallback(save_path="heatmap.html", refresh_rate=10, verbose=1), # 实时热力图 (开启 verbose 以显示 Attention Debug)
        DetailedLogCallback(log_freq=500), # 新增：详细数据日志
    ]
    
    # 优化1: 动态探索系数
    dynamic_entropy = DynamicEntropyCallback(
        start_ent_coef=args.start_ent,
        end_ent_coef=args.end_ent,
        decay_type=args.ent_decay,
        total_timesteps=args.timesteps,
        warmup_steps=min(10000, args.timesteps // 10),
        verbose=0,  # 静默
    )
    callbacks.append(dynamic_entropy)
    
    # 开始训练
    print(f"开始训练...\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=False,  # 关闭进度条，用自己的输出
        )
    except KeyboardInterrupt:
        print("\n 训练被中断")
    finally:
        model.save(args.save_path)
        print(f"\n模型已保存: {args.save_path}")
        env.close()


if __name__ == "__main__":
    main()
