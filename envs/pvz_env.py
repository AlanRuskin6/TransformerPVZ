"""
PVZ Gym Environment for Survival Endless Day (草地无尽)
使用 Maskable PPO 训练 AI 玩 PVZ
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
import yaml
import time
import logging
import os

from hook_client import HookClient
from hook_client.injector import find_pvz_process, inject_dll
from core import PVZInterface, InterfaceMode
from data.plants import PlantType, PLANT_COST
from data.zombies import ZombieType, ZOMBIE_BASE_SPEED
from data.projectiles import ProjectileType, PROJECTILE_DAMAGE
import subprocess
import psutil

logger = logging.getLogger(__name__)

# 升级植物映射表 (升级植物ID -> 原植物ID)
UPGRADE_PLANTS = {
    40: 7,   # Gatling Pea -> Repeater
    41: 1,   # Twin Sunflower -> Sunflower
    42: 10,  # Gloom-shroom -> Fume-shroom
    43: 16,  # Cattail -> Lily Pad
    44: 39,  # Winter Melon -> Melon-pult
    45: 31,  # Gold Magnet -> Magnet-shroom
    46: 21,  # Spikerock -> Spikeweed
}

# 水生植物集合
AQUATIC_PLANTS = {16, 19, 24}  # Lily Pad, Tangle Kelp, Sea-shroom

_PLANT_POTENTIAL_BASE = {
    PlantType.SUNFLOWER: 0.65,
    PlantType.TWINSUNFLOWER: 0.75,
    PlantType.PEASHOOTER: 0.8,
    PlantType.REPEATER: 0.9,
    PlantType.SNOW_PEA: 0.85,
    PlantType.THREEPEATER: 1.0,
    PlantType.STARFRUIT: 0.85,
    PlantType.GATLINGPEA: 1.1,
    PlantType.CACTUS: 0.7,
    PlantType.TORCHWOOD: 0.7,
    PlantType.WALLNUT: 0.65,
    PlantType.TALLNUT: 0.9,
    PlantType.PUMPKIN: 0.9,
    PlantType.POTATO_MINE: 0.45,
    PlantType.SQUASH: 0.5,
    PlantType.JALAPENO: 0.5,
    PlantType.CHERRY_BOMB: 0.4,
}

_ZOMBIE_THREAT_PRIORITY = {
    ZombieType.GIGA_GARGANTUAR: 3.0,
    ZombieType.GARGANTUAR: 2.6,
    ZombieType.ZOMBOSS: 3.5,
    ZombieType.CATAPULT: 1.9,
    ZombieType.ZOMBONI: 1.8,
    ZombieType.FOOTBALL: 2.2,  # 提高橄榄球僵尸威胁度 (1.6 -> 2.2)
    ZombieType.BUCKETHEAD: 1.5, # 略微提高铁桶 (1.4 -> 1.5)
    ZombieType.SCREENDOOR: 1.3,
    ZombieType.POLEVAULTER: 1.3,
    ZombieType.POGO: 1.2,
    ZombieType.DIGGER: 1.4,
}


class PVZEnv(gym.Env):
    """
    PVZ Gym 环境
    
    观测空间:
        - grid: (5, 9, 4) 网格特征
        - global_features: (23,) 全局特征
    
    动作空间:
        - 0-449: 种植动作 (10卡 × 45格)
        - 450-494: 铲除动作 (45格)
        - 495: 等待
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(
        self,
        config_path: str = "config/training_config.yaml",
        render_mode: Optional[str] = None,
        frame_skip: int = 24,  # 每4帧决策一次，减少通信开销
        max_steps: int = 10000,
        game_speed: float = 20.0,  # 游戏速度倍率 (20x 超快)
        hook_port: int = 12345,  # Hook DLL 端口（多实例时指定不同端口）
        verbose: int = 1,  # 日志级别: 0=静默, 1=关键信息, 2=详细调试
    ):
        """
        初始化环境
        
        Args:
            config_path: 配置文件路径
            render_mode: 渲染模式
            frame_skip: 帧跳过 (每N帧决策一次)
            max_steps: 最大步数
            game_speed: 游戏速度倍率 (1.0-10.0, 默认5.0)
            hook_port: Hook DLL TCP 端口（多实例并行时使用不同端口）
            verbose: 日志级别 (0=静默, 1=关键信息, 2=详细调试)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.game_speed = max(0.1, min(100.0, game_speed))  # 支持最高 100x 速度
        self.hook_port = hook_port  # 保存端口
        self.verbose = verbose  # 日志级别
        
        # 加载配置
        self._load_config(config_path)
        
        # 游戏接口
        self.hook_client: Optional[HookClient] = None
        self.pvz: Optional[PVZInterface] = None
        
        # 游戏参数
        self.rows = self.config['game']['rows']  # 5
        self.cols = self.config['game']['cols']  # 9
        self.num_cards = self.config['cards']['slot_count']  # 10
        self.game_mode = self.config['game']['game_mode_id']  # 11
        self.initial_sun = self.config['game'].get('initial_sun')
        if self.initial_sun is None:
            self.initial_sun = 50  # 默认50
        # 无尽模式不设置目标波数，从游戏状态获取
        
        # 卡片信息
        self.card_plant_ids = [p['id'] for p in self.config['cards']['plants']]
        self.card_costs = [p['cost'] for p in self.config['cards']['plants']]
        
        # 动作空间: 10卡×45格 + 铲子×0格 + 等待 = 451
        self.n_plant_actions = self.num_cards * self.rows * self.cols  # 450
        self.n_shovel_actions = 0 # self.rows * self.cols  # 禁用铲子
        self.n_actions = self.n_plant_actions + self.n_shovel_actions + 1  # 451
        
        self.action_space = spaces.Discrete(self.n_actions)
        
        # 观测空间 (增强版)
        obs_config = self.config.get('observation_space', {})
        # 默认升级为 13 通道
        grid_shape = obs_config.get('grid', {}).get('shape', [5, 9, 13])
        global_dim = obs_config.get('global', {}).get('total_dim', 71)  # 增加新特征
        card_attr_shape = obs_config.get('card_attributes', {}).get('shape', [10, 7])  # 增加子弹类型
        
        self.observation_space = spaces.Dict({
            # 网格特征: 5行×9列×13通道 (增强)
            "grid": spaces.Box(
                low=0.0, high=1.0,
                shape=tuple(grid_shape),
                dtype=np.float32
            ),
            # 全局特征: 54维 (增强)
            "global_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(global_dim,),
                dtype=np.float32
            ),
            # 卡片属性特征: 10张卡×7属性 (Cost, HP, Damage, Range, Cooldown, Role, ProjectileType)
            "card_attributes": spaces.Box(
                low=0.0, high=1.0,
                shape=tuple(card_attr_shape),
                dtype=np.float32
            ),
            # 动作掩码
            "action_mask": spaces.Box(
                low=0, high=1,
                shape=(self.n_actions,),
                dtype=np.int8
            ),
        })
        
        # 初始化植物属性表 (用于构建 card_attributes)
        # [Cost, HP, Damage, Range, Cooldown, Role, ProjectileType]
        # Role: 0=Producer, 1=Attacker, 2=Defender, 3=Instant, 4=Support
        # ProjectileType: 0=None, 1=NormalPea, 2=IcePea, 3=Star, 4=Instant, 5=Special
        self.plant_stats = {
            0:  [100, 300,  20,  9, 750,  1, 1],  # Peashooter - Normal Pea
            1:  [50,  300,  0,   0, 750,  0, 0],  # Sunflower - No projectile
            2:  [150, 300,  1800,1, 5000, 3, 4],  # Cherry Bomb - Instant damage
            3:  [50,  4000, 0,   0, 3000, 2, 0],  # Wall-nut - No projectile
            4:  [25,  300,  1800,0, 3000, 3, 4],  # Potato Mine - Instant damage
            5:  [175, 300,  20,  9, 750,  1, 2],  # Snow Pea - Ice Pea (slow)
            6:  [150, 300,  1800,1, 750,  3, 5],  # Chomper - Special (devour)
            7:  [200, 300,  40,  9, 750,  1, 1],  # Repeater - Normal Pea (double)
            16: [25,  300,  0,   0, 750,  4, 0],  # Lily Pad - Support
            17: [50,  300,  1800,1, 3000, 3, 4],  # Squash - Instant
            19: [25,  300,  1800,1, 3000, 3, 4],  # Tangle Kelp - Instant
            20: [125, 300,  1800,9, 5000, 3, 4],  # Jalapeno - Instant
            21: [100, 300,  20,  1, 750,  1, 0],  # Spikeweed - Ground attack
            22: [175, 300,  0,   0, 750,  4, 5],  # Torchwood - Special (buff)
            29: [125, 300,  20,  9, 750,  1, 3],  # Starfruit - Star projectile
            34: [100, 300,  20,  9, 750,  1, 5],  # Kernel-pult - Butter (control)
            39: [300, 300,  80,  9, 750,  1, 3],  # Melon-pult - Heavy AOE
            43: [225, 300,  20,  9, 5000, 1, 5],  # Cattail - Homing
            # 默认值
            -1: [100, 300,  20,  1, 750,  1, 1],
        }
        
        # 奖励配置
        self.rewards = self.config['rewards']
        
        # 状态变量
        self.steps = 0
        self.total_reward = 0.0
        self.last_sun = 0
        self.last_zombie_count = 0
        self.last_plant_count = 0
        self.last_wave = 0
        self.sunflower_count = 0
        self._cached_game_state = None  # 缓存游戏状态，减少内存读取
        self.last_potential = 0.0
        
        # 小推车状态跟踪 (每行是否还有小推车)
        self.lawnmower_available = [True] * self.rows
        
        # 连胜追踪
        self.win_streak = 0  # 当前连胜数
        self.max_win_streak = 0  # 历史最高连胜

        # 统计
        self.zombies_killed = 0
        self.plants_lost = 0

        # 击杀效率追踪 (位置 -> 击杀数)
        self.kill_heatmap = np.zeros((self.rows, self.cols), dtype=np.float32)

        # 游戏路径配置 (用于自动重启)
        self.pvz_exe_path = self.config.get('game', {}).get('exe_path', None)
        self._last_restart_time = 0  # 上次重启时间，防止频繁重启
        
        # 回合状态追踪
        self._episode_win = None  # 回合是否胜利
        self._victory_printed = False  # 是否已打印胜利信息
        self._no_zombie_steps = 0  # 连续无僵尸的步数
        
    def _load_config(self, config_path: str):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def _is_pvz_running(self) -> bool:
        """检查 PVZ 进程是否在运行"""
        return find_pvz_process() is not None

    def _restart_pvz(self) -> bool:
        """
        重启 PVZ 游戏并重新注入 DLL

        Returns:
            True if successful
        """
        # 防止频繁重启 (至少间隔 30 秒)
        now = time.time()
        if now - self._last_restart_time < 30:
            print(f"[DEBUG] 距离上次重启不足30秒，跳过")
            return False
        self._last_restart_time = now

        print(f"[DEBUG] 尝试重启 PVZ 游戏...")

        # 先关闭现有连接
        if self.hook_client:
            self.hook_client.disconnect()
            self.hook_client = None
        if self.pvz:
            self.pvz = None

        # 检查游戏是否还在运行
        pid = find_pvz_process()
        if pid:
            print(f"[DEBUG] PVZ 进程存在 (PID={pid})，尝试重新注入 DLL...")
        else:
            # 游戏不在运行，尝试启动
            if self.pvz_exe_path and os.path.exists(self.pvz_exe_path):
                print(f"[DEBUG] 启动游戏: {self.pvz_exe_path}")
                try:
                    subprocess.Popen([self.pvz_exe_path], cwd=os.path.dirname(self.pvz_exe_path))
                    # 等待游戏启动
                    for _ in range(30):  # 最多等待 30 秒
                        time.sleep(1)
                        if find_pvz_process():
                            print(f"[DEBUG] 游戏已启动!")
                            break
                    else:
                        print(f"[DEBUG] 游戏启动超时!")
                        return False
                except Exception as e:
                    print(f"[DEBUG] 启动游戏失败: {e}")
                    return False
            else:
                print(f"[DEBUG] 游戏路径未配置或不存在: {self.pvz_exe_path}")
                print(f"[DEBUG] 请在 config/training_config.yaml 中设置 game.exe_path")
                return False

        # 等待游戏完全加载
        time.sleep(3)

        # 注入 DLL
        print(f"[DEBUG] 注入 Hook DLL...")
        if not inject_dll():
            print(f"[DEBUG] DLL 注入失败!")
            return False

        # 等待 DLL 初始化
        time.sleep(2)

        print(f"[DEBUG] 重启完成!")
        return True
    
    def _connect(self, max_retries: int = 3) -> bool:
        """连接游戏，带重试机制和自动重启"""
        if self.hook_client is None:
            self.hook_client = HookClient(port=self.hook_port, timeout=10.0)  # 增加超时到10秒

        if not self.hook_client.connected:
            print(f"[PVZEnv] 正在连接 Hook DLL (port: {self.hook_port})...")

            # 重试连接
            for attempt in range(max_retries):
                if self.hook_client.connect():
                    print(f"[PVZEnv] Hook 连接成功 (port: {self.hook_port})!")
                    break

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2秒, 4秒, 6秒...
                    print(f"[PVZEnv] 连接失败，{wait_time}秒后重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
            else:
                # 所有重试都失败，尝试自动重启游戏
                print(f"[PVZEnv] 连接失败，尝试自动重启游戏...")
                if self._restart_pvz():
                    # 重启成功，重新创建客户端并连接
                    self.hook_client = HookClient(port=self.hook_port, timeout=10.0)
                    if self.hook_client.connect():
                        print(f"[PVZEnv] 重启后连接成功!")
                    else:
                        print(f"[PVZEnv] ERROR: 重启后仍无法连接!")
                        return False
                else:
                    print(f"[PVZEnv] ERROR: 无法连接到 Hook DLL (port: {self.hook_port})!")
                    print("[PVZEnv] 请确保:")
                    print("  1. PVZ 游戏已启动")
                    print("  2. 已运行 python inject.py 注入 DLL")
                    return False
        
        if self.pvz is None:
            self.pvz = PVZInterface(mode=InterfaceMode.HOOK)
            if not self.pvz.attach():
                if self.verbose >= 1:
                    print("[PVZEnv] 错误: 无法附加到PVZ进程")
                return False
        elif not self.pvz.is_attached():
            if self.verbose >= 2:
                print("[PVZEnv] 重新附加进程...")
            if not self.pvz.attach():
                if self.verbose >= 1:
                    print("[PVZEnv] 错误: 重新附加失败")
                return False
        
        return True
    
    def _set_game_speed(self, speed: float):
        """设置游戏速度（仅在游戏中有效）"""
        if self.hook_client and self.hook_client.connected:
            self.hook_client.set_game_speed(speed)
    
    def _restore_normal_speed(self):
        """恢复正常游戏速度"""
        self._set_game_speed(1.0)
        
        if self.pvz is None:
            self.pvz = PVZInterface(mode=InterfaceMode.HOOK)
            if not self.pvz.attach():
                if self.verbose >= 1:
                    print("[PVZEnv] 错误: 无法附加到PVZ进程")
                return False
        
        return True
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境
        
        Returns:
            observation: 初始观测
            info: 额外信息
        """
        super().reset(seed=seed)
        
        # 连接游戏
        if not self._connect():
            raise RuntimeError("Failed to connect to PVZ game. 请先启动游戏并注入DLL!")
        
        # 检查当前UI状态
        ui = self.hook_client.get_ui()
        
        if ui == -1:
            # 尝试直接用内存读取判断是否在游戏中
            if self.pvz and self.pvz.is_in_game():
                ui = 3
            else:
                raise RuntimeError("无法确定游戏状态")
        
        # 处理各种UI状态 (静默处理)
        if ui == 0:  # 加载中 (Click to Start 界面)
            self.hook_client.click(400, 300)
            time.sleep(0.5)
            for _ in range(50):
                time.sleep(0.2)
                ui = self.hook_client.get_ui()
                if ui != 0:
                    break
                if _ % 10 == 0:
                    self.hook_client.click(400, 300)
        
        if ui == 7:  # 选项界面
            self.hook_client._send_command("CLOSEOPTS")
            time.sleep(0.5)
            ui = self.hook_client.get_ui()
        
        if ui == 4:  # ZOMBIES_WON
            for attempt in range(10):
                self.hook_client.click_scaled(280, 370)
                time.sleep(0.3)
                new_ui = self.hook_client.get_ui()
                if new_ui != 4:
                    ui = new_ui
                    break
            if ui == 4:
                self.hook_client.back_to_main()
                time.sleep(1.0)
                ui = self.hook_client.get_ui()
        
        if ui == 5:  # AWARD
            self.hook_client.back_to_main()
            time.sleep(1.0)
            ui = self.hook_client.get_ui()
        
        if ui == 3:  # 游戏中
            self.hook_client.back_to_main()
            time.sleep(1.0)
            ui = self.hook_client.get_ui()
        
        if ui == 2:  # 选卡界面
            # 等待卡片界面就绪（最多3秒）
            for _ in range(30):
                if self.hook_client.is_card_select_ready():
                    break
                time.sleep(0.1)
            
            self.hook_client.select_cards(self.card_plant_ids)
            time.sleep(len(self.card_plant_ids) * 0.1 + 0.5)
            self.hook_client.rock()
            time.sleep(1.0)
            ui = self.hook_client.get_ui()
        
        if ui == 1:  # 主菜单
            if not self.hook_client.auto_start_game(
                mode=self.game_mode,
                cards=self.card_plant_ids,
                timeout=10.0
            ):
                time.sleep(0.5)
                ui = self.hook_client.get_ui()
                if ui == 2:
                    # 等待卡片界面就绪（最多3秒）
                    for _ in range(30):
                        if self.hook_client.is_card_select_ready():
                            break
                        time.sleep(0.1)
                    self.hook_client.select_cards(self.card_plant_ids)
                    time.sleep(len(self.card_plant_ids) * 0.1 + 0.5)
                    self.hook_client.rock()
                    time.sleep(1.0)
        
        # 等待游戏开始
        self.hook_client.wait_for_ui(3, timeout=10.0)
        
        # 设置初始阳光 (如果是教学模式)
        if self.initial_sun != 50:
            # 等待一小会儿确保内存结构已初始化
            time.sleep(0.5)
            if self.pvz and self.pvz.set_sun(self.initial_sun):
                # 验证是否设置成功
                time.sleep(0.1)
                state = self.pvz.get_game_state()
                current_sun = state.sun if state else -1
                if self.verbose >= 1:
                    print(f"[PVZEnv] 初始阳光设置: 目标={self.initial_sun}, 实际={current_sun}")
            else:
                if self.verbose >= 1:
                    print(f"[PVZEnv] 警告: 设置初始阳光失败")

        # 设置游戏速度
        if self.game_speed != 1.0:
            tick_ms = max(1, int(10.0 / self.game_speed))
            if not self.hook_client.set_tick_ms(tick_ms):
                self.hook_client.set_game_speed(min(self.game_speed, 10.0))
        
        # 发送 FLUSH 命令确保没有残留的命令
        self.hook_client._send_command("FLUSH")
        time.sleep(0.1)
        
        # 重置状态
        self.steps = 0
        self.total_reward = 0.0
        self.zombies_killed = 0
        self.plants_lost = 0
        self._episode_win = None  # 重置胜负状态
        self._victory_printed = False  # 重置胜利打印标记
        self._no_zombie_steps = 0  # 重置无僵尸计数
        
        # 重置热力图
        if hasattr(self, 'kill_heatmap'):
            self.kill_heatmap.fill(0)
        
        # 重置小推车状态
        self.lawnmower_available = [True] * self.rows
        
        # 获取初始状态并缓存
        game_state = self.pvz.get_game_state()
        self._cached_game_state = game_state  # 缓存供第一步使用
        if game_state:
            self.last_sun = game_state.sun
            self.last_zombie_count = len(game_state.zombies)
            self.last_zombies_state = list(game_state.zombies)  # 保存僵尸列表副本用于追踪击杀位置
            self.last_plant_count = len(game_state.plants)
            self.last_wave = game_state.wave
            self.sunflower_count = sum(1 for p in game_state.plants if p.type == PlantType.SUNFLOWER)
            # 初始化小推车状态
            for lm in game_state.lawnmowers:
                if 0 <= lm.row < self.rows:
                    self.lawnmower_available[lm.row] = (lm.state == 1)  # READY=1

        self.last_potential = self._calculate_potential(game_state)
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: 动作索引
        
        Returns:
            observation: 新观测
            reward: 奖励
            terminated: 是否结束 (胜利/失败)
            truncated: 是否截断 (超时)
            info: 额外信息
        """

        # 轻量级保活：如果 Hook/TCP 断开或附加丢失，尝试即时重连/重新附加
        if self.hook_client:
            if not self.hook_client.connected:
                print(f"[DEBUG] Step {self.steps}: Hook连接断开，尝试重连...")
                # 等待一小段时间再重连，避免立即重试失败
                time.sleep(0.5)
                for retry in range(3):
                    print(f"[DEBUG] 重连尝试 {retry + 1}/3...")
                    if self.hook_client.connect():
                        print(f"[DEBUG] Hook重连成功!")
                        break
                    print(f"[DEBUG] 重连失败，等待1秒...")
                    time.sleep(1.0)
                else:
                    print(f"[DEBUG] 3次重连均失败!")

        if self.pvz and not self.pvz.is_attached():
            print(f"[DEBUG] Step {self.steps}: 游戏进程附加丢失，尝试重新附加...")
            time.sleep(0.5)
            if self.pvz.attach():
                print(f"[DEBUG] 进程重新附加成功!")
            else:
                print(f"[DEBUG] 进程重新附加失败!")

        self.steps += 1
        reward = 0.0
        
        # 调试：记录本步奖励详情
        step_reward_details = {}
        
        # 统计动作类型
        if not hasattr(self, '_action_stats'):
            self._action_stats = {'plant': 0, 'shovel': 0, 'wait': 0, 'invalid': 0}
        
        # 解析动作 - 使用上一帧的缓存状态检查
        success = self._execute_action(action, self._cached_game_state)
        
        if not success:
            # 无效动作：小惩罚（鼓励探索但不鼓励乱点）
            r_invalid = self.rewards.get('invalid_action', -0.01)
            reward += r_invalid
            step_reward_details['invalid'] = r_invalid
            self._action_stats['invalid'] += 1
        else:
            if action == self.n_actions - 1:  # 等待动作
                self._action_stats['wait'] += 1
                # 等待时如果阳光充足，给惩罚（鼓励种植）
                # 修正：阈值提高到 300，允许 AI 攒钱买双发/寒冰 (原 100)
                if self._cached_game_state:
                    sun = self._cached_game_state.sun
                    if sun >= 300:  # 只有阳光溢出才惩罚
                        reward += -0.02
                        step_reward_details['wait_with_sun'] = -0.02
            elif action < self.n_plant_actions:
                self._action_stats['plant'] += 1
                # 成功种植有奖励（在 _compute_reward 中）
            else:
                self._action_stats['shovel'] += 1
        
        # 帧跳过 - 零延迟（游戏速度已加速，去除等待减少瓶颈）
        # time.sleep(0.002 * self.frame_skip)  # 去除延迟，依赖游戏自身节奏
        
        # 自动收集阳光 - 每步都收集，高速下防止漏收
        if self.hook_client:
            res = self.hook_client.collect()
            if self.verbose >= 1 and res > 0:
                print(f"[PVZEnv] Step {self.steps}: 收集了 {res} 个物品")
            elif self.verbose >= 2:
                print(f"[PVZEnv] Step {self.steps}: collect() returned {res}")
        


        # 获取游戏状态（每步只读取一次）
        game_state = self.pvz.get_game_state() if self.pvz else None
        self._cached_game_state = game_state  # 缓存供下一步使用
        
        terminated = False
        truncated = False
        self._episode_win = None  # 记录胜负
        
        if game_state is None:
            # 游戏状态读取失败
            truncated = True
        else:
            # [调试] 输出关键状态
            level_end_cd = getattr(game_state, 'level_end_countdown', -999)
            if self.verbose >= 2 or (self.steps % 50 == 0 and self.verbose >= 1):
                print(f"[Step {self.steps}] wave={game_state.wave}/{game_state.total_waves}, level_end_countdown={level_end_cd}, zombies={len(game_state.zombies)}")
                if self.verbose >= 2:
                    print(f"  game_state类型={type(game_state).__name__}, 有level_end_countdown属性={hasattr(game_state, 'level_end_countdown')}")
            
            # 计算奖励
            r_compute, compute_details, potential = self._compute_reward_debug(game_state)
            reward += r_compute
            step_reward_details.update(compute_details)
            
            # 检查终止条件
            terminated, win = self._check_terminated(game_state)
            
            if terminated:
                self._episode_win = win  # 记录胜负
                # 游戏结束，先恢复正常速度
                self._restore_normal_speed()
                
                if win:
                    # 连胜奖励：越连胜奖励越高
                    self.win_streak += 1
                    self.max_win_streak = max(self.max_win_streak, self.win_streak)
                    
                    base_win = self.rewards.get('game_win', 3.0)
                    streak_bonus = self.win_streak * self.rewards.get('streak_bonus', 1.0)
                    # 无僵尸空窗计数，用于额外胜利兜底
                    self._no_zombie_steps = 0
                    step_reward_details['win'] = base_win + streak_bonus
                else:
                    # 失败，连胜归零
                    self.win_streak = 0
                    r_lose = self.rewards.get('game_lose', -3.0)
                    reward += r_lose
                    step_reward_details['lose'] = r_lose
                    # 失败时主动返回主菜单，准备下一次 reset
                    self._handle_game_failure()
            
            # 更新状态
            self._update_last_state(game_state, potential)
        
        # 检查超时
        if self.steps >= self.max_steps:
            truncated = True

        # 如果本步截断或终止，打印一次原因（便于定位4000步附近的停顿）
        if (terminated or truncated) and self.verbose >= 2:
            reason = "状态读取失败" if game_state is None else "达到最大步数" if self.steps >= self.max_steps else "其他"
            print(f"[调试] 回合结束: {reason} (步数={self.steps}, 终止={terminated}, 截断={truncated})")
        
        self.total_reward += reward
        
        # 累计本局奖励详情
        if not hasattr(self, '_episode_reward_stats'):
            self._episode_reward_stats = {}
        for k, v in step_reward_details.items():
            self._episode_reward_stats[k] = self._episode_reward_stats.get(k, 0.0) + v
            
        # 打印本局总结（仅在 verbose >= 1 时显示）
        if (terminated or truncated) and self.verbose >= 1:
            print(f"[回合结束] 总奖励: {self.total_reward:.1f}, 步数: {self.steps}")
            if self.verbose >= 2:
                # 详细模式：显示奖励详情
                sorted_stats = sorted(self._episode_reward_stats.items(), key=lambda x: abs(x[1]), reverse=True)
                for k, v in sorted_stats:
                    if abs(v) > 0.1:
                        print(f"   {k}: {v:.1f}")
            self._episode_reward_stats = {}  # 重置
        
        # 传入已获取的 game_state，避免重复读取
        obs = self._get_observation(game_state)
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

    def _calculate_potential(self, game_state) -> float:
        """
        计算势能函数 (Potential Function)，用于奖励塑形
        利用阳光、植物布局、小推车、波数和僵尸威胁构建更平滑的潜势。
        """
        if game_state is None:
            return 0.0

        cfg = self.rewards.get('potential', {})
        sun_scale = cfg.get('sun_scale', 0.18)
        sun_cap = max(1.0, cfg.get('sun_cap', 400.0))
        plant_scale = cfg.get('plant_scale', 1.0)
        spread_bonus = cfg.get('spread_bonus', 0.15)
        lawnmower_scale = cfg.get('lawnmower_scale', 2.0)
        zombie_scale = cfg.get('zombie_threat_scale', 0.9)
        zombie_distance_bonus = cfg.get('zombie_distance_bonus', 1.2)
        wave_scale = cfg.get('wave_scale', 0.25)

        sun_potential = sun_scale * (game_state.sun / (game_state.sun + sun_cap))

        plant_potential = 0.0
        column_coverage = set()
        row_counts = [0] * self.rows
        max_col_index = max(1, self.cols - 1)

        for plant in game_state.plants:
            base_value = _PLANT_POTENTIAL_BASE.get(plant.type, 0.35)
            column_bias = 1.0 - (plant.col / max_col_index) if max_col_index > 0 else 0.5
            column_factor = 1.0 + 0.3 * column_bias

            hp_max = getattr(plant, 'hp_max', None) or getattr(plant, 'hpMax', None)
            hp_ratio = 1.0
            if hp_max and hp_max > 0:
                hp_ratio = max(0.0, min(1.0, plant.hp / hp_max))

            plant_potential += base_value * hp_ratio * column_factor
            column_coverage.add(plant.col)
            if 0 <= plant.row < self.rows:
                row_counts[plant.row] += 1

        coverage_score = sum(min(1.0, count / 3.0) for count in row_counts) / max(1, self.rows)
        coverage_bonus = coverage_score * spread_bonus
        column_bonus = (len(column_coverage) / max(1, self.cols)) * spread_bonus

        lawnmowers_ready = 0
        for lm in getattr(game_state, 'lawnmowers', []):
            if getattr(lm, 'state', 0) == 1:
                lawnmowers_ready += 1

        total_waves = getattr(game_state, 'total_waves', 0)
        wave_progress = (game_state.wave / max(1, total_waves)) if total_waves > 0 else 0.0
        wave_component = wave_progress * wave_scale

        zombie_threat = 0.0
        for zombie in game_state.zombies:
            if getattr(zombie, 'hp', 0) <= 0:
                continue

            dist = max(0.0, min(1.0, 1.0 - (zombie.x + 70.0) / 900.0))
            base_threat = 0.35 + dist * zombie_distance_bonus
            try:
                z_type = ZombieType(zombie.type)
            except Exception:
                z_type = None
            type_multiplier = _ZOMBIE_THREAT_PRIORITY.get(z_type, 1.0)
            speed_factor = 1.0 + 0.12 * (getattr(zombie, 'speed', 0.0) or 0.0)
            zombie_threat += zombie_scale * base_threat * type_multiplier * speed_factor

        potential = (
            sun_potential
            + (plant_potential * plant_scale)
            + coverage_bonus
            + column_bonus
            + lawnmowers_ready * lawnmower_scale
            + wave_component
            - zombie_threat
        )

        return potential

    def _compute_reward_debug(self, game_state) -> Tuple[float, Dict[str, float], float]:
        """计算奖励并返回详情 (调试版)"""
        reward = 0.0
        details = {}
        
        # 存活奖励
        r_survival = self.rewards.get('survival_per_step', 0.0)
        if r_survival != 0:
            reward += r_survival
            details['survival'] = r_survival
            
        # === 新增机制: 防线覆盖奖励 ===
        # 鼓励 AI 在每一行都部署"非经济类"植物，防止被单点突破
        covered_rows = set()
        for plant in game_state.plants:
            # 排除向日葵(1)和阳光菇(4)，其他都算防守力量
            if plant.type not in [1, 4]: 
                covered_rows.add(plant.row)
        
        # 每覆盖一行，每步给一点点奖励 (0.002 * 5 = 0.01/step)
        r_coverage = len(covered_rows) * 0.002
        if r_coverage > 0:
            reward += r_coverage
            details['coverage'] = r_coverage

        # === 新增机制: 僵尸逼近压力 ===
        # 如果有僵尸越过警戒线 (X < 200)，给予持续压力惩罚
        proximity_penalty = 0.0
        for zombie in game_state.zombies:
            if zombie.x < 200:
                # 距离越近惩罚越大，最大约 -0.005/step
                proximity_penalty += ((200 - zombie.x) / 200.0) * 0.005
        
        if proximity_penalty > 0:
            reward -= proximity_penalty
            details['proximity'] = -proximity_penalty
        
        # 阳光收集奖励
        sun_diff = game_state.sun - self.last_sun
        if sun_diff > 0:
            r_sun = sun_diff * self.rewards.get('sun_collect', 0.001)
            reward += r_sun
            details['sun'] = r_sun
        
        # 僵尸击杀奖励与热力图更新
        current_zombie_indices = {z.index for z in game_state.zombies}
        killed_zombies = []
        
        # 确保 last_zombies_state 存在
        if not hasattr(self, 'last_zombies_state'):
            self.last_zombies_state = []

        for old_z in self.last_zombies_state:
            if old_z.index not in current_zombie_indices:
                # 僵尸消失了
                # 排除到达终点的僵尸 (x < -30，通常进家是 -50 左右)
                if old_z.x > -30:
                    killed_zombies.append(old_z)
                    
                    # 更新击杀热力图
                    # 找到最近的格子
                    r = int(old_z.row)
                    c = int((old_z.x - 10) / 80)  # 估算列
                    if 0 <= r < 5 and 0 <= c < 9:
                        self.kill_heatmap[r, c] += 1.0
        
        # 热力图衰减 (每步衰减，保持动态性)
        self.kill_heatmap *= 0.995
        
        if killed_zombies:
            count = len(killed_zombies)
            r_kill = count * self.rewards['zombie_kill'].get('normal', 0.3)
            reward += r_kill
            self.zombies_killed += count
            details['kill'] = r_kill
        
        # 波次完成奖励
        if game_state.wave > self.last_wave:
            r_wave = self.rewards.get('wave_complete', 1.0)
            reward += r_wave
            details['wave'] = r_wave
        
        # 小推车状态检测
        for lm in game_state.lawnmowers:
            if 0 <= lm.row < self.rows:
                was_available = self.lawnmower_available[lm.row]
                is_available = (lm.state == 1)
                
                if was_available and not is_available:
                    penalty = self.rewards.get('lawnmower_triggered', -3.0)
                    reward += penalty
                    self.lawnmower_available[lm.row] = False
                    details['lawnmower'] = penalty
        
        # 种植奖励
        plant_diff = len(game_state.plants) - self.last_plant_count
        if plant_diff > 0:
            new_plant = None
            if game_state.plants:
                new_plant = game_state.plants[-1]
            
            if new_plant:
                base_reward = 0.0
                if new_plant.type == PlantType.SUNFLOWER:
                    base_reward = self.rewards.get('plant_sunflower', 0.3)
                    if new_plant.col <= 2: base_reward += 0.2
                    elif new_plant.col >= 5: base_reward -= 0.2
                elif new_plant.type == 16:  # Lily Pad (睡莲)
                    base_reward = 0.4  # 鼓励在水域铺睡莲
                    # 在水域中间行(第3、4行)种植睡莲给额外奖励
                    if 2 <= new_plant.row <= 3:
                        base_reward += 0.2
                elif new_plant.type == PlantType.WALLNUT:
                    base_reward = self.rewards.get('plant_wall', 0.2)
                    if new_plant.col >= 5: base_reward += 0.2
                elif new_plant.type == 30: # PUMPKIN (南瓜头)
                    # 南瓜头逻辑：鼓励保护核心植物，严惩后排空放
                    base_reward = 0.1
                    
                    # 检查同位置是否有其他植物 (被保护对象)
                    inner_plant = None
                    for p in game_state.plants:
                        if p.row == new_plant.row and p.col == new_plant.col and p.type != 30:
                            inner_plant = p
                            break
                    
                    if inner_plant:
                        # 成功套盾：给予高额奖励
                        base_reward += 0.3
                        # 保护核心植物 (双发/火炬/高坚果/杨桃) 额外加分
                        if inner_plant.type in [7, 22, 23, 29]: 
                            base_reward += 0.3
                        # 保护前排受威胁植物
                        if new_plant.col >= 4:
                            base_reward += 0.2
                    else:
                        # 空放南瓜头 (没有套在植物上)
                        if new_plant.col >= 7: 
                            # 只有在最前排当临时墙用才给一点分
                            base_reward += 0.1
                        else:
                            # 在后排空地乱放南瓜头：给予惩罚
                            base_reward -= 0.3
                else:
                    base_reward = self.rewards.get('plant_attacker', 0.25)
                    # 额外奖励：如果种的是昂贵的重武器 (Cost > 150)
                    # 34=Melon-pult, 43=Cattail, 35=Winter Melon
                    if new_plant.type in [34, 35]:
                        base_reward += 0.5  # 适度奖励重武器
                    elif new_plant.type == 43: # Cattail (香蒲)
                        base_reward += 0.8  # 香蒲奖励,但不过度激进
                    
                    if 2 <= new_plant.col <= 6: base_reward += 0.1
                
                reward += base_reward
                details['plant'] = base_reward
            else:
                reward += 0.3
                details['plant'] = 0.3
            
            self.sunflower_count = sum(1 for p in game_state.plants if p.type == PlantType.SUNFLOWER)
        
        # 植物损失惩罚 (降低惩罚强度,避免AI过度保守)
        elif plant_diff < 0:
            r_lost = abs(plant_diff) * self.rewards.get('plant_lost', -5.0)  # -15 -> -5
            reward += r_lost
            self.plants_lost += abs(plant_diff)
            details['plant_lost'] = r_lost
            
            current_sunflowers = sum(1 for p in game_state.plants if p.type == PlantType.SUNFLOWER)
            sunflower_lost = self.sunflower_count - current_sunflowers
            if sunflower_lost > 0:
                r_sf_lost = sunflower_lost * self.rewards.get('sunflower_lost', -10.0)  # -30 -> -10
                reward += r_sf_lost
                details['sunflower_lost'] = r_sf_lost
            self.sunflower_count = current_sunflowers
        
        potential = self._calculate_potential(game_state)
        delta = potential - self.last_potential
        delta_scale = self.rewards.get('potential', {}).get('delta_scale', 1.3)
        delta = np.clip(delta, -5.0, 5.0)  # 避免崩盘时势能过大
        potential_shaping = delta * delta_scale
        reward += potential_shaping
        if abs(potential_shaping) > 1e-6:
            details['potential_delta'] = potential_shaping
        self.last_potential = potential

        return reward, details, potential

    def _compute_reward(self, game_state) -> float:
        """兼容旧接口"""
        r, _ = self._compute_reward_debug(game_state)
        return r
    
    def _handle_game_failure(self):
        """处理游戏失败，返回主菜单"""
        if not self.hook_client:
            return
        
        time.sleep(0.5)
        ui = self.hook_client.get_ui()
        
        if ui == 4:  # ZOMBIES_WON
            for attempt in range(10):
                self.hook_client.click_scaled(280, 370)
                time.sleep(0.3)
                ui = self.hook_client.get_ui()
                if ui != 4:
                    break
        
        if ui == 3 or ui == 4:
            self.hook_client.back_to_main()
            time.sleep(1.0)
    
    def _execute_action(self, action: int, game_state=None) -> bool:
        """
        执行动作
        
        Args:
            action: 动作索引
            game_state: 游戏状态（可选，避免重复读取）
            
        Returns:
            是否成功
        """
        # 使用传入的 game_state，避免重复读取
        if game_state is None:
            game_state = self.pvz.get_game_state() if self.pvz else None
        
        if action < self.n_plant_actions:
            # 种植动作: action = card_idx * 45 + row * 9 + col
            card_idx = action // (self.rows * self.cols)
            cell_idx = action % (self.rows * self.cols)
            row = cell_idx // self.cols
            col = cell_idx % self.cols
            
            # 执行种植前检查
            if not self._can_plant(card_idx, row, col, game_state):
                return False
            
            # 使用 plant_card 而不是 plant，这样会扣除阳光
            return self.hook_client.plant_card(row, col, card_idx)
            
        elif action < self.n_plant_actions + self.n_shovel_actions:
            # 铲除动作
            shovel_idx = action - self.n_plant_actions
            row = shovel_idx // self.cols
            col = shovel_idx % self.cols
            
            # 执行铲除前检查
            if not self._can_shovel(row, col, game_state):
                return False
            
            return self.hook_client.shovel(row, col)
            
        else:
            # 等待动作
            return True
    
    def _can_plant(self, card_idx: int, row: int, col: int, game_state) -> bool:
        """
        检查是否可以在指定位置种植 (修复版 - 支持泳池和升级植物)
        """
        if game_state is None:
            return False
        
        if card_idx < 0 or card_idx >= len(self.card_costs):
            return False
        
        # 检查阳光
        cost = self.card_costs[card_idx]
        if game_state.sun < cost:
            return False
        
        # 检查冷却
        if card_idx < len(game_state.seeds):
            seed = game_state.seeds[card_idx]
            if not seed.is_ready:
                return False
        else:
            return False
        
        # 获取卡片植物类型
        card_plant_id = self.card_plant_ids[card_idx] if hasattr(self, 'card_plant_ids') else -1
        is_pumpkin_card = (card_plant_id == 30)
        
        # 获取格子上的植物信息
        plants_on_cell = []
        has_pumpkin_on_cell = False
        has_lily_pad = False
        
        for plant in game_state.plants:
            if plant.row == row and plant.col == col:
                plants_on_cell.append(plant)
                if plant.type == 30:
                    has_pumpkin_on_cell = True
                if plant.type == 16: # Lily Pad
                    has_lily_pad = True
        
        # === 1. 升级植物检查 ===
        if card_plant_id in UPGRADE_PLANTS:
            base_id = UPGRADE_PLANTS[card_plant_id]
            has_base = False
            for p in plants_on_cell:
                if p.type == base_id:
                    has_base = True
                    break
            
            # 必须有基础植物才能升级
            if not has_base:
                return False
            
            # 检查是否被其他植物占据 (除了南瓜头)
            # 允许: [Base], [Base, Pumpkin]
            # 不允许: [Base, Other] (e.g. [Lily Pad, Peashooter] -> Cannot upgrade Lily Pad to Cattail)
            for p in plants_on_cell:
                if p.type != base_id and p.type != 30: # 30 is Pumpkin
                    return False
            
            return True

        # === 2. 地形与水生检查 ===
        # 泳池模式下，行 2 和 3 是水路 (0-indexed)
        is_water_row = (self.rows == 6 and (row == 2 or row == 3))
        is_aquatic_card = (card_plant_id in AQUATIC_PLANTS)
        
        if is_water_row:
            if is_aquatic_card:
                # 水生植物种在水上
                # 特例：睡莲不能种在睡莲上
                if card_plant_id == 16 and has_lily_pad:
                    return False
                
                # 如果格子为空，或者只有南瓜头(虽然水上南瓜头必须依附植物，但逻辑上允许种睡莲在空水面)
                # 实际上水上只有睡莲能直接种，缠绕海草也能直接种
                if not plants_on_cell:
                    return True
                
                # 如果已经有植物(非睡莲)，通常不能再种水生植物(除了南瓜头)
                # 但这里简化：如果只有南瓜头，允许种睡莲？不，南瓜头必须套在植物上。
                # 所以水上如果为空，可以种睡莲/海草。
                return True
            else:
                # 非水生植物种在水上：必须有睡莲
                if not has_lily_pad:
                    return False
                
                # 有睡莲，检查是否已有其他植物占据 (除了南瓜头)
                # 格子上允许：[睡莲], [睡莲, 南瓜头]
                # 如果已经有 [睡莲, 豌豆]，则不能再种
                count_occupants = 0
                for p in plants_on_cell:
                    if p.type != 16 and p.type != 30:
                        count_occupants += 1
                
                if count_occupants > 0:
                    return False
                
                # 允许种植 (叠加在睡莲上)
                # 此时如果是南瓜头卡片，下面逻辑会处理
                pass
        else:
            # 陆地行
            if is_aquatic_card:
                return False # 水生植物不能种在陆地
            
            # 陆地正常种植逻辑，继续往下
            pass

        # === 3. 普通叠加/空地检查 ===
        if not plants_on_cell:
            return True
            
        if is_pumpkin_card:
            # 种南瓜头：只要没南瓜头就行
            return not has_pumpkin_on_cell
        else:
            # 种普通植物：
            # 如果是水上且有睡莲，上面已经检查过是否被占据
            if is_water_row:
                # 此时一定有睡莲 (否则上面返回False了)
                # 且没有其他占据者 (否则上面返回False了)
                # 所以可以种
                return True
            
            # 陆地：只要格子上只有南瓜头就行 (或者完全为空，已处理)
            # 如果有其他植物，不能种
            # 除非是铲除重种，但这里是 _can_plant，不自动铲除
            return len(plants_on_cell) == 1 and has_pumpkin_on_cell
    
    def _can_shovel(self, row: int, col: int, game_state) -> bool:
        """
        检查是否可以铲除指定位置的植物
        
        检查条件:
        1. 游戏状态有效
        2. 格子有植物
        
        Args:
            row: 行
            col: 列
            game_state: 游戏状态
            
        Returns:
            是否可以铲除
        """
        if game_state is None:
            return False
        
        # 检查格子是否有植物
        for plant in game_state.plants:
            if plant.row == row and plant.col == col:
                return True
        
        return False
    

    def _check_lawnmower_fail(self, game_state) -> bool:
        """
        检查是否因僵尸进屋而失败
        
        条件: 任何僵尸 X < -70 (已进屋，游戏失败)
        
        Returns:
            True 如果检测到失败条件
        """
        if not game_state:
            return False
        
        # 僵尸进屋阈值: X < -70 表示已进屋，游戏失败
        ZOMBIE_ENTERED_X = -70
        
        # 检查是否有僵尸已经进屋
        for zombie in game_state.zombies:
            if zombie.x < ZOMBIE_ENTERED_X:
                return True
        
        return False
    
    def _check_terminated(self, game_state) -> Tuple[bool, bool]:
        """
        检查是否终止
        
        终止条件:
        - 僵尸 X < -70: 失败 (僵尸进屋)
        - level_end_countdown > 0: 胜利 (关卡即将结束)
        
        Returns:
            (terminated, win)
        """
        # 检查失败条件: 僵尸进屋 (X < -70)
        if self._check_lawnmower_fail(game_state):
            return True, False
        
        # 检查胜利条件: level_end_countdown > 0
        level_end_cd = getattr(game_state, 'level_end_countdown', 0)
        
        if level_end_cd > 0:
            # 只有在最后一波(或更后)才算真正的胜利
            if game_state.total_waves > 0 and game_state.wave >= game_state.total_waves:
                return True, True
        
        return False, False
    
    def _update_last_state(self, game_state, potential=None):
        """更新上一帧状态"""
        self.last_sun = game_state.sun
        self.last_zombie_count = len(game_state.zombies)
        self.last_zombies_state = list(game_state.zombies)  # 更新僵尸列表副本
        self.last_plant_count = len(game_state.plants)
        self.last_wave = game_state.wave
        if potential is None:
            potential = self._calculate_potential(game_state)
        self.last_potential = potential
    
    def _get_observation(self, game_state=None) -> Dict[str, np.ndarray]:
        """获取观测"""
        if game_state is None:
            game_state = self.pvz.get_game_state() if self.pvz else None
        
        # 从配置获取维度
        obs_config = self.config.get('observation_space', {})
        grid_shape = obs_config.get('grid', {}).get('shape', [5, 9, 8])
        global_dim = obs_config.get('global', {}).get('total_dim', 71)  # 增加新特征
        
        # 网格特征 (增强到8通道)
        grid = np.zeros(tuple(grid_shape), dtype=np.float32)
        
        # 全局特征 (增强到54维)
        global_features = np.zeros(global_dim, dtype=np.float32)
        
        # 动作掩码
        action_mask = self._get_action_mask(game_state)
        
        # 卡片属性特征 (10张卡 × 7属性)
        card_attributes = np.zeros((10, 7), dtype=np.float32)
        
        if game_state:
            # 填充网格特征
            grid = self._build_grid_features(game_state)
            
            # 填充全局特征
            global_features = self._build_global_features(game_state)
            
            # 填充卡片属性特征
            # 归一化因子: Cost/500, HP/4000, Dmg/1800, Range/9, CD/5000, Role/5, ProjType/5
            for i, card_id in enumerate(self.card_plant_ids):
                if i < 10:
                    stats = self.plant_stats.get(card_id, self.plant_stats[-1])
                    card_attributes[i, 0] = stats[0] / 500.0   # Cost
                    card_attributes[i, 1] = stats[1] / 4000.0  # HP
                    card_attributes[i, 2] = stats[2] / 1800.0  # Damage
                    card_attributes[i, 3] = stats[3] / 9.0     # Range
                    card_attributes[i, 4] = stats[4] / 5000.0  # Cooldown
                    card_attributes[i, 5] = stats[5] / 5.0     # Role
                    card_attributes[i, 6] = stats[6] / 5.0     # Projectile Type

        return {
            "grid": grid,
            "global_features": global_features,
            "card_attributes": card_attributes,
            "action_mask": action_mask,
        }
    
    def _build_grid_features(self, game_state) -> np.ndarray:
        """
        构建增强网格特征 (13通道) - 修正版
        
        Channel 0: 植物类型 (归一化)
        Channel 1: 植物血量 (比例)
        Channel 2: 植物功能 (0=防御, 0.3=经济, 0.6=控制, 1=攻击)
        Channel 3: 僵尸数量 (密度)
        Channel 4: 僵尸血量 (总和)
        Channel 5: 僵尸威胁 (类型加权)
        Channel 6: 僵尸距离 (Time to Impact)
        Channel 7: 僵尸状态 (冰冻0.5/减速0.3/啃食0.8/魅惑-1.0)
        Channel 8: 子弹密度 (数量)
        Channel 9: 子弹威胁 (伤害/特效加权)
        Channel 10: DPS热力图 (火力覆盖向右扩散, 含火炬加成, 考虑射程和三线/杨桃)
        Channel 11: 威胁热力图 (僵尸威胁双向扩散: 前方高强度, 后方提示部署)
        Channel 12: 击杀效率图 (历史击杀热力, 动态衰减)
        
        修复项:
        - DPS 热力: 火炬位置判断修正, 增加射程限制, 支持杨桃
        - 威胁热力: 双向扩散 (前方威胁 + 后方建议部署区)
        数据利用率：96%
        """
        obs_config = self.config.get('observation_space', {})
        grid_shape = obs_config.get('grid', {}).get('shape', [5, 9, 13])
        grid = np.zeros(tuple(grid_shape), dtype=np.float32)
        n_channels = grid_shape[2]
        
        # === 植物特征 (Ch 0-2) ===
        for plant in game_state.plants:
            if 0 <= plant.row < self.rows and 0 <= plant.col < self.cols:
                # Ch 0: 类型
                grid[plant.row, plant.col, 0] = (plant.type + 1) / 50.0
                
                # Ch 1: 血量
                if plant.hp_max > 0:
                    grid[plant.row, plant.col, 1] = plant.hp / plant.hp_max
                
                # Ch 2: 功能分类
                role_val = 0.0
                if plant.type in [1, 9, 41, 38]: # 经济 (向日葵/阳光菇/双子/金盏花)
                    role_val = 0.3
                elif plant.type in [5, 14, 44, 19]: # 控制 (寒冰/冰菇/冰瓜/缠绕)
                    role_val = 0.6
                elif plant.type in [0, 7, 18, 22, 39, 40, 47]: # 攻击 (豌豆/双发/三线/火炬/西瓜/机枪/玉米炮)
                    role_val = 1.0
                grid[plant.row, plant.col, 2] = role_val

        # === 僵尸特征 (Ch 3-7) & 威胁热力 (Ch 11) ===
        threat_heat = np.zeros((self.rows, self.cols), dtype=np.float32)
        max_zombie_hp = 3000.0  # 归一化基准
        
        for zombie in game_state.zombies:
            if 0 <= zombie.row < self.rows:
                # 计算所在列
                col = int((zombie.x - 40) / 80)
                col = max(0, min(self.cols - 1, col))
                
                # Ch 3: 数量密度
                grid[zombie.row, col, 3] = min(1.0, grid[zombie.row, col, 3] + 0.2)
                
                # Ch 4: 血量总和
                grid[zombie.row, col, 4] = min(1.0, grid[zombie.row, col, 4] + zombie.total_hp / max_zombie_hp)
                
                # Ch 5: 类型威胁
                threat_level = self._get_zombie_threat(zombie.type)
                grid[zombie.row, col, 5] = max(grid[zombie.row, col, 5], threat_level)
                
                # Ch 6: 距离 (Time to Impact)
                # 修复: 速度为0时使用默认值
                base_speed = ZOMBIE_BASE_SPEED.get(ZombieType(zombie.type), 0.23)
                speed = getattr(zombie, 'speed', None)
                if speed is None or speed <= 0.01:
                    speed = base_speed
                
                dist_pixels = max(1.0, zombie.x + 70) # +70是进屋线
                time_to_impact = dist_pixels / max(0.1, speed * 100)
                # 归一化: 10秒内到达为1.0, 越近越大
                urgency = min(1.0, 10.0 / time_to_impact)
                grid[zombie.row, col, 6] = max(grid[zombie.row, col, 6], urgency)
                
                # Ch 7: 状态 (叠加多种状态)
                status_val = 0.0
                
                # 控制效果 (正值)
                if hasattr(zombie, 'is_frozen') and zombie.is_frozen:
                    status_val += 0.5
                if hasattr(zombie, 'is_slowed') and zombie.is_slowed:
                    status_val += 0.3
                
                # 正在啃植物 (紧急信号，高优先级)
                if hasattr(zombie, 'is_eating') and zombie.is_eating:
                    status_val = max(status_val, 0.8)  # 优先级高于其他状态
                
                # 魅惑状态 (负值表示友军)
                if hasattr(zombie, 'is_hypno') and zombie.is_hypno:
                    status_val = -1.0
                
                grid[zombie.row, col, 7] = np.clip(status_val, -1.0, 1.0)
                
                # === 威胁热力扩散 (Ch 11) - 修正版 ===
                # 威胁 = 类型系数 × (1 + 血量系数) × 距离系数
                hp_factor = zombie.total_hp / 1000.0
                distance_factor = max(0.1, 1.0 - zombie.x / 800.0)  # 越近威胁越大
                impact_threat = threat_level * (1.0 + hp_factor) * distance_factor * (1.0 + urgency)
                
                # 双向扩散：
                # 1. 向左扩散 (僵尸前方) - 表示僵尸即将到达的区域
                # 2. 向右扩散 (僵尸后方) - 提醒 AI 在后方提前部署防御
                
                # 向左扩散 (僵尸前进方向，高强度)
                for c in range(col, -1, -1):
                    dist = col - c
                    decay = 0.7 ** dist
                    threat_heat[zombie.row, c] += impact_threat * decay
                
                # 向右扩散 (建议部署防御的区域，低强度)
                for c in range(col + 1, min(col + 4, self.cols)):
                    dist = c - col
                    decay = 0.5 ** dist  # 快速衰减
                    threat_heat[zombie.row, c] += impact_threat * decay * 0.3  # 30% 强度

        # === 子弹特征 (Ch 8-9) ===
        if hasattr(game_state, 'projectiles'):
            for proj in game_state.projectiles:
                if 0 <= proj.row < self.rows:
                    col = int((proj.x - 40) / 80)
                    col = max(0, min(self.cols - 1, col))
                    
                    # Ch 8: 密度
                    grid[proj.row, col, 8] = min(1.0, grid[proj.row, col, 8] + 0.1)
                    
                    # Ch 9: 伤害/特效
                    dmg = PROJECTILE_DAMAGE.get(proj.type, 20)
                    val = dmg / 100.0
                    if proj.type in [1, 5, 12]: # 寒冰/冰瓜/黄油 (控制)
                        val += 0.5
                    elif proj.type in [3, 5, 11]: # 西瓜/玉米炮 (溅射)
                        val += 0.3
                    grid[proj.row, col, 9] = max(grid[proj.row, col, 9], val)

        # === DPS 热力 (Ch 10) - 修正版 ===
        dps_heat = np.zeros((self.rows, self.cols), dtype=np.float32)
        
        # 先找出所有火炬的位置 (优化查找效率)
        torch_positions = set()
        for p in game_state.plants:
            if p.type == 22 and 0 <= p.row < self.rows and 0 <= p.col < self.cols:
                torch_positions.add((p.row, p.col))
        
        for plant in game_state.plants:
            if not (0 <= plant.row < self.rows and 0 <= plant.col < self.cols):
                continue
                
            base_dps = 0.0
            attack_range = self.cols  # 默认全屏
            is_pea_shooter = False  # 是否是豌豆类（可被火炬加成）
            
            # 射手植物 DPS 配置
            if plant.type == 0:  # 豌豆射手
                base_dps = 20.0
                is_pea_shooter = True
            elif plant.type == 5:  # 寒冰射手
                base_dps = 20.0
                is_pea_shooter = True
            elif plant.type == 7:  # 双发射手
                base_dps = 40.0
                is_pea_shooter = True
            elif plant.type == 18:  # 三线射手
                base_dps = 60.0  # 20*3
                is_pea_shooter = True
            elif plant.type == 40:  # 机枪射手
                base_dps = 80.0
                is_pea_shooter = True
            elif plant.type == 39:  # 西瓜投手
                base_dps = 80.0
                attack_range = 8  # 西瓜射程有限
            elif plant.type == 44:  # 冰冻西瓜
                base_dps = 80.0
                attack_range = 8
            elif plant.type == 47:  # 玉米加农炮
                base_dps = 500.0
                attack_range = self.cols  # 全屏
            elif plant.type == 22:  # 火炬树
                base_dps = 10.0  # 火炬自身微弱伤害
            elif plant.type == 27:  # 杨桃
                base_dps = 40.0  # 五向射击
            elif plant.type == 34:  # 玉米投手
                base_dps = 25.0  # 考虑黄油概率，平均略高于20
            elif plant.type == 32:  # 卷心菜投手
                base_dps = 40.0  # 攻速慢，单发40，DPS约20-30

            
            if base_dps <= 0:
                continue
            
            # 检查火炬加成 (火炬必须在射手右侧才有效)
            torch_multiplier = 1.0
            if is_pea_shooter:
                # 查找该行右侧最近的火炬
                nearest_torch_col = None
                for tc in range(plant.col + 1, self.cols):
                    if (plant.row, tc) in torch_positions:
                        nearest_torch_col = tc
                        break
                
                if nearest_torch_col is not None:
                    torch_multiplier = 2.0  # 火炬加成 x2
            
            effective_dps = base_dps * torch_multiplier
            
            # DPS 向右扩散 (射程内)
            if effective_dps > 0:
                max_col = min(plant.col + attack_range, self.cols)
                for c in range(plant.col, max_col):
                    dist = c - plant.col
                    # 衰减系数 (考虑子弹飞行时间和僵尸移动)
                    decay = max(0.5, 1.0 - 0.06 * dist)
                    dps_heat[plant.row, c] += effective_dps * decay
                
                # 三线射手特殊处理 (上下两行各 1/3 DPS)
                if plant.type == 18:
                    for offset in [-1, 1]:
                        target_row = plant.row + offset
                        if 0 <= target_row < self.rows:
                            for c in range(plant.col, max_col):
                                dist = c - plant.col
                                decay = max(0.5, 1.0 - 0.06 * dist)
                                dps_heat[target_row, c] += (effective_dps / 3.0) * decay
                
                # 杨桃特殊处理 (五向射击，简化为同行 + 上下行)
                if plant.type == 27:
                    for offset in [-1, 1]:
                        target_row = plant.row + offset
                        if 0 <= target_row < self.rows:
                            for c in range(plant.col, max_col):
                                dist = c - plant.col
                                decay = max(0.3, 1.0 - 0.08 * dist)
                                dps_heat[target_row, c] += (effective_dps * 0.4) * decay

        # 归一化热力图
        if n_channels > 10:
            grid[:, :, 10] = np.clip(dps_heat / 200.0, 0.0, 1.0)
        
        if n_channels > 11:
            grid[:, :, 11] = np.clip(threat_heat / 5.0, 0.0, 1.0)
            
        # === 击杀效率 (Ch 12) ===
        if n_channels > 12:
            # 归一化历史击杀数据 (最大值动态调整)
            max_kills = np.max(self.kill_heatmap) if np.max(self.kill_heatmap) > 0 else 1.0
            grid[:, :, 12] = self.kill_heatmap / max_kills
        
        return grid
    
    def _get_zombie_threat(self, zombie_type: int) -> float:
        """获取僵尸类型威胁度 (完整版)"""
        threat_map = {
            0: 0.2,   # 普通
            1: 0.3,   # 旗帜
            2: 0.4,   # 路障
            3: 0.5,   # 撑杆
            4: 0.6,   # 铁桶
            5: 0.4,   # 读报
            6: 0.5,   # 铁门
            7: 0.8,   # 橄榄球 (高威胁)
            8: 0.5,   # 舞王
            9: 0.3,   # 伴舞
            10: 0.3,  # 鸭子
            11: 0.3,  # 潜水
            12: 0.8,  # 冰车 (碾压)
            13: 0.6,  # 雪橇
            14: 0.5,  # 海豚
            15: 0.9,  # 小丑 (爆炸极高威胁)
            16: 0.4,  # 气球
            17: 0.6,  # 矿工 (后排威胁)
            18: 0.5,  # 跳跳
            19: 0.6,  # 雪人
            20: 0.6,  # 蹦极
            21: 0.6,  # 扶梯
            22: 0.7,  # 投篮
            23: 1.0,  # 巨人
            24: 0.3,  # 小鬼
            25: 1.0,  # 僵王
            32: 1.2,  # 红眼巨人 (最高威胁)
        }
        return threat_map.get(zombie_type, 0.3)
    
    def _build_global_features(self, game_state) -> np.ndarray:
        """构建增强全局特征 (71维)"""
        obs_config = self.config.get('observation_space', {})
        global_dim = obs_config.get('global', {}).get('total_dim', 71)
        features = np.zeros(global_dim, dtype=np.float32)
        
        idx = 0
        
        # 1. 阳光 (归一化，假设最大9999)
        features[idx] = min(1.0, game_state.sun / 9999.0)
        idx += 1
        
        # 2. 向日葵数量 (归一化，假设最多20)
        sunflower_count = sum(1 for p in game_state.plants if p.type == PlantType.SUNFLOWER)
        features[idx] = min(1.0, sunflower_count / 20.0)
        idx += 1
        
        # 3. 当前波数 (归一化)
        features[idx] = game_state.wave / max(1, game_state.total_waves) if game_state.total_waves > 0 else 0
        idx += 1
        
        # 4. 总波数 (归一化，假设最多100)
        features[idx] = min(1.0, game_state.total_waves / 100.0)
        idx += 1
        
        # 5. 下一波倒计时 (归一化)
        if hasattr(game_state, 'refresh_countdown'):
            features[idx] = min(1.0, game_state.refresh_countdown / 2000.0)  # 假设最大2000厘秒
        idx += 1
        
        # 6. 游戏时钟 (周期性编码)
        if hasattr(game_state, 'game_clock'):
            features[idx] = (game_state.game_clock % 1000) / 1000.0  # 周期1000厘秒
        idx += 1
        
        # 7. 僵尸总数 (归一化)
        features[idx] = min(1.0, len(game_state.zombies) / 100.0)
        idx += 1
        
        # 8. 僵尸总血量 (归一化)
        total_zombie_hp = sum(z.total_hp for z in game_state.zombies)
        features[idx] = min(1.0, total_zombie_hp / 50000.0)  # 假设最大50000
        idx += 1
        
        # 9. 植物总数 (归一化)
        features[idx] = min(1.0, len(game_state.plants) / 45.0)  # 最多45格
        idx += 1
        
        # 10. 小推车状态 (5维，每行一个)
        if hasattr(game_state, 'lawnmowers'):
            for row in range(self.rows):
                has_mower = any(lm.row == row and not lm.is_dead for lm in game_state.lawnmowers)
                if idx < global_dim:
                    features[idx] = 1.0 if has_mower else 0.0
                    idx += 1
        else:
            idx += self.rows  # 跳过
        
        # 11. 卡片CD比例 (10维，连续值0-1)
        for i in range(self.num_cards):
            if i < len(game_state.seeds):
                seed = game_state.seeds[i]
                if seed.recharge_time > 0:
                    cd_ratio = 1.0 - (seed.recharge_countdown / seed.recharge_time)
                else:
                    cd_ratio = 1.0 if seed.is_ready else 0.0
                if idx < global_dim:
                    features[idx] = max(0.0, min(1.0, cd_ratio))
            idx += 1
        
        # 12. 卡片是否买得起 (10维)
        for i, cost in enumerate(self.card_costs):
            if idx < global_dim:
                features[idx] = 1.0 if game_state.sun >= cost else 0.0
            idx += 1
        
        # 13. 每行威胁度 (5维)
        row_threats = [0.0] * self.rows
        for zombie in game_state.zombies:
            if 0 <= zombie.row < self.rows:
                # 威胁度 = 血量 × 类型系数 × 距离系数
                threat = zombie.total_hp / 1000.0  # 血量归一化
                threat *= self._get_zombie_threat(zombie.type)  # 类型系数
                threat *= max(0.1, 1.0 - zombie.x / 800.0)  # 距离越近威胁越大
                row_threats[zombie.row] += threat
        for row in range(self.rows):
            if idx < global_dim:
                features[idx] = min(1.0, row_threats[row])
            idx += 1
        
        # 14. 每行防御力 (5维)
        row_defense = [0.0] * self.rows
        for plant in game_state.plants:
            if 0 <= plant.row < self.rows:
                # 根据植物类型给防御值
                if plant.type == 3:  # 坚果墙
                    row_defense[plant.row] += 0.3
                elif plant.type == 0 or plant.type == 7:  # 豌豆/双发
                    row_defense[plant.row] += 0.1
                elif plant.type == 5:  # 寒冰射手
                    row_defense[plant.row] += 0.15
                else:
                    row_defense[plant.row] += 0.05
        for row in range(self.rows):
            if idx < global_dim:
                features[idx] = min(1.0, row_defense[row])
            idx += 1
        
        # 15. 大波倒计时 (1维)
        if hasattr(game_state, 'huge_wave_countdown') and idx < global_dim:
            features[idx] = min(1.0, game_state.huge_wave_countdown / 5000.0)  # 归一化，假设最大5000厘秒
        idx += 1
        
        # 16. 出怪血量阈值 (1维)
        if hasattr(game_state, 'zombie_refresh_hp') and idx < global_dim:
            features[idx] = min(1.0, game_state.zombie_refresh_hp / 10000.0)  # 归一化
        idx += 1
        
        # 17. 冰道状态 (5维，每行一个)
        if hasattr(game_state, 'ice_trails'):
            for row in range(self.rows):
                has_ice = any(trail.get('row') == row and trail.get('timer', 0) > 0 for trail in game_state.ice_trails)
                if idx < global_dim:
                    features[idx] = 1.0 if has_ice else 0.0
                    idx += 1
        else:
            idx += self.rows        # 18. 下一波出怪预告 (10维，各类型僵尸数量)
        # spawn_lists: List[List[int]], spawn_lists[wave] = [zombie_type1, zombie_type2, ...]
        if hasattr(game_state, 'spawn_lists') and game_state.spawn_lists:
            next_wave = game_state.wave  # 下一波索引
            if 0 <= next_wave < len(game_state.spawn_lists):
                next_zombies = game_state.spawn_lists[next_wave]
                # 统计各类型数量 (0=普通, 1=路障, 2=铁桶, 3=撑杆, 4=读报, ...)
                type_counts = [0] * 10  # 前10种常见类型
                for ztype in next_zombies:
                    if 0 <= ztype < 10:
                        type_counts[ztype] += 1
                # 归一化 (假设每种最多20个)
                for i in range(10):
                    if idx < global_dim:
                        features[idx] = min(1.0, type_counts[i] / 20.0)
                    idx += 1
            else:
                idx += 10  # 跳过
        else:
            idx += 10
        
        return features
    
    def _get_action_mask(self, game_state) -> np.ndarray:
        """
        获取动作掩码 (修复版 - 支持泳池和升级植物)
        """
        mask = np.zeros(self.n_actions, dtype=np.int8)
        
        if game_state is None:
            mask[-1] = 1  # 只允许等待
            return mask
        
        # 获取每个格子的植物列表
        cell_plants = {}
        for plant in game_state.plants:
            pos = (plant.row, plant.col)
            if pos not in cell_plants:
                cell_plants[pos] = []
            cell_plants[pos].append(plant.type)
        
        # 1. 种植动作掩码
        seeds = game_state.seeds
        for card_idx in range(self.num_cards):
            if card_idx >= len(seeds):
                continue
                
            seed = seeds[card_idx]
            
            # 检查卡片是否可用 (CD好且阳光够)
            if not seed.is_ready or game_state.sun < self.card_costs[card_idx]:
                continue
            
            # 获取当前卡片的植物ID
            card_plant_id = self.card_plant_ids[card_idx] if hasattr(self, 'card_plant_ids') else -1
            is_pumpkin_card = (card_plant_id == 30)
            is_aquatic_card = (card_plant_id in AQUATIC_PLANTS)
            is_upgrade_card = (card_plant_id in UPGRADE_PLANTS)
                
            for row in range(self.rows):
                for col in range(self.cols):
                    action_idx = card_idx * (self.rows * self.cols) + row * self.cols + col
                    
                    plants_on_cell = cell_plants.get((row, col), [])
                    is_empty = len(plants_on_cell) == 0
                    has_pumpkin_on_cell = 30 in plants_on_cell
                    has_lily_pad = 16 in plants_on_cell
                    
                    can_plant = False
                    
                    # === 1. 升级植物检查 ===
                    if is_upgrade_card:
                        base_id = UPGRADE_PLANTS[card_plant_id]
                        if base_id in plants_on_cell:
                            # 检查是否被其他植物占据
                            is_blocked = False
                            for pid in plants_on_cell:
                                if pid != base_id and pid != 30:
                                    is_blocked = True
                                    break
                            if not is_blocked:
                                can_plant = True
                        # 如果没有基础植物，不能种 (即使是空地也不能种升级植物)
                        # 注意：这里简化了，如果格子上只有南瓜头，也不能种升级植物(除非南瓜头里有基础植物)
                        # 上面的 base_id in plants_on_cell 已经涵盖了
                    
                    # === 2. 地形与水生检查 ===
                    elif self.rows == 6 and (row == 2 or row == 3): # 泳池行
                        if is_aquatic_card:
                            # 水生植物 (睡莲/海草)
                            if card_plant_id == 16: # 睡莲
                                # 睡莲不能种在睡莲上
                                if not has_lily_pad:
                                    # 可以种在空水面，或者只有南瓜头的水面(虽然少见)
                                    # 实际上如果只有南瓜头，说明下面肯定有植物(或者bug)，这里假设空水面
                                    if is_empty:
                                        can_plant = True
                            else:
                                # 其他水生 (海草)
                                if is_empty:
                                    can_plant = True
                        else:
                            # 非水生植物在泳池：必须有睡莲
                            if has_lily_pad:
                                # 有睡莲，检查是否被占据
                                # 允许：[睡莲] -> 种豌豆
                                # 允许：[睡莲, 豌豆] -> 种南瓜头
                                # 不允许：[睡莲, 豌豆] -> 种西瓜
                                
                                # 计算非睡莲非南瓜头的植物数量
                                occupants = [p for p in plants_on_cell if p != 16 and p != 30]
                                
                                if is_pumpkin_card:
                                    if not has_pumpkin_on_cell:
                                        can_plant = True
                                else:
                                    if len(occupants) == 0:
                                        can_plant = True
                    
                    # === 3. 陆地行检查 ===
                    else:
                        if is_aquatic_card:
                            can_plant = False # 水生不能种陆地
                        else:
                            # 普通陆地逻辑
                            if is_empty:
                                can_plant = True
                            elif is_pumpkin_card:
                                if not has_pumpkin_on_cell:
                                    can_plant = True
                            else:
                                if len(plants_on_cell) == 1 and has_pumpkin_on_cell:
                                    can_plant = True
                    
                    if can_plant:
                        mask[action_idx] = 1
        
        # 2. 铲除动作掩码 (已禁用)
        # for row in range(self.rows):
        #     for col in range(self.cols):
        #         if (row, col) in cell_plants:
        #             shovel_idx = self.n_plant_actions + row * self.cols + col
        #             mask[shovel_idx] = 1
        
        # 3. 等待动作始终可用
        mask[-1] = 1
        
        return mask
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        # 从缓存状态获取游戏数据
        game_state = self._cached_game_state
        lawnmowers = [0] * self.rows
        if game_state and hasattr(game_state, 'lawnmowers'):
            for i, lm in enumerate(game_state.lawnmowers):
                if i < self.rows:
                    lawnmowers[i] = 1 if getattr(lm, 'state', 0) == 1 else 0
        
        info = {
            "steps": self.steps,
            "total_reward": self.total_reward,
            "zombies_killed": self.zombies_killed,
            "plants_lost": self.plants_lost,
            "win": self._episode_win if self._episode_win is not None else False,
            "game_ended": self._episode_win is not None,  # True=游戏正常结束, False=超时/中断
            # Debug/diagnostic fields
            "hook_connected": bool(self.hook_client.connected) if self.hook_client else False,
            "pvz_attached": bool(self.pvz.is_attached()) if self.pvz else False,
            "last_collect_result": getattr(self, "_last_collect_result", None),
            "sun": game_state.sun if game_state else 0,
            "wave": game_state.wave if game_state else 0,
            "zombie_count": len(game_state.zombies) if game_state else 0,
            "plant_count": len(game_state.plants) if game_state else 0,
            "lawnmowers": lawnmowers,
            "is_paused": False,  # PVZ doesn't expose pause state directly
        }
        return info
    
    def render(self):
        """渲染 (游戏本身已渲染)"""
        pass
    
    def close(self):
        """关闭环境"""
        if self.hook_client:
            self.hook_client.disconnect()
            self.hook_client = None
        self.pvz = None
    
    def action_meanings(self) -> List[str]:
        """动作含义"""
        meanings = []
        
        # 种植动作
        for card_idx in range(self.num_cards):
            plant_name = self.config['cards']['plants'][card_idx]['name']
            for row in range(self.rows):
                for col in range(self.cols):
                    meanings.append(f"Plant {plant_name} at ({row},{col})")
        
        # 铲除动作
        for row in range(self.rows):
            for col in range(self.cols):
                meanings.append(f"Shovel at ({row},{col})")
        
        # 等待
        meanings.append("Wait")
        
        return meanings


# 注册环境
gym.register(
    id="PVZ-SurvivalEndlessDay-v0",
    entry_point="envs.pvz_env:PVZEnv",
)
