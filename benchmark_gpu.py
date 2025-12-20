"""
GPU 算力测试脚本
测试 GPU 最大吞吐量，帮助分析训练瓶颈
"""

import torch
import torch.nn as nn
import time
import numpy as np

def get_gpu_info():
    """获取 GPU 信息"""
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return None
    
    print("=" * 60)
    print("GPU 信息")
    print("=" * 60)
    
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    
    print(f"设备名称: {props.name}")
    print(f"显存总量: {props.total_memory / 1024**3:.2f} GB")
    print(f"SM 数量: {props.multi_processor_count}")
    print(f"CUDA 核心数: ~{props.multi_processor_count * 128}")  # 估算
    print(f"计算能力: {props.major}.{props.minor}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    
    return device


def benchmark_matmul(device, sizes=[1024, 2048, 4096], iterations=100):
    """测试矩阵乘法性能"""
    print("\n" + "=" * 60)
    print("矩阵乘法性能测试")
    print("=" * 60)
    
    results = []
    
    for size in sizes:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # 预热
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # 计时
        start = time.perf_counter()
        for _ in range(iterations):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # 计算 TFLOPS (矩阵乘法: 2*N^3 次浮点运算)
        flops = 2 * size**3 * iterations
        tflops = flops / elapsed / 1e12
        
        print(f"矩阵大小 {size}x{size}: {tflops:.2f} TFLOPS, {elapsed/iterations*1000:.2f} ms/iter")
        results.append((size, tflops))
    
    return results


def benchmark_neural_network(device, batch_sizes=[64, 256, 512, 1024], iterations=100):
    """测试神经网络前向+反向传播性能"""
    print("\n" + "=" * 60)
    print("神经网络训练性能测试 (类似 PPO 的网络结构)")
    print("=" * 60)
    
    # 模拟 PVZ 环境的输入
    grid_shape = (5, 9, 8)  # 5行9列8通道
    global_dim = 54
    action_dim = 496
    
    # 构建类似 PPO 的网络
    class PPONet(nn.Module):
        def __init__(self):
            super().__init__()
            # 网格特征提取
            self.grid_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(5*9*8, 256),
                nn.ReLU(),
            )
            # 全局特征
            self.global_net = nn.Sequential(
                nn.Linear(global_dim, 128),
                nn.ReLU(),
            )
            # 合并后的网络 (512-512-256 结构)
            self.shared = nn.Sequential(
                nn.Linear(256 + 128, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            )
            # 策略头和价值头
            self.policy = nn.Linear(256, action_dim)
            self.value = nn.Linear(256, 1)
        
        def forward(self, grid, global_feat):
            g = self.grid_net(grid)
            gl = self.global_net(global_feat)
            x = torch.cat([g, gl], dim=1)
            x = self.shared(x)
            return self.policy(x), self.value(x)
    
    model = PPONet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    results = []
    
    for batch_size in batch_sizes:
        # 生成假数据
        grid = torch.randn(batch_size, 5, 9, 8, device=device)
        global_feat = torch.randn(batch_size, global_dim, device=device)
        target_actions = torch.randint(0, action_dim, (batch_size,), device=device)
        
        # 预热
        for _ in range(10):
            policy, value = model(grid, global_feat)
            loss = nn.functional.cross_entropy(policy, target_actions) + value.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        
        # 计时
        start = time.perf_counter()
        for _ in range(iterations):
            policy, value = model(grid, global_feat)
            loss = nn.functional.cross_entropy(policy, target_actions) + value.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        samples_per_sec = batch_size * iterations / elapsed
        ms_per_iter = elapsed / iterations * 1000
        
        print(f"Batch {batch_size:4d}: {samples_per_sec:,.0f} samples/s, {ms_per_iter:.2f} ms/iter")
        results.append((batch_size, samples_per_sec))
    
    return results


def benchmark_inference_only(device, batch_sizes=[64, 256, 512, 1024], iterations=500):
    """测试推理性能 (不含反向传播)"""
    print("\n" + "=" * 60)
    print("神经网络推理性能测试 (仅前向传播)")
    print("=" * 60)
    
    grid_shape = (5, 9, 8)
    global_dim = 54
    action_dim = 496
    
    class PPONet(nn.Module):
        def __init__(self):
            super().__init__()
            self.grid_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(5*9*8, 256),
                nn.ReLU(),
            )
            self.global_net = nn.Sequential(
                nn.Linear(global_dim, 128),
                nn.ReLU(),
            )
            self.shared = nn.Sequential(
                nn.Linear(256 + 128, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
            )
            self.policy = nn.Linear(256, action_dim)
            self.value = nn.Linear(256, 1)
        
        def forward(self, grid, global_feat):
            g = self.grid_net(grid)
            gl = self.global_net(global_feat)
            x = torch.cat([g, gl], dim=1)
            x = self.shared(x)
            return self.policy(x), self.value(x)
    
    model = PPONet().to(device)
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for batch_size in batch_sizes:
            grid = torch.randn(batch_size, 5, 9, 8, device=device)
            global_feat = torch.randn(batch_size, global_dim, device=device)
            
            # 预热
            for _ in range(10):
                policy, value = model(grid, global_feat)
            torch.cuda.synchronize()
            
            # 计时
            start = time.perf_counter()
            for _ in range(iterations):
                policy, value = model(grid, global_feat)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            samples_per_sec = batch_size * iterations / elapsed
            ms_per_iter = elapsed / iterations * 1000
            
            print(f"Batch {batch_size:4d}: {samples_per_sec:,.0f} samples/s, {ms_per_iter:.3f} ms/iter")
            results.append((batch_size, samples_per_sec))
    
    return results


def analyze_bottleneck(inference_rate, env_rate=50):
    """分析训练瓶颈"""
    print("\n" + "=" * 60)
    print("瓶颈分析")
    print("=" * 60)
    
    print(f"环境交互速度: ~{env_rate} steps/s (受游戏限制)")
    print(f"GPU 推理速度: ~{inference_rate:,.0f} samples/s")
    print(f"GPU 比环境快: {inference_rate/env_rate:.0f}x")
    print()
    print("结论:")
    if inference_rate / env_rate > 100:
        print("  ⚠️ 严重瓶颈在 CPU/游戏端，GPU 大部分时间空闲")
        print("  💡 建议: 多实例并行 或 进一步优化游戏交互速度")
    elif inference_rate / env_rate > 10:
        print("  ⚠️ 瓶颈在 CPU/游戏端")
        print("  💡 建议: 增加 n_epochs 让 GPU 多训练几轮")
    else:
        print("  ✅ CPU/GPU 较为平衡")


def main():
    print("🚀 GPU 算力测试")
    print()
    
    device = get_gpu_info()
    if device is None:
        return
    
    # 显存使用
    print(f"\n当前显存使用: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
    
    # 矩阵乘法测试
    benchmark_matmul(device)
    
    # 神经网络训练测试
    train_results = benchmark_neural_network(device)
    
    # 推理测试
    inference_results = benchmark_inference_only(device)
    
    # 分析瓶颈
    # 用 batch=1 的推理速度估算
    single_inference_rate = inference_results[0][1] if inference_results else 10000
    analyze_bottleneck(single_inference_rate, env_rate=50)
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
