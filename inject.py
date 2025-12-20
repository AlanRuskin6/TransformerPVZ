"""
注入 Hook DLL 到 PVZ 进程
使用方法: python inject.py
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hook_client import inject_dll

def main():
    print("=" * 50)
    print("PVZ Hook DLL 注入器")
    print("=" * 50)
    print()
    print("请确保 Plants vs. Zombies 游戏已运行！")
    print()
    
    # 注入
    if inject_dll():
        print()
        print("✓ DLL 注入成功！")
        print("  Hook 服务已启动在 127.0.0.1:12345")
        print()
        print("现在可以运行训练脚本:")
        print("  python train.py train")
    else:
        print()
        print("✗ DLL 注入失败！")
        print()
        print("可能的原因:")
        print("  1. PVZ 游戏未运行")
        print("  2. 权限不足 (尝试以管理员身份运行)")
        print("  3. DLL 文件不存在")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
