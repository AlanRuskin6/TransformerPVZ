#!/usr/bin/env python3
"""
设置金币工具

使用方法：
    python tools/set_coins.py [金币数量]
    
示例：
    python tools/set_coins.py 1000000

注意：需要先启动游戏并加载存档！
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.process import ProcessAttacher
from memory.reader import MemoryReader
from memory.writer import MemoryWriter
from data.offsets import Offset


def set_coins(target_coins: int = 1000000):
    """设置金币数量"""
    print("=" * 50)
    print("  PVZ 金币修改工具")
    print("=" * 50)
    print()
    
    # 连接游戏进程
    attacher = ProcessAttacher()
    if not attacher.attach():
        print("❌ 无法连接到游戏进程！请确保游戏正在运行。")
        return False
    
    print("✅ 已连接到游戏进程")
    
    kernel32 = attacher.kernel32
    handle = attacher.handle
    
    reader = MemoryReader(kernel32, handle)
    writer = MemoryWriter(kernel32, handle)
    
    # 获取基础地址
    base = reader.get_pvz_base()
    if base == 0:
        print("❌ 无法获取游戏基址")
        return False
    
    # 获取 PlayerInfo 指针
    player_info = reader.read_int(base + Offset.PLAYER_INFO)
    if player_info == 0:
        print("❌ 无法获取玩家信息，请确保已加载存档！")
        return False
    
    print(f"✅ 找到玩家信息: 0x{player_info:X}")
    
    # 读取当前金币
    current_coins = reader.read_int(player_info + Offset.PI_COINS)
    print(f"   当前金币: {current_coins}")
    
    # 设置金币
    # 注意：游戏内显示的金币 = 实际值 / 10
    # 所以要设置100万显示金币，实际写入 1000000 * 10 = 10000000
    actual_value = target_coins * 10
    
    success = writer.write_int(player_info + Offset.PI_COINS, actual_value)
    if success:
        print(f"✅ 金币已设置为: {target_coins} (显示值)")
        print(f"   内存实际值: {actual_value}")
    else:
        print("❌ 设置金币失败")
        return False
    
    # 验证
    new_coins = reader.read_int(player_info + Offset.PI_COINS)
    print(f"   验证 - 当前金币: {new_coins // 10} (显示值)")
    
    return True


def main():
    # 默认100万金币
    target = 1000000
    
    if len(sys.argv) > 1:
        try:
            target = int(sys.argv[1])
        except ValueError:
            print(f"❌ 无效的金币数量: {sys.argv[1]}")
            print("使用方法: python tools/set_coins.py [金币数量]")
            return
    
    set_coins(target)


if __name__ == "__main__":
    main()
