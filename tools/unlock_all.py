#!/usr/bin/env python3
"""
解锁所有植物工具

使用方法：
    python tools/unlock_all.py

注意：需要先启动游戏！
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.process import ProcessAttacher
from memory.reader import MemoryReader
from memory.writer import MemoryWriter
from data.offsets import Offset
from data.plants import STORE_PLANTS, PlantType


def main():
    print("=" * 50)
    print("  PVZ 解锁所有植物工具")
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
    
    # 读取当前关卡
    current_level = reader.read_int(player_info + Offset.PI_LEVEL)
    print(f"   当前关卡进度: {current_level}")
    
    # 读取冒险模式通关次数
    finished_adv = reader.read_int(player_info + Offset.PI_FINISHED_ADV)
    print(f"   冒险模式通关次数: {finished_adv}")
    
    # 1. 设置关卡进度为60（解锁所有冒险模式植物）
    target_level = 60
    if current_level < target_level:
        success = writer.write_int(player_info + Offset.PI_LEVEL, target_level)
        if success:
            print(f"✅ 关卡进度已设置为: {target_level} (解锁所有冒险模式植物)")
        else:
            print("❌ 设置关卡进度失败")
    else:
        print(f"✅ 关卡进度已足够: {current_level}")
    
    # 2. 设置冒险模式通关次数（解锁Survival模式和其他模式）
    if finished_adv < 1:
        success = writer.write_int(player_info + Offset.PI_FINISHED_ADV, 2)
        if success:
            print(f"✅ 冒险模式通关次数已设置为: 2 (解锁Survival等所有模式)")
        else:
            print("❌ 设置通关次数失败")
    else:
        print(f"✅ 已通关冒险模式: {finished_adv} 次")
    
    # 2. 解锁所有商店植物
    print()
    print("解锁商店植物...")
    
    store_plant_names = {
        0: "机枪豌豆 (Gatling Pea)",
        1: "双子向日葵 (Twin Sunflower)",
        2: "忧郁蘑菇 (Gloom-shroom)",
        3: "香蒲 (Cattail)",
        4: "冰瓜 (Winter Melon)",
        5: "吸金磁 (Gold Magnet)",
        6: "地刺王 (Spikerock)",
        7: "玉米加农炮 (Cob Cannon)",
        8: "模仿者 (Imitater)",
    }
    
    base_addr = player_info + Offset.PI_PURCHASES
    
    for store_idx in range(9):  # 0-8 是植物
        addr = base_addr + store_idx * Offset.PI_PURCHASE_SIZE
        current_value = reader.read_int(addr)
        
        if current_value == 0:
            # 设置为1表示已购买
            success = writer.write_int(addr, 1)
            if success:
                print(f"  ✅ 已解锁: {store_plant_names.get(store_idx, f'商店物品 {store_idx}')}")
            else:
                print(f"  ❌ 解锁失败: {store_plant_names.get(store_idx, f'商店物品 {store_idx}')}")
        else:
            print(f"  ✓ 已拥有: {store_plant_names.get(store_idx, f'商店物品 {store_idx}')}")
    
    # 3. 解锁额外卡槽（可选）
    print()
    print("解锁额外卡槽...")
    
    # 卡槽购买索引 (STORE_ITEM_PACKET_UPGRADE = 10)
    slot_indices = [10, 11, 12, 13]  # 4个额外卡槽升级
    for i, slot_idx in enumerate(slot_indices):
        addr = base_addr + slot_idx * Offset.PI_PURCHASE_SIZE
        current_value = reader.read_int(addr)
        
        if current_value == 0:
            success = writer.write_int(addr, 1)
            if success:
                print(f"  ✅ 已解锁: 额外卡槽 {i + 1}")
            else:
                print(f"  ❌ 解锁失败: 额外卡槽 {i + 1}")
        else:
            print(f"  ✓ 已拥有: 额外卡槽 {i + 1}")
    
    # 4. 解锁耙子等道具（可选）
    print()
    print("解锁其他道具...")
    
    # 耙子索引 (STORE_ITEM_RAKE = 14)
    rake_idx = 14
    addr = base_addr + rake_idx * Offset.PI_PURCHASE_SIZE
    current_value = reader.read_int(addr)
    if current_value == 0:
        success = writer.write_int(addr, 3)  # 给3个耙子
        if success:
            print(f"  ✅ 已添加: 耙子 x3")
    else:
        print(f"  ✓ 已拥有: 耙子 x{current_value}")
    
    print()
    print("=" * 50)
    print("  解锁完成！")
    print("  请重新选择存档或重启游戏以生效")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    main()
