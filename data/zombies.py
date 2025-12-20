"""
Zombie Data from AsmVsZombies and re-plants-vs-zombies
Contains all 33 zombie types with HP, movement speed, and special abilities

重要说明：
1. 僵尸血量 (ZOMBIE_HP_DATA / ZOMBIE_HEALTH) 是精确值
   - 来源: re-plants-vs-zombies Zombie.cpp 构造函数
   - body: 本体血量 (mBodyHealth)
   - armor: 护甲血量 (mHelmHealth) - 路障/铁桶/橄榄球头盔等
   - shield: 盾牌血量 (mShieldHealth) - 铁门/梯子/报纸等
2. 僵尸速度 (ZOMBIE_BASE_SPEED) 是参考值，实际应从内存实时读取
   - 内存偏移: Z_SPEED = 0x34
   - 速度会受减速/冻结/攻击动画影响
   - 来源: re-plants-vs-zombies Zombie.cpp PickRandomSpeed()
3. 数据来源:
   - https://github.com/Patoke/re-plants-vs-zombies
   - https://github.com/vector-wlc/AsmVsZombies
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, NamedTuple


class ZombieType(IntEnum):
    """All zombie types in PVZ"""
    ZOMBIE = 0  # 普通僵尸
    FLAG = 1  # 旗帜僵尸
    CONEHEAD = 2  # 路障僵尸
    POLEVAULTER = 3  # 撑杆僵尸
    BUCKETHEAD = 4  # 铁桶僵尸
    NEWSPAPER = 5  # 读报僵尸
    SCREENDOOR = 6  # 铁栅门僵尸
    FOOTBALL = 7  # 橄榄球僵尸
    DANCING = 8  # 舞王僵尸
    BACKUP = 9  # 伴舞僵尸
    DUCKYTUBE = 10  # 鸭子救生圈僵尸
    SNORKEL = 11  # 潜水僵尸
    ZOMBONI = 12  # 冰车僵尸
    BOBSLED = 13  # 雪橇僵尸
    DOLPHIN = 14  # 海豚骑士僵尸
    JACKINBOX = 15  # 小丑僵尸
    BALLOON = 16  # 气球僵尸
    DIGGER = 17  # 矿工僵尸
    POGO = 18  # 跳跳僵尸
    YETI = 19  # 雪人僵尸
    BUNGEE = 20  # 蹦极僵尸
    LADDER = 21  # 扶梯僵尸
    CATAPULT = 22  # 投篮僵尸
    GARGANTUAR = 23  # 巨人僵尸
    IMP = 24  # 小鬼僵尸
    ZOMBOSS = 25  # 僵尸博士
    # 26-31 unused
    GIGA_GARGANTUAR = 32  # 红眼巨人


@dataclass
class ZombieData:
    """Complete data for a zombie type"""
    type: ZombieType
    name_en: str
    name_cn: str
    body_hp: int  # 本体血量
    accessory_hp: int  # 防具血量 (路障/铁桶/门/梯子等)
    speed: float  # 基础速度 (像素/cs)
    has_flying: bool  # 是否可飞行
    has_special: bool  # 是否有特殊能力


class ZombieHealth(NamedTuple):
    """
    Zombie health values (body/armor/shield separation)
    Source: re-plants-vs-zombies Zombie.cpp constructor
    
    Attributes:
        body: Body health (mBodyHealth) - damage goes here after armor/shield depleted
        armor: Armor health (mHelmHealth) - 路障/铁桶/橄榄球头盔/矿工帽等
        shield: Shield health (mShieldHealth) - 铁门/梯子/报纸等
    """
    body: int
    armor: int = 0
    shield: int = 0


# ============================================================================
# Zombie Health Data (with body/armor/shield separation)
# Source: re-plants-vs-zombies Zombie.cpp ZombieInitialize()
# Note: armor = mHelmHealth, shield = mShieldHealth
# ============================================================================

ZOMBIE_HEALTH: dict[ZombieType, ZombieHealth] = {
    # Source: Zombie.cpp line 168: mBodyHealth = 270
    ZombieType.ZOMBIE: ZombieHealth(body=270, armor=0, shield=0),
    ZombieType.FLAG: ZombieHealth(body=270, armor=0, shield=0),
    
    # Source: Zombie.cpp line 193: mHelmHealth = 370
    ZombieType.CONEHEAD: ZombieHealth(body=270, armor=370, shield=0),
    
    # Source: Zombie.cpp line 201: mHelmHealth = 1100
    ZombieType.BUCKETHEAD: ZombieHealth(body=270, armor=1100, shield=0),
    
    # Source: Zombie.cpp line 543: mShieldHealth = 150
    ZombieType.NEWSPAPER: ZombieHealth(body=270, armor=0, shield=150),
    
    # Source: Zombie.cpp line 206: mShieldHealth = 1100
    ZombieType.SCREENDOOR: ZombieHealth(body=270, armor=0, shield=1100),
    
    # Source: Zombie.cpp line 275: mHelmHealth = 1400
    ZombieType.FOOTBALL: ZombieHealth(body=270, armor=1400, shield=0),
    
    # Source: Zombie.cpp line 308: mBodyHealth = 500
    ZombieType.POLEVAULTER: ZombieHealth(body=500, armor=0, shield=0),
    
    # Source: Zombie.cpp line 592: mBodyHealth = 500
    ZombieType.DANCING: ZombieHealth(body=500, armor=0, shield=0),
    ZombieType.BACKUP: ZombieHealth(body=270, armor=0, shield=0),
    
    ZombieType.DUCKYTUBE: ZombieHealth(body=270, armor=0, shield=0),
    
    # Snorkel has no explicit HP set, defaults to 270
    ZombieType.SNORKEL: ZombieHealth(body=270, armor=0, shield=0),
    
    # Source: Zombie.cpp line 389: mBodyHealth = 1350
    ZombieType.ZOMBONI: ZombieHealth(body=1350, armor=0, shield=0),
    
    # Source: Zombie.cpp line 488: mHelmHealth = 300
    ZombieType.BOBSLED: ZombieHealth(body=270, armor=300, shield=0),
    
    # Source: Zombie.cpp line 330: mBodyHealth = 500
    ZombieType.DOLPHIN: ZombieHealth(body=500, armor=0, shield=0),
    
    # Source: Zombie.cpp line 428: mBodyHealth = 500
    ZombieType.JACKINBOX: ZombieHealth(body=500, armor=0, shield=0),
    
    # Balloon HP is in body, mFlyingHealth=20 is separate
    ZombieType.BALLOON: ZombieHealth(body=270, armor=0, shield=0),
    
    # Source: Zombie.cpp line 283: mHelmHealth = 100
    ZombieType.DIGGER: ZombieHealth(body=270, armor=100, shield=0),
    
    # Source: Zombie.cpp line 533: mBodyHealth = 500
    ZombieType.POGO: ZombieHealth(body=500, armor=0, shield=0),
    
    # Source: Zombie.cpp line 212: mBodyHealth = 1350
    ZombieType.YETI: ZombieHealth(body=1350, armor=0, shield=0),
    
    # Source: Zombie.cpp line 233: mBodyHealth = 450
    ZombieType.BUNGEE: ZombieHealth(body=450, armor=0, shield=0),
    
    # Source: Zombie.cpp line 219-220: mBodyHealth = 500, mShieldHealth = 500
    ZombieType.LADDER: ZombieHealth(body=500, armor=0, shield=500),
    
    # Source: Zombie.cpp line 401: mBodyHealth = 850
    ZombieType.CATAPULT: ZombieHealth(body=850, armor=0, shield=0),
    
    # Source: Zombie.cpp line 348: mBodyHealth = 3000
    ZombieType.GARGANTUAR: ZombieHealth(body=3000, armor=0, shield=0),
    
    # IMP: 270 HP normally, but 70 HP in I, Zombie levels (line 614)
    # Using 270 for standard gameplay
    ZombieType.IMP: ZombieHealth(body=270, armor=0, shield=0),
    
    # ZOMBOSS: 40000 HP (adventure mode) or 60000 HP (other modes)
    # Source: Zombie.cpp line 624
    # Using adventure mode value for standard gameplay
    ZombieType.ZOMBOSS: ZombieHealth(body=40000, armor=0, shield=0),
    
    # Source: Zombie.cpp line 382: mBodyHealth = 6000
    ZombieType.GIGA_GARGANTUAR: ZombieHealth(body=6000, armor=0, shield=0),
}


# Zombie HP (body + accessory) - Legacy format for backward compatibility
# Format: (body_hp, accessory_hp)
# Body HP values from AVZ community standard (includes dying state HP)
ZOMBIE_HP_DATA = {
    ZombieType.ZOMBIE: (270, 0),
    ZombieType.FLAG: (270, 0),
    ZombieType.CONEHEAD: (270, 370),
    ZombieType.POLEVAULTER: (500, 0),
    ZombieType.BUCKETHEAD: (270, 1100),
    ZombieType.NEWSPAPER: (270, 150),
    ZombieType.SCREENDOOR: (270, 1100),
    ZombieType.FOOTBALL: (270, 1400),  # body + helmet (Source: Zombie.cpp)
    ZombieType.DANCING: (500, 0),
    ZombieType.BACKUP: (270, 0),
    ZombieType.DUCKYTUBE: (270, 0),  # HP depends on base zombie
    ZombieType.SNORKEL: (270, 0),
    ZombieType.ZOMBONI: (1350, 0),  # Ice machine HP
    ZombieType.BOBSLED: (270, 300),  # body + bobsled helmet
    ZombieType.DOLPHIN: (500, 0),
    ZombieType.JACKINBOX: (500, 0),
    ZombieType.BALLOON: (270, 0),  # Balloon is part of body, not accessory
    ZombieType.DIGGER: (270, 100),  # Mining helmet
    ZombieType.POGO: (500, 0),  # Pogo stick is not armor
    ZombieType.YETI: (1350, 0),
    ZombieType.BUNGEE: (450, 0),
    ZombieType.LADDER: (500, 500),  # Body + ladder (shield)
    ZombieType.CATAPULT: (850, 0),  # Basketball machine
    ZombieType.GARGANTUAR: (3000, 0),
    ZombieType.IMP: (270, 0),  # 270 standard, 70 in I, Zombie
    ZombieType.ZOMBOSS: (40000, 0),  # Adventure mode; 60000 in other modes
    ZombieType.GIGA_GARGANTUAR: (6000, 0),
}


def get_zombie_total_hp(zombie_type: ZombieType) -> int:
    """Get total HP (body + accessory) for a zombie type"""
    body, accessory = ZOMBIE_HP_DATA.get(zombie_type, (200, 0))
    return body + accessory


def get_zombie_body_hp(zombie_type: ZombieType) -> int:
    """Get body HP for a zombie type"""
    body, _ = ZOMBIE_HP_DATA.get(zombie_type, (200, 0))
    return body


def get_zombie_accessory_hp(zombie_type: ZombieType) -> int:
    """Get accessory HP for a zombie type"""
    _, accessory = ZOMBIE_HP_DATA.get(zombie_type, (200, 0))
    return accessory


# ============================================================================
# Zombie Base Speed (pixels per centisecond)
# Source: re-plants-vs-zombies Zombie.cpp PickRandomSpeed() (line 1110-1165)
# Note: Actual speed should be read from memory Z_SPEED (0x34) at runtime
#       Speed is affected by slow/freeze effects and attack animations
# ============================================================================

ZOMBIE_BASE_SPEED: dict[ZombieType, float] = {
    # Source: Zombie.cpp line 1158: mVelX = RandRangeFloat(0.23f, 0.32f)
    ZombieType.ZOMBIE: 0.23,
    # Source: Zombie.cpp line 1140: mVelX = 0.45f (flag/dancer/pogo)
    ZombieType.FLAG: 0.45,
    ZombieType.CONEHEAD: 0.23,
    # Source: Zombie.cpp line 1145: PHASE_POLEVAULTER_PRE_VAULT mVelX = 0.66-0.68
    ZombieType.POLEVAULTER: 0.67,     # Before vault; walks normal after
    ZombieType.BUCKETHEAD: 0.23,
    # Source: Zombie.cpp line 1154: PHASE_NEWSPAPER_MAD mVelX = 0.89-0.91
    ZombieType.NEWSPAPER: 0.23,       # Normal; 0.90 when angry
    ZombieType.SCREENDOOR: 0.23,
    # Source: Zombie.cpp line 1145: ZOMBIE_FOOTBALL mVelX = 0.66-0.68
    ZombieType.FOOTBALL: 0.67,
    # Source: Zombie.cpp line 1140: ZOMBIE_DANCER mVelX = 0.45f
    ZombieType.DANCING: 0.45,
    ZombieType.BACKUP: 0.23,
    ZombieType.DUCKYTUBE: 0.23,
    # Source: Zombie.cpp line 1145: ZOMBIE_SNORKEL mVelX = 0.66-0.68
    ZombieType.SNORKEL: 0.67,
    # Fixed speed based on ice track
    ZombieType.ZOMBONI: 0.44,
    # Source: Zombie.cpp line 492: mVelX = 0.6f (bobsled sliding)
    ZombieType.BOBSLED: 0.60,
    # Source: Zombie.cpp line 1154: PHASE_DOLPHIN_WALKING mVelX = 0.89-0.91
    ZombieType.DOLPHIN: 0.90,
    # Source: Zombie.cpp line 1145: ZOMBIE_JACK_IN_THE_BOX mVelX = 0.66-0.68
    ZombieType.JACKINBOX: 0.67,
    # Balloon floats
    ZombieType.BALLOON: 0.30,
    # Source: Zombie.cpp line 1122: mVelX = 0.12f (tunneling)
    ZombieType.DIGGER: 0.12,          # Underground; 0.23 after emerging
    # Source: Zombie.cpp line 1140: ZOMBIE_POGO mVelX = 0.45f
    ZombieType.POGO: 0.45,
    # Source: Zombie.cpp line 1135: ZOMBIE_YETI mVelX = 0.4f; 0.8f running
    ZombieType.YETI: 0.40,
    ZombieType.BUNGEE: 0.0,           # Descends from sky
    # Source: Zombie.cpp line 1149: PHASE_LADDER_CARRYING mVelX = 0.79-0.81
    ZombieType.LADDER: 0.80,
    # Fixed catapult speed
    ZombieType.CATAPULT: 0.22,
    # Gargantuars are slow
    ZombieType.GARGANTUAR: 0.15,
    # Source: Zombie.cpp line 1127: mVelX = 0.9f (IZombie IMP)
    ZombieType.IMP: 0.60,             # Thrown IMPs move faster
    ZombieType.ZOMBOSS: 0.0,          # Stationary
    ZombieType.GIGA_GARGANTUAR: 0.15,
}

# 红眼巨人平均速度 (像素/厘秒)
# 来自 AVZ judge.h: float(484) / 3158 * 1.25
# 作者注释："1.25 是我瞎调的参数"
# 这是考虑了锤击动画停顿后的平均移动速度
GIGA_AVG_SPEED = 484 / 3158 * 1.25  # ≈ 0.192 px/cs

# Speed when slowed (50% of normal)
SLOW_SPEED_MULTIPLIER = 0.5

# Speed when frozen (0)
FROZEN_SPEED = 0.0


# ============================================================================
# Special Zombie Abilities
# ============================================================================

# Zombies that can jump over plants
JUMPING_ZOMBIES = {
    ZombieType.POLEVAULTER,
    ZombieType.POGO,
    ZombieType.DOLPHIN,
}

# Zombies that fly (need Blover/Cactus)
FLYING_ZOMBIES = {
    ZombieType.BALLOON,
    ZombieType.BUNGEE,
}

# Zombies with shields (need Magnet-shroom or Fume-shroom)
SHIELD_ZOMBIES = {
    ZombieType.SCREENDOOR,
    ZombieType.LADDER,
}

# Metal zombies (can be affected by Magnet-shroom)
METAL_ZOMBIES = {
    ZombieType.BUCKETHEAD,
    ZombieType.SCREENDOOR,
    ZombieType.FOOTBALL,
    ZombieType.LADDER,
    ZombieType.POGO,
    ZombieType.JACKINBOX,
    ZombieType.DIGGER,
}

# Zombies immune to instant kill (Gargantuar takes half damage)
GARGANTUAR_ZOMBIES = {
    ZombieType.GARGANTUAR,
    ZombieType.GIGA_GARGANTUAR,
}

# Zombies that crush plants (don't eat, just destroy)
CRUSHING_ZOMBIES = {
    ZombieType.ZOMBONI,
    ZombieType.GARGANTUAR,
    ZombieType.GIGA_GARGANTUAR,
    ZombieType.CATAPULT,
}

# Underground zombies (need special plants)
UNDERGROUND_ZOMBIES = {
    ZombieType.DIGGER,
}

# Water zombies
WATER_ZOMBIES = {
    ZombieType.DUCKYTUBE,
    ZombieType.SNORKEL,
    ZombieType.DOLPHIN,
}

# ============================================================================
# Gargantuar Specific Constants
# ============================================================================

# HP thresholds for throwing imp
GARG_THROW_IMP_HP_THRESHOLD_1 = 0.5  # First throw at 50% HP
GARG_THROW_IMP_HP_THRESHOLD_2 = 0.25  # Second throw at 25% HP (Giga only)

# Hammer attack range (from zombie x position)
GARG_HAMMER_RANGE_LEFT = -30
GARG_HAMMER_RANGE_RIGHT = 59

# Time for hammer attack animation (cs)
GARG_HAMMER_TIME = 105  # Also 210 for second imp throw


# ============================================================================
# Zombie Animation Timing
# ============================================================================

# Eating speed (damage per cs to plants)
ZOMBIE_BITE_DAMAGE = 100 / 70  # About 1.43 damage per cs
ZOMBIE_BITE_INTERVAL = 70  # cs between bites (100 damage per bite at 70cs interval)

# Gargantuar smash damage (instant kill plants)
GARG_SMASH_DAMAGE = 300  # Kills any plant in one hit


# ============================================================================
# Helper Functions
# ============================================================================

def is_dangerous_zombie(zombie_type: ZombieType) -> bool:
    """Check if zombie type is considered dangerous (priority target)"""
    return zombie_type in {
        ZombieType.GARGANTUAR,
        ZombieType.GIGA_GARGANTUAR,
        ZombieType.FOOTBALL,
        ZombieType.ZOMBONI,
        ZombieType.JACKINBOX,
        ZombieType.DOLPHIN,
        ZombieType.POGO,
        ZombieType.DIGGER,
    }


def is_fast_zombie(zombie_type: ZombieType) -> bool:
    """Check if zombie type is fast moving"""
    return ZOMBIE_BASE_SPEED.get(zombie_type, 0.23) >= 0.4


def can_be_frozen(zombie_type: ZombieType) -> bool:
    """Check if zombie type can be frozen"""
    # All zombies except Zomboss can be frozen
    return zombie_type != ZombieType.ZOMBOSS


def can_be_slowed(zombie_type: ZombieType) -> bool:
    """Check if zombie type can be slowed"""
    # All zombies except Zomboss can be slowed
    return zombie_type != ZombieType.ZOMBOSS


def is_gargantuar(zombie_type: ZombieType) -> bool:
    """Check if zombie type is a Gargantuar variant"""
    return zombie_type in GARGANTUAR_ZOMBIES


def get_threat_multiplier(zombie_type: ZombieType) -> float:
    """Get threat level multiplier for a zombie type"""
    if zombie_type in {ZombieType.GARGANTUAR, ZombieType.GIGA_GARGANTUAR}:
        return 3.0
    elif zombie_type in {ZombieType.FOOTBALL, ZombieType.ZOMBONI}:
        return 2.0
    elif zombie_type in {ZombieType.BUCKETHEAD, ZombieType.SCREENDOOR}:
        return 1.5
    elif zombie_type == ZombieType.BALLOON:
        return 1.8
    elif zombie_type in {ZombieType.JACKINBOX, ZombieType.DIGGER, ZombieType.POGO}:
        return 1.7
    else:
        return 1.0


# ============================================================================
# Endless Mode Health Multiplier
# Source: Based on game mechanics for Survival: Endless mode
# Zombie health increases with each flag/wave
# ============================================================================

def get_endless_health_multiplier(flag: int) -> float:
    """
    Get zombie health multiplier for Endless mode based on flag count.
    
    In Survival: Endless mode, zombie health scales with wave progression:
    - Flags 1-2: 1.0x (normal health)
    - After flag 2: increases by 0.1x per flag
    - Maximum multiplier: 6.0x (at flag 52+)
    
    Args:
        flag: Current flag number (1-indexed, starts at 1)
        
    Returns:
        Health multiplier to apply to zombie HP values
        
    Example:
        >>> get_endless_health_multiplier(1)
        1.0
        >>> get_endless_health_multiplier(10)
        1.8
        >>> get_endless_health_multiplier(52)
        6.0
    """
    if flag <= 2:
        return 1.0
    # Linear increase starting from flag 3
    multiplier = 1.0 + (flag - 2) * 0.1
    return min(multiplier, 6.0)


def get_zombie_health_for_endless(
    zombie_type: ZombieType,
    flag: int
) -> ZombieHealth:
    """
    Get zombie health values scaled for Endless mode.
    
    Args:
        zombie_type: Type of zombie
        flag: Current flag number in Endless mode
        
    Returns:
        ZombieHealth with scaled body/armor/shield values
    """
    base_health = ZOMBIE_HEALTH.get(
        zombie_type,
        ZombieHealth(body=270, armor=0, shield=0)
    )
    multiplier = get_endless_health_multiplier(flag)
    
    return ZombieHealth(
        body=int(base_health.body * multiplier),
        armor=int(base_health.armor * multiplier),
        shield=int(base_health.shield * multiplier)
    )
