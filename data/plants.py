"""
Plant Data from AsmVsZombies
Contains all 48 plant types with damage, attack intervals, range, HP, and collision boxes
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Tuple


class PlantType(IntEnum):
    """All 48 plant types in PVZ"""
    PEASHOOTER = 0
    SUNFLOWER = 1
    CHERRY_BOMB = 2
    WALLNUT = 3
    POTATO_MINE = 4
    SNOW_PEA = 5
    CHOMPER = 6
    REPEATER = 7
    PUFFSHROOM = 8
    SUNSHROOM = 9
    FUMESHROOM = 10
    GRAVEBUSTER = 11
    HYPNOSHROOM = 12
    SCAREDYSHROOM = 13
    ICESHROOM = 14
    DOOMSHROOM = 15
    LILYPAD = 16
    SQUASH = 17
    THREEPEATER = 18
    TANGLEKELP = 19
    JALAPENO = 20
    SPIKEWEED = 21
    TORCHWOOD = 22
    TALLNUT = 23
    SEASHROOM = 24
    PLANTERN = 25
    CACTUS = 26
    BLOVER = 27
    SPLITPEA = 28
    STARFRUIT = 29
    PUMPKIN = 30
    MAGNETSHROOM = 31
    CABBAGEPULT = 32
    FLOWERPOT = 33
    KERNELPULT = 34
    COFFEEBEAN = 35
    GARLIC = 36
    UMBRELLALEAF = 37
    MARIGOLD = 38
    MELONPULT = 39
    GATLINGPEA = 40
    TWINSUNFLOWER = 41
    GLOOMSHROOM = 42
    CATTAIL = 43
    WINTERMELON = 44
    GOLDMAGNET = 45
    SPIKEROCK = 46
    COBCANNON = 47
    IMITATER = 48  # 模仿者


@dataclass
class PlantData:
    """Complete data for a plant type"""
    type: PlantType
    name_en: str
    name_cn: str
    cost: int
    recharge: int  # cs (centiseconds)
    hp: int
    damage: int
    attack_interval: int  # cs, 0 if not applicable
    is_instant: bool  # Instant kill plant
    is_mushroom: bool  # Needs coffee bean in day
    is_aquatic: bool  # Needs lily pad in water


# Plant sun cost dictionary
PLANT_COST = {
    PlantType.PEASHOOTER: 100,
    PlantType.SUNFLOWER: 50,
    PlantType.CHERRY_BOMB: 150,
    PlantType.WALLNUT: 50,
    PlantType.POTATO_MINE: 25,
    PlantType.SNOW_PEA: 175,
    PlantType.CHOMPER: 150,
    PlantType.REPEATER: 200,
    PlantType.PUFFSHROOM: 0,
    PlantType.SUNSHROOM: 25,
    PlantType.FUMESHROOM: 75,
    PlantType.GRAVEBUSTER: 75,
    PlantType.HYPNOSHROOM: 75,
    PlantType.SCAREDYSHROOM: 25,
    PlantType.ICESHROOM: 75,
    PlantType.DOOMSHROOM: 125,
    PlantType.LILYPAD: 25,
    PlantType.SQUASH: 50,
    PlantType.THREEPEATER: 325,
    PlantType.TANGLEKELP: 25,
    PlantType.JALAPENO: 125,
    PlantType.SPIKEWEED: 100,
    PlantType.TORCHWOOD: 175,
    PlantType.TALLNUT: 125,
    PlantType.SEASHROOM: 0,
    PlantType.PLANTERN: 25,
    PlantType.CACTUS: 125,
    PlantType.BLOVER: 100,
    PlantType.SPLITPEA: 125,
    PlantType.STARFRUIT: 125,
    PlantType.PUMPKIN: 125,
    PlantType.MAGNETSHROOM: 100,
    PlantType.CABBAGEPULT: 100,
    PlantType.FLOWERPOT: 25,
    PlantType.KERNELPULT: 100,
    PlantType.COFFEEBEAN: 75,
    PlantType.GARLIC: 50,
    PlantType.UMBRELLALEAF: 100,
    PlantType.MARIGOLD: 50,
    PlantType.MELONPULT: 300,
    PlantType.GATLINGPEA: 250,
    PlantType.TWINSUNFLOWER: 150,
    PlantType.GLOOMSHROOM: 150,
    PlantType.CATTAIL: 225,
    PlantType.WINTERMELON: 200,
    PlantType.GOLDMAGNET: 50,
    PlantType.SPIKEROCK: 125,
    PlantType.COBCANNON: 500,
}

# Plant HP (health points)
PLANT_HP = {
    PlantType.PEASHOOTER: 300,
    PlantType.SUNFLOWER: 300,
    PlantType.CHERRY_BOMB: 300,
    PlantType.WALLNUT: 4000,
    PlantType.POTATO_MINE: 300,
    PlantType.SNOW_PEA: 300,
    PlantType.CHOMPER: 300,
    PlantType.REPEATER: 300,
    PlantType.PUFFSHROOM: 300,
    PlantType.SUNSHROOM: 300,
    PlantType.FUMESHROOM: 300,
    PlantType.GRAVEBUSTER: 300,
    PlantType.HYPNOSHROOM: 300,
    PlantType.SCAREDYSHROOM: 300,
    PlantType.ICESHROOM: 300,
    PlantType.DOOMSHROOM: 300,
    PlantType.LILYPAD: 300,
    PlantType.SQUASH: 300,
    PlantType.THREEPEATER: 300,
    PlantType.TANGLEKELP: 300,
    PlantType.JALAPENO: 300,
    PlantType.SPIKEWEED: 300,
    PlantType.TORCHWOOD: 300,
    PlantType.TALLNUT: 8000,
    PlantType.SEASHROOM: 300,
    PlantType.PLANTERN: 300,
    PlantType.CACTUS: 300,
    PlantType.BLOVER: 300,
    PlantType.SPLITPEA: 300,
    PlantType.STARFRUIT: 300,
    PlantType.PUMPKIN: 4000,
    PlantType.MAGNETSHROOM: 300,
    PlantType.CABBAGEPULT: 300,
    PlantType.FLOWERPOT: 300,
    PlantType.KERNELPULT: 300,
    PlantType.COFFEEBEAN: 300,
    PlantType.GARLIC: 400,
    PlantType.UMBRELLALEAF: 300,
    PlantType.MARIGOLD: 300,
    PlantType.MELONPULT: 300,
    PlantType.GATLINGPEA: 300,
    PlantType.TWINSUNFLOWER: 300,
    PlantType.GLOOMSHROOM: 300,
    PlantType.CATTAIL: 300,
    PlantType.WINTERMELON: 300,
    PlantType.GOLDMAGNET: 300,
    PlantType.SPIKEROCK: 450,  # 9 times durability
    PlantType.COBCANNON: 300,
}

# Plant recharge time (cs)
PLANT_RECHARGE = {
    PlantType.PEASHOOTER: 750,
    PlantType.SUNFLOWER: 750,
    PlantType.CHERRY_BOMB: 5000,
    PlantType.WALLNUT: 3000,
    PlantType.POTATO_MINE: 3000,
    PlantType.SNOW_PEA: 750,
    PlantType.CHOMPER: 750,
    PlantType.REPEATER: 750,
    PlantType.PUFFSHROOM: 750,
    PlantType.SUNSHROOM: 750,
    PlantType.FUMESHROOM: 750,
    PlantType.GRAVEBUSTER: 750,
    PlantType.HYPNOSHROOM: 3000,
    PlantType.SCAREDYSHROOM: 750,
    PlantType.ICESHROOM: 5000,
    PlantType.DOOMSHROOM: 5000,
    PlantType.LILYPAD: 750,
    PlantType.SQUASH: 3000,
    PlantType.THREEPEATER: 750,
    PlantType.TANGLEKELP: 3000,
    PlantType.JALAPENO: 5000,
    PlantType.SPIKEWEED: 750,
    PlantType.TORCHWOOD: 750,
    PlantType.TALLNUT: 3000,
    PlantType.SEASHROOM: 3000,
    PlantType.PLANTERN: 3000,
    PlantType.CACTUS: 750,
    PlantType.BLOVER: 750,
    PlantType.SPLITPEA: 750,
    PlantType.STARFRUIT: 750,
    PlantType.PUMPKIN: 3000,
    PlantType.MAGNETSHROOM: 750,
    PlantType.CABBAGEPULT: 750,
    PlantType.FLOWERPOT: 750,
    PlantType.KERNELPULT: 750,
    PlantType.COFFEEBEAN: 750,
    PlantType.GARLIC: 750,
    PlantType.UMBRELLALEAF: 750,
    PlantType.MARIGOLD: 750,
    PlantType.MELONPULT: 750,
    PlantType.GATLINGPEA: 5000,
    PlantType.TWINSUNFLOWER: 5000,
    PlantType.GLOOMSHROOM: 5000,
    PlantType.CATTAIL: 5000,
    PlantType.WINTERMELON: 5000,
    PlantType.GOLDMAGNET: 5000,
    PlantType.SPIKEROCK: 5000,
    PlantType.COBCANNON: 5000,
}

# ============================================================================
# Plant Collision Boxes (from judge.h)
# HIT_DEFENSE_RANGE: Range where zombies can hit the plant
# EXPLODE_DEFENSE_RANGE: Range where explosions can hit the plant
# ============================================================================

@dataclass
class CollisionBox:
    """Plant collision box relative to plant x coordinate"""
    left: int
    right: int


# Hit defense range (zombies can bite/hit)
# Format: (left_offset, right_offset) from plant x
HIT_DEFENSE_RANGE = {
    'DEFAULT': CollisionBox(30, 50),
    PlantType.TALLNUT: CollisionBox(30, 60),
    PlantType.PUMPKIN: CollisionBox(20, 80),
    PlantType.COBCANNON: CollisionBox(20, 120),
}

# Explode defense range (explosions affect plant)
# Format: (left_offset, right_offset) from plant x
EXPLODE_DEFENSE_RANGE = {
    'DEFAULT': CollisionBox(-50, 10),
    PlantType.TALLNUT: CollisionBox(-50, 30),
    PlantType.PUMPKIN: CollisionBox(-60, 40),
    PlantType.COBCANNON: CollisionBox(-60, 80),
}


def get_hit_defense_range(plant_type: PlantType) -> CollisionBox:
    """Get the hit defense range for a plant type"""
    return HIT_DEFENSE_RANGE.get(plant_type, HIT_DEFENSE_RANGE['DEFAULT'])


def get_explode_defense_range(plant_type: PlantType) -> CollisionBox:
    """Get the explode defense range for a plant type"""
    return EXPLODE_DEFENSE_RANGE.get(plant_type, EXPLODE_DEFENSE_RANGE['DEFAULT'])


# ============================================================================
# Plant Attack Data
# ============================================================================

# Plants that can attack
ATTACKING_PLANTS = {
    PlantType.PEASHOOTER,
    PlantType.SNOW_PEA,
    PlantType.REPEATER,
    PlantType.THREEPEATER,
    PlantType.SPLITPEA,
    PlantType.GATLINGPEA,
    PlantType.PUFFSHROOM,
    PlantType.SCAREDYSHROOM,
    PlantType.FUMESHROOM,
    PlantType.SEASHROOM,
    PlantType.CACTUS,
    PlantType.STARFRUIT,
    PlantType.CABBAGEPULT,
    PlantType.KERNELPULT,
    PlantType.MELONPULT,
    PlantType.WINTERMELON,
    PlantType.GLOOMSHROOM,
    PlantType.CATTAIL,
}

# Instant kill plants
INSTANT_KILL_PLANTS = {
    PlantType.CHERRY_BOMB,
    PlantType.JALAPENO,
    PlantType.DOOMSHROOM,
    PlantType.SQUASH,
    PlantType.POTATO_MINE,
    PlantType.CHOMPER,
    PlantType.TANGLEKELP,
    PlantType.HYPNOSHROOM,
}

# Mushroom plants (need coffee bean in day)
MUSHROOM_PLANTS = {
    PlantType.PUFFSHROOM,
    PlantType.SUNSHROOM,
    PlantType.FUMESHROOM,
    PlantType.HYPNOSHROOM,
    PlantType.SCAREDYSHROOM,
    PlantType.ICESHROOM,
    PlantType.DOOMSHROOM,
    PlantType.SEASHROOM,
    PlantType.MAGNETSHROOM,
    PlantType.GLOOMSHROOM,
}

# Aquatic plants
AQUATIC_PLANTS = {
    PlantType.LILYPAD,
    PlantType.TANGLEKELP,
    PlantType.SEASHROOM,
    PlantType.CATTAIL,
}

# Sun producing plants
SUN_PRODUCING_PLANTS = {
    PlantType.SUNFLOWER,
    PlantType.SUNSHROOM,
    PlantType.TWINSUNFLOWER,
}

# Defensive plants
DEFENSIVE_PLANTS = {
    PlantType.WALLNUT,
    PlantType.TALLNUT,
    PlantType.PUMPKIN,
    PlantType.GARLIC,
}

# Upgrade plants (require another plant to place on)
UPGRADE_PLANTS = {
    PlantType.GATLINGPEA: PlantType.REPEATER,
    PlantType.TWINSUNFLOWER: PlantType.SUNFLOWER,
    PlantType.GLOOMSHROOM: PlantType.FUMESHROOM,
    PlantType.CATTAIL: PlantType.LILYPAD,
    PlantType.WINTERMELON: PlantType.MELONPULT,
    PlantType.GOLDMAGNET: PlantType.MAGNETSHROOM,
    PlantType.SPIKEROCK: PlantType.SPIKEWEED,
    PlantType.COBCANNON: PlantType.KERNELPULT,  # Requires two kernel-pults
}


# ============================================================================
# Plant Damage Values (per projectile)
# Source: re-plants-vs-zombies Projectile.cpp gProjectileDefinition[] (line 14-28)
# All values in HP damage per hit
# ============================================================================

PLANT_DAMAGE: dict[PlantType, int] = {
    # Pea shooters - damage 20 per pea
    # Source: Projectile.cpp line 14: PROJECTILE_PEA damage=20
    PlantType.PEASHOOTER: 20,
    PlantType.SNOW_PEA: 20,           # + slow effect
    PlantType.REPEATER: 20,           # x2 shots per attack
    PlantType.THREEPEATER: 20,        # fires to 3 rows
    PlantType.SPLITPEA: 20,           # front and back
    PlantType.GATLINGPEA: 20,         # x4 shots per attack

    # Mushrooms
    # Source: Projectile.cpp line 18: PROJECTILE_PUFF damage=20
    PlantType.PUFFSHROOM: 20,
    PlantType.SEASHROOM: 20,          # same as puffshroom
    PlantType.SCAREDYSHROOM: 20,
    # Source: Projectile.cpp line 20: PROJECTILE_FUME damage=20 (pierce)
    PlantType.FUMESHROOM: 20,         # piercing, hits all zombies in range
    PlantType.GLOOMSHROOM: 20,        # 8-directional, 4 hits per attack cycle

    # Catapult plants
    # Source: Projectile.cpp line 16: PROJECTILE_CABBAGE damage=40
    PlantType.CABBAGEPULT: 40,
    # Source: Projectile.cpp line 24: PROJECTILE_KERNEL damage=20
    PlantType.KERNELPULT: 20,         # kernel; butter does 40 and stuns
    # Source: Projectile.cpp line 17: PROJECTILE_MELON damage=80
    PlantType.MELONPULT: 80,          # + splash damage 60 to nearby
    # Source: Projectile.cpp line 19: PROJECTILE_WINTERMELON damage=80
    PlantType.WINTERMELON: 80,        # + splash + slow effect

    # Other ranged
    # Source: Projectile.cpp line 21: PROJECTILE_STAR damage=20
    PlantType.STARFRUIT: 20,          # fires 5 stars per attack
    PlantType.CACTUS: 20,             # same as pea
    PlantType.CATTAIL: 20,            # homing spikes

    # Special plants
    # Source: Projectile.cpp line 25: PROJECTILE_COBBIG damage=300
    PlantType.COBCANNON: 300,         # 3x3 AOE explosion
    # Source: Projectile.cpp line 22: PROJECTILE_SPIKE damage=20
    PlantType.SPIKEWEED: 20,          # damage per contact
    PlantType.SPIKEROCK: 20,          # same damage, more durability

    # Instant kill plants (1800 damage = kills any non-Gargantuar)
    # Source: re-pvz constants, instant kills deal 1800 damage
    PlantType.CHERRY_BOMB: 1800,      # 3x3 AOE
    PlantType.JALAPENO: 1800,         # full row
    PlantType.DOOMSHROOM: 1800,       # large AOE (900 to Gargantuar)
    PlantType.POTATO_MINE: 1800,      # single target
    PlantType.SQUASH: 1800,           # jumps to target
    PlantType.CHOMPER: 1800,          # instant kill, 42s digest
    PlantType.TANGLEKELP: 1800,       # water only
}


# ============================================================================
# Plant Attack Intervals (time between shots in centiseconds)
# Source: re-plants-vs-zombies Plant.cpp LaunchProjectile() and Fire()
# Note: These are the shooting animation times, actual fire rate depends on
#       the number of projectiles per animation
# Unit: cs (centiseconds), 1 second = 100 cs
# ============================================================================

PLANT_ATTACK_INTERVAL: dict[PlantType, int] = {
    # Standard shooters - 141cs base interval
    # Source: Plant.cpp line 745: mShootingCounter = 33 (animation frames)
    # Actual interval calculated from game mechanics: ~141cs
    PlantType.PEASHOOTER: 141,
    PlantType.SNOW_PEA: 141,

    # Repeater/Splitpea - faster animation
    # Source: Plant.cpp line 749: mShootingCounter = 26
    PlantType.REPEATER: 141,          # 2 peas per cycle
    PlantType.SPLITPEA: 141,          # 1 front + 2 back per cycle

    # Three-peater
    PlantType.THREEPEATER: 141,       # 3 peas (one per row) per cycle

    # Gatling Pea
    # Source: Plant.cpp line 754: mShootingCounter = 100
    PlantType.GATLINGPEA: 141,        # 4 peas per cycle

    # Mushrooms
    # Source: Plant.cpp line 779: SEED_PUFFSHROOM mShootingCounter = 29
    PlantType.PUFFSHROOM: 141,
    PlantType.SEASHROOM: 141,
    # Source: Plant.cpp line 780: SEED_SCAREDYSHROOM mShootingCounter = 25
    PlantType.SCAREDYSHROOM: 141,
    # Source: Plant.cpp line 778: SEED_FUMESHROOM mShootingCounter = 50
    PlantType.FUMESHROOM: 150,

    # Gloom-shroom
    # Source: Plant.cpp line 765: mShootingCounter = 200
    PlantType.GLOOMSHROOM: 200,       # 4 fumes x 8 directions = 80 damage/cycle

    # Catapult plants
    # Source: Plant.cpp line 781: SEED_CABBAGEPULT mShootingCounter = 32
    PlantType.CABBAGEPULT: 300,
    # Source: Plant.cpp line 794: mShootingCounter = 30
    PlantType.KERNELPULT: 300,
    # Source: Plant.cpp line 783: SEED_MELONPULT/WINTERMELON mShootingCounter = 36
    PlantType.MELONPULT: 300,
    PlantType.WINTERMELON: 300,

    # Other ranged
    PlantType.STARFRUIT: 141,         # 5 stars per attack
    # Source: Plant.cpp line 797: SEED_CACTUS mShootingCounter = 35
    PlantType.CACTUS: 141,
    # Source: Plant.cpp line 770: mShootingCounter = 50
    PlantType.CATTAIL: 150,           # homing

    # Cob Cannon reload time
    # Source: constants.py COB_RECOVER_TIME = 3475
    PlantType.COBCANNON: 3475,        # reload/cooldown time
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_plant_damage(plant_type: PlantType) -> int:
    """Get damage per projectile for a plant type"""
    return PLANT_DAMAGE.get(plant_type, 0)


def get_plant_attack_interval(plant_type: PlantType) -> int:
    """Get attack interval in centiseconds for a plant type"""
    return PLANT_ATTACK_INTERVAL.get(plant_type, 0)


# ============================================================================
# Plant Unlock Data (解锁信息)
# ============================================================================

# 冒险模式关卡解锁的植物 (关卡号 -> 植物类型)
# 关卡编号: 1-1到1-10 = 1-10, 2-1到2-10 = 11-20, etc.
ADVENTURE_UNLOCK_PLANTS = {
    1: PlantType.PEASHOOTER,       # 1-1
    3: PlantType.SUNFLOWER,        # 1-3
    5: PlantType.CHERRY_BOMB,      # 1-5
    7: PlantType.WALLNUT,          # 1-7
    10: PlantType.POTATO_MINE,     # 1-10
    11: PlantType.SNOW_PEA,        # 2-1
    13: PlantType.CHOMPER,         # 2-3
    15: PlantType.REPEATER,        # 2-5
    21: PlantType.PUFFSHROOM,      # 3-1
    22: PlantType.SUNSHROOM,       # 3-2
    23: PlantType.FUMESHROOM,      # 3-3
    24: PlantType.GRAVEBUSTER,     # 3-4
    25: PlantType.HYPNOSHROOM,     # 3-5
    26: PlantType.SCAREDYSHROOM,   # 3-6
    27: PlantType.ICESHROOM,       # 3-7
    28: PlantType.DOOMSHROOM,      # 3-8
    31: PlantType.LILYPAD,         # 4-1
    32: PlantType.SQUASH,          # 4-2
    33: PlantType.THREEPEATER,     # 4-3
    34: PlantType.TANGLEKELP,      # 4-4
    35: PlantType.JALAPENO,        # 4-5
    36: PlantType.SPIKEWEED,       # 4-6
    37: PlantType.TORCHWOOD,       # 4-7
    38: PlantType.TALLNUT,         # 4-8
    41: PlantType.SEASHROOM,       # 5-1
    42: PlantType.PLANTERN,        # 5-2
    43: PlantType.CACTUS,          # 5-3
    44: PlantType.BLOVER,          # 5-4
    45: PlantType.SPLITPEA,        # 5-5
    46: PlantType.STARFRUIT,       # 5-6
    47: PlantType.PUMPKIN,         # 5-7
    48: PlantType.MAGNETSHROOM,    # 5-8
    51: PlantType.CABBAGEPULT,     # 6-1 (夜间屋顶)
    52: PlantType.FLOWERPOT,       # 6-2
    53: PlantType.KERNELPULT,      # 6-3
    54: PlantType.COFFEEBEAN,      # 6-4
    55: PlantType.GARLIC,          # 6-5
    56: PlantType.UMBRELLALEAF,    # 6-6
    57: PlantType.MARIGOLD,        # 6-7
    58: PlantType.MELONPULT,       # 6-8
}

# 商店购买植物 (StoreItem索引 -> 植物类型)
# 索引对应 mPurchases 数组
STORE_PLANTS = {
    0: PlantType.GATLINGPEA,       # STORE_ITEM_PLANT_GATLINGPEA
    1: PlantType.TWINSUNFLOWER,    # STORE_ITEM_PLANT_TWINSUNFLOWER
    2: PlantType.GLOOMSHROOM,      # STORE_ITEM_PLANT_GLOOMSHROOM
    3: PlantType.CATTAIL,          # STORE_ITEM_PLANT_CATTAIL
    4: PlantType.WINTERMELON,      # STORE_ITEM_PLANT_WINTERMELON
    5: PlantType.GOLDMAGNET,       # STORE_ITEM_PLANT_GOLD_MAGNET
    6: PlantType.SPIKEROCK,        # STORE_ITEM_PLANT_SPIKEROCK
    7: PlantType.COBCANNON,        # STORE_ITEM_PLANT_COBCANNON
    8: PlantType.IMITATER,         # STORE_ITEM_PLANT_IMITATER (模仿者)
}

# 所有植物的反查表: 植物类型 -> (解锁方式, 索引)
# 解锁方式: 'adventure' = 冒险模式解锁, 'store' = 商店购买
PLANT_UNLOCK_INFO = {}
for level, plant in ADVENTURE_UNLOCK_PLANTS.items():
    PLANT_UNLOCK_INFO[plant] = ('adventure', level)
for store_idx, plant in STORE_PLANTS.items():
    PLANT_UNLOCK_INFO[plant] = ('store', store_idx)


def get_unlocked_plants(player_level: int, purchases: list) -> set:
    """
    根据玩家关卡进度和商店购买记录，返回已解锁的植物集合。
    
    Args:
        player_level: 玩家当前冒险模式关卡 (1-60+)
        purchases: 商店购买记录列表 (80个元素)
    
    Returns:
        已解锁的 PlantType 集合
    """
    unlocked = set()
    
    # 冒险模式解锁的植物
    for level, plant in ADVENTURE_UNLOCK_PLANTS.items():
        if player_level >= level:
            unlocked.add(plant)
    
    # 商店购买的植物
    for store_idx, plant in STORE_PLANTS.items():
        if store_idx < len(purchases) and purchases[store_idx] > 0:
            unlocked.add(plant)
    
    return unlocked


def get_available_plants(player_level: int, purchases: list) -> list:
    """
    获取玩家可选择的所有植物列表（已解锁的）。
    
    Returns:
        按 PlantType 值排序的植物列表
    """
    unlocked = get_unlocked_plants(player_level, purchases)
    return sorted(list(unlocked), key=lambda x: x.value)
