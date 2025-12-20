"""
Damage Calculation Utilities
伤害计算工具

Provides damage calculations for:
- Weapon damage values (pea, cob, cherry, etc.)
- Kill requirements (damage needed to kill zombies)
- DPS (damage per second) calculations
- Damage efficiency evaluation

All damage values are in HP units.
All time values are in centiseconds (cs) = 1/100 second.

Reference: 
- data/constants.py for damage constants
- data/zombies.py for zombie HP data
"""

import math
from typing import Optional, List, Tuple

from data.constants import (
    # Basic damage
    PEA_DAMAGE,
    SNOW_PEA_DAMAGE,
    FIRE_PEA_DAMAGE,
    MELON_DAMAGE,
    WINTER_MELON_DAMAGE,
    # Instant kill damage
    CHERRY_DAMAGE,
    JALAPENO_DAMAGE,
    DOOM_DAMAGE,
    DOOM_DAMAGE_GARG,
    # Attack intervals
    PEASHOOTER_ATTACK_INTERVAL,
    # DPS constants
    GLOOM_DAMAGE_PER_CS,
)
from data.zombies import (
    ZombieType,
    get_zombie_total_hp,
    get_zombie_body_hp,
    get_zombie_accessory_hp,
    is_gargantuar,
    GARGANTUAR_ZOMBIES,
    ZOMBIE_HP_DATA,
)


# ============================================================================
# Weapon Damage Values
# ============================================================================

# Standard projectile damage
DAMAGE_VALUES = {
    'pea': PEA_DAMAGE,                    # 20
    'snow_pea': SNOW_PEA_DAMAGE,          # 20
    'fire_pea': FIRE_PEA_DAMAGE,          # 40
    'melon': MELON_DAMAGE,                # 80
    'winter_melon': WINTER_MELON_DAMAGE,  # 80
    'fume': 20,                           # Fume-shroom
    'cactus': 20,                         # Cactus
    'starfruit': 20,                      # Starfruit
    'cabbage': 40,                        # Cabbage-pult
    'kernel': 20,                         # Kernel-pult (direct, butter = 80)
    'butter': 80,                         # Kernel butter hit
    'spikeweed': 20,                      # Per tick
    'spikerock': 20,                      # Per tick
    'gloom': 80,                          # Per attack (200cs interval)
    'cattail': 20,                        # Per spike
}

# Instant kill damage (to normal zombies)
INSTANT_DAMAGE = {
    'cherry': CHERRY_DAMAGE,      # 1800
    'jalapeno': JALAPENO_DAMAGE,  # 1800
    'doom': DOOM_DAMAGE,          # 1800
    'cob': 1800,                  # Cob cannon
    'squash': 1800,               # Squash (instant kill)
    'chomper': 9999,              # Chomper (one-shot)
    'tangle': 9999,               # Tangle kelp (one-shot)
    'potato': 1800,               # Potato mine
}


def get_weapon_damage(weapon_type: str) -> int:
    """
    Get base damage for a weapon type
    
    Args:
        weapon_type: Type of weapon (see DAMAGE_VALUES keys)
        
    Returns:
        Damage value
    """
    if weapon_type in INSTANT_DAMAGE:
        return INSTANT_DAMAGE[weapon_type]
    return DAMAGE_VALUES.get(weapon_type, 20)


def get_instant_damage_to_zombie(weapon_type: str, zombie_type: int) -> int:
    """
    Get instant kill damage to a specific zombie type
    
    Gargantuars take half damage from instant kill plants.
    
    Args:
        weapon_type: Type of instant weapon
        zombie_type: Zombie type ID
        
    Returns:
        Actual damage dealt
    """
    base_damage = INSTANT_DAMAGE.get(weapon_type, 1800)
    
    if is_gargantuar(zombie_type):
        return base_damage // 2  # 900 for normal instant kills
    
    return base_damage


def calculate_cob_damage(zombie_type: int) -> int:
    """
    Calculate cob cannon damage to a zombie
    
    Args:
        zombie_type: Zombie type ID
        
    Returns:
        Damage value (1800 normal, 900 for gargantuars)
    """
    return get_instant_damage_to_zombie('cob', zombie_type)


def calculate_cherry_damage(zombie_type: int) -> int:
    """
    Calculate cherry bomb damage to a zombie
    
    Args:
        zombie_type: Zombie type ID
        
    Returns:
        Damage value
    """
    return get_instant_damage_to_zombie('cherry', zombie_type)


def calculate_doom_damage(zombie_type: int) -> int:
    """
    Calculate doom shroom damage to a zombie
    
    Args:
        zombie_type: Zombie type ID
        
    Returns:
        Damage value
    """
    return get_instant_damage_to_zombie('doom', zombie_type)


def calculate_jalapeno_damage(zombie_type: int) -> int:
    """
    Calculate jalapeno damage to a zombie
    
    Args:
        zombie_type: Zombie type ID
        
    Returns:
        Damage value
    """
    return get_instant_damage_to_zombie('jalapeno', zombie_type)


# ============================================================================
# Kill Requirements
# ============================================================================

def get_damage_to_kill(zombie_type: int, current_hp: Optional[int] = None,
                       accessory_hp: Optional[int] = None) -> int:
    """
    Calculate total damage needed to kill a zombie
    
    Args:
        zombie_type: Zombie type ID
        current_hp: Current HP (if None, uses max HP)
        accessory_hp: Current accessory HP (if None, uses max)
        
    Returns:
        Total damage needed
    """
    if current_hp is not None:
        total_hp = current_hp
        if accessory_hp is not None:
            total_hp += accessory_hp
        return total_hp
    
    return get_zombie_total_hp(zombie_type)


def cobs_needed_to_kill(zombie_type: int, current_hp: Optional[int] = None) -> int:
    """
    Calculate number of cob hits needed to kill a zombie
    
    Args:
        zombie_type: Zombie type ID
        current_hp: Current HP (if None, uses max HP)
        
    Returns:
        Number of cob hits needed
    """
    hp = current_hp if current_hp is not None else get_zombie_total_hp(zombie_type)
    damage_per_cob = calculate_cob_damage(zombie_type)
    
    if damage_per_cob <= 0:
        return 999
    
    return math.ceil(hp / damage_per_cob)


def can_instant_kill(weapon_type: str, zombie_type: int,
                     current_hp: Optional[int] = None) -> bool:
    """
    Check if an instant kill weapon can kill a zombie
    
    Args:
        weapon_type: Type of instant weapon
        zombie_type: Zombie type ID
        current_hp: Current HP (if None, uses max HP)
        
    Returns:
        True if weapon can kill the zombie
    """
    damage = get_instant_damage_to_zombie(weapon_type, zombie_type)
    hp = current_hp if current_hp is not None else get_zombie_total_hp(zombie_type)
    return damage >= hp


def get_remaining_hp_after_hit(zombie_type: int, weapon_type: str,
                                current_hp: int) -> int:
    """
    Calculate remaining HP after a weapon hit
    
    Args:
        zombie_type: Zombie type ID
        weapon_type: Type of weapon
        current_hp: Current HP
        
    Returns:
        Remaining HP (0 if killed)
    """
    if weapon_type in INSTANT_DAMAGE:
        damage = get_instant_damage_to_zombie(weapon_type, zombie_type)
    else:
        damage = get_weapon_damage(weapon_type)
    
    return max(0, current_hp - damage)


# ============================================================================
# DPS Calculations
# ============================================================================

def calculate_dps(damage: int, attack_interval: int) -> float:
    """
    Calculate damage per centisecond
    
    Args:
        damage: Damage per attack
        attack_interval: Time between attacks (cs)
        
    Returns:
        DPS value (hp/cs)
    """
    if attack_interval <= 0:
        return 0.0
    return damage / attack_interval


def get_peashooter_dps() -> float:
    """
    Get peashooter DPS
    
    Returns:
        DPS value (hp/cs) - 20/141 ≈ 0.142
    """
    return calculate_dps(PEA_DAMAGE, PEASHOOTER_ATTACK_INTERVAL)


def get_repeater_dps() -> float:
    """
    Get repeater DPS (2 peas per shot)
    
    Returns:
        DPS value (hp/cs) - 40/141 ≈ 0.284
    """
    return calculate_dps(PEA_DAMAGE * 2, PEASHOOTER_ATTACK_INTERVAL)


def get_gatling_dps() -> float:
    """
    Get gatling pea DPS (4 peas per shot)
    
    Returns:
        DPS value (hp/cs) - 80/141 ≈ 0.567
    """
    return calculate_dps(PEA_DAMAGE * 4, PEASHOOTER_ATTACK_INTERVAL)


def get_gloom_dps() -> float:
    """
    Get gloom-shroom DPS
    
    Gloom attacks every 200cs, dealing 80 damage.
    
    Returns:
        DPS value (hp/cs) - 80/200 = 0.4
    """
    return GLOOM_DAMAGE_PER_CS


def get_melon_dps() -> float:
    """
    Get melon-pult DPS (approximate)
    
    Melon attacks every ~300cs.
    
    Returns:
        DPS value (hp/cs)
    """
    return calculate_dps(MELON_DAMAGE, 300)


def calculate_time_to_kill(target_hp: int, dps: float) -> float:
    """
    Calculate time to kill a target with sustained DPS
    
    Args:
        target_hp: Target HP
        dps: Damage per centisecond
        
    Returns:
        Time in centiseconds (cs)
    """
    if dps <= 0:
        return float('inf')
    return target_hp / dps


def calculate_gloom_time_to_kill(target_hp: int, num_glooms: int = 1) -> float:
    """
    Calculate time for gloom-shrooms to kill a target
    
    Args:
        target_hp: Target HP
        num_glooms: Number of glooms hitting the target
        
    Returns:
        Time in centiseconds (cs)
    """
    total_dps = get_gloom_dps() * num_glooms
    return calculate_time_to_kill(target_hp, total_dps)


# ============================================================================
# Damage Efficiency Evaluation
# ============================================================================

def calculate_overkill(damage_dealt: int, target_hp: int) -> int:
    """
    Calculate wasted damage (overkill)
    
    Args:
        damage_dealt: Total damage dealt
        target_hp: Target HP before attack
        
    Returns:
        Overkill amount (positive = wasted, negative = not killed)
    """
    return damage_dealt - target_hp


def calculate_damage_efficiency(damage_dealt: int, target_hp: int) -> float:
    """
    Calculate damage efficiency (useful damage / total damage)
    
    Args:
        damage_dealt: Total damage dealt
        target_hp: Target HP before attack
        
    Returns:
        Efficiency ratio (0.0 to 1.0)
    """
    if damage_dealt <= 0:
        return 0.0
    
    useful_damage = min(damage_dealt, target_hp)
    return useful_damage / damage_dealt


def evaluate_cob_efficiency(zombies_hit: List[Tuple[int, int]]) -> dict:
    """
    Evaluate efficiency of a cob cannon hit
    
    Args:
        zombies_hit: List of (hp, zombie_type) tuples
        
    Returns:
        Dictionary with:
        - total_hp: Total HP of targets
        - total_damage: Total damage dealt
        - useful_damage: Damage that actually affected HP
        - wasted_damage: Overkill damage
        - efficiency: Ratio of useful/total damage
        - kills: Number of zombies killed
    """
    if not zombies_hit:
        return {
            'total_hp': 0,
            'total_damage': 0,
            'useful_damage': 0,
            'wasted_damage': 0,
            'efficiency': 0.0,
            'kills': 0,
        }
    
    total_hp = sum(hp for hp, _ in zombies_hit)
    total_damage = 0
    useful_damage = 0
    kills = 0
    
    for hp, zombie_type in zombies_hit:
        damage = calculate_cob_damage(zombie_type)
        total_damage += damage
        useful_damage += min(damage, hp)
        if damage >= hp:
            kills += 1
    
    return {
        'total_hp': total_hp,
        'total_damage': total_damage,
        'useful_damage': useful_damage,
        'wasted_damage': total_damage - useful_damage,
        'efficiency': useful_damage / total_damage if total_damage > 0 else 0.0,
        'kills': kills,
    }


def compare_weapon_efficiency(weapon1: str, weapon2: str, 
                              target_type: int) -> dict:
    """
    Compare efficiency of two weapons against a target
    
    Args:
        weapon1: First weapon type
        weapon2: Second weapon type
        target_type: Target zombie type
        
    Returns:
        Dictionary with comparison results
    """
    hp = get_zombie_total_hp(target_type)
    
    damage1 = get_instant_damage_to_zombie(weapon1, target_type) \
              if weapon1 in INSTANT_DAMAGE else get_weapon_damage(weapon1)
    damage2 = get_instant_damage_to_zombie(weapon2, target_type) \
              if weapon2 in INSTANT_DAMAGE else get_weapon_damage(weapon2)
    
    return {
        'weapon1': {
            'name': weapon1,
            'damage': damage1,
            'can_kill': damage1 >= hp,
            'hits_to_kill': (hp + damage1 - 1) // damage1 if damage1 > 0 else 999,
        },
        'weapon2': {
            'name': weapon2,
            'damage': damage2,
            'can_kill': damage2 >= hp,
            'hits_to_kill': (hp + damage2 - 1) // damage2 if damage2 > 0 else 999,
        },
        'target_hp': hp,
    }


# ============================================================================
# Gargantuar Specific Damage
# ============================================================================

def get_garg_damage_reduction() -> float:
    """
    Get Gargantuar damage reduction ratio for instant kills
    
    Returns:
        Damage multiplier (0.5 = takes half damage)
    """
    return 0.5


def calculate_garg_instant_damage(weapon_type: str) -> int:
    """
    Calculate instant kill damage to Gargantuar
    
    Args:
        weapon_type: Type of instant weapon
        
    Returns:
        Damage value (half of normal)
    """
    base = INSTANT_DAMAGE.get(weapon_type, 1800)
    return base // 2


def cobs_to_kill_garg(is_giga: bool = False, current_hp: Optional[int] = None) -> int:
    """
    Calculate cobs needed to kill a Gargantuar
    
    Args:
        is_giga: True for Giga Gargantuar (red-eye)
        current_hp: Current HP (if None, uses max)
        
    Returns:
        Number of cob hits needed
    """
    zombie_type = ZombieType.GIGA_GARGANTUAR if is_giga else ZombieType.GARGANTUAR
    return cobs_needed_to_kill(zombie_type, current_hp)
