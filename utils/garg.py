"""
Gargantuar Special Handling Utilities
巨人特殊处理工具

Provides calculations for:
- Hammer timing detection (animation circulation rate 0.644)
- Hammer damage range (-30 to +59 pixels)
- Imp throwing timing (when HP < 50%)
- Gargantuar instant kill damage reduction (half damage)

All time values are in centiseconds (cs) = 1/100 second.

Reference:
- data/constants.py for HAMMER_CIRCULATION_RATE
- data/zombies.py for Gargantuar HP and ranges
- AVZ judge.h for hammer detection logic
"""

import math
from typing import Optional, Tuple, Dict, List

from data.constants import (
    HAMMER_CIRCULATION_RATE,
    GIGA_THROW_IMP_TIME,
    GIGA_AVG_SPEED,
    DOOM_DAMAGE_GARG,
)
from data.zombies import (
    ZombieType,
    ZOMBIE_HP_DATA,
    GARGANTUAR_ZOMBIES,
    GARG_HAMMER_RANGE_LEFT,
    GARG_HAMMER_RANGE_RIGHT,
    GARG_THROW_IMP_HP_THRESHOLD_1,
    GARG_THROW_IMP_HP_THRESHOLD_2,
    GARG_HAMMER_TIME,
    is_gargantuar,
)


# ============================================================================
# Gargantuar Constants
# ============================================================================

# Hammer attack animation threshold
# When animation progress > HAMMER_CIRCULATION_RATE, hammer is coming down
# Reference: AVZ judge.h - isGigaHammer()

# Gargantuar HP values
GARG_MAX_HP = ZOMBIE_HP_DATA[ZombieType.GARGANTUAR][0]  # 3000
GIGA_MAX_HP = ZOMBIE_HP_DATA[ZombieType.GIGA_GARGANTUAR][0]  # 6000

# Damage values for Gargantuars
GARG_INSTANT_DAMAGE = 900  # Half of normal 1800


# ============================================================================
# Hammer Attack Timing
# ============================================================================

def is_hammer_coming(animation_progress: float) -> bool:
    """
    Check if Gargantuar hammer attack is in progress
    
    The hammer comes down when animation progress exceeds the circulation rate.
    
    Args:
        animation_progress: Animation progress from 0.0 to 1.0
        
    Returns:
        True if hammer is about to hit
    """
    return animation_progress > HAMMER_CIRCULATION_RATE


def get_hammer_circulation_rate() -> float:
    """
    Get the hammer animation circulation rate threshold
    
    Returns:
        Circulation rate - 0.644
    """
    return HAMMER_CIRCULATION_RATE


def calculate_time_to_hammer(animation_progress: float, 
                              animation_speed: float = 0.01) -> float:
    """
    Estimate time until hammer hits
    
    Args:
        animation_progress: Current animation progress (0.0-1.0)
        animation_speed: Animation progress per cs (approximate)
        
    Returns:
        Estimated time until hammer hit (cs)
    """
    if animation_progress >= HAMMER_CIRCULATION_RATE:
        return 0.0  # Hammer already coming
    
    remaining_progress = HAMMER_CIRCULATION_RATE - animation_progress
    if animation_speed <= 0:
        return float('inf')
    
    return remaining_progress / animation_speed


def is_giga_hammer_attack(zombie_type: int, animation_progress: float) -> bool:
    """
    Check if a zombie is a Gargantuar performing a hammer attack
    
    Args:
        zombie_type: Zombie type ID
        animation_progress: Animation progress
        
    Returns:
        True if this is a hammer attack
    """
    if not is_gargantuar(zombie_type):
        return False
    return is_hammer_coming(animation_progress)


# ============================================================================
# Hammer Damage Range
# ============================================================================

def get_hammer_range() -> Tuple[int, int]:
    """
    Get the hammer attack x-range relative to Gargantuar position
    
    Returns:
        (left_offset, right_offset) in pixels
        Left is -30, right is +59
    """
    return (GARG_HAMMER_RANGE_LEFT, GARG_HAMMER_RANGE_RIGHT)


def is_in_hammer_range(zombie_x: float, target_x: float) -> bool:
    """
    Check if a target is within Gargantuar hammer range
    
    Args:
        zombie_x: Gargantuar x position
        target_x: Target (plant) x position
        
    Returns:
        True if target is in hammer range
    """
    left_bound = zombie_x + GARG_HAMMER_RANGE_LEFT   # -30
    right_bound = zombie_x + GARG_HAMMER_RANGE_RIGHT  # +59
    return left_bound <= target_x <= right_bound


def get_hammer_danger_zone(zombie_x: float) -> Tuple[float, float]:
    """
    Get the x-coordinates of the hammer danger zone
    
    Args:
        zombie_x: Gargantuar x position
        
    Returns:
        (left_x, right_x) danger zone boundaries
    """
    return (zombie_x + GARG_HAMMER_RANGE_LEFT, 
            zombie_x + GARG_HAMMER_RANGE_RIGHT)


def will_plant_be_hammered(zombie_x: float, zombie_row: int,
                            plant_x: float, plant_row: int,
                            animation_progress: float) -> bool:
    """
    Check if a plant will be hammered by a Gargantuar
    
    Args:
        zombie_x: Gargantuar x position
        zombie_row: Gargantuar row
        plant_x: Plant x position
        plant_row: Plant row
        animation_progress: Current animation progress
        
    Returns:
        True if plant is about to be hammered
    """
    # Must be same row
    if zombie_row != plant_row:
        return False
    
    # Must be in range
    if not is_in_hammer_range(zombie_x, plant_x):
        return False
    
    # Hammer must be coming
    return is_hammer_coming(animation_progress)


# ============================================================================
# Imp Throwing Timing
# ============================================================================

def get_throw_imp_hp_threshold(is_giga: bool = False) -> int:
    """
    Get HP threshold for throwing imp
    
    Normal Gargantuar throws imp at 50% HP.
    Giga Gargantuar throws imp at 50% HP (first) and 25% HP (second).
    
    Args:
        is_giga: True for Giga Gargantuar
        
    Returns:
        HP threshold for first imp throw
    """
    max_hp = GIGA_MAX_HP if is_giga else GARG_MAX_HP
    return int(max_hp * GARG_THROW_IMP_HP_THRESHOLD_1)


def will_throw_imp(current_hp: int, is_giga: bool = False,
                   has_thrown_first: bool = False) -> bool:
    """
    Check if Gargantuar will throw an imp based on HP
    
    Args:
        current_hp: Current HP
        is_giga: True for Giga Gargantuar
        has_thrown_first: Whether first imp has been thrown (Giga only)
        
    Returns:
        True if will throw imp
    """
    max_hp = GIGA_MAX_HP if is_giga else GARG_MAX_HP
    hp_ratio = current_hp / max_hp
    
    if is_giga and has_thrown_first:
        # Giga's second imp at 25%
        return hp_ratio <= GARG_THROW_IMP_HP_THRESHOLD_2
    else:
        # First imp at 50%
        return hp_ratio <= GARG_THROW_IMP_HP_THRESHOLD_1


def get_imp_throw_timings() -> List[int]:
    """
    Get the timing values for imp throwing animation
    
    Returns:
        List of timing values (cs) - [105, 210]
    """
    return list(GIGA_THROW_IMP_TIME)


def calculate_damage_to_trigger_imp(current_hp: int, is_giga: bool = False,
                                     has_thrown_first: bool = False) -> int:
    """
    Calculate damage needed to trigger imp throw
    
    Args:
        current_hp: Current HP
        is_giga: True for Giga Gargantuar
        has_thrown_first: Whether first imp has been thrown
        
    Returns:
        Damage needed to reach imp throw threshold
    """
    max_hp = GIGA_MAX_HP if is_giga else GARG_MAX_HP
    
    if is_giga and has_thrown_first:
        threshold_hp = int(max_hp * GARG_THROW_IMP_HP_THRESHOLD_2)
    else:
        threshold_hp = int(max_hp * GARG_THROW_IMP_HP_THRESHOLD_1)
    
    damage_needed = current_hp - threshold_hp
    return max(0, damage_needed)


# ============================================================================
# Gargantuar Damage Reduction
# ============================================================================

def get_garg_damage_reduction() -> float:
    """
    Get Gargantuar damage reduction multiplier for instant kills
    
    Returns:
        Damage multiplier - 0.5 (takes half damage)
    """
    return 0.5


def calculate_garg_instant_damage() -> int:
    """
    Get instant kill damage dealt to Gargantuars
    
    Returns:
        Damage value - 900 (half of 1800)
    """
    return DOOM_DAMAGE_GARG


def cobs_to_kill_garg(current_hp: Optional[int] = None) -> int:
    """
    Calculate cob hits needed to kill a normal Gargantuar
    
    Args:
        current_hp: Current HP (if None, uses max HP 3000)
        
    Returns:
        Number of cob hits needed
    """
    hp = current_hp if current_hp is not None else GARG_MAX_HP
    return math.ceil(hp / GARG_INSTANT_DAMAGE)


def cobs_to_kill_giga(current_hp: Optional[int] = None) -> int:
    """
    Calculate cob hits needed to kill a Giga Gargantuar
    
    Args:
        current_hp: Current HP (if None, uses max HP 6000)
        
    Returns:
        Number of cob hits needed
    """
    hp = current_hp if current_hp is not None else GIGA_MAX_HP
    return math.ceil(hp / GARG_INSTANT_DAMAGE)


def calculate_remaining_hp_after_cobs(current_hp: int, num_cobs: int,
                                       zombie_type: int) -> int:
    """
    Calculate remaining HP after cob hits
    
    Args:
        current_hp: Current HP
        num_cobs: Number of cob hits
        zombie_type: Zombie type
        
    Returns:
        Remaining HP
    """
    if not is_gargantuar(zombie_type):
        # Non-gargs take full damage
        damage = 1800 * num_cobs
    else:
        damage = GARG_INSTANT_DAMAGE * num_cobs
    
    return max(0, current_hp - damage)


# ============================================================================
# Gargantuar Speed and Movement
# ============================================================================

def get_garg_average_speed() -> float:
    """
    Get Gargantuar average speed (accounting for hammer animations)
    
    The average speed is calculated from AVZ:
    float(484) / 3158 * 1.25 ≈ 0.192 px/cs
    
    Returns:
        Average speed in pixels/cs
    """
    return GIGA_AVG_SPEED


def predict_garg_position(zombie_x: float, time_cs: float,
                          is_attacking: bool = False) -> float:
    """
    Predict Gargantuar position after given time
    
    Uses average speed that accounts for attack animations.
    
    Args:
        zombie_x: Current x position
        time_cs: Time to predict (cs)
        is_attacking: Whether currently attacking (not moving)
        
    Returns:
        Predicted x position
    """
    if is_attacking:
        return zombie_x  # No movement during attack
    
    return zombie_x - GIGA_AVG_SPEED * time_cs


def estimate_garg_arrival_time(zombie_x: float, target_x: float) -> float:
    """
    Estimate time for Gargantuar to reach target position
    
    Args:
        zombie_x: Current x position
        target_x: Target x position
        
    Returns:
        Estimated time in cs
    """
    distance = zombie_x - target_x
    if distance <= 0:
        return 0.0
    
    return distance / GIGA_AVG_SPEED


# ============================================================================
# Gargantuar Combat Analysis
# ============================================================================

def analyze_garg_threat(zombie_x: float, zombie_hp: int, zombie_row: int,
                        zombie_type: int, animation_progress: float = 0.0,
                        target_x: float = 200) -> Dict[str, any]:
    """
    Analyze Gargantuar threat level
    
    Args:
        zombie_x: Current x position
        zombie_hp: Current HP
        zombie_row: Zombie row
        zombie_type: Zombie type
        animation_progress: Current animation progress
        target_x: Target position to defend
        
    Returns:
        Dictionary with threat analysis
    """
    is_giga = zombie_type == ZombieType.GIGA_GARGANTUAR
    max_hp = GIGA_MAX_HP if is_giga else GARG_MAX_HP
    
    # Calculate cobs needed
    cobs_needed = cobs_to_kill_giga(zombie_hp) if is_giga else cobs_to_kill_garg(zombie_hp)
    
    # Estimate arrival time
    arrival_time = estimate_garg_arrival_time(zombie_x, target_x)
    
    # Check if hammer is coming
    hammer_imminent = is_hammer_coming(animation_progress)
    
    # Check imp throw status
    can_throw_imp = will_throw_imp(zombie_hp, is_giga)
    
    # HP percentage
    hp_percentage = zombie_hp / max_hp * 100
    
    return {
        'zombie_type': zombie_type,
        'is_giga': is_giga,
        'current_hp': zombie_hp,
        'max_hp': max_hp,
        'hp_percentage': hp_percentage,
        'cobs_needed': cobs_needed,
        'arrival_time': arrival_time,
        'hammer_imminent': hammer_imminent,
        'can_throw_imp': can_throw_imp,
        'row': zombie_row,
        'threat_level': 'high' if is_giga or zombie_x < 400 else 'medium',
    }


def get_optimal_cob_count_for_gargs(garg_count: int, giga_count: int,
                                     avg_hp_ratio: float = 1.0) -> int:
    """
    Calculate optimal number of cobs to handle multiple Gargantuars
    
    Args:
        garg_count: Number of normal Gargantuars
        giga_count: Number of Giga Gargantuars
        avg_hp_ratio: Average HP ratio (1.0 = full HP)
        
    Returns:
        Recommended number of cobs
    """
    # Normal Gargs need ~4 cobs at full HP (3000/900 = 3.33)
    # Gigas need ~7 cobs at full HP (6000/900 = 6.67)
    
    garg_cobs = garg_count * math.ceil(GARG_MAX_HP * avg_hp_ratio / GARG_INSTANT_DAMAGE)
    giga_cobs = giga_count * math.ceil(GIGA_MAX_HP * avg_hp_ratio / GARG_INSTANT_DAMAGE)
    
    return garg_cobs + giga_cobs
