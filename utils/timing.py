"""
Timing Calculation Utilities
时机计算工具

Provides timing calculations for:
- Cob cannon flight time (normal and roof scenes)
- Instant kill plant activation delay
- Ice mushroom effect and duration
- Zombie travel time (with status effects)
- Cob cannon cooldown

All time values are in centiseconds (cs) = 1/100 second.

Reference: data/constants.py for timing constants
"""

import math
from typing import Optional

from data.constants import (
    # Cob cannon
    COB_FLY_TIME,
    ROOF_COB_FLY_TIME,
    COB_RECOVER_TIME,
    # Instant plants
    CHERRY_DELAY,
    JALAPENO_DELAY,
    DOOM_DELAY,
    SQUASH_DELAY,
    POTATO_MINE_ARM_TIME,
    # Ice effects
    ICE_EFFECT_TIME,
    ICE_DURATION,
    SLOW_DURATION,
    # Scene types
    SCENE_ROOF,
    SCENE_ROOF_NIGHT,
    # Grid
    GRID_WIDTH,
    LAWN_LEFT_X,
)
from data.zombies import SLOW_SPEED_MULTIPLIER


# ============================================================================
# Cob Cannon Flight Time
# ============================================================================

def get_cob_fly_time(scene: int = 0, target_col: Optional[int] = None) -> int:
    """
    Get cob cannon flight time based on scene and target column
    
    For non-roof scenes: constant 373cs
    For roof scenes: varies by target column (359-373cs)
    
    Args:
        scene: Scene type (0-5)
        target_col: Target column (0-8), only used for roof scenes
        
    Returns:
        Flight time in centiseconds (cs)
    """
    if scene in (SCENE_ROOF, SCENE_ROOF_NIGHT):
        if target_col is not None and 0 <= target_col <= 7:
            # ROOF_COB_FLY_TIME[0-6] corresponds to columns 1-7
            # For col 0, use col 1's time (359)
            # For col 7, use index 6 (373)
            if target_col == 0:
                return ROOF_COB_FLY_TIME[0]  # 359
            elif target_col <= 7:
                return ROOF_COB_FLY_TIME[min(target_col - 1, 6)]
        else:
            # Default to rightmost column time
            return COB_FLY_TIME  # 373
    
    return COB_FLY_TIME  # 373cs for non-roof scenes


def get_roof_cob_fly_times() -> dict:
    """
    Get all roof cob fly times by column
    
    Returns:
        Dictionary mapping column (1-7) to flight time (cs)
    """
    return {
        1: ROOF_COB_FLY_TIME[0],  # 359
        2: ROOF_COB_FLY_TIME[1],  # 362
        3: ROOF_COB_FLY_TIME[2],  # 364
        4: ROOF_COB_FLY_TIME[3],  # 367
        5: ROOF_COB_FLY_TIME[4],  # 369
        6: ROOF_COB_FLY_TIME[5],  # 372
        7: ROOF_COB_FLY_TIME[6],  # 373
    }


# ============================================================================
# Instant Kill Plant Delay
# ============================================================================

def get_instant_plant_delay(plant_type: str) -> int:
    """
    Get activation delay for instant kill plants
    
    These plants have a 100cs delay from placement to effect.
    
    Args:
        plant_type: 'cherry', 'jalapeno', 'doom', or 'squash'
        
    Returns:
        Delay in centiseconds (cs)
    """
    delays = {
        'cherry': CHERRY_DELAY,      # 100cs
        'jalapeno': JALAPENO_DELAY,  # 100cs
        'doom': DOOM_DELAY,          # 100cs
        'squash': SQUASH_DELAY,      # 100cs
    }
    return delays.get(plant_type, 100)


def get_potato_mine_arm_time() -> int:
    """
    Get potato mine arming time
    
    Returns:
        Arming time in centiseconds (cs) - 1500cs
    """
    return POTATO_MINE_ARM_TIME


def calculate_instant_plant_effect_time(placement_time: int, plant_type: str) -> int:
    """
    Calculate when an instant kill plant will take effect
    
    Args:
        placement_time: Game time when plant is placed (cs)
        plant_type: Type of instant plant
        
    Returns:
        Game time when plant takes effect (cs)
    """
    delay = get_instant_plant_delay(plant_type)
    return placement_time + delay


# ============================================================================
# Ice Mushroom Timing
# ============================================================================

def get_ice_effect_delay() -> int:
    """
    Get ice mushroom effect activation delay
    
    Returns:
        Delay from placement to effect in centiseconds (cs) - 298cs
    """
    return ICE_EFFECT_TIME


def get_ice_duration() -> int:
    """
    Get ice freeze duration
    
    Returns:
        Duration in centiseconds (cs) - 400cs
    """
    return ICE_DURATION


def get_slow_duration() -> int:
    """
    Get slow effect duration
    
    Returns:
        Duration in centiseconds (cs) - 1000cs
    """
    return SLOW_DURATION


def calculate_ice_effect_timing(placement_time: int) -> dict:
    """
    Calculate all timing related to ice mushroom effect
    
    Args:
        placement_time: Game time when ice mushroom is placed
        
    Returns:
        Dictionary with timing information:
        - effect_start: When freeze takes effect
        - freeze_end: When freeze effect ends
        - slow_end: When slow effect ends (after freeze)
    """
    effect_start = placement_time + ICE_EFFECT_TIME
    freeze_end = effect_start + ICE_DURATION
    slow_end = freeze_end + SLOW_DURATION
    
    return {
        'effect_start': effect_start,
        'freeze_end': freeze_end,
        'slow_end': slow_end,
        'total_effect_duration': ICE_DURATION + SLOW_DURATION,
    }


def get_ice_status_at_time(ice_effect_start: int, current_time: int) -> str:
    """
    Get the ice effect status at a given time
    
    Args:
        ice_effect_start: When ice effect started
        current_time: Current game time
        
    Returns:
        'frozen', 'slowed', or 'none'
    """
    elapsed = current_time - ice_effect_start
    
    if elapsed < 0:
        return 'none'  # Ice hasn't taken effect yet
    elif elapsed < ICE_DURATION:
        return 'frozen'
    elif elapsed < ICE_DURATION + SLOW_DURATION:
        return 'slowed'
    else:
        return 'none'


# ============================================================================
# Zombie Travel Time
# ============================================================================

def calculate_travel_time(distance: float, speed: float,
                          is_slowed: bool = False, is_frozen: bool = False) -> float:
    """
    Calculate time for a zombie to travel a given distance
    
    Args:
        distance: Distance to travel in pixels
        speed: Zombie speed in pixels/cs
        is_slowed: Whether zombie is slowed (50% speed)
        is_frozen: Whether zombie is frozen (0 speed)
        
    Returns:
        Time in centiseconds (cs), or inf if frozen
    """
    if is_frozen or speed <= 0:
        return float('inf')
    
    effective_speed = speed * SLOW_SPEED_MULTIPLIER if is_slowed else speed
    if effective_speed <= 0:
        return float('inf')
    
    return distance / effective_speed


def calculate_time_to_target_x(zombie_x: float, target_x: float, speed: float,
                                is_slowed: bool = False, is_frozen: bool = False) -> float:
    """
    Calculate time for zombie to reach a target x position
    
    Args:
        zombie_x: Current zombie x position
        target_x: Target x position
        speed: Zombie speed in pixels/cs
        is_slowed: Whether zombie is slowed
        is_frozen: Whether zombie is frozen
        
    Returns:
        Time in centiseconds (cs), or inf if cannot reach
    """
    distance = zombie_x - target_x  # Zombies move left (decreasing x)
    
    if distance <= 0:
        return 0.0  # Already at or past target
    
    return calculate_travel_time(distance, speed, is_slowed, is_frozen)


def calculate_time_to_column(zombie_x: float, target_col: int, speed: float,
                             is_slowed: bool = False, is_frozen: bool = False) -> float:
    """
    Calculate time for zombie to reach a target column
    
    Args:
        zombie_x: Current zombie x position
        target_col: Target column (0-8)
        speed: Zombie speed in pixels/cs
        is_slowed: Whether zombie is slowed
        is_frozen: Whether zombie is frozen
        
    Returns:
        Time in centiseconds (cs)
    """
    target_x = LAWN_LEFT_X + target_col * GRID_WIDTH
    return calculate_time_to_target_x(zombie_x, target_x, speed, is_slowed, is_frozen)


def calculate_travel_time_with_effects(distance: float, speed: float,
                                        freeze_remaining: int = 0,
                                        slow_remaining: int = 0) -> float:
    """
    Calculate travel time considering status effect durations
    
    This accounts for effects wearing off during travel.
    
    Args:
        distance: Distance to travel in pixels
        speed: Base zombie speed in pixels/cs
        freeze_remaining: Remaining freeze time (cs)
        slow_remaining: Remaining slow time (cs)
        
    Returns:
        Total time in centiseconds (cs)
    """
    if distance <= 0:
        return 0.0
    
    if speed <= 0:
        return float('inf')
    
    total_time = 0.0
    remaining_distance = distance
    
    # Phase 1: Frozen (no movement)
    if freeze_remaining > 0:
        total_time += freeze_remaining
    
    # Phase 2: Slowed movement (after freeze wears off)
    slow_after_freeze = max(0, slow_remaining - freeze_remaining)
    if slow_after_freeze > 0:
        slowed_speed = speed * SLOW_SPEED_MULTIPLIER
        distance_while_slowed = slowed_speed * slow_after_freeze
        
        if distance_while_slowed >= remaining_distance:
            # Reaches destination while slowed
            return total_time + remaining_distance / slowed_speed
        else:
            total_time += slow_after_freeze
            remaining_distance -= distance_while_slowed
    
    # Phase 3: Normal speed
    if remaining_distance > 0:
        total_time += remaining_distance / speed
    
    return total_time


# ============================================================================
# Cob Cannon Cooldown
# ============================================================================

def get_cob_cooldown() -> int:
    """
    Get cob cannon cooldown time
    
    Returns:
        Cooldown time in centiseconds (cs) - 3475cs
    """
    return COB_RECOVER_TIME


def calculate_next_cob_ready_time(last_fire_time: int) -> int:
    """
    Calculate when cob cannon will be ready to fire again
    
    Args:
        last_fire_time: Game time when cob was last fired
        
    Returns:
        Game time when cob will be ready
    """
    return last_fire_time + COB_RECOVER_TIME


def calculate_cob_availability(cob_countdown: int, current_time: int) -> dict:
    """
    Calculate cob cannon availability information
    
    Args:
        cob_countdown: Remaining cooldown from memory (P_COB_COUNTDOWN)
        current_time: Current game time
        
    Returns:
        Dictionary with:
        - is_ready: Whether cob can fire now
        - ready_at: Game time when ready (or current time if ready)
        - countdown: Remaining cooldown
    """
    is_ready = cob_countdown <= 0
    
    return {
        'is_ready': is_ready,
        'ready_at': current_time if is_ready else current_time + cob_countdown,
        'countdown': max(0, cob_countdown),
    }


# ============================================================================
# Combined Timing Calculations
# ============================================================================

def calculate_cob_intercept_timing(zombie_x: float, zombie_speed: float,
                                    target_col: float, scene: int = 0) -> dict:
    """
    Calculate timing for cob cannon to intercept a moving zombie
    
    Args:
        zombie_x: Current zombie x position
        zombie_speed: Zombie speed in pixels/cs
        target_col: Target column for cob
        scene: Scene type
        
    Returns:
        Dictionary with:
        - fire_delay: When to fire cob (cs from now)
        - impact_time: When cob will land (cs from now)
        - zombie_x_at_impact: Predicted zombie x at impact
    """
    target_x = LAWN_LEFT_X + target_col * GRID_WIDTH
    fly_time = get_cob_fly_time(scene, int(target_col))
    
    if zombie_speed <= 0:
        # Zombie not moving, fire immediately
        return {
            'fire_delay': 0,
            'impact_time': fly_time,
            'zombie_x_at_impact': zombie_x,
        }
    
    # Calculate when zombie will be at target
    time_to_target = (zombie_x - target_x) / zombie_speed
    
    # Fire delay = time to target - fly time
    fire_delay = max(0, time_to_target - fly_time)
    
    # Predicted zombie position at impact
    impact_time = fire_delay + fly_time
    zombie_x_at_impact = zombie_x - zombie_speed * impact_time
    
    return {
        'fire_delay': fire_delay,
        'impact_time': impact_time,
        'zombie_x_at_impact': zombie_x_at_impact,
    }


def calculate_instant_plant_intercept_timing(zombie_x: float, zombie_speed: float,
                                              target_col: int, plant_type: str) -> dict:
    """
    Calculate timing for instant kill plant to intercept a moving zombie
    
    Args:
        zombie_x: Current zombie x position
        zombie_speed: Zombie speed in pixels/cs
        target_col: Target column for plant
        plant_type: Type of instant plant
        
    Returns:
        Dictionary with:
        - place_delay: When to place plant (cs from now)
        - effect_time: When plant will take effect (cs from now)
        - zombie_x_at_effect: Predicted zombie x at effect
    """
    target_x = LAWN_LEFT_X + target_col * GRID_WIDTH
    effect_delay = get_instant_plant_delay(plant_type)
    
    if zombie_speed <= 0:
        return {
            'place_delay': 0,
            'effect_time': effect_delay,
            'zombie_x_at_effect': zombie_x,
        }
    
    time_to_target = (zombie_x - target_x) / zombie_speed
    place_delay = max(0, time_to_target - effect_delay)
    
    effect_time = place_delay + effect_delay
    zombie_x_at_effect = zombie_x - zombie_speed * effect_time
    
    return {
        'place_delay': place_delay,
        'effect_time': effect_time,
        'zombie_x_at_effect': zombie_x_at_effect,
    }
