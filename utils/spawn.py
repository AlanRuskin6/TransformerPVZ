"""
Spawn Prediction Utilities
出怪预测工具

Provides spawn-related calculations for:
- Reading spawn list (ZOMBIE_LIST offset 0x6B4)
- Reading spawn type list (ZOMBIE_TYPE_LIST offset 0x54D4)
- Predicting next wave zombie types
- Wave refresh time prediction
- Huge wave detection (every 10th wave)

All time values are in centiseconds (cs) = 1/100 second.

Reference:
- data/offsets.py for memory offsets
- data/zombies.py for zombie types
"""

import math
from typing import List, Optional, Dict, Set, Tuple
from enum import IntEnum

from data.zombies import ZombieType, is_dangerous_zombie, GARGANTUAR_ZOMBIES
from data.offsets import Offset


# ============================================================================
# Spawn List Constants
# ============================================================================

# Memory offsets for spawn data
ZOMBIE_LIST_OFFSET = Offset.ZOMBIE_LIST  # 0x6B4
ZOMBIE_TYPE_LIST_OFFSET = Offset.ZOMBIE_TYPE_LIST  # 0x54D4

# Wave constants
WAVES_PER_FLAG = 10  # Huge wave every 10 waves
MAX_ZOMBIE_TYPES = 33  # Total zombie types in game


class WaveType(IntEnum):
    """Types of waves"""
    NORMAL = 0
    HUGE = 1  # Every 10th wave (flag wave)


# ============================================================================
# Spawn Type List Functions
# ============================================================================

def parse_zombie_type_list(type_list_data: List[bool]) -> List[ZombieType]:
    """
    Parse zombie type list from memory data
    
    The zombie type list is a bool array indicating which zombie types
    will appear in the level.
    
    Args:
        type_list_data: List of booleans (one per zombie type)
        
    Returns:
        List of ZombieType that will appear
    """
    enabled_types = []
    for i, enabled in enumerate(type_list_data):
        if enabled and i <= MAX_ZOMBIE_TYPES:
            try:
                zombie_type = ZombieType(i)
                enabled_types.append(zombie_type)
            except ValueError:
                # Invalid zombie type ID
                pass
    return enabled_types


def get_dangerous_types_in_level(type_list_data: List[bool]) -> List[ZombieType]:
    """
    Get dangerous zombie types that will appear in the level
    
    Args:
        type_list_data: Zombie type list from memory
        
    Returns:
        List of dangerous ZombieType
    """
    all_types = parse_zombie_type_list(type_list_data)
    return [z for z in all_types if is_dangerous_zombie(z)]


def has_gargantuar_in_level(type_list_data: List[bool]) -> bool:
    """
    Check if Gargantuars will appear in the level
    
    Args:
        type_list_data: Zombie type list from memory
        
    Returns:
        True if any Gargantuar type is present
    """
    all_types = parse_zombie_type_list(type_list_data)
    return any(z in GARGANTUAR_ZOMBIES for z in all_types)


def has_giga_in_level(type_list_data: List[bool]) -> bool:
    """
    Check if Giga Gargantuars will appear in the level
    
    Args:
        type_list_data: Zombie type list from memory
        
    Returns:
        True if Giga Gargantuar is present
    """
    if len(type_list_data) > ZombieType.GIGA_GARGANTUAR:
        return type_list_data[ZombieType.GIGA_GARGANTUAR]
    return False


# ============================================================================
# Wave Prediction
# ============================================================================

def is_huge_wave(wave_number: int) -> bool:
    """
    Check if a wave is a huge wave (flag wave)
    
    Huge waves occur every 10th wave (10, 20, etc.)
    
    Args:
        wave_number: Wave number (1-indexed)
        
    Returns:
        True if this is a huge wave
    """
    return wave_number > 0 and wave_number % WAVES_PER_FLAG == 0


def get_wave_type(wave_number: int) -> WaveType:
    """
    Get the type of a wave
    
    Args:
        wave_number: Wave number (1-indexed)
        
    Returns:
        WaveType (NORMAL or HUGE)
    """
    return WaveType.HUGE if is_huge_wave(wave_number) else WaveType.NORMAL


def get_next_huge_wave(current_wave: int) -> int:
    """
    Get the next huge wave number
    
    Args:
        current_wave: Current wave number
        
    Returns:
        Next huge wave number
    """
    if current_wave <= 0:
        return WAVES_PER_FLAG
    
    return ((current_wave // WAVES_PER_FLAG) + 1) * WAVES_PER_FLAG


def waves_until_huge(current_wave: int) -> int:
    """
    Calculate waves remaining until next huge wave
    
    Args:
        current_wave: Current wave number
        
    Returns:
        Number of waves until huge wave
    """
    next_huge = get_next_huge_wave(current_wave)
    return next_huge - current_wave


# ============================================================================
# Spawn List Functions
# ============================================================================

# Spawn list structure:
# Each wave has a list of zombie types that will spawn
# The spawn list is located at MainObject + ZOMBIE_LIST_OFFSET

def parse_wave_spawn_list(spawn_data: List[int], wave_index: int,
                          zombies_per_wave: int = 50) -> List[ZombieType]:
    """
    Parse spawn list for a specific wave
    
    Args:
        spawn_data: Raw spawn list data
        wave_index: Wave index (0-indexed)
        zombies_per_wave: Maximum zombies per wave
        
    Returns:
        List of ZombieType for the wave
    """
    start_index = wave_index * zombies_per_wave
    end_index = start_index + zombies_per_wave
    
    if start_index >= len(spawn_data):
        return []
    
    wave_zombies = []
    for i in range(start_index, min(end_index, len(spawn_data))):
        zombie_id = spawn_data[i]
        if zombie_id == -1:  # End of wave marker
            break
        try:
            zombie_type = ZombieType(zombie_id)
            wave_zombies.append(zombie_type)
        except ValueError:
            pass
    
    return wave_zombies


def predict_next_wave_zombies(spawn_data: List[int], current_wave: int,
                               total_waves: int = 20) -> List[ZombieType]:
    """
    Predict zombie types for the next wave
    
    Args:
        spawn_data: Raw spawn list data
        current_wave: Current wave number (1-indexed)
        total_waves: Total waves in level
        
    Returns:
        List of predicted ZombieType for next wave
    """
    next_wave_index = current_wave  # 0-indexed for next wave
    
    if next_wave_index >= total_waves:
        return []  # No more waves
    
    return parse_wave_spawn_list(spawn_data, next_wave_index)


def count_zombie_types_in_wave(spawn_data: List[int], wave_index: int) -> Dict[ZombieType, int]:
    """
    Count occurrences of each zombie type in a wave
    
    Args:
        spawn_data: Raw spawn list data
        wave_index: Wave index (0-indexed)
        
    Returns:
        Dictionary mapping ZombieType to count
    """
    wave_zombies = parse_wave_spawn_list(spawn_data, wave_index)
    
    counts = {}
    for zombie in wave_zombies:
        counts[zombie] = counts.get(zombie, 0) + 1
    
    return counts


def get_garg_count_in_wave(spawn_data: List[int], wave_index: int) -> Tuple[int, int]:
    """
    Get count of Gargantuars in a wave
    
    Args:
        spawn_data: Raw spawn list data
        wave_index: Wave index (0-indexed)
        
    Returns:
        Tuple of (normal_garg_count, giga_count)
    """
    counts = count_zombie_types_in_wave(spawn_data, wave_index)
    
    normal_gargs = counts.get(ZombieType.GARGANTUAR, 0)
    giga_gargs = counts.get(ZombieType.GIGA_GARGANTUAR, 0)
    
    return (normal_gargs, giga_gargs)


# ============================================================================
# Wave Refresh Time Prediction
# ============================================================================

# Wave refresh timing constants
INITIAL_WAVE_DELAY = 600  # Initial delay before first wave (cs)
WAVE_REFRESH_BASE_TIME = 600  # Base time between waves (cs)
HUGE_WAVE_COUNTDOWN = 750  # Extra delay for huge waves (cs)


def predict_wave_refresh_time(wave_number: int, zombie_count: int = 0,
                               base_refresh_time: int = WAVE_REFRESH_BASE_TIME) -> int:
    """
    Predict time until next wave refresh
    
    The actual refresh time depends on current zombie count and wave type.
    
    Args:
        wave_number: Current wave number
        zombie_count: Current zombie count on field
        base_refresh_time: Base refresh time (cs)
        
    Returns:
        Estimated refresh time in cs
    """
    # Huge waves have additional countdown
    if is_huge_wave(wave_number + 1):
        return base_refresh_time + HUGE_WAVE_COUNTDOWN
    
    return base_refresh_time


def get_wave_timing_info(current_wave: int, total_waves: int,
                         refresh_countdown: int) -> Dict[str, any]:
    """
    Get comprehensive wave timing information
    
    Args:
        current_wave: Current wave number (1-indexed)
        total_waves: Total waves in level
        refresh_countdown: Current refresh countdown from memory
        
    Returns:
        Dictionary with wave timing info
    """
    is_final_wave = current_wave >= total_waves
    next_is_huge = is_huge_wave(current_wave + 1) if not is_final_wave else False
    waves_to_huge = waves_until_huge(current_wave) if not is_final_wave else 0
    
    return {
        'current_wave': current_wave,
        'total_waves': total_waves,
        'is_huge_wave': is_huge_wave(current_wave),
        'next_is_huge': next_is_huge,
        'waves_until_huge': waves_to_huge,
        'is_final_wave': is_final_wave,
        'refresh_countdown': refresh_countdown,
        'wave_type': get_wave_type(current_wave),
    }


# ============================================================================
# Spawn Analysis Functions
# ============================================================================

def analyze_level_difficulty(type_list_data: List[bool],
                              total_waves: int = 20) -> Dict[str, any]:
    """
    Analyze overall level difficulty based on spawn types
    
    Args:
        type_list_data: Zombie type list from memory
        total_waves: Total waves in level
        
    Returns:
        Dictionary with difficulty analysis
    """
    all_types = parse_zombie_type_list(type_list_data)
    dangerous_types = get_dangerous_types_in_level(type_list_data)
    
    has_garg = has_gargantuar_in_level(type_list_data)
    has_giga = has_giga_in_level(type_list_data)
    
    # Calculate difficulty score
    difficulty_score = len(all_types) * 2
    difficulty_score += len(dangerous_types) * 5
    difficulty_score += 20 if has_garg else 0
    difficulty_score += 30 if has_giga else 0
    
    # Determine difficulty level
    if difficulty_score >= 100:
        difficulty = 'extreme'
    elif difficulty_score >= 70:
        difficulty = 'hard'
    elif difficulty_score >= 40:
        difficulty = 'medium'
    else:
        difficulty = 'easy'
    
    return {
        'total_types': len(all_types),
        'dangerous_types': len(dangerous_types),
        'dangerous_list': dangerous_types,
        'has_gargantuar': has_garg,
        'has_giga': has_giga,
        'difficulty_score': difficulty_score,
        'difficulty': difficulty,
        'total_waves': total_waves,
        'huge_waves': total_waves // WAVES_PER_FLAG,
    }


def get_priority_targets_for_wave(spawn_data: List[int], wave_index: int) -> List[ZombieType]:
    """
    Get priority target zombie types for a wave
    
    Args:
        spawn_data: Raw spawn list data
        wave_index: Wave index (0-indexed)
        
    Returns:
        List of ZombieType that should be prioritized
    """
    wave_zombies = parse_wave_spawn_list(spawn_data, wave_index)
    
    # Priority order: Giga > Garg > Other dangerous
    priority_zombies = []
    
    # Add Gigas first
    if ZombieType.GIGA_GARGANTUAR in wave_zombies:
        priority_zombies.append(ZombieType.GIGA_GARGANTUAR)
    
    # Add normal Gargs
    if ZombieType.GARGANTUAR in wave_zombies:
        priority_zombies.append(ZombieType.GARGANTUAR)
    
    # Add other dangerous types
    for zombie in wave_zombies:
        if is_dangerous_zombie(zombie) and zombie not in priority_zombies:
            priority_zombies.append(zombie)
    
    return priority_zombies


def recommend_cob_count_for_wave(spawn_data: List[int], wave_index: int) -> int:
    """
    Recommend number of cobs to prepare for a wave
    
    Args:
        spawn_data: Raw spawn list data
        wave_index: Wave index (0-indexed)
        
    Returns:
        Recommended cob count
    """
    normal_gargs, giga_gargs = get_garg_count_in_wave(spawn_data, wave_index)
    
    # Each normal Garg needs ~4 cobs (3000/900 = 3.33)
    # Each Giga needs ~7 cobs (6000/900 = 6.67)
    cobs_for_gargs = normal_gargs * 4 + giga_gargs * 7
    
    # Base cobs for other zombies
    base_cobs = 2 if is_huge_wave(wave_index + 1) else 1
    
    return max(base_cobs, cobs_for_gargs)
