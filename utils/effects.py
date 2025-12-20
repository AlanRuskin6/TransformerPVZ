"""
Status Effect Calculation Utilities
状态效果计算工具

Provides calculations for:
- Freeze effect (400cs duration, complete stop)
- Slow effect (1000cs duration, 50% speed)
- Butter effect (fixed duration)
- Effect remaining time calculation
- Effect stacking rules

All time values are in centiseconds (cs) = 1/100 second.

Reference: data/constants.py for effect constants
"""

from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import IntEnum

from data.constants import (
    ICE_DURATION,      # 400cs freeze
    SLOW_DURATION,     # 1000cs slow
)
from data.zombies import SLOW_SPEED_MULTIPLIER


# ============================================================================
# Effect Types and Constants
# ============================================================================

class EffectType(IntEnum):
    """Types of status effects"""
    NONE = 0
    FROZEN = 1
    SLOWED = 2
    BUTTERED = 3


# Effect durations (cs)
EFFECT_DURATIONS = {
    EffectType.FROZEN: ICE_DURATION,    # 400cs
    EffectType.SLOWED: SLOW_DURATION,   # 1000cs
    EffectType.BUTTERED: 400,           # Butter duration (same as freeze)
}

# Speed multipliers
SPEED_MULTIPLIERS = {
    EffectType.NONE: 1.0,
    EffectType.FROZEN: 0.0,
    EffectType.SLOWED: SLOW_SPEED_MULTIPLIER,  # 0.5
    EffectType.BUTTERED: 0.0,
}


@dataclass
class StatusEffect:
    """Represents a status effect on a zombie"""
    effect_type: EffectType
    remaining_time: int  # cs
    
    @property
    def is_active(self) -> bool:
        """Check if effect is currently active"""
        return self.remaining_time > 0
    
    @property
    def speed_multiplier(self) -> float:
        """Get speed multiplier for this effect"""
        if not self.is_active:
            return 1.0
        return SPEED_MULTIPLIERS.get(self.effect_type, 1.0)


# ============================================================================
# Freeze Effect
# ============================================================================

def get_freeze_duration() -> int:
    """
    Get freeze effect duration
    
    Returns:
        Duration in centiseconds (cs) - 400cs
    """
    return ICE_DURATION


def calculate_freeze_remaining(freeze_start_time: int, current_time: int) -> int:
    """
    Calculate remaining freeze time
    
    Args:
        freeze_start_time: When freeze started
        current_time: Current game time
        
    Returns:
        Remaining time in cs (0 if expired)
    """
    elapsed = current_time - freeze_start_time
    remaining = ICE_DURATION - elapsed
    return max(0, remaining)


def is_frozen(freeze_countdown: int) -> bool:
    """
    Check if zombie is currently frozen
    
    Args:
        freeze_countdown: Remaining freeze time from memory
        
    Returns:
        True if frozen
    """
    return freeze_countdown > 0


def get_freeze_end_time(freeze_start_time: int) -> int:
    """
    Calculate when freeze effect will end
    
    Args:
        freeze_start_time: When freeze started
        
    Returns:
        Game time when freeze ends
    """
    return freeze_start_time + ICE_DURATION


# ============================================================================
# Slow Effect
# ============================================================================

def get_slow_duration() -> int:
    """
    Get slow effect duration
    
    Returns:
        Duration in centiseconds (cs) - 1000cs
    """
    return SLOW_DURATION


def get_slow_speed_multiplier() -> float:
    """
    Get speed multiplier when slowed
    
    Returns:
        Speed multiplier - 0.5 (50% speed)
    """
    return SLOW_SPEED_MULTIPLIER


def calculate_slow_remaining(slow_start_time: int, current_time: int) -> int:
    """
    Calculate remaining slow time
    
    Args:
        slow_start_time: When slow started
        current_time: Current game time
        
    Returns:
        Remaining time in cs (0 if expired)
    """
    elapsed = current_time - slow_start_time
    remaining = SLOW_DURATION - elapsed
    return max(0, remaining)


def is_slowed(slow_countdown: int) -> bool:
    """
    Check if zombie is currently slowed
    
    Args:
        slow_countdown: Remaining slow time from memory
        
    Returns:
        True if slowed
    """
    return slow_countdown > 0


def get_slow_end_time(slow_start_time: int) -> int:
    """
    Calculate when slow effect will end
    
    Args:
        slow_start_time: When slow started
        
    Returns:
        Game time when slow ends
    """
    return slow_start_time + SLOW_DURATION


# ============================================================================
# Butter Effect
# ============================================================================

# Butter freezes zombie for ~400cs (same as ice freeze)
BUTTER_DURATION = 400


def get_butter_duration() -> int:
    """
    Get butter effect duration
    
    Returns:
        Duration in centiseconds (cs) - 400cs
    """
    return BUTTER_DURATION


def calculate_butter_remaining(butter_countdown: int) -> int:
    """
    Calculate remaining butter time
    
    Args:
        butter_countdown: Remaining butter time from memory
        
    Returns:
        Remaining time in cs
    """
    return max(0, butter_countdown)


def is_buttered(butter_countdown: int) -> bool:
    """
    Check if zombie is currently buttered (frozen by butter)
    
    Args:
        butter_countdown: Remaining butter time from memory
        
    Returns:
        True if buttered
    """
    return butter_countdown > 0


# ============================================================================
# Combined Effect Calculations
# ============================================================================

def get_effective_speed(base_speed: float, freeze_countdown: int = 0,
                        slow_countdown: int = 0, butter_countdown: int = 0) -> float:
    """
    Calculate effective zombie speed considering all status effects
    
    Priority: Frozen/Buttered (0 speed) > Slowed (50% speed) > Normal
    
    Args:
        base_speed: Base zombie speed
        freeze_countdown: Remaining freeze time
        slow_countdown: Remaining slow time
        butter_countdown: Remaining butter time
        
    Returns:
        Effective speed in pixels/cs
    """
    # Frozen or buttered = no movement
    if freeze_countdown > 0 or butter_countdown > 0:
        return 0.0
    
    # Slowed = half speed
    if slow_countdown > 0:
        return base_speed * SLOW_SPEED_MULTIPLIER
    
    # No effects
    return base_speed


def get_current_status(freeze_countdown: int = 0, slow_countdown: int = 0,
                       butter_countdown: int = 0) -> EffectType:
    """
    Get the current dominant status effect
    
    Args:
        freeze_countdown: Remaining freeze time
        slow_countdown: Remaining slow time
        butter_countdown: Remaining butter time
        
    Returns:
        Current effect type
    """
    if freeze_countdown > 0:
        return EffectType.FROZEN
    if butter_countdown > 0:
        return EffectType.BUTTERED
    if slow_countdown > 0:
        return EffectType.SLOWED
    return EffectType.NONE


def get_status_summary(freeze_countdown: int = 0, slow_countdown: int = 0,
                       butter_countdown: int = 0) -> Dict[str, any]:
    """
    Get a summary of all status effects
    
    Args:
        freeze_countdown: Remaining freeze time
        slow_countdown: Remaining slow time
        butter_countdown: Remaining butter time
        
    Returns:
        Dictionary with effect information
    """
    current = get_current_status(freeze_countdown, slow_countdown, butter_countdown)
    
    return {
        'current_effect': current,
        'is_immobile': current in (EffectType.FROZEN, EffectType.BUTTERED),
        'speed_multiplier': SPEED_MULTIPLIERS.get(current, 1.0),
        'freeze_remaining': freeze_countdown,
        'slow_remaining': slow_countdown,
        'butter_remaining': butter_countdown,
        'any_effect_active': current != EffectType.NONE,
    }


# ============================================================================
# Effect Stacking and Timing
# ============================================================================

def calculate_effect_timeline(freeze_countdown: int = 0, slow_countdown: int = 0,
                              butter_countdown: int = 0) -> List[Dict[str, any]]:
    """
    Calculate timeline of effect changes
    
    Args:
        freeze_countdown: Remaining freeze time
        slow_countdown: Remaining slow time
        butter_countdown: Remaining butter time
        
    Returns:
        List of dictionaries describing effect phases
    """
    timeline = []
    current_time = 0
    
    # Phase 1: Frozen/Buttered (if applicable)
    immobile_time = max(freeze_countdown, butter_countdown)
    if immobile_time > 0:
        timeline.append({
            'phase': 'immobile',
            'effect': EffectType.FROZEN if freeze_countdown >= butter_countdown else EffectType.BUTTERED,
            'start_time': current_time,
            'end_time': immobile_time,
            'duration': immobile_time,
            'speed_multiplier': 0.0,
        })
        current_time = immobile_time
    
    # Phase 2: Slowed (if applicable, and lasts beyond freeze)
    if slow_countdown > immobile_time:
        slow_remaining = slow_countdown - immobile_time
        timeline.append({
            'phase': 'slowed',
            'effect': EffectType.SLOWED,
            'start_time': current_time,
            'end_time': current_time + slow_remaining,
            'duration': slow_remaining,
            'speed_multiplier': SLOW_SPEED_MULTIPLIER,
        })
        current_time += slow_remaining
    
    # Phase 3: Normal (all effects expired)
    timeline.append({
        'phase': 'normal',
        'effect': EffectType.NONE,
        'start_time': current_time,
        'end_time': None,  # Continues indefinitely
        'duration': None,
        'speed_multiplier': 1.0,
    })
    
    return timeline


def calculate_travel_with_effects(distance: float, base_speed: float,
                                   freeze_countdown: int = 0,
                                   slow_countdown: int = 0,
                                   butter_countdown: int = 0) -> float:
    """
    Calculate travel time considering all status effects wearing off
    
    Args:
        distance: Distance to travel (pixels)
        base_speed: Base zombie speed (pixels/cs)
        freeze_countdown: Remaining freeze time
        slow_countdown: Remaining slow time
        butter_countdown: Remaining butter time
        
    Returns:
        Total travel time in cs
    """
    if distance <= 0:
        return 0.0
    
    if base_speed <= 0:
        return float('inf')
    
    remaining_distance = distance
    total_time = 0.0
    
    timeline = calculate_effect_timeline(freeze_countdown, slow_countdown, butter_countdown)
    
    for phase in timeline:
        if remaining_distance <= 0:
            break
        
        speed = base_speed * phase['speed_multiplier']
        
        if phase['duration'] is None:
            # Normal phase - travel at full speed
            if speed > 0:
                total_time += remaining_distance / speed
            else:
                return float('inf')
            remaining_distance = 0
        elif speed > 0:
            # Can move during this phase
            distance_in_phase = speed * phase['duration']
            if distance_in_phase >= remaining_distance:
                # Reaches destination in this phase
                total_time += remaining_distance / speed
                remaining_distance = 0
            else:
                # Uses entire phase duration
                total_time += phase['duration']
                remaining_distance -= distance_in_phase
        else:
            # Cannot move (frozen/buttered) - just add time
            total_time += phase['duration']
    
    return total_time


# ============================================================================
# Ice Mushroom Effect Chain
# ============================================================================

def calculate_ice_chain_effect(ice_effect_time: int, current_time: int) -> Dict[str, any]:
    """
    Calculate ice mushroom effect chain (freeze -> slow)
    
    Ice mushroom first freezes zombies (400cs), then slows them (1000cs).
    
    Args:
        ice_effect_time: When ice effect activated
        current_time: Current game time
        
    Returns:
        Dictionary with current status and remaining times
    """
    elapsed = current_time - ice_effect_time
    
    if elapsed < 0:
        # Ice hasn't taken effect yet
        return {
            'phase': 'pending',
            'freeze_remaining': ICE_DURATION,
            'slow_remaining': SLOW_DURATION,
            'time_until_effect': -elapsed,
        }
    elif elapsed < ICE_DURATION:
        # Currently frozen
        return {
            'phase': 'frozen',
            'freeze_remaining': ICE_DURATION - elapsed,
            'slow_remaining': SLOW_DURATION,
            'speed_multiplier': 0.0,
        }
    elif elapsed < ICE_DURATION + SLOW_DURATION:
        # Currently slowed
        return {
            'phase': 'slowed',
            'freeze_remaining': 0,
            'slow_remaining': ICE_DURATION + SLOW_DURATION - elapsed,
            'speed_multiplier': SLOW_SPEED_MULTIPLIER,
        }
    else:
        # Effect expired
        return {
            'phase': 'expired',
            'freeze_remaining': 0,
            'slow_remaining': 0,
            'speed_multiplier': 1.0,
        }


def can_refreeze(last_ice_effect_time: int, current_time: int,
                 refreeze_threshold: int = 100) -> bool:
    """
    Check if it's beneficial to apply another ice effect
    
    Args:
        last_ice_effect_time: When last ice effect activated
        current_time: Current game time
        refreeze_threshold: Minimum time remaining before refreeze is useful
        
    Returns:
        True if refreezing would be beneficial
    """
    status = calculate_ice_chain_effect(last_ice_effect_time, current_time)
    
    # Refreeze if effect expired or about to expire
    if status['phase'] == 'expired':
        return True
    if status['phase'] == 'slowed' and status['slow_remaining'] < refreeze_threshold:
        return True
    if status['phase'] == 'frozen' and status['freeze_remaining'] < refreeze_threshold:
        return True
    
    return False
