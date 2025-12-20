"""
Zombie Info Class
Represents a zombie entity in the game
"""

from dataclasses import dataclass
from typing import Optional

from data.zombies import (
    ZombieType, 
    get_zombie_total_hp,
    get_threat_multiplier,
    is_gargantuar
)


@dataclass
class ZombieInfo:
    """
    Information about a zombie on the field
    
    Contains position, health, status, and calculated threat level.
    """
    
    index: int  # Index in zombie array
    row: int  # Current row (0-5)
    x: float  # X position (pixels from left)
    y: float  # Y position (pixels from top)
    type: int  # Zombie type ID
    hp: int  # Current body HP
    hp_max: int  # Maximum body HP
    accessory_hp: int  # Current accessory HP (cone/bucket/etc)
    state: int  # Current state (walking, eating, etc)
    speed: float  # Current speed (pixels per cs)
    slow_countdown: int  # Remaining slow effect time (cs)
    freeze_countdown: int  # Remaining freeze effect time (cs)
    butter_countdown: int  # Remaining butter effect time (cs)
    at_wave: int  # Wave this zombie spawned in
    height: float = 0.0  # Height offset (for flying zombies)
    
    # Additional fields from offsets
    exist_time: int = 0          # Z_EXIST_TIME (0x60)
    state_countdown: int = 0     # Z_STATE_COUNTDOWN (0x68)
    is_eating: bool = False      # Z_IS_EAT (0x51)
    hurt_width: int = 0          # Z_HURT_WIDTH (0x94)
    hurt_height: int = 0         # Z_HURT_HEIGHT (0x98)
    bullet_x: int = 0            # Z_BULLET_X (0x8C)
    bullet_y: int = 0            # Z_BULLET_Y (0x90)
    attack_x: int = 0            # Z_ATTACK_X (0x9C)
    attack_y: int = 0            # Z_ATTACK_Y (0xA0)
    
    # ========================================================================
    # Status Properties
    # ========================================================================
    
    @property
    def is_slowed(self) -> bool:
        """Check if zombie is currently slowed"""
        return self.slow_countdown > 0
    
    @property
    def is_frozen(self) -> bool:
        """Check if zombie is currently frozen"""
        return self.freeze_countdown > 0
    
    @property
    def is_buttered(self) -> bool:
        """Check if zombie is currently buttered"""
        return self.butter_countdown > 0
    
    @property
    def is_immobilized(self) -> bool:
        """Check if zombie cannot move"""
        return self.is_frozen or self.is_buttered
    
    @property
    def is_dying(self) -> bool:
        """Check if zombie is in dying state"""
        return self.state in [1, 2, 3]
    
    @property
    def is_hammering(self) -> bool:
        """Check if Gargantuar is in hammering state"""
        return self.state == 70
    
    @property
    def total_hp(self) -> int:
        """Get total HP including accessory"""
        return self.hp + self.accessory_hp
    
    @property
    def hp_ratio(self) -> float:
        """Get HP as ratio of max (0-1)"""
        if self.hp_max <= 0:
            return 0.0
        return min(1.0, max(0.0, self.hp / self.hp_max))
    
    @property
    def effective_speed(self) -> float:
        """Get effective speed considering status effects"""
        if self.is_frozen or self.is_buttered:
            return 0.0
        if self.is_slowed:
            return self.speed * 0.5
        return self.speed
    
    # ========================================================================
    # Position Properties
    # ========================================================================
    
    @property
    def col(self) -> int:
        """Get approximate column (0-8)"""
        from data.constants import GRID_WIDTH, LAWN_LEFT_X
        return max(0, min(8, int((self.x - LAWN_LEFT_X) / GRID_WIDTH)))
    
    @property
    def distance_to_left(self) -> float:
        """Get distance to left edge of lawn"""
        return self.x
    
    @property
    def is_on_lawn(self) -> bool:
        """Check if zombie is on the playable area"""
        return self.x > 0 and self.x < 850
    
    @property
    def is_near_plants(self) -> bool:
        """Check if zombie is close to plant columns"""
        return self.x < 600
    
    @property
    def is_critical(self) -> bool:
        """Check if zombie is critically close to left edge"""
        return self.x < 200
    
    # ========================================================================
    # Type Properties
    # ========================================================================
    
    @property
    def is_gargantuar(self) -> bool:
        """Check if this is a Gargantuar variant"""
        return is_gargantuar(self.type)
    
    @property
    def type_name(self) -> str:
        """Get zombie type name"""
        try:
            return ZombieType(self.type).name
        except ValueError:
            return f"UNKNOWN_{self.type}"
    
    # ========================================================================
    # Threat Calculation
    # ========================================================================
    
    @property
    def threat_level(self) -> float:
        """
        Calculate overall threat level (0-10+)
        
        Considers:
        - Distance to left (closer = more threat)
        - HP (higher = more threat)
        - Type (some types more dangerous)
        - Speed (faster = more threat)
        """
        # Distance threat (0-1): closer to left = higher threat
        distance_threat = max(0, (800 - self.x) / 800)
        
        # HP threat (0-1): based on total HP vs typical max
        hp_threat = min(1.0, self.total_hp / 3000)
        
        # Type multiplier
        type_mult = get_threat_multiplier(self.type)
        
        # Speed multiplier (1.0 - 1.5)
        speed_mult = 1.0 + abs(self.effective_speed) * 0.5
        
        # Combine factors
        base_threat = distance_threat * (1 + hp_threat) * type_mult * speed_mult
        
        # Bonus for critical distance
        if self.is_critical:
            base_threat *= 1.5
        
        return base_threat
    
    # ========================================================================
    # Prediction
    # ========================================================================
    
    def time_to_reach(self, target_x: float) -> float:
        """
        Estimate time (cs) to reach a target x position
        
        Args:
            target_x: Target x coordinate
            
        Returns:
            Estimated time in centiseconds, or float('inf') if immobilized
        """
        if self.is_immobilized:
            return float('inf')
        
        speed = self.effective_speed
        if speed <= 0:
            return float('inf')
        
        distance = self.x - target_x
        if distance <= 0:
            return 0.0
        
        return distance / speed
    
    def position_at(self, time_cs: float) -> float:
        """
        Predict x position after a given time
        
        Args:
            time_cs: Time in centiseconds
            
        Returns:
            Predicted x position
        """
        if self.is_immobilized:
            return self.x
        
        return self.x - self.effective_speed * time_cs
    
    def __repr__(self) -> str:
        return (f"ZombieInfo(type={self.type_name}, row={self.row}, "
                f"x={self.x:.1f}, hp={self.hp}, threat={self.threat_level:.2f})")
