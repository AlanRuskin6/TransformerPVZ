"""
Plant Info Class
Represents a plant entity in the game
"""

from dataclasses import dataclass
from typing import Optional

from data.plants import (
    PlantType,
    PLANT_COST,
    PLANT_HP,
    ATTACKING_PLANTS,
    DEFENSIVE_PLANTS,
    INSTANT_KILL_PLANTS,
    SUN_PRODUCING_PLANTS,
)
from data.constants import GRID_WIDTH, LAWN_LEFT_X


@dataclass
class PlantInfo:
    """
    Information about a plant on the field
    
    Contains position, health, state, and type information.
    """
    
    index: int  # Index in plant array
    row: int  # Row (0-5)
    col: int  # Column (0-8)
    type: int  # Plant type ID
    hp: int  # Current HP
    hp_max: int  # Maximum HP
    state: int  # Current state
    shoot_countdown: int  # Time until next shot (for attackers)
    effective: bool  # Is awake/effective (mushrooms may be sleeping)
    
    # Optional fields for special plants
    pumpkin_hp: int = 0  # HP of pumpkin shield if present
    cob_countdown: int = 0  # Cob cannon reload time
    cob_ready: bool = False  # Cob cannon ready to fire
    
    # Additional fields from offsets
    visible: bool = True              # P_VISIBLE (0x18)
    explode_countdown: int = 0        # P_EXPLODE_COUNTDOWN (0x50)
    blover_countdown: int = 0         # P_BLOVER_COUNTDOWN (0x4C)
    mushroom_countdown: int = 0       # P_MUSHROOM_COUNTDOWN (0x130)
    bungee_state: int = 0             # P_BUNGEE_STATE (0x134)
    hurt_width: int = 0               # P_HURT_WIDTH (0x10)
    hurt_height: int = 0              # P_HURT_HEIGHT (0x14)
    
    # ========================================================================
    # Position Properties
    # ========================================================================
    
    @property
    def x(self) -> float:
        """Get x position in pixels"""
        return LAWN_LEFT_X + self.col * GRID_WIDTH
    
    @property
    def y(self) -> float:
        """Get y position in pixels"""
        return 80 + self.row * 100  # Approximate
    
    # ========================================================================
    # Status Properties
    # ========================================================================
    
    @property
    def hp_ratio(self) -> float:
        """Get HP as ratio of max (0-1)"""
        if self.hp_max <= 0:
            return 0.0
        return min(1.0, max(0.0, self.hp / self.hp_max))
    
    @property
    def is_damaged(self) -> bool:
        """Check if plant has taken damage"""
        return self.hp < self.hp_max
    
    @property
    def is_critical(self) -> bool:
        """Check if plant HP is critically low (<25%)"""
        return self.hp_ratio < 0.25
    
    @property
    def total_hp(self) -> int:
        """Get total HP including pumpkin"""
        return self.hp + self.pumpkin_hp
    
    # ========================================================================
    # Type Properties
    # ========================================================================
    
    @property
    def type_name(self) -> str:
        """Get plant type name"""
        try:
            return PlantType(self.type).name
        except ValueError:
            return f"UNKNOWN_{self.type}"
    
    @property
    def cost(self) -> int:
        """Get plant sun cost"""
        return PLANT_COST.get(self.type, 100)
    
    @property
    def is_attacker(self) -> bool:
        """Check if plant can attack zombies"""
        return self.type in ATTACKING_PLANTS
    
    @property
    def is_defender(self) -> bool:
        """Check if plant is defensive (wall/nut)"""
        return self.type in DEFENSIVE_PLANTS
    
    @property
    def is_instant_kill(self) -> bool:
        """Check if plant is instant kill type"""
        return self.type in INSTANT_KILL_PLANTS
    
    @property
    def is_sun_producer(self) -> bool:
        """Check if plant produces sun"""
        return self.type in SUN_PRODUCING_PLANTS
    
    @property
    def is_cob_cannon(self) -> bool:
        """Check if plant is a cob cannon"""
        return self.type == PlantType.COBCANNON
    
    @property
    def is_grabbed_by_bungee(self) -> bool:
        """Check if plant is being grabbed by a bungee zombie"""
        return self.bungee_state > 0
    
    @property
    def time_to_explode(self) -> int:
        """Get the countdown until plant explodes (for instant plants)"""
        return self.explode_countdown
    
    # ========================================================================
    # Special Plant Checks
    # ========================================================================
    
    def can_fire_cob(self) -> bool:
        """Check if this cob cannon can fire"""
        return self.is_cob_cannon and self.cob_ready
    
    def is_shooting_soon(self) -> bool:
        """Check if plant will shoot soon (within 50cs)"""
        return self.is_attacker and self.shoot_countdown <= 50
    
    # ========================================================================
    # Defense Value Calculation
    # ========================================================================
    
    @property
    def defense_value(self) -> float:
        """
        Calculate defensive value of this plant
        
        Higher for:
        - Defensive plants (walls)
        - Higher HP
        - Has pumpkin
        """
        if not self.is_defender:
            return 0.0
        
        base_value = self.total_hp / 1000  # Normalize to reasonable range
        
        # Tall-nut is more valuable (blocks jumping zombies)
        if self.type == PlantType.TALLNUT:
            base_value *= 1.5
        
        return base_value
    
    @property
    def attack_value(self) -> float:
        """
        Calculate attack value of this plant
        
        Based on DPS and effectiveness
        """
        if not self.is_attacker:
            return 0.0
        
        # Base DPS values (approximate)
        dps_map = {
            PlantType.PEASHOOTER: 20 / 1.41,  # 20 damage per 141cs
            PlantType.SNOW_PEA: 20 / 1.41,
            PlantType.REPEATER: 40 / 1.41,
            PlantType.THREEPEATER: 60 / 1.41,
            PlantType.GATLINGPEA: 80 / 1.41,
            PlantType.MELONPULT: 80 / 3.0,  # Slower but splash
            PlantType.WINTERMELON: 80 / 3.0,
            PlantType.GLOOMSHROOM: 80 / 2.0,  # AOE
            PlantType.FUMESHROOM: 20 / 1.5,  # Piercing
        }
        
        dps = dps_map.get(self.type, 10)
        return dps if self.effective else 0
    
    def __repr__(self) -> str:
        return (f"PlantInfo(type={self.type_name}, row={self.row}, col={self.col}, "
                f"hp={self.hp}/{self.hp_max})")
