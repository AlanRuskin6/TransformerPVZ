"""
Action Definitions
Defines all possible actions the bot can take
"""

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Optional, Dict, Any


class ActionType(IntEnum):
    """Types of actions the bot can perform"""
    WAIT = auto()  # Do nothing this tick
    PLANT = auto()  # Plant a seed
    SHOVEL = auto()  # Remove a plant
    USE_COB = auto()  # Fire cob cannon
    USE_CHERRY = auto()  # Use cherry bomb
    USE_JALAPENO = auto()  # Use jalapeno
    USE_ICE = auto()  # Use ice-shroom
    USE_DOOM = auto()  # Use doom-shroom
    USE_SQUASH = auto()  # Use squash
    COLLECT_SUN = auto()  # Collect sun/items


@dataclass
class Action:
    """
    Represents an action to be performed
    
    Contains the action type, parameters, and metadata for evaluation.
    """
    
    action_type: ActionType
    row: int = 0
    col: int = 0
    plant_type: int = -1
    target_x: float = 0.0
    target_y: float = 0.0
    priority: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ========================================================================
    # Factory Methods
    # ========================================================================
    
    @staticmethod
    def wait(reason: str = "No action needed") -> 'Action':
        """Create a wait action"""
        return Action(
            action_type=ActionType.WAIT,
            reason=reason
        )
    
    @staticmethod
    def plant(row: int, col: int, plant_type: int, 
              priority: float = 0.0, reason: str = "") -> 'Action':
        """Create a plant action"""
        return Action(
            action_type=ActionType.PLANT,
            row=row,
            col=col,
            plant_type=plant_type,
            priority=priority,
            reason=reason
        )
    
    @staticmethod
    def shovel(row: int, col: int, priority: float = 0.0, 
               reason: str = "") -> 'Action':
        """Create a shovel action"""
        return Action(
            action_type=ActionType.SHOVEL,
            row=row,
            col=col,
            priority=priority,
            reason=reason
        )
    
    @staticmethod
    def use_cob(target_x: float, target_row: int,
                priority: float = 0.0, reason: str = "") -> 'Action':
        """Create a cob cannon fire action"""
        return Action(
            action_type=ActionType.USE_COB,
            row=target_row,
            target_x=target_x,
            target_y=80 + target_row * 100,
            priority=priority,
            reason=reason
        )
    
    @staticmethod
    def use_cherry(row: int, col: int, 
                   priority: float = 0.0, reason: str = "") -> 'Action':
        """Create a cherry bomb action"""
        from data.plants import PlantType
        return Action(
            action_type=ActionType.USE_CHERRY,
            row=row,
            col=col,
            plant_type=PlantType.CHERRY_BOMB,
            priority=priority,
            reason=reason
        )
    
    @staticmethod
    def use_jalapeno(row: int, col: int = 0, priority: float = 0.0, 
                     reason: str = "") -> 'Action':
        """Create a jalapeno action"""
        from data.plants import PlantType
        return Action(
            action_type=ActionType.USE_JALAPENO,
            row=row,
            col=col,
            plant_type=PlantType.JALAPENO,
            priority=priority,
            reason=reason
        )
    
    @staticmethod
    def use_ice(row: int = 0, col: int = 0,
                priority: float = 0.0, reason: str = "") -> 'Action':
        """Create an ice-shroom action"""
        from data.plants import PlantType
        return Action(
            action_type=ActionType.USE_ICE,
            row=row,
            col=col,
            plant_type=PlantType.ICESHROOM,
            priority=priority,
            reason=reason
        )
    
    @staticmethod
    def collect_sun(priority: float = 0.0, reason: str = "") -> 'Action':
        """Create a collect sun action"""
        return Action(
            action_type=ActionType.COLLECT_SUN,
            priority=priority,
            reason=reason
        )
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def is_wait(self) -> bool:
        """Check if this is a wait action"""
        return self.action_type == ActionType.WAIT
    
    @property
    def is_plant_action(self) -> bool:
        """Check if this action plants something"""
        return self.action_type in {
            ActionType.PLANT,
            ActionType.USE_CHERRY,
            ActionType.USE_JALAPENO,
            ActionType.USE_ICE,
            ActionType.USE_DOOM,
            ActionType.USE_SQUASH,
        }
    
    @property
    def is_instant_kill(self) -> bool:
        """Check if this action uses an instant kill plant"""
        return self.action_type in {
            ActionType.USE_CHERRY,
            ActionType.USE_JALAPENO,
            ActionType.USE_DOOM,
            ActionType.USE_COB,
        }
    
    @property
    def sun_cost(self) -> int:
        """Get sun cost of this action"""
        if not self.is_plant_action or self.plant_type < 0:
            return 0
        from data.plants import PLANT_COST
        return PLANT_COST.get(self.plant_type, 100)
    
    @property
    def type_name(self) -> str:
        """Get action type name"""
        return self.action_type.name
    
    def __repr__(self) -> str:
        if self.action_type == ActionType.WAIT:
            return f"Action(WAIT: {self.reason})"
        elif self.action_type == ActionType.PLANT:
            from data.plants import PlantType
            try:
                plant_name = PlantType(self.plant_type).name
            except ValueError:
                plant_name = f"Type{self.plant_type}"
            return f"Action(PLANT {plant_name} at ({self.row}, {self.col}), priority={self.priority:.1f})"
        elif self.action_type == ActionType.USE_COB:
            return f"Action(COB at x={self.target_x:.0f}, row={self.row}, priority={self.priority:.1f})"
        else:
            return f"Action({self.type_name} at ({self.row}, {self.col}), priority={self.priority:.1f})"
