"""
Place Item Info Class
Represents a field item (grave, crater, etc.) in the game
"""

from dataclasses import dataclass


class PlaceItemType:
    """Place item types in PVZ"""
    NONE = 0
    GRAVE = 1
    CRATER = 2
    BRAIN = 3  # ZomBotany brain target


@dataclass
class PlaceItemInfo:
    """
    Information about a place item on the field
    
    Contains position, type, and value information for field items
    like graves and craters.
    """
    
    index: int  # Index in place item array
    row: int  # Row (0-5)
    col: int  # Column (0-8)
    type: int  # Item type (grave, crater, etc.)
    value: int  # Value (grave emerge count, crater countdown, brain HP)
    is_dead: bool  # Whether item is dead/removed
    
    @property
    def is_grave(self) -> bool:
        """Check if this is a grave"""
        return self.type == PlaceItemType.GRAVE
    
    @property
    def is_crater(self) -> bool:
        """Check if this is a crater"""
        return self.type == PlaceItemType.CRATER
    
    @property
    def is_brain(self) -> bool:
        """Check if this is a brain target"""
        return self.type == PlaceItemType.BRAIN
    
    def __repr__(self) -> str:
        type_names = {
            PlaceItemType.NONE: "NONE",
            PlaceItemType.GRAVE: "GRAVE",
            PlaceItemType.CRATER: "CRATER",
            PlaceItemType.BRAIN: "BRAIN",
        }
        type_name = type_names.get(self.type, f"TYPE_{self.type}")
        return f"PlaceItemInfo(type={type_name}, row={self.row}, col={self.col})"
