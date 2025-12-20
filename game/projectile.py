"""
Projectile Info Class
Represents a projectile (bullet/pea/cob) in the game
"""

from dataclasses import dataclass


class ProjectileType:
    """Projectile types in PVZ"""
    PEA = 0
    SNOW_PEA = 1
    CABBAGE = 2
    MELON = 3
    PUFF = 4
    WINTERMELON = 5
    FIRE_PEA = 6
    STAR = 7
    CACTUS = 8
    FUME = 9
    BASKETBALL = 10
    COB = 11
    BUTTER = 12
    KERNEL = 13


@dataclass
class ProjectileInfo:
    """
    Information about a projectile on the field
    
    Contains position, type, and cob cannon targeting data.
    """
    
    index: int  # Index in projectile array
    x: float  # X position (pixels)
    y: float  # Y position (pixels)
    row: int  # Current row
    type: int  # Projectile type ID
    exist_time: int  # Time since creation (cs)
    is_dead: bool  # Whether projectile is dead/removed
    cob_target_x: float = 0.0  # Cob target X (needs +87.5 for actual landing)
    cob_target_row: int = 0  # Cob target row
    
    @property
    def is_cob(self) -> bool:
        """Check if this is a cob cannon projectile"""
        return self.type == ProjectileType.COB
    
    @property
    def actual_cob_target_x(self) -> float:
        """Get the actual landing X position for cob cannon"""
        return self.cob_target_x + 87.5
    
    @property
    def is_butter(self) -> bool:
        """Check if this is a butter projectile"""
        return self.type == ProjectileType.BUTTER
    
    @property
    def is_frozen(self) -> bool:
        """Check if this is a freezing projectile"""
        return self.type in [ProjectileType.SNOW_PEA, ProjectileType.WINTERMELON]
    
    def __repr__(self) -> str:
        type_names = {
            ProjectileType.PEA: "PEA",
            ProjectileType.SNOW_PEA: "SNOW_PEA",
            ProjectileType.CABBAGE: "CABBAGE",
            ProjectileType.MELON: "MELON",
            ProjectileType.COB: "COB",
            ProjectileType.BUTTER: "BUTTER",
        }
        type_name = type_names.get(self.type, f"TYPE_{self.type}")
        return f"ProjectileInfo(type={type_name}, row={self.row}, x={self.x:.1f})"
