"""
Lawnmower Info Class
Represents a lawnmower entity in the game

Reference: re-plants-vs-zombies/Lawn/LawnMower.h
           re-plants-vs-zombies/ConstEnums.h
"""

from dataclasses import dataclass
from enum import IntEnum


class LawnMowerState(IntEnum):
    """
    Lawnmower state enum
    Reference: ConstEnums.h - LawnMowerState
    """
    ROLLING_IN = 0   # 正在滚入场地（刚出现）
    READY = 1        # 准备就绪（待命）
    TRIGGERED = 2    # 已触发（正在移动攻击僵尸）
    SQUISHED = 3     # 被压扁（被巨人僵尸压扁）


class LawnMowerType(IntEnum):
    """
    Lawnmower type enum
    Reference: ConstEnums.h - LawnMowerType
    """
    LAWN = 0         # 普通草地小推车
    POOL = 1         # 水池清道夫
    ROOF = 2         # 屋顶滚车
    SUPER_MOWER = 3  # 超级小推车


@dataclass
class LawnmowerInfo:
    """
    Information about a lawnmower on the field
    
    Reference: LawnMower.h struct layout
    - mPosX: float at +0x08
    - mRow: int at +0x14
    - mMowerState: LawnMowerState at +0x2C
    - mDead: bool at +0x30
    - mMowerType: LawnMowerType at +0x34
    """
    
    index: int              # Index in lawnmower array
    row: int                # Row (0-5)
    x: float                # X position (pixels)
    state: int              # Current state (LawnMowerState)
    is_dead: bool           # Whether lawnmower is dead/used
    mower_type: int = 0     # Lawnmower type (LawnMowerType)
    
    @property
    def state_enum(self) -> LawnMowerState:
        """Get state as enum"""
        try:
            return LawnMowerState(self.state)
        except ValueError:
            return LawnMowerState.READY
    
    @property
    def type_enum(self) -> LawnMowerType:
        """Get type as enum"""
        try:
            return LawnMowerType(self.mower_type)
        except ValueError:
            return LawnMowerType.LAWN
    
    @property
    def is_available(self) -> bool:
        """Check if lawnmower is available (not dead and ready)"""
        return not self.is_dead and self.state == LawnMowerState.READY
    
    @property
    def is_rolling_in(self) -> bool:
        """Check if lawnmower is rolling into position"""
        return not self.is_dead and self.state == LawnMowerState.ROLLING_IN
    
    @property
    def is_triggered(self) -> bool:
        """Check if lawnmower is triggered and moving"""
        return not self.is_dead and self.state == LawnMowerState.TRIGGERED
    
    @property
    def is_squished(self) -> bool:
        """Check if lawnmower was squished by Gargantuar"""
        return not self.is_dead and self.state == LawnMowerState.SQUISHED
    
    @property
    def is_moving(self) -> bool:
        """Check if lawnmower is currently moving (rolling in or triggered)"""
        return not self.is_dead and self.state in (LawnMowerState.ROLLING_IN, LawnMowerState.TRIGGERED)
    
    def __repr__(self) -> str:
        if self.is_dead:
            status = "dead"
        else:
            status = self.state_enum.name.lower()
        type_name = self.type_enum.name.lower()
        return f"LawnmowerInfo(row={self.row}, x={self.x:.1f}, type={type_name}, status={status})"
