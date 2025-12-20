"""
Position and Coordinate Utilities
坐标转换和位置计算工具

Provides conversion between grid coordinates and pixel coordinates,
distance calculations, and roof scene slope handling.

All coordinates are in pixels unless otherwise specified.
Time values are in centiseconds (cs) = 1/100 second.

Reference: data/constants.py for GRID_WIDTH, LAWN_LEFT_X, etc.
"""

import math
from typing import Tuple, Optional

from data.constants import (
    GRID_WIDTH,
    GRID_HEIGHT,
    GRID_HEIGHT_ROOF,
    GRID_COLS,
    GRID_ROWS,
    LAWN_LEFT_X,
    LAWN_RIGHT_X,
    SCENE_ROOF,
    SCENE_ROOF_NIGHT,
)


# ============================================================================
# Column ↔ X Pixel Coordinate Conversion
# ============================================================================

def col_to_x(col: int, center: bool = True) -> float:
    """
    Convert column index to x pixel coordinate
    
    Args:
        col: Column index (0-8)
        center: If True, return center of cell; if False, return left edge
        
    Returns:
        X pixel coordinate
    """
    x = LAWN_LEFT_X + col * GRID_WIDTH
    if center:
        x += GRID_WIDTH / 2
    return x


def x_to_col(x: float) -> int:
    """
    Convert x pixel coordinate to column index
    
    Args:
        x: X pixel coordinate
        
    Returns:
        Column index (0-8), clamped to valid range
    """
    col = int((x - LAWN_LEFT_X) / GRID_WIDTH)
    return max(0, min(GRID_COLS - 1, col))


def x_to_col_float(x: float) -> float:
    """
    Convert x pixel coordinate to column index (float, for precision)
    
    Args:
        x: X pixel coordinate
        
    Returns:
        Column index as float
    """
    return (x - LAWN_LEFT_X) / GRID_WIDTH


# ============================================================================
# Row ↔ Y Pixel Coordinate Conversion
# ============================================================================

# Y offset for the first row (top of lawn)
LAWN_TOP_Y = 80


def row_to_y(row: int, scene: int = 0, center: bool = True) -> float:
    """
    Convert row index to y pixel coordinate
    
    Args:
        row: Row index (0-5)
        scene: Scene type (for roof scenes, affects y calculation)
        center: If True, return center of cell; if False, return top edge
        
    Returns:
        Y pixel coordinate
    """
    height = GRID_HEIGHT_ROOF if scene in (SCENE_ROOF, SCENE_ROOF_NIGHT) else GRID_HEIGHT
    y = LAWN_TOP_Y + row * height
    if center:
        y += height / 2
    return y


def y_to_row(y: float, scene: int = 0) -> int:
    """
    Convert y pixel coordinate to row index
    
    Args:
        y: Y pixel coordinate
        scene: Scene type
        
    Returns:
        Row index (0-5), clamped to valid range
    """
    height = GRID_HEIGHT_ROOF if scene in (SCENE_ROOF, SCENE_ROOF_NIGHT) else GRID_HEIGHT
    row = int((y - LAWN_TOP_Y) / height)
    return max(0, min(GRID_ROWS - 1, row))


def y_to_row_float(y: float, scene: int = 0) -> float:
    """
    Convert y pixel coordinate to row index (float, for precision)
    
    Args:
        y: Y pixel coordinate
        scene: Scene type
        
    Returns:
        Row index as float
    """
    height = GRID_HEIGHT_ROOF if scene in (SCENE_ROOF, SCENE_ROOF_NIGHT) else GRID_HEIGHT
    return (y - LAWN_TOP_Y) / height


# ============================================================================
# Grid ↔ Pixel Coordinate Conversion
# ============================================================================

def grid_to_pixel(col: int, row: int, scene: int = 0, 
                  center: bool = True) -> Tuple[float, float]:
    """
    Convert grid coordinates (col, row) to pixel coordinates (x, y)
    
    Args:
        col: Column index (0-8)
        row: Row index (0-5)
        scene: Scene type
        center: If True, return center of cell
        
    Returns:
        (x, y) pixel coordinates
    """
    x = col_to_x(col, center)
    y = row_to_y(row, scene, center)
    return (x, y)


def pixel_to_grid(x: float, y: float, scene: int = 0) -> Tuple[int, int]:
    """
    Convert pixel coordinates (x, y) to grid coordinates (col, row)
    
    Args:
        x: X pixel coordinate
        y: Y pixel coordinate
        scene: Scene type
        
    Returns:
        (col, row) grid coordinates
    """
    col = x_to_col(x)
    row = y_to_row(y, scene)
    return (col, row)


def pixel_to_grid_float(x: float, y: float, scene: int = 0) -> Tuple[float, float]:
    """
    Convert pixel coordinates to grid coordinates (float, for precision)
    
    Args:
        x: X pixel coordinate
        y: Y pixel coordinate
        scene: Scene type
        
    Returns:
        (col, row) as floats
    """
    col = x_to_col_float(x)
    row = y_to_row_float(y, scene)
    return (col, row)


# ============================================================================
# Distance Calculations
# ============================================================================

def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        x1, y1: First point coordinates
        x2, y2: Second point coordinates
        
    Returns:
        Distance in pixels
    """
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)


def distance_x(x1: float, x2: float) -> float:
    """
    Calculate horizontal distance between two x coordinates
    
    Args:
        x1: First x coordinate
        x2: Second x coordinate
        
    Returns:
        Absolute distance in pixels
    """
    return abs(x2 - x1)


def distance_grid(col1: int, row1: int, col2: int, row2: int) -> float:
    """
    Calculate distance between two grid positions
    
    Args:
        col1, row1: First grid position
        col2, row2: Second grid position
        
    Returns:
        Distance in grid units
    """
    dx = col2 - col1
    dy = row2 - row1
    return math.sqrt(dx * dx + dy * dy)


def manhattan_distance(col1: int, row1: int, col2: int, row2: int) -> int:
    """
    Calculate Manhattan distance between two grid positions
    
    Args:
        col1, row1: First grid position
        col2, row2: Second grid position
        
    Returns:
        Manhattan distance in grid units
    """
    return abs(col2 - col1) + abs(row2 - row1)


# ============================================================================
# Roof Scene Slope Handling
# ============================================================================

# Roof slope parameters
# The roof has a slope from column 1-5, then flattens out
ROOF_SLOPE_START_COL = 1  # Slope starts at column 1
ROOF_SLOPE_END_COL = 5    # Slope ends at column 5
ROOF_HEIGHT_PER_COL = 20  # Height change per column (pixels)

# Base height at the highest point of roof (column 5+)
ROOF_BASE_HEIGHT = 0


def get_roof_height_offset(col: int) -> float:
    """
    Get the height offset for a column on the roof scene
    
    The roof slopes downward from left to right for columns 1-5,
    then flattens out for columns 6+.
    
    Args:
        col: Column index (0-8)
        
    Returns:
        Height offset in pixels (positive = higher)
    """
    if col <= ROOF_SLOPE_START_COL:
        # Before slope or at start - highest point
        return (ROOF_SLOPE_END_COL - ROOF_SLOPE_START_COL) * ROOF_HEIGHT_PER_COL
    elif col >= ROOF_SLOPE_END_COL:
        # After slope - flat
        return ROOF_BASE_HEIGHT
    else:
        # On the slope
        return (ROOF_SLOPE_END_COL - col) * ROOF_HEIGHT_PER_COL


def get_roof_adjusted_y(col: int, row: int, scene: int) -> float:
    """
    Get y coordinate adjusted for roof slope
    
    Args:
        col: Column index
        row: Row index
        scene: Scene type
        
    Returns:
        Adjusted y coordinate
    """
    base_y = row_to_y(row, scene)
    
    if scene in (SCENE_ROOF, SCENE_ROOF_NIGHT):
        height_offset = get_roof_height_offset(col)
        return base_y - height_offset  # Subtract because higher = lower y
    
    return base_y


def is_on_roof(scene: int) -> bool:
    """
    Check if scene is a roof scene
    
    Args:
        scene: Scene type
        
    Returns:
        True if roof scene
    """
    return scene in (SCENE_ROOF, SCENE_ROOF_NIGHT)


# ============================================================================
# Utility Functions
# ============================================================================

def is_valid_grid_position(col: int, row: int) -> bool:
    """
    Check if a grid position is valid
    
    Args:
        col: Column index
        row: Row index
        
    Returns:
        True if position is valid
    """
    return 0 <= col < GRID_COLS and 0 <= row < GRID_ROWS


def clamp_to_lawn(x: float) -> float:
    """
    Clamp x coordinate to lawn boundaries
    
    Args:
        x: X pixel coordinate
        
    Returns:
        Clamped x coordinate
    """
    return max(LAWN_LEFT_X, min(LAWN_RIGHT_X, x))


def get_plant_position(col: int, row: int, scene: int = 0) -> Tuple[float, float]:
    """
    Get the pixel position where a plant should be placed
    
    Plants are placed at the center of their grid cell.
    
    Args:
        col: Column index (0-8)
        row: Row index (0-5)
        scene: Scene type
        
    Returns:
        (x, y) pixel coordinates for plant placement
    """
    x = col_to_x(col, center=True)
    y = row_to_y(row, scene, center=True)
    
    # Adjust for roof slope if applicable
    if scene in (SCENE_ROOF, SCENE_ROOF_NIGHT):
        y = get_roof_adjusted_y(col, row, scene)
    
    return (x, y)


def get_cob_target_position(col: float, row: int, scene: int = 0) -> Tuple[float, float]:
    """
    Get the pixel position for a cob cannon target
    
    Cob cannon can target any position, not just grid centers.
    The actual landing position is target_x + 87.5 (from PR_COB_TARGET_X offset).
    
    Args:
        col: Target column (can be float for precision)
        row: Target row
        scene: Scene type
        
    Returns:
        (x, y) pixel coordinates for cob target
    """
    x = LAWN_LEFT_X + col * GRID_WIDTH
    y = row_to_y(row, scene, center=True)
    return (x, y)
