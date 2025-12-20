"""
Grid Class
Represents the game grid for quick plant lookup
"""

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class Grid:
    """
    Grid representation for quick plant lookup
    
    Standard PVZ grid is 9 columns x 6 rows (pool levels)
    or 9 columns x 5 rows (day/night/roof levels)
    """
    
    rows: int = 6
    cols: int = 9
    _grid: Dict[Tuple[int, int], any] = field(default_factory=dict)
    
    def set(self, row: int, col: int, plant) -> None:
        """Set a plant at a grid position"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self._grid[(row, col)] = plant
    
    def get(self, row: int, col: int) -> Optional[any]:
        """Get plant at a grid position"""
        return self._grid.get((row, col), None)
    
    def clear(self, row: int, col: int) -> None:
        """Clear a grid position"""
        if (row, col) in self._grid:
            del self._grid[(row, col)]
    
    def is_empty(self, row: int, col: int) -> bool:
        """Check if a grid position is empty"""
        return self.get(row, col) is None
    
    def get_row(self, row: int) -> List[any]:
        """Get all plants in a row"""
        return [self._grid[(row, col)] for col in range(self.cols) 
                if (row, col) in self._grid]
    
    def get_col(self, col: int) -> List[any]:
        """Get all plants in a column"""
        return [self._grid[(row, col)] for row in range(self.rows) 
                if (row, col) in self._grid]
    
    def get_all_plants(self) -> List[any]:
        """Get all plants in the grid"""
        return list(self._grid.values())
    
    def get_empty_positions(self) -> List[Tuple[int, int]]:
        """Get all empty grid positions"""
        return [(row, col) for row in range(self.rows) for col in range(self.cols)
                if (row, col) not in self._grid]
    
    def get_occupied_positions(self) -> List[Tuple[int, int]]:
        """Get all occupied grid positions"""
        return list(self._grid.keys())
    
    def count(self) -> int:
        """Get total number of plants in grid"""
        return len(self._grid)
    
    def count_in_row(self, row: int) -> int:
        """Count plants in a specific row"""
        return sum(1 for r, c in self._grid.keys() if r == row)
    
    def count_in_col(self, col: int) -> int:
        """Count plants in a specific column"""
        return sum(1 for r, c in self._grid.keys() if c == col)
    
    def clear_all(self) -> None:
        """Clear the entire grid"""
        self._grid.clear()
    
    def copy(self) -> 'Grid':
        """Create a copy of the grid"""
        new_grid = Grid(self.rows, self.cols)
        new_grid._grid = self._grid.copy()
        return new_grid
    
    def __repr__(self) -> str:
        return f"Grid({self.rows}x{self.cols}, {self.count()} plants)"
    
    def visualize(self) -> str:
        """Create a text visualization of the grid"""
        lines = []
        lines.append("  " + " ".join(str(c) for c in range(self.cols)))
        lines.append("  " + "-" * (self.cols * 2 - 1))
        
        for row in range(self.rows):
            row_str = f"{row}|"
            for col in range(self.cols):
                plant = self.get(row, col)
                if plant:
                    # Use first letter of plant type
                    row_str += plant.type_name[0] if hasattr(plant, 'type_name') else "P"
                else:
                    row_str += "."
                row_str += " "
            lines.append(row_str)
        
        return "\n".join(lines)
