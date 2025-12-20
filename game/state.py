"""
Game State Class
Represents the complete state of a PVZ game at a point in time
"""

from dataclasses import dataclass, field
from typing import List, Optional

from game.zombie import ZombieInfo
from game.plant import PlantInfo
from game.grid import Grid
from game.projectile import ProjectileInfo
from game.lawnmower import LawnmowerInfo
from game.place_item import PlaceItemInfo


@dataclass
class SeedInfo:
    """Information about a seed card in the player's hand"""
    index: int
    type: int
    recharge_countdown: int  # Remaining cooldown (cs)
    recharge_time: int  # Total cooldown time (cs)
    usable: bool  # Can be planted now
    imitator_type: int = -1  # Type if imitator (-1 if not)
    
    @property
    def is_ready(self) -> bool:
        """Check if the seed is ready to use"""
        return self.usable and self.recharge_countdown <= 0
    
    @property
    def cooldown_percent(self) -> float:
        """Get cooldown progress as percentage (0-100)"""
        if self.recharge_time <= 0:
            return 100.0
        return max(0, min(100, 100 * (1 - self.recharge_countdown / self.recharge_time)))


@dataclass
class GameState:
    """
    Complete game state snapshot
    
    Contains all information about the current game state including:
    - Resources (sun)
    - Wave progress
    - All zombies on field
    - All plants on field
    - Seed cards in hand
    - Grid representation of plants
    """
    
    # Resources
    sun: int = 0
    
    # Wave info
    wave: int = 0
    total_waves: int = 0
    refresh_countdown: int = 0  # Time until next wave
    huge_wave_countdown: int = 0  # Time until huge wave
    level_end_countdown: int = 0  # 关卡结束倒计时 (>0 表示即将通关)
    
    # Timing
    game_clock: int = 0  # Game time in cs
    global_clock: int = 0  # Global clock (battle and selection)
    initial_countdown: int = 0  # Initial zombie refresh countdown
    click_pao_countdown: int = 0  # Cob cannon click cooldown (30cs anti-misclick)
    zombie_refresh_hp: int = 0  # Zombie refresh HP threshold
    
    # Scene
    scene: int = 0
    
    # Entities
    zombies: List[ZombieInfo] = field(default_factory=list)
    plants: List[PlantInfo] = field(default_factory=list)
    seeds: List[SeedInfo] = field(default_factory=list)
    projectiles: List[ProjectileInfo] = field(default_factory=list)
    lawnmowers: List[LawnmowerInfo] = field(default_factory=list)
    place_items: List[PlaceItemInfo] = field(default_factory=list)
    
    # Ice trails (冰道，冰车留下的)
    ice_trails: List[dict] = field(default_factory=list)  # [{row, min_x, timer}]
    
    # Grid types (场地格子类型)
    grid_types: List[List[int]] = field(default_factory=list)  # 6x9, GridSquareType
    
    # Spawn lists (出怪列表，每波僵尸的类型)
    spawn_lists: List[List[int]] = field(default_factory=list)  # [wave][zombie_types]
    
    # Player inventory (仓库/已解锁植物)
    player_level: int = 0  # 冒险模式关卡进度
    player_coins: int = 0  # 金币
    unlocked_plants: set = field(default_factory=set)  # 已解锁的植物类型集合
    
    # Grid representation (quick plant lookup)
    plant_grid: Optional[Grid] = None
    
    # ========================================================================
    # Utility Properties
    # ========================================================================
    
    @property
    def alive_zombies(self) -> List[ZombieInfo]:
        """Get all alive zombies"""
        return [z for z in self.zombies if z.hp > 0]
    
    @property
    def alive_plants(self) -> List[PlantInfo]:
        """Get all alive plants"""
        return [p for p in self.plants if p.hp > 0]
    
    @property
    def zombie_count(self) -> int:
        """Get number of alive zombies"""
        return len(self.alive_zombies)
    
    @property
    def plant_count(self) -> int:
        """Get number of alive plants"""
        return len(self.alive_plants)
    
    @property
    def is_final_wave(self) -> bool:
        """Check if on final wave"""
        return self.wave >= self.total_waves
    
    @property
    def is_huge_wave(self) -> bool:
        """Check if current wave is a huge wave (every 10th)"""
        return self.wave > 0 and self.wave % 10 == 0
    
    # ========================================================================
    # Zombie Queries
    # ========================================================================
    
    def get_zombies_in_row(self, row: int) -> List[ZombieInfo]:
        """Get all zombies in a specific row"""
        return [z for z in self.alive_zombies if z.row == row]
    
    def get_closest_zombie_in_row(self, row: int) -> Optional[ZombieInfo]:
        """Get the closest zombie to the left in a row"""
        row_zombies = self.get_zombies_in_row(row)
        if not row_zombies:
            return None
        return min(row_zombies, key=lambda z: z.x)
    
    def get_zombies_by_type(self, zombie_type: int) -> List[ZombieInfo]:
        """Get all zombies of a specific type"""
        return [z for z in self.alive_zombies if z.type == zombie_type]
    
    def get_dangerous_zombies(self) -> List[ZombieInfo]:
        """Get all zombies with high threat level"""
        from data.zombies import is_dangerous_zombie
        return [z for z in self.alive_zombies if is_dangerous_zombie(z.type)]
    
    # ========================================================================
    # Plant Queries
    # ========================================================================
    
    def get_plant_at(self, row: int, col: int) -> Optional[PlantInfo]:
        """Get plant at a specific grid position"""
        if self.plant_grid:
            return self.plant_grid.get(row, col)
        # Fallback to linear search
        for plant in self.alive_plants:
            if plant.row == row and plant.col == col:
                return plant
        return None
    
    def get_plants_in_row(self, row: int) -> List[PlantInfo]:
        """Get all plants in a specific row"""
        return [p for p in self.alive_plants if p.row == row]
    
    def get_plants_by_type(self, plant_type: int) -> List[PlantInfo]:
        """Get all plants of a specific type"""
        return [p for p in self.alive_plants if p.type == plant_type]
    
    def is_cell_empty(self, row: int, col: int) -> bool:
        """Check if a grid cell is empty"""
        return self.get_plant_at(row, col) is None
    
    # ========================================================================
    # Seed Queries
    # ========================================================================
    
    def get_seed_by_type(self, plant_type: int) -> Optional[SeedInfo]:
        """Get seed card for a specific plant type"""
        for seed in self.seeds:
            if seed.type == plant_type:
                return seed
        return None
    
    def get_usable_seeds(self) -> List[SeedInfo]:
        """Get all seeds that are currently usable"""
        return [s for s in self.seeds if s.usable]
    
    def can_plant(self, plant_type: int) -> bool:
        """Check if we can plant a specific type (have card and enough sun)"""
        from data.plants import PLANT_COST
        seed = self.get_seed_by_type(plant_type)
        if not seed or not seed.usable:
            return False
        cost = PLANT_COST.get(plant_type, 100)
        return self.sun >= cost
    
    # ========================================================================
    # Row Analysis
    # ========================================================================
    
    def get_row_threat(self, row: int) -> float:
        """Calculate total threat level for a row"""
        return sum(z.threat_level for z in self.get_zombies_in_row(row))
    
    def get_most_threatened_row(self) -> int:
        """Get the row with highest threat"""
        from data.offsets import SceneType
        max_threat = -1
        max_row = 0
        row_count = SceneType.get_row_count(self.scene)
        for row in range(row_count):
            threat = self.get_row_threat(row)
            if threat > max_threat:
                max_threat = threat
                max_row = row
        return max_row
    
    def get_row_defense_count(self, row: int) -> int:
        """Count defensive plants in a row"""
        from data.plants import DEFENSIVE_PLANTS
        return sum(1 for p in self.get_plants_in_row(row) if p.type in DEFENSIVE_PLANTS)
    
    def get_row_attacker_count(self, row: int) -> int:
        """Count attacking plants in a row"""
        from data.plants import ATTACKING_PLANTS
        return sum(1 for p in self.get_plants_in_row(row) if p.type in ATTACKING_PLANTS)
    
    # ========================================================================
    # Projectile Queries
    # ========================================================================
    
    def get_flying_cobs(self) -> List[ProjectileInfo]:
        """Get all cob cannon projectiles currently in flight"""
        from game.projectile import ProjectileType
        return [p for p in self.projectiles 
                if not p.is_dead and p.type == ProjectileType.COB]
    
    # ========================================================================
    # Cob Cannon Queries
    # ========================================================================
    
    def get_ready_cobs(self) -> List[PlantInfo]:
        """Get all cob cannons that are ready to fire"""
        return [p for p in self.alive_plants if p.is_cob_cannon and p.cob_ready]
    
    def can_fire_cob(self) -> bool:
        """Check if any cob cannon can be fired (ready and no click cooldown)"""
        return self.click_pao_countdown <= 0 and len(self.get_ready_cobs()) > 0
    
    # ========================================================================
    # Lawnmower Queries
    # ========================================================================
    
    def has_lawnmower(self, row: int) -> bool:
        """Check if a specific row has an available lawnmower"""
        for lm in self.lawnmowers:
            if lm.row == row and lm.is_available:
                return True
        return False
    
    # ========================================================================
    # Ice Trail Queries (冰道查询)
    # ========================================================================
    
    def has_ice_trail(self, row: int) -> bool:
        """Check if a specific row has an active ice trail"""
        for trail in self.ice_trails:
            if trail['row'] == row and trail['timer'] > 0:
                return True
        return False
    
    def get_ice_trail(self, row: int) -> Optional[dict]:
        """Get ice trail info for a specific row, or None if no trail"""
        for trail in self.ice_trails:
            if trail['row'] == row:
                return trail
        return None
    
    def is_on_ice(self, row: int, x: float) -> bool:
        """Check if a position is on ice trail"""
        trail = self.get_ice_trail(row)
        if trail is None:
            return False
        # 冰道从 min_x 延伸到场地右边
        return x >= trail['min_x']
    
    # ========================================================================
    # Grid Type Queries (场地格子查询)
    # ========================================================================
    
    def get_grid_type(self, row: int, col: int) -> int:
        """
        Get grid square type at position
        Returns: 0=none, 1=grass, 2=dirt, 3=pool, 4=high_ground
        """
        if not self.grid_types:
            return 0
        if 0 <= row < len(self.grid_types) and 0 <= col < len(self.grid_types[0]):
            return self.grid_types[row][col]
        return 0
    
    def is_pool(self, row: int, col: int) -> bool:
        """Check if grid position is a pool tile"""
        return self.get_grid_type(row, col) == 3  # GRIDSQUARE_POOL
    
    def is_roof(self, row: int, col: int) -> bool:
        """Check if grid position is high ground (roof)"""
        return self.get_grid_type(row, col) == 4  # GRIDSQUARE_HIGH_GROUND
    
    # ========================================================================
    # Place Item Queries
    # ========================================================================
    
    def get_graves(self) -> List[PlaceItemInfo]:
        """Get all grave items on the field"""
        return [item for item in self.place_items 
                if not item.is_dead and item.is_grave]
    
    # ========================================================================
    # Spawn List Queries (出怪列表查询)
    # ========================================================================
    
    def get_wave_zombies(self, wave: int) -> List[int]:
        """
        Get zombie types for a specific wave (0-indexed)
        
        Returns:
            List of ZombieType values for that wave
        """
        if 0 <= wave < len(self.spawn_lists):
            return self.spawn_lists[wave]
        return []
    
    def get_wave_zombies_named(self, wave: int) -> List[str]:
        """
        Get zombie type names for a specific wave (0-indexed)
        
        Returns:
            List of zombie type names (e.g., ['普通僵尸', '路障僵尸'])
        """
        from data.zombies import ZombieType
        types = self.get_wave_zombies(wave)
        result = []
        for z_type in types:
            try:
                result.append(ZombieType(z_type).name)
            except ValueError:
                result.append(f'UNKNOWN_{z_type}')
        return result
    
    def get_current_wave_zombies(self) -> List[int]:
        """Get zombie types for the current wave"""
        return self.get_wave_zombies(self.wave - 1)  # wave is 1-indexed in game
    
    def get_next_wave_zombies(self) -> List[int]:
        """Get zombie types for the next wave"""
        return self.get_wave_zombies(self.wave)  # wave is 1-indexed, so wave = next wave index
    
    def get_all_waves_summary(self) -> dict:
        """
        Get a summary of all waves with zombie counts
        
        Returns:
            Dict with wave index as key and {types: [...], count: N} as value
        """
        from data.zombies import ZombieType
        summary = {}
        for wave_idx, zombies in enumerate(self.spawn_lists):
            named = []
            for z_type in zombies:
                try:
                    named.append(ZombieType(z_type).name)
                except ValueError:
                    named.append(f'UNKNOWN_{z_type}')
            summary[wave_idx] = {
                'types': named,
                'count': len(zombies)
            }
        return summary
