"""
Game Reader Module
Factory class for reading game objects from memory
"""

from typing import List

from data.offsets import Offset
from memory.reader import MemoryReader
from game.zombie import ZombieInfo
from game.plant import PlantInfo
from game.projectile import ProjectileInfo, ProjectileType
from game.lawnmower import LawnmowerInfo
from game.place_item import PlaceItemInfo
from game.state import GameState, SeedInfo
from game.grid import Grid


class GameReader:
    """
    Factory class for reading game entities from memory
    
    Converts raw memory addresses into structured Python objects.
    """
    
    def __init__(self, reader: MemoryReader):
        """
        Initialize GameReader
        
        Args:
            reader: MemoryReader instance for reading memory
        """
        self.reader = reader
    
    # ========================================================================
    # Single Entity Readers
    # ========================================================================
    
    def read_zombie(self, addr: int, index: int) -> ZombieInfo:
        """
        Read a single zombie from memory
        
        Args:
            addr: Base address of zombie structure
            index: Index in zombie array
            
        Returns:
            ZombieInfo instance
        """
        return ZombieInfo(
            index=index,
            row=self.reader.read_int(addr + Offset.Z_ROW),
            x=self.reader.read_float(addr + Offset.Z_X),
            y=self.reader.read_float(addr + Offset.Z_Y),
            type=self.reader.read_int(addr + Offset.Z_TYPE),
            hp=self.reader.read_int(addr + Offset.Z_HP),
            hp_max=self.reader.read_int(addr + Offset.Z_HP_MAX),
            accessory_hp=self.reader.read_int(addr + Offset.Z_ACCESSORY_HP_1),
            state=self.reader.read_int(addr + Offset.Z_STATE),
            speed=self.reader.read_float(addr + Offset.Z_SPEED),
            slow_countdown=self.reader.read_int(addr + Offset.Z_SLOW_COUNTDOWN),
            freeze_countdown=self.reader.read_int(addr + Offset.Z_FREEZE_COUNTDOWN),
            butter_countdown=self.reader.read_int(addr + Offset.Z_BUTTER_COUNTDOWN),
            at_wave=self.reader.read_int(addr + Offset.Z_AT_WAVE),
            height=self.reader.read_float(addr + Offset.Z_HEIGHT),
            exist_time=self.reader.read_int(addr + Offset.Z_EXIST_TIME),
            state_countdown=self.reader.read_int(addr + Offset.Z_STATE_COUNTDOWN),
            is_eating=self.reader.read_bool(addr + Offset.Z_IS_EAT),
            hurt_width=self.reader.read_int(addr + Offset.Z_HURT_WIDTH),
            hurt_height=self.reader.read_int(addr + Offset.Z_HURT_HEIGHT),
            bullet_x=self.reader.read_int(addr + Offset.Z_BULLET_X),
            bullet_y=self.reader.read_int(addr + Offset.Z_BULLET_Y),
            attack_x=self.reader.read_int(addr + Offset.Z_ATTACK_X),
            attack_y=self.reader.read_int(addr + Offset.Z_ATTACK_Y),
        )
    
    def read_plant(self, addr: int, index: int) -> PlantInfo:
        """
        Read a single plant from memory
        
        Args:
            addr: Base address of plant structure
            index: Index in plant array
            
        Returns:
            PlantInfo instance
        """
        return PlantInfo(
            index=index,
            row=self.reader.read_int(addr + Offset.P_ROW),
            col=self.reader.read_int(addr + Offset.P_COL),
            type=self.reader.read_int(addr + Offset.P_TYPE),
            hp=self.reader.read_int(addr + Offset.P_HP),
            hp_max=self.reader.read_int(addr + Offset.P_HP_MAX),
            state=self.reader.read_int(addr + Offset.P_STATE),
            shoot_countdown=self.reader.read_int(addr + Offset.P_SHOOT_COUNTDOWN),
            effective=self.reader.read_int(addr + Offset.P_EFFECTIVE) != 0,
            pumpkin_hp=self.reader.read_int(addr + Offset.P_PUMPKIN_HP),
            cob_countdown=self.reader.read_int(addr + Offset.P_COB_COUNTDOWN),
            cob_ready=self.reader.read_bool(addr + Offset.P_COB_READY),
            visible=self.reader.read_bool(addr + Offset.P_VISIBLE),
            explode_countdown=self.reader.read_int(addr + Offset.P_EXPLODE_COUNTDOWN),
            blover_countdown=self.reader.read_int(addr + Offset.P_BLOVER_COUNTDOWN),
            mushroom_countdown=self.reader.read_int(addr + Offset.P_MUSHROOM_COUNTDOWN),
            bungee_state=self.reader.read_int(addr + Offset.P_BUNGEE_STATE),
            hurt_width=self.reader.read_int(addr + Offset.P_HURT_WIDTH),
            hurt_height=self.reader.read_int(addr + Offset.P_HURT_HEIGHT),
        )
    
    def read_projectile(self, addr: int, index: int) -> ProjectileInfo:
        """
        Read a single projectile from memory
        
        Args:
            addr: Base address of projectile structure
            index: Index in projectile array
            
        Returns:
            ProjectileInfo instance
        """
        return ProjectileInfo(
            index=index,
            x=self.reader.read_float(addr + Offset.PR_X),
            y=self.reader.read_float(addr + Offset.PR_Y),
            row=self.reader.read_int(addr + Offset.PR_ROW),
            type=self.reader.read_int(addr + Offset.PR_TYPE),
            exist_time=self.reader.read_int(addr + Offset.PR_EXIST_TIME),
            is_dead=self.reader.read_bool(addr + Offset.PR_DEAD),
            cob_target_x=self.reader.read_float(addr + Offset.PR_COB_TARGET_X),
            cob_target_row=self.reader.read_int(addr + Offset.PR_COB_TARGET_ROW),
        )
    
    def read_lawnmower(self, addr: int, index: int) -> LawnmowerInfo:
        """
        Read a single lawnmower from memory
        
        Args:
            addr: Base address of lawnmower structure
            index: Index in lawnmower array
            
        Returns:
            LawnmowerInfo instance
        """
        return LawnmowerInfo(
            index=index,
            row=self.reader.read_int(addr + Offset.LM_ROW),
            x=self.reader.read_float(addr + Offset.LM_X),
            state=self.reader.read_int(addr + Offset.LM_STATE),
            is_dead=self.reader.read_bool(addr + Offset.LM_DEAD),
            mower_type=self.reader.read_int(addr + Offset.LM_TYPE),
        )
    
    def read_place_item(self, addr: int, index: int) -> PlaceItemInfo:
        """
        Read a single place item from memory
        
        Args:
            addr: Base address of place item structure
            index: Index in place item array
            
        Returns:
            PlaceItemInfo instance
        """
        return PlaceItemInfo(
            index=index,
            row=self.reader.read_int(addr + Offset.PI_ROW),
            col=self.reader.read_int(addr + Offset.PI_COL),
            type=self.reader.read_int(addr + Offset.PI_TYPE),
            value=self.reader.read_int(addr + Offset.PI_VALUE),
            is_dead=self.reader.read_bool(addr + Offset.PI_DEAD),
        )
    
    def read_seed(self, addr: int, index: int) -> SeedInfo:
        """
        Read a single seed card from memory
        
        Args:
            addr: Base address of seed structure
            index: Index in seed array
            
        Returns:
            SeedInfo instance
        """
        return SeedInfo(
            index=index,
            type=self.reader.read_int(addr + Offset.S_TYPE),
            recharge_countdown=self.reader.read_int(addr + Offset.S_RECHARGE_COUNTDOWN),
            recharge_time=self.reader.read_int(addr + Offset.S_RECHARGE_TIME),
            usable=self.reader.read_bool(addr + Offset.S_USABLE),
            imitator_type=self.reader.read_int(addr + Offset.S_IMITATOR_TYPE),
        )
    
    # ========================================================================
    # Array Readers
    # ========================================================================
    
    def read_all_zombies(self) -> List[ZombieInfo]:
        """
        Read all zombies from memory
        
        Returns:
            List of ZombieInfo instances (alive zombies only)
        """
        zombies = []
        zombie_array = self.reader.get_zombie_array()
        if zombie_array == 0:
            return zombies
        
        zombie_count_max = self.reader.get_zombie_count_max()
        
        for i in range(zombie_count_max):
            addr = zombie_array + i * Offset.ZOMBIE_SIZE
            is_dead = self.reader.read_bool(addr + Offset.Z_DEAD)
            if not is_dead:
                zombies.append(self.read_zombie(addr, i))
        
        return zombies
    
    def read_all_plants(self) -> List[PlantInfo]:
        """
        Read all plants from memory
        
        Returns:
            List of PlantInfo instances (alive plants only)
        """
        plants = []
        plant_array = self.reader.get_plant_array()
        if plant_array == 0:
            return plants
        
        plant_count_max = self.reader.get_plant_count_max()
        
        for i in range(plant_count_max):
            addr = plant_array + i * Offset.PLANT_SIZE
            is_dead = self.reader.read_bool(addr + Offset.P_DEAD)
            if not is_dead:
                plants.append(self.read_plant(addr, i))
        
        return plants
    
    def read_all_projectiles(self) -> List[ProjectileInfo]:
        """
        Read all projectiles from memory
        
        Returns:
            List of ProjectileInfo instances (alive projectiles only)
        """
        projectiles = []
        board = self.reader.get_board()
        if board == 0:
            return projectiles
        
        projectile_array = self.reader.read_int(board + Offset.PROJECTILE_ARRAY)
        if projectile_array == 0:
            return projectiles
        
        projectile_count_max = self.reader.read_int(board + Offset.PROJECTILE_COUNT_MAX)
        
        for i in range(projectile_count_max):
            addr = projectile_array + i * Offset.PROJECTILE_SIZE
            is_dead = self.reader.read_bool(addr + Offset.PR_DEAD)
            if not is_dead:
                projectiles.append(self.read_projectile(addr, i))
        
        return projectiles
    
    def read_all_lawnmowers(self) -> List[LawnmowerInfo]:
        """
        Read all lawnmowers from memory
        
        Returns:
            List of LawnmowerInfo instances (alive lawnmowers only)
        """
        lawnmowers = []
        board = self.reader.get_board()
        if board == 0:
            return lawnmowers
        
        lawnmower_array = self.reader.read_int(board + Offset.LAWNMOWER_ARRAY)
        if lawnmower_array == 0:
            return lawnmowers
        
        lawnmower_count_max = self.reader.read_int(board + Offset.LAWNMOWER_COUNT_MAX)
        
        for i in range(lawnmower_count_max):
            addr = lawnmower_array + i * Offset.LAWNMOWER_SIZE
            is_dead = self.reader.read_bool(addr + Offset.LM_DEAD)
            if not is_dead:
                lawnmowers.append(self.read_lawnmower(addr, i))
        
        return lawnmowers
    
    def read_all_place_items(self) -> List[PlaceItemInfo]:
        """
        Read all place items from memory
        
        Returns:
            List of PlaceItemInfo instances (alive items only)
        """
        items = []
        board = self.reader.get_board()
        if board == 0:
            return items
        
        place_item_array = self.reader.read_int(board + Offset.PLACE_ITEM_ARRAY)
        if place_item_array == 0:
            return items
        
        place_item_count_max = self.reader.read_int(board + Offset.PLACE_ITEM_COUNT_MAX)
        
        for i in range(place_item_count_max):
            addr = place_item_array + i * Offset.PLACE_ITEM_SIZE
            is_dead = self.reader.read_bool(addr + Offset.PI_DEAD)
            if not is_dead:
                items.append(self.read_place_item(addr, i))
        
        return items
    
    def read_ice_trails(self) -> List[dict]:
        """
        Read ice trail data for each row (left by Zomboni)
        
        Returns:
            List of dicts with 'row', 'min_x', 'timer' for each active ice trail
        """
        ice_trails = []
        board = self.reader.get_board()
        if board == 0:
            return ice_trails
        
        for row in range(6):
            min_x = self.reader.read_int(board + Offset.ICE_MIN_X + row * 4)
            timer = self.reader.read_int(board + Offset.ICE_TIMER + row * 4)
            
            # 如果 timer > 0，说明这行有冰道
            if timer > 0:
                ice_trails.append({
                    'row': row,
                    'min_x': min_x,  # 冰道左边界X坐标
                    'timer': timer,  # 剩余时间 (cs)
                })
        
        return ice_trails
    
    def read_zombie_spawn_list(self, wave: int) -> List[int]:
        """
        Read zombie types for a specific wave
        
        Args:
            wave: Wave number (0-indexed)
            
        Returns:
            List of ZombieType values for that wave (-1 = empty slot)
        """
        zombies = []
        board = self.reader.get_board()
        if board == 0:
            return zombies
        
        if wave < 0 or wave >= Offset.ZOMBIE_LIST_MAX_WAVES:
            return zombies
        
        base = board + Offset.ZOMBIE_LIST + wave * Offset.ZOMBIE_LIST_WAVE_SIZE
        
        for i in range(Offset.ZOMBIE_LIST_MAX_PER_WAVE):
            zombie_type = self.reader.read_int(base + i * 4)
            if zombie_type != -1:  # -1 表示空槽位
                zombies.append(zombie_type)
        
        return zombies
    
    def read_all_spawn_lists(self, total_waves: int) -> List[List[int]]:
        """
        Read zombie spawn lists for all waves
        
        Args:
            total_waves: Total number of waves in this level
            
        Returns:
            List of lists, each containing ZombieType values for that wave
        """
        all_waves = []
        for wave in range(min(total_waves, Offset.ZOMBIE_LIST_MAX_WAVES)):
            all_waves.append(self.read_zombie_spawn_list(wave))
        return all_waves
    
    def read_grid_types(self) -> List[List[int]]:
        """
        Read grid square types for the entire field
        
        Returns:
            6x9 grid of GridSquareType values
            grid[row][col] = type (0=none, 1=grass, 2=dirt, 3=pool, 4=high_ground)
        """
        grid = []
        board = self.reader.get_board()
        if board == 0:
            return [[0] * 9 for _ in range(6)]
        
        for row in range(6):
            row_data = []
            for col in range(9):
                # Grid is stored as int[6][9]
                offset = (row * 9 + col) * 4
                grid_type = self.reader.read_int(board + Offset.GRID_TYPE_LIST + offset)
                row_data.append(grid_type)
            grid.append(row_data)
        
        return grid

    def read_all_seeds(self, seed_count: int = 10) -> List[SeedInfo]:
        """
        Read all seed cards from memory
        
        Args:
            seed_count: Number of seed cards to read (default 10)
            
        Returns:
            List of SeedInfo instances
        """
        seeds = []
        seed_array = self.reader.get_seed_array()
        if seed_array == 0:
            return seeds
        
        for i in range(seed_count):
            addr = seed_array + i * Offset.SEED_SIZE
            seeds.append(self.read_seed(addr, i))
        
        return seeds
    
    # ========================================================================
    # Full State Reader
    # ========================================================================
    
    def read_game_state(self) -> GameState:
        """
        Read complete game state from memory
        
        Returns:
            GameState instance with all game data
        """
        board = self.reader.get_board()
        if board == 0:
            return GameState()
        
        # Read all entities
        zombies = self.read_all_zombies()
        plants = self.read_all_plants()
        projectiles = self.read_all_projectiles()
        lawnmowers = self.read_all_lawnmowers()
        place_items = self.read_all_place_items()
        seeds = self.read_all_seeds()
        ice_trails = self.read_ice_trails()
        grid_types = self.read_grid_types()
        
        # Build plant grid
        plant_grid = Grid()
        for plant in plants:
            plant_grid.set(plant.row, plant.col, plant)
        
        return GameState(
            sun=self.reader.read_int(board + Offset.SUN),
            wave=self.reader.read_int(board + Offset.WAVE),
            total_waves=self.reader.read_int(board + Offset.TOTAL_WAVE),
            refresh_countdown=self.reader.read_int(board + Offset.REFRESH_COUNTDOWN),
            huge_wave_countdown=self.reader.read_int(board + Offset.HUGE_WAVE_COUNTDOWN),
            level_end_countdown=self.reader.read_int(board + Offset.LEVEL_END_COUNTDOWN),
            game_clock=self.reader.read_int(board + Offset.GAME_CLOCK),
            global_clock=self.reader.read_int(board + Offset.GLOBAL_CLOCK),
            initial_countdown=self.reader.read_int(board + Offset.INITIAL_COUNTDOWN),
            click_pao_countdown=self.reader.read_int(board + Offset.CLICK_PAO_COUNTDOWN),
            zombie_refresh_hp=self.reader.read_int(board + Offset.ZOMBIE_REFRESH_HP),
            scene=self.reader.read_int(board + Offset.SCENE),
            zombies=zombies,
            plants=plants,
            seeds=seeds,
            projectiles=projectiles,
            lawnmowers=lawnmowers,
            place_items=place_items,
            ice_trails=ice_trails,
            grid_types=grid_types,
            plant_grid=plant_grid,
        )
