"""
PVZ Interface - Unified interface supporting both Hook and Legacy modes
"""

import sys
import time
from typing import Optional
from enum import Enum

from config import BotConfig
from utils.logger import get_logger

# Import game state modules
from game.state import GameState, SeedInfo
from game.zombie import ZombieInfo
from game.plant import PlantInfo
from game.grid import Grid

from data.offsets import Offset
from data.plants import get_unlocked_plants


class InterfaceMode(Enum):
    """Interface mode"""
    HOOK = "hook"       # Hook DLL mode (recommended, stable)
    LEGACY = "legacy"   # Legacy shellcode injection mode (deprecated)


class PVZInterface:
    """
    Unified PVZ Interface
    
    Supports both Hook DLL mode (recommended) and legacy shellcode mode.
    
    Usage:
        # Hook mode (recommended)
        pvz = PVZInterface(mode=InterfaceMode.HOOK)
        
        # Legacy mode (deprecated)
        pvz = PVZInterface(mode=InterfaceMode.LEGACY)
    """
    
    def __init__(self, mode: InterfaceMode = InterfaceMode.HOOK):
        """
        Initialize interface
        
        Args:
            mode: Interface mode (HOOK or LEGACY)
        """
        self.mode = mode
        self.logger = get_logger()
        
        if mode == InterfaceMode.HOOK:
            self._init_hook_mode()
        else:
            self._init_legacy_mode()
    
    def _init_hook_mode(self):
        """Initialize Hook mode"""
        try:
            from hook_client import HookClient
            from memory.process import ProcessAttacher
            from memory.reader import MemoryReader
            
            self.hook_client: Optional[HookClient] = None
            self.attacher = ProcessAttacher()
            self.reader: Optional[MemoryReader] = None
            self.writer = None
            self.injector = None
            
            self.logger.info("Using Hook DLL mode (recommended)")
        except ImportError as e:
            self.logger.error(f"Failed to initialize Hook mode: {e}")
            sys.exit(1)
    
    def _init_legacy_mode(self):
        """Initialize Legacy mode"""
        import warnings
        warnings.warn(
            "Legacy mode is deprecated and may cause crashes. Use Hook mode instead.",
            DeprecationWarning
        )
        
        from memory.process import ProcessAttacher
        from memory.reader import MemoryReader
        from memory.writer import MemoryWriter
        from memory.injector import AsmInjector
        
        self.hook_client = None
        self.attacher = ProcessAttacher()
        self.reader: Optional[MemoryReader] = None
        self.writer: Optional[MemoryWriter] = None
        self.injector: Optional[AsmInjector] = None
        
        self.logger.info("Using Legacy mode (deprecated)")
    
    def attach(self) -> bool:
        """Attach to PVZ process"""
        if self.mode == InterfaceMode.HOOK:
            return self._attach_hook()
        else:
            return self._attach_legacy()
    
    def _attach_hook(self) -> bool:
        """Attach in Hook mode"""
        from hook_client import HookClient
        
        # Attach for reading
        if not self.attacher.attach():
            return False
        
        kernel32 = self.attacher.kernel32
        handle = self.attacher.handle
        
        from memory.reader import MemoryReader
        from memory.writer import MemoryWriter
        
        self.reader = MemoryReader(kernel32, handle)
        self.writer = MemoryWriter(kernel32, handle)  # Enable writing in Hook mode too
        
        # Connect to Hook DLL
        self.hook_client = HookClient()
        if not self.hook_client.connect():
            self.logger.warning("Hook DLL not connected. Make sure DLL is injected.")
            # Don't fail - still allow reading
        
        return True
    
    def _attach_legacy(self) -> bool:
        """Attach in Legacy mode"""
        if not self.attacher.attach():
            return False
        
        kernel32 = self.attacher.kernel32
        handle = self.attacher.handle
        
        from memory.reader import MemoryReader
        from memory.writer import MemoryWriter
        from memory.injector import AsmInjector
        
        self.reader = MemoryReader(kernel32, handle)
        self.writer = MemoryWriter(kernel32, handle)
        self.injector = AsmInjector(kernel32, handle, self.reader)
        
        return True
    
    def is_attached(self) -> bool:
        """Check if attached to process"""
        return self.attacher.is_attached()
    
    def is_in_game(self) -> bool:
        """Check if in game"""
        if not self.reader:
            return False
        return self.reader.is_in_game()
    
    def plant(self, row: int, col: int, plant_type: int) -> bool:
        """Plant at position"""
        if self.mode == InterfaceMode.HOOK:
            if not self.hook_client:
                return False
            return self.hook_client.plant(row, col, plant_type)
        else:
            if not self.injector:
                return False
            return self.injector.plant(row, col, plant_type)
    
    def shovel(self, row: int, col: int) -> bool:
        """Remove plant at position"""
        if self.mode == InterfaceMode.HOOK:
            if not self.hook_client:
                return False
            return self.hook_client.shovel(row, col)
        else:
            if not self.injector:
                return False
            return self.injector.shovel(row, col)
    
    def restart_level(self) -> bool:
        """Restart current level"""
        if self.mode == InterfaceMode.HOOK:
            if not self.hook_client:
                return False
            return self.hook_client.reset()
        else:
            if not self.injector:
                return False
            return self.injector.make_new_board()
    
    def enable_auto_collect_patch(self, enable: bool = True) -> bool:
        """
        Enable/disable auto-collect using code patch method (like PvZ Tools)
        
        This patches the game's collection check code at 0x0043158f:
        - Enable: Change 0x75 (jnz) to 0xEB (jmp) - skip collection check
        - Disable: Restore 0x75 (jnz) - normal behavior
        
        This is more efficient than manual collection as it hooks the game logic.
        
        Args:
            enable: True to enable auto-collect, False to disable
            
        Returns:
            True if successful
        """
        if not self.writer:
            self.logger.error("Writer not initialized, cannot apply patch")
            return False
        
        AUTO_COLLECT_ADDR = 0x0043158F
        
        if enable:
            # Patch: 0x75 -> 0xEB (jnz -> jmp)
            success = self.writer.write_byte(AUTO_COLLECT_ADDR, 0xEB)
            if success:
                self.logger.info("Auto-collect patch enabled (code patching method)")
            else:
                self.logger.error("Failed to enable auto-collect patch")
            return success
        else:
            # Restore: 0xEB -> 0x75 (jmp -> jnz)
            success = self.writer.write_byte(AUTO_COLLECT_ADDR, 0x75)
            if success:
                self.logger.info("Auto-collect patch disabled")
            else:
                self.logger.error("Failed to disable auto-collect patch")
            return success
    
    def collect_all_items(self) -> int:
        """
        Collect all items (sun, coins)
        
        Note: For better performance, use enable_auto_collect_patch() instead.
        This method manually marks each item as collected.
        """
        if not self.reader:
            return 0
        
        board = self.reader.get_board()
        if board == 0:
            return 0
        
        count = 0
        item_array = self.reader.read_int(board + Offset.ITEM_ARRAY)
        item_max = self.reader.read_int(board + Offset.ITEM_COUNT_MAX)
        
        if self.mode == InterfaceMode.HOOK:
            # Auto-collect not implemented in Hook DLL yet
            # Use enable_auto_collect_patch() or Legacy mode
            self.logger.warning("Auto-collect not supported in Hook mode. Use enable_auto_collect_patch() or Legacy mode.")
            return 0
        else:
            # Legacy mode - direct memory write
            if not self.writer:
                return 0
            
            for i in range(min(item_max, 100)):
                addr = item_array + i * Offset.ITEM_SIZE
                
                if self.reader.read_byte(addr + Offset.I_DISAPPEARED):
                    continue
                if self.reader.read_byte(addr + Offset.I_COLLECTED):
                    continue
                
                self.writer.write_byte(addr + Offset.I_COLLECTED, 1)
                count += 1
        
        return count
    
    def set_sun(self, amount: int) -> bool:
        """Set sun amount directly"""
        if not self.writer or not self.reader:
            return False
            
        board = self.reader.get_board()
        if board == 0:
            return False
            
        return self.writer.write_int(board + Offset.SUN, amount)

    def get_game_state(self) -> Optional[GameState]:
        """Read complete game state"""
        if not self.reader or not self.reader.is_in_game():
            return None
        
        board = self.reader.get_board()
        if board == 0:
            return None
        
        # Read basic info
        sun = self.reader.get_sun()
        
        # Sanity check for Sun (avoid garbage memory)
        if sun < 0 or sun > 99999:
            # self.logger.warning(f"Invalid Sun value detected: {sun}. Ignoring state.")
            return None
            
        wave = self.reader.get_wave()
        
        # Sanity check for Wave
        if wave < 0 or wave > 1000:
            return None
            
        total_waves = self.reader.get_total_waves()
        game_clock = self.reader.get_game_clock()
        scene = self.reader.get_scene()
        refresh_cd = self.reader.read_int(board + Offset.REFRESH_COUNTDOWN)
        huge_wave_cd = self.reader.read_int(board + Offset.HUGE_WAVE_COUNTDOWN)
        
        # Read zombies
        zombies = self._read_zombies(board)
        
        # Read plants and build grid
        plants, plant_grid = self._read_plants(board)
        
        # Read seeds
        seeds = self._read_seeds(board)
        
        # Read click_pao_countdown
        click_pao_cd = self.reader.read_int(board + Offset.CLICK_PAO_COUNTDOWN)
        
        # Read player inventory
        player_level = self.reader.get_player_level()
        player_coins = self.reader.get_player_coins()
        purchases = self.reader.get_all_purchases()
        unlocked_plants = get_unlocked_plants(player_level, purchases)
        
        # Read spawn lists
        spawn_lists = self._read_spawn_lists(board, total_waves)
        
        # Read lawnmowers
        lawnmowers = self._read_lawnmowers(board)
        
        return GameState(
            sun=sun,
            wave=wave,
            total_waves=total_waves,
            game_clock=game_clock,
            scene=scene,
            refresh_countdown=refresh_cd,
            huge_wave_countdown=huge_wave_cd,
            click_pao_countdown=click_pao_cd,
            zombies=zombies,
            plants=plants,
            seeds=seeds,
            plant_grid=plant_grid,
            player_level=player_level,
            player_coins=player_coins,
            unlocked_plants=unlocked_plants,
            spawn_lists=spawn_lists,
            lawnmowers=lawnmowers,
        )
    
    def _read_zombies(self, board: int) -> list:
        """Read all zombies from memory"""
        zombies = []
        zombie_array = self.reader.read_int(board + Offset.ZOMBIE_ARRAY)
        zombie_max = self.reader.read_int(board + Offset.ZOMBIE_COUNT_MAX)
        
        for i in range(min(zombie_max, 200)):
            addr = zombie_array + i * Offset.ZOMBIE_SIZE
            
            if self.reader.read_byte(addr + Offset.Z_DEAD):
                continue
            
            zombies.append(ZombieInfo(
                index=i,
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
            ))
        
        return zombies
    
    def _read_plants(self, board: int) -> tuple:
        """Read all plants and build grid"""
        plants = []
        grid = Grid()
        
        plant_array = self.reader.read_int(board + Offset.PLANT_ARRAY)
        plant_max = self.reader.read_int(board + Offset.PLANT_COUNT_MAX)
        
        from data.plants import PlantType
        
        for i in range(min(plant_max, 200)):
            addr = plant_array + i * Offset.PLANT_SIZE
            
            if self.reader.read_byte(addr + Offset.P_DEAD):
                continue
            
            row = self.reader.read_int(addr + Offset.P_ROW)
            col = self.reader.read_int(addr + Offset.P_COL)
            plant_type = self.reader.read_int(addr + Offset.P_TYPE)
            
            plant = PlantInfo(
                index=i,
                row=row,
                col=col,
                type=plant_type,
                hp=self.reader.read_int(addr + Offset.P_HP),
                hp_max=self.reader.read_int(addr + Offset.P_HP_MAX),
                state=self.reader.read_int(addr + Offset.P_STATE),
                shoot_countdown=self.reader.read_int(addr + Offset.P_SHOOT_COUNTDOWN),
                effective=self.reader.read_int(addr + Offset.P_EFFECTIVE) != 0,
                pumpkin_hp=self.reader.read_int(addr + Offset.P_PUMPKIN_HP),
                cob_countdown=self.reader.read_int(addr + Offset.P_COB_COUNTDOWN) if plant_type == PlantType.COBCANNON else 0,
                cob_ready=self.reader.read_bool(addr + Offset.P_COB_READY) if plant_type == PlantType.COBCANNON else False,
            )
            
            plants.append(plant)
            grid.set(row, col, plant)
        
        return plants, grid
    
    def _read_seeds(self, board: int) -> list:
        """Read seed cards"""
        seeds = []
        seed_array = self.reader.read_int(board + Offset.SEED_ARRAY)
        
        for i in range(10):
            addr = seed_array + i * Offset.SEED_SIZE
            seeds.append(SeedInfo(
                index=i,
                type=self.reader.read_int(addr + Offset.S_TYPE),
                recharge_countdown=self.reader.read_int(addr + Offset.S_RECHARGE_COUNTDOWN),
                recharge_time=self.reader.read_int(addr + Offset.S_RECHARGE_TIME),
                usable=self.reader.read_byte(addr + Offset.S_USABLE) != 0,
                imitator_type=self.reader.read_int(addr + Offset.S_IMITATOR_TYPE),
            ))
        
        return seeds
    
    def _read_spawn_lists(self, board: int, total_waves: int) -> list:
        """Read zombie spawn lists for all waves"""
        all_waves = []
        max_waves = min(total_waves, Offset.ZOMBIE_LIST_MAX_WAVES)
        
        for wave in range(max_waves):
            wave_zombies = []
            base = board + Offset.ZOMBIE_LIST + wave * Offset.ZOMBIE_LIST_WAVE_SIZE
            
            for i in range(Offset.ZOMBIE_LIST_MAX_PER_WAVE):
                zombie_type = self.reader.read_int(base + i * 4)
                if zombie_type != -1:
                    wave_zombies.append(zombie_type)
            
            all_waves.append(wave_zombies)
        
        return all_waves
    
    def _read_lawnmowers(self, board: int) -> list:
        """Read all lawnmowers from memory"""
        from game.lawnmower import LawnmowerInfo
        
        lawnmowers = []
        lawnmower_array = self.reader.read_int(board + Offset.LAWNMOWER_ARRAY)
        if lawnmower_array == 0:
            return lawnmowers
        
        lawnmower_max = self.reader.read_int(board + Offset.LAWNMOWER_COUNT_MAX)
        
        for i in range(min(lawnmower_max, 10)):
            addr = lawnmower_array + i * Offset.LAWNMOWER_SIZE
            
            if self.reader.read_bool(addr + Offset.LM_DEAD):
                continue
            
            lawnmowers.append(LawnmowerInfo(
                index=i,
                row=self.reader.read_int(addr + Offset.LM_ROW),
                x=self.reader.read_float(addr + Offset.LM_X),
                state=self.reader.read_int(addr + Offset.LM_STATE),
                is_dead=False,
                mower_type=self.reader.read_int(addr + Offset.LM_TYPE),
            ))
        
        return lawnmowers
    
    @property
    def pid(self) -> Optional[int]:
        """Get process ID"""
        return self.attacher.pid
