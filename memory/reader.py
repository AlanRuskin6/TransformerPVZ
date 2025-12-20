"""
Memory Reader Module
Handles reading values from PVZ process memory
"""

import ctypes
from typing import Optional, List
from data.offsets import Offset


class MemoryReadError(Exception):
    """Exception raised when memory read fails"""
    pass


class MemoryReader:
    """Reads values from process memory"""
    
    def __init__(self, kernel32, process_handle: int):
        self.kernel32 = kernel32
        self.process = process_handle
        
    def read_int(self, address: int) -> int:
        """Read a 4-byte integer from memory"""
        buf = ctypes.c_int()
        bytes_read = ctypes.c_size_t()
        result = self.kernel32.ReadProcessMemory(
            self.process, address, ctypes.byref(buf), 4, ctypes.byref(bytes_read)
        )
        if not result or bytes_read.value != 4:
            return 0  # Return default value instead of raising to avoid crashes
        return buf.value
    
    def read_uint(self, address: int) -> int:
        """Read a 4-byte unsigned integer from memory"""
        buf = ctypes.c_uint()
        self.kernel32.ReadProcessMemory(
            self.process, address, ctypes.byref(buf), 4, None
        )
        return buf.value
    
    def read_float(self, address: int) -> float:
        """Read a 4-byte float from memory"""
        buf = ctypes.c_float()
        bytes_read = ctypes.c_size_t()
        result = self.kernel32.ReadProcessMemory(
            self.process, address, ctypes.byref(buf), 4, ctypes.byref(bytes_read)
        )
        if not result or bytes_read.value != 4:
            return 0.0
        return buf.value
    
    def read_byte(self, address: int) -> int:
        """Read a single byte from memory"""
        buf = ctypes.c_byte()
        bytes_read = ctypes.c_size_t()
        result = self.kernel32.ReadProcessMemory(
            self.process, address, ctypes.byref(buf), 1, ctypes.byref(bytes_read)
        )
        if not result or bytes_read.value != 1:
            return 0
        return buf.value
    
    def read_bool(self, address: int) -> bool:
        """Read a boolean (single byte) from memory"""
        return self.read_byte(address) != 0
    
    def read_bytes(self, address: int, size: int) -> bytes:
        """Read multiple bytes from memory"""
        buf = (ctypes.c_byte * size)()
        self.kernel32.ReadProcessMemory(
            self.process, address, ctypes.byref(buf), size, None
        )
        return bytes(buf)
    
    def read_short(self, address: int) -> int:
        """Read a 2-byte short from memory"""
        buf = ctypes.c_short()
        self.kernel32.ReadProcessMemory(
            self.process, address, ctypes.byref(buf), 2, None
        )
        return buf.value
    
    def read_double(self, address: int) -> float:
        """Read an 8-byte double from memory"""
        buf = ctypes.c_double()
        self.kernel32.ReadProcessMemory(
            self.process, address, ctypes.byref(buf), 8, None
        )
        return buf.value
    
    # ========================================================================
    # PVZ Specific Reading Methods
    # ========================================================================
    
    def get_pvz_base(self) -> int:
        """Get the PVZ base pointer"""
        return self.read_int(Offset.BASE)
    
    def get_base(self) -> int:
        """Alias for get_pvz_base"""
        return self.get_pvz_base()
    
    def get_board(self) -> int:
        """Get the Board/MainObject pointer"""
        base = self.get_pvz_base()
        if base == 0:
            return 0
        return self.read_int(base + Offset.MAIN_OBJECT)
    
    def get_game_ui(self) -> int:
        """Get the current game UI state"""
        base = self.get_pvz_base()
        if base == 0:
            return 0
        return self.read_int(base + Offset.GAME_UI)
    
    def is_in_game(self) -> bool:
        """Check if player is currently in a game"""
        return self.get_game_ui() == 3
    
    def get_sun(self) -> int:
        """Get current sun amount"""
        board = self.get_board()
        if board == 0:
            return 0
        return self.read_int(board + Offset.SUN)
    
    def get_wave(self) -> int:
        """Get current wave number"""
        board = self.get_board()
        if board == 0:
            return 0
        return self.read_int(board + Offset.WAVE)
    
    def get_total_waves(self) -> int:
        """Get total number of waves"""
        board = self.get_board()
        if board == 0:
            return 0
        return self.read_int(board + Offset.TOTAL_WAVE)
    
    def get_game_clock(self) -> int:
        """Get game clock (time in cs)"""
        board = self.get_board()
        if board == 0:
            return 0
        return self.read_int(board + Offset.GAME_CLOCK)
    
    def get_scene(self) -> int:
        """Get current scene type"""
        board = self.get_board()
        if board == 0:
            return 0
        return self.read_int(board + Offset.SCENE)
    
    def get_zombie_array(self) -> int:
        """Get zombie array base address"""
        board = self.get_board()
        if board == 0:
            return 0
        return self.read_int(board + Offset.ZOMBIE_ARRAY)
    
    def get_zombie_count_max(self) -> int:
        """Get maximum zombie count (array size)"""
        board = self.get_board()
        if board == 0:
            return 0
        return self.read_int(board + Offset.ZOMBIE_COUNT_MAX)
    
    def get_plant_array(self) -> int:
        """Get plant array base address"""
        board = self.get_board()
        if board == 0:
            return 0
        return self.read_int(board + Offset.PLANT_ARRAY)
    
    def get_plant_count_max(self) -> int:
        """Get maximum plant count (array size)"""
        board = self.get_board()
        if board == 0:
            return 0
        return self.read_int(board + Offset.PLANT_COUNT_MAX)
    
    def get_seed_array(self) -> int:
        """Get seed/card array base address"""
        board = self.get_board()
        if board == 0:
            return 0
        return self.read_int(board + Offset.SEED_ARRAY)
    
    def get_item_array(self) -> int:
        """Get item/collectible array base address"""
        board = self.get_board()
        if board == 0:
            return 0
        return self.read_int(board + Offset.ITEM_ARRAY)
    
    def get_item_count_max(self) -> int:
        """Get maximum item count (array size)"""
        board = self.get_board()
        if board == 0:
            return 0
        return self.read_int(board + Offset.ITEM_COUNT_MAX)
    
    # ========================================================================
    # PlayerInfo Methods (玩家存档信息)
    # ========================================================================
    
    def get_player_info(self) -> int:
        """Get PlayerInfo pointer"""
        base = self.get_pvz_base()
        if base == 0:
            return 0
        return self.read_int(base + Offset.PLAYER_INFO)
    
    def get_player_level(self) -> int:
        """Get player's adventure level (1-50+)"""
        player_info = self.get_player_info()
        if player_info == 0:
            return 0
        return self.read_int(player_info + Offset.PI_LEVEL)
    
    def get_player_coins(self) -> int:
        """Get player's coin count"""
        player_info = self.get_player_info()
        if player_info == 0:
            return 0
        return self.read_int(player_info + Offset.PI_COINS)
    
    def get_finished_adventure(self) -> int:
        """Get number of times player has finished adventure mode"""
        player_info = self.get_player_info()
        if player_info == 0:
            return 0
        return self.read_int(player_info + Offset.PI_FINISHED_ADV)
    
    def get_purchase(self, index: int) -> int:
        """Get purchase count for a store item (0 = not purchased)"""
        if index < 0 or index >= Offset.PI_PURCHASE_COUNT:
            return 0
        player_info = self.get_player_info()
        if player_info == 0:
            return 0
        addr = player_info + Offset.PI_PURCHASES + index * Offset.PI_PURCHASE_SIZE
        return self.read_int(addr)
    
    def get_all_purchases(self) -> List[int]:
        """Get all purchase counts as a list"""
        player_info = self.get_player_info()
        if player_info == 0:
            return [0] * Offset.PI_PURCHASE_COUNT
        
        purchases = []
        base_addr = player_info + Offset.PI_PURCHASES
        for i in range(Offset.PI_PURCHASE_COUNT):
            purchases.append(self.read_int(base_addr + i * Offset.PI_PURCHASE_SIZE))
        return purchases
