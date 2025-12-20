"""
ASM Injector Module
Handles shellcode injection for calling game functions directly

⚠️ DEPRECATED: This module is deprecated and should not be used for new code.
Use the Hook DLL (hook_client) instead for stable and crash-free operations.
Shellcode injection has timing issues that can cause game crashes.
This module is kept for backward compatibility only.
"""

import warnings
import struct
import ctypes
from ctypes import wintypes
from typing import Optional, List

from data.offsets import Offset
from memory.reader import MemoryReader

# Issue deprecation warning
warnings.warn(
    "memory.injector is deprecated. Use hook_client for stable operations without crashes.",
    DeprecationWarning,
    stacklevel=2
)


# Windows API constants
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_EXECUTE_READWRITE = 0x40


class AsmInjector:
    """
    Injects and executes ASM shellcode in the game process
    
    This allows direct calling of game functions for reliable operations
    like planting, shoveling, and firing cob cannons.
    """
    
    def __init__(self, kernel32, process_handle: int, reader: MemoryReader):
        self.kernel32 = kernel32
        self.process = process_handle
        self.reader = reader
    
    def alloc_memory(self, size: int) -> int:
        """
        Allocate executable memory in the game process
        
        Args:
            size: Number of bytes to allocate
            
        Returns:
            Address of allocated memory, or 0 on failure
        """
        addr = self.kernel32.VirtualAllocEx(
            self.process,
            None,
            size,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE
        )
        return addr or 0
    
    def free_memory(self, address: int):
        """Free previously allocated memory"""
        self.kernel32.VirtualFreeEx(self.process, address, 0, MEM_RELEASE)
    
    def write_bytes(self, address: int, data: bytes) -> bool:
        """Write bytes to process memory"""
        written = ctypes.c_size_t()
        return self.kernel32.WriteProcessMemory(
            self.process, address, data, len(data), ctypes.byref(written)
        )
    
    def execute_shellcode(self, shellcode: bytes, timeout: int = 1000, debug_name: str = "unknown") -> bool:
        """
        Execute shellcode in the game process
        
        Process:
        1. Allocate executable memory in game process
        2. Write shellcode to that memory
        3. Create remote thread to execute it
        4. Wait for completion
        5. Free memory
        
        Args:
            shellcode: The machine code to execute
            timeout: Maximum time to wait for execution (ms)
            debug_name: Name of operation for debugging
            
        Returns:
            True if execution succeeded, False otherwise
        """
        # Allocate memory for shellcode
        addr = self.alloc_memory(len(shellcode) + 16)
        if not addr:
            print(f"[Shellcode] {debug_name}: 内存分配失败")
            return False
        
        try:
            # Write shellcode to game memory
            if not self.write_bytes(addr, shellcode):
                print(f"[Shellcode] {debug_name}: 写入失败")
                return False
            
            # Create remote thread to execute
            thread_id = wintypes.DWORD()
            thread = self.kernel32.CreateRemoteThread(
                self.process,
                None, 0,
                addr, None, 0,
                ctypes.byref(thread_id)
            )
            
            if not thread:
                print(f"[Shellcode] {debug_name}: 创建线程失败")
                return False
            
            # Wait for thread to complete
            WAIT_OBJECT_0 = 0x00000000
            WAIT_TIMEOUT = 0x00000102
            wait_result = self.kernel32.WaitForSingleObject(thread, timeout)
            self.kernel32.CloseHandle(thread)
            
            # Check if thread completed successfully
            if wait_result == WAIT_OBJECT_0:
                return True
            elif wait_result == WAIT_TIMEOUT:
                # Thread timed out - game may have crashed
                print(f"[Shellcode] {debug_name}: 超时 (游戏可能崩溃)")
                return False
            else:
                # Other error
                print(f"[Shellcode] {debug_name}: 等待失败 (code={wait_result})")
                return False
            
        finally:
            # Always free the allocated memory
            self.free_memory(addr)
    
    # ========================================================================
    # High-Level Game Functions
    # ========================================================================
    
    def _find_seed_index_by_type(self, plant_type: int) -> int:
        """
        Find the seed slot index for a given plant type.
        
        Args:
            plant_type: The plant type ID to find
            
        Returns:
            Seed slot index (0-based), or -1 if not found
        """
        board = self.reader.get_board()
        if board == 0:
            return -1
        
        seed_array = self.reader.read_int(board + Offset.SEED_ARRAY)
        if seed_array == 0:
            return -1
        
        # Search through seed slots (max 10)
        for i in range(10):
            addr = seed_array + i * Offset.SEED_SIZE
            s_type = self.reader.read_int(addr + Offset.S_TYPE)
            if s_type == plant_type:
                return i
            # Also check imitator type
            s_imitator = self.reader.read_int(addr + Offset.S_IMITATOR_TYPE)
            if s_imitator == plant_type:
                return i
        
        return -1
    
    def _grid_to_pixel(self, row: int, col: int) -> tuple:
        """
        Convert grid coordinates to pixel coordinates.
        
        Based on re-plants-vs-zombies source:
        - x = col * 80 + LAWN_XMIN (40)
        - y = row * ROW_HEIGHT + LAWN_YMIN (80)
        
        ROW_HEIGHT varies by scene:
        - Normal/Night: 100
        - Pool/Fog: 85
        - Roof: 85 (with slope offset)
        
        Args:
            row: Row index (0-based)
            col: Column index (0-based)
            
        Returns:
            (x, y) pixel coordinates
        """
        LAWN_XMIN = 40
        LAWN_YMIN = 80
        
        x = col * 80 + LAWN_XMIN
        
        # Get scene to determine row height
        scene = self.reader.get_scene()
        
        if scene in [2, 3]:  # Pool, Fog
            row_height = 85
            y = row * row_height + LAWN_YMIN
        elif scene in [4, 5]:  # Roof, Roof Night
            row_height = 85
            # Roof has slope: higher columns are lower
            slope_offset = max(0, (5 - col) * 20) if col < 5 else 0
            y = row * row_height + LAWN_YMIN - 10 + slope_offset
        else:  # Day, Night (0, 1)
            row_height = 100
            y = row * row_height + LAWN_YMIN
        
        return (x, y)
    
    def plant(self, row: int, col: int, plant_type: int, imitator_type: int = -1) -> bool:
        """
        Plant at a specific position using PlantCard function.
        
        This properly uses the game's card system:
        1. Find the seed slot index for the plant type
        2. Convert grid to pixel coordinates
        3. Call PlantCard (0x488590) to properly plant
        
        This will:
        - Check if card is available
        - Deduct sun cost
        - Trigger cooldown
        
        Based on LLMvsZ implementation.
        
        Args:
            row: Row to plant (0-5)
            col: Column to plant (0-8)
            plant_type: Plant type ID
            imitator_type: Type if using imitator (-1 if not)
            
        Returns:
            True if successful, False otherwise
        """
        board = self.reader.get_board()
        if board == 0:
            return False
        
        # Find seed slot index for this plant type
        seed_index = self._find_seed_index_by_type(plant_type)
        if seed_index < 0:
            return False
        
        # Convert grid to pixel coordinates
        x, y = self._grid_to_pixel(row, col)
        
        # Get seed array address
        seed_array = self.reader.read_int(board + Offset.SEED_ARRAY)
        if seed_array == 0:
            return False
        
        # Calculate seed address (seed_array + index * 0x50 + 0x28)
        # The +0x28 is the offset to the ASeed data within the slot
        seed_addr = seed_array + seed_index * Offset.SEED_SIZE + 0x28
        
        # Build shellcode based on AVZ PlantCard function
        # This calls SeedPacket::MouseDown (0x488590) then checks plant ready (0x40fd30)
        # Finally calls ReleaseMouse (0x40cd80) to release mouse control
        shellcode = bytes([
            # push y
            0x68, *struct.pack('<i', y),
            
            # push x
            0x68, *struct.pack('<i', x),
            
            # push board
            0x68, *struct.pack('<I', board),
            
            # push seed_addr (the seed packet)
            0x68, *struct.pack('<I', seed_addr),
            
            # mov ecx, 0x488590 (SeedPacket::MouseDown / PlantCard)
            0xB9, *struct.pack('<I', 0x488590),
            
            # call ecx
            0xFF, 0xD1,
            
            # mov ecx, 1 (for plant ready check)
            0xB9, 0x01, 0x00, 0x00, 0x00,
            
            # mov edx, 0x40fd30 (check plant ready)
            0xBA, *struct.pack('<I', 0x40fd30),
            
            # call edx
            0xFF, 0xD2,
            
            # ReleaseMouse - release mouse control so user can click
            # mov eax, [0x6a9ec0]
            0xA1, *struct.pack('<I', 0x6a9ec0),
            
            # mov eax, [eax + 0x768]
            0x8B, 0x80, *struct.pack('<I', 0x768),
            
            # mov edx, 0x40cd80 (ReleaseMouse)
            0xBA, *struct.pack('<I', 0x40cd80),
            
            # call edx
            0xFF, 0xD2,
            
            # ret
            0xC3
        ])
        
        return self.execute_shellcode(shellcode, timeout=2000, debug_name="plant")
    
    def shovel(self, row: int, col: int) -> bool:
        """
        Remove/shovel a plant at a specific position
        
        First finds the plant at the position, then calls RemovePlant.
        
        Args:
            row: Row of plant
            col: Column of plant
            
        Returns:
            True if successful, False otherwise
        """
        board = self.reader.get_board()
        if board == 0:
            return False
        
        # Find the plant at this position
        plant_array = self.reader.read_int(board + Offset.PLANT_ARRAY)
        if plant_array == 0:
            return False
            
        plant_max = self.reader.read_int(board + Offset.PLANT_COUNT_MAX)
        # Validate plant_max is within reasonable bounds (cap at 200 for safety)
        if plant_max <= 0:
            return False
        plant_max = min(plant_max, 200)
        
        plant_addr = None
        for i in range(plant_max):
            addr = plant_array + i * Offset.PLANT_SIZE
            if self.reader.read_byte(addr + Offset.P_DEAD):
                continue
            p_row = self.reader.read_int(addr + Offset.P_ROW)
            p_col = self.reader.read_int(addr + Offset.P_COL)
            if p_row == row and p_col == col:
                plant_addr = addr
                break
        
        if plant_addr is None:
            return False
        
        # Call RemovePlant
        shellcode = bytes([
            # push plant_addr
            0x68, *struct.pack('<I', plant_addr),
            
            # mov edx, FUNC_REMOVE_PLANT
            0xBA, *struct.pack('<I', Offset.FUNC_REMOVE_PLANT),
            
            # call edx
            0xFF, 0xD2,
            
            # ret
            0xC3
        ])
        
        return self.execute_shellcode(shellcode, debug_name="shovel")
    
    def refresh_seed_cooldowns(self) -> bool:
        """
        Refresh all seed card cooldowns
        
        Returns:
            True if successful, False otherwise
        """
        board = self.reader.get_board()
        if board == 0:
            return False
        
        seed_bank = self.reader.read_int(board + Offset.SEED_ARRAY)
        if seed_bank == 0:
            return False
        
        shellcode = bytes([
            # push seed_bank
            0x68, *struct.pack('<I', seed_bank),
            
            # mov eax, FUNC_REFRESH_SEEDS
            0xB8, *struct.pack('<I', Offset.FUNC_REFRESH_SEEDS),
            
            # call eax
            0xFF, 0xD0,
            
            # ret
            0xC3
        ])
        
        return self.execute_shellcode(shellcode, debug_name="refresh_seeds")
    
    def fire_cob(self, cob_index: int, target_x: float, target_y: float) -> bool:
        """
        Fire a cob cannon at a specific position
        
        Uses ASM injection to call the game's cob fire function.
        Based on AVZ cob_manager implementation.
        
        Args:
            cob_index: Index of the cob cannon plant in plant array
            target_x: Target x coordinate (pixels)
            target_y: Target y coordinate (pixels)
            
        Returns:
            True if successful, False otherwise
        """
        board = self.reader.get_board()
        if board == 0:
            return False
        
        # Get plant array and calculate cob address
        plant_array = self.reader.read_int(board + Offset.PLANT_ARRAY)
        if plant_array == 0:
            return False
        
        cob_addr = plant_array + cob_index * Offset.PLANT_SIZE
        
        # Verify the plant is a cob cannon and is ready
        plant_type = self.reader.read_int(cob_addr + Offset.P_TYPE)
        from data.plants import PlantType
        if plant_type != PlantType.COBCANNON:
            return False
        
        cob_ready = self.reader.read_bool(cob_addr + Offset.P_COB_READY)
        if not cob_ready:
            return False
        
        # Convert target to integers for the function call
        target_x_int = int(target_x)
        target_y_int = int(target_y)
        
        # Shellcode to fire cob cannon
        # Based on PVZ 1.0.0.1051 reverse engineering
        # The fire function: void __thiscall FireCob(Plant* this, int x, int y)
        shellcode = bytes([
            # push target_y
            0x68, *struct.pack('<I', target_y_int),
            
            # push target_x
            0x68, *struct.pack('<I', target_x_int),
            
            # mov ecx, cob_addr (this pointer)
            0xB9, *struct.pack('<I', cob_addr),
            
            # mov eax, FUNC_COB_FIRE
            0xB8, *struct.pack('<I', Offset.FUNC_COB_FIRE),
            
            # call eax
            0xFF, 0xD0,
            
            # ret
            0xC3
        ])
        
        return self.execute_shellcode(shellcode, debug_name="fire_cob")
    
    def collect_sun(self, item_addr: int) -> bool:
        """
        Collect a specific sun/item
        
        This is a simple method that sets the collected flag.
        The game will automatically add the sun value.
        
        Args:
            item_addr: Memory address of the item
            
        Returns:
            True if successful
        """
        # Write directly to the collected flag
        buf = ctypes.c_byte(1)
        return self.kernel32.WriteProcessMemory(
            self.process, 
            item_addr + Offset.I_COLLECTED, 
            ctypes.byref(buf), 
            1, 
            None
        )
    
    def make_new_board(self) -> bool:
        """
        Reset the current level by calling MakeNewBoard.
        
        This function:
        1. Clears all plants, zombies, projectiles
        2. Resets sun to starting value
        3. Resets wave to 0
        4. Keeps the same scene/level type
        
        Based on AsmVsZombies: AAsm::MakeNewBoard()
        Address: 0x44F5F0
        
        Returns:
            True if successful
        """
        # First save the current scene so we can restore it
        board = self.reader.get_board()
        if board == 0:
            return False
        
        scene = self.reader.read_int(board + Offset.SCENE)
        
        # Shellcode to call MakeNewBoard
        # Based on avz_asm.cpp:
        #   movl 0x6a9ec0, %ecx
        #   movl $0x44f5f0, %eax
        #   call *%eax
        #   (then ProcessSafeDeleteList)
        #   movl 0x6a9ec0, %ecx
        #   pushl %ecx
        #   movl $0x5518f0, %eax
        #   call *%eax
        
        shellcode = bytes([
            # mov ecx, [0x6a9ec0]  ; Get PvzBase
            0x8B, 0x0D, *struct.pack('<I', Offset.BASE),
            
            # mov eax, 0x44f5f0  ; MakeNewBoard
            0xB8, *struct.pack('<I', 0x44F5F0),
            
            # call eax
            0xFF, 0xD0,
            
            # ProcessSafeDeleteList
            # mov ecx, [0x6a9ec0]
            0x8B, 0x0D, *struct.pack('<I', Offset.BASE),
            
            # push ecx
            0x51,
            
            # mov eax, 0x5518f0
            0xB8, *struct.pack('<I', 0x5518F0),
            
            # call eax
            0xFF, 0xD0,
            
            # add esp, 4 (clean up push)
            0x83, 0xC4, 0x04,
            
            # ret
            0xC3
        ])
        
        success = self.execute_shellcode(shellcode, timeout=3000, debug_name="make_new_board")
        
        if success:
            # Restore scene (MakeNewBoard may change it)
            # Get new board address (it might have changed)
            new_board = self.reader.get_board()
            if new_board:
                # Write scene back
                buf = ctypes.c_int(scene)
                self.kernel32.WriteProcessMemory(
                    self.process,
                    new_board + Offset.SCENE,
                    ctypes.byref(buf),
                    4,
                    None
                )
        
        return success
    
    def enter_game(self, game_mode: int) -> bool:
        """
        Enter a specific game mode directly.
        
        Game modes:
            0 = Adventure
            1-5 = Survival Normal (Day, Night, Pool, Fog, Roof)
            6-10 = Survival Hard
            11-15 = Survival Endless
        
        Based on AAsm::EnterGame()
        Address: 0x44F560
        
        Note: Only works from main menu, not during gameplay.
        
        Args:
            game_mode: Game mode ID
            
        Returns:
            True if successful
        """
        # Check current UI state
        base = self.reader.read_int(Offset.BASE)
        if not base:
            return False
        
        game_ui = self.reader.read_int(base + Offset.GAME_UI)
        
        # Only works from main menu (UI = 1) or loading (UI = 0)
        if game_ui == 2 or game_ui == 3:  # In select cards or playing
            return False
        
        # Shellcode to enter game
        # Based on avz_asm.cpp EnterGame:
        #   push 1 (ok)
        #   push gameMode
        #   mov esi, [0x6a9ec0]
        #   mov eax, 0x44f560
        #   call eax
        
        ok = 1
        shellcode = bytes([
            # push 1 (ok = true)
            0x6A, 0x01,
            
            # push game_mode
            0x68, *struct.pack('<I', game_mode),
            
            # mov esi, [0x6a9ec0]
            0x8B, 0x35, *struct.pack('<I', Offset.BASE),
            
            # mov eax, 0x44f560
            0xB8, *struct.pack('<I', 0x44F560),
            
            # call eax
            0xFF, 0xD0,
            
            # ret
            0xC3
        ])
        
        return self.execute_shellcode(shellcode, timeout=5000, debug_name="enter_game")

    def back_to_main(self) -> bool:
        """
        Return to main menu from gameplay.
        
        Based on AAsm::DoBackToMain()
        Address: 0x44FEB0
        
        Returns:
            True if successful
        """
        # Check if in game
        base = self.reader.read_int(Offset.BASE)
        if not base:
            return False
        
        game_ui = self.reader.read_int(base + Offset.GAME_UI)
        if game_ui != 3:  # Not in playing state
            return False
        
        # Shellcode:
        #   mov eax, [0x6a9ec0]
        #   mov ecx, 0x44feb0
        #   call ecx
        shellcode = bytes([
            # mov eax, [0x6a9ec0]
            0xA1, *struct.pack('<I', Offset.BASE),
            
            # mov ecx, 0x44feb0
            0xB9, *struct.pack('<I', 0x44FEB0),
            
            # call ecx
            0xFF, 0xD1,
            
            # ret
            0xC3
        ])
        
        return self.execute_shellcode(shellcode, timeout=3000, debug_name="back_to_main")

    def click_seed_chooser_button(self) -> bool:
        """
        Click the "Let's Rock" button in seed chooser to start the game.
        
        Based on AAsm::Rock() from avz_asm.cpp (line 111-123)
        Uses correct function address 0x486D20 and required register setup
        
        Returns:
            True if successful
        """
        base = self.reader.read_int(Offset.BASE)
        if not base:
            return False
        
        game_ui = self.reader.read_int(base + Offset.GAME_UI)
        if game_ui != 2:  # Not in seed chooser
            return False
        
        # Get seed chooser screen pointer: [base + 0x774]
        seed_chooser = self.reader.read_int(base + Offset.SEED_CHOOSER)
        if not seed_chooser:
            return False
        
        # Based on AAsm::Rock() (avz_asm.cpp line 111-123):
        # mov ebx, [0x6a9ec0]
        # mov ebx, [ebx+0x774]  ; seed chooser
        # mov eax, FUNC_ROCK (0x486d20)
        # mov esi, [0x6a9ec0]
        # mov edi, 1
        # mov ebp, 1
        # call eax
        # With ASaveAllRegister macro
        
        shellcode = bytes([
            # Save registers (ASaveAllRegister)
            0x55,                           # push ebp
            0x53,                           # push ebx
            0x56,                           # push esi
            0x57,                           # push edi
            
            # mov ebx, [0x6a9ec0]
            0x8B, 0x1D, *struct.pack('<I', Offset.BASE),
            
            # mov ebx, [ebx+0x774]  ; seed chooser
            0x8B, 0x9B, *struct.pack('<I', Offset.SEED_CHOOSER),
            
            # mov eax, FUNC_ROCK (0x486d20)
            0xB8, *struct.pack('<I', Offset.FUNC_ROCK),
            
            # mov esi, [0x6a9ec0]
            0x8B, 0x35, *struct.pack('<I', Offset.BASE),
            
            # mov edi, 1
            0xBF, 0x01, 0x00, 0x00, 0x00,
            
            # mov ebp, 1
            0xBD, 0x01, 0x00, 0x00, 0x00,
            
            # call eax
            0xFF, 0xD0,
            
            # Restore registers
            0x5F,                           # pop edi
            0x5E,                           # pop esi
            0x5B,                           # pop ebx
            0x5D,                           # pop ebp
            
            # ret
            0xC3
        ])
        
        return self.execute_shellcode(shellcode, timeout=3000, debug_name="click_lets_rock")

    def choose_seed(self, plant_type: int) -> bool:
        """
        Choose a seed in the seed chooser screen.
        
        Based on AAsm::ChooseCard() from avz_asm.cpp
        Uses ASaveAllRegister (saves/restores ebp, ebx, esi, edi)
        
        Args:
            plant_type: Plant type ID to choose (0=Peashooter, 1=Sunflower, etc.)
            
        Returns:
            True if successful
        """
        base = self.reader.read_int(Offset.BASE)
        if not base:
            return False
        
        game_ui = self.reader.read_int(base + Offset.GAME_UI)
        if game_ui != 2:  # Not in seed chooser
            return False
        
        # Based on AAsm::ChooseCard (avz_asm.cpp line 258-275):
        # mov eax, [0x6a9ec0]
        # mov eax, [eax+0x774]  ; seed chooser screen
        # edx = cardType * 15 * 4 + 0xa4 + eax  ; calculate card address
        # push edx
        # call 0x486030
        # With ASaveAllRegister macro (push/pop ebp, ebx, esi, edi)
        
        seed_chooser = self.reader.read_int(base + Offset.SEED_CHOOSER)
        if not seed_chooser:
            return False
        
        # Calculate card address: cardType * 60 + 0xa4 + seed_chooser
        card_addr = plant_type * 60 + 0xa4 + seed_chooser
        
        shellcode = bytes([
            # Save registers (ASaveAllRegister)
            0x55,                           # push ebp
            0x53,                           # push ebx
            0x56,                           # push esi
            0x57,                           # push edi
            
            # push card_addr
            0x68, *struct.pack('<I', card_addr),
            
            # mov ecx, FUNC_CHOOSE_CARD (0x486030)
            0xB9, *struct.pack('<I', Offset.FUNC_CHOOSE_CARD),
            
            # call ecx
            0xFF, 0xD1,
            
            # Restore registers
            0x5F,                           # pop edi
            0x5E,                           # pop esi
            0x5B,                           # pop ebx
            0x5D,                           # pop ebp
            
            # ret
            0xC3
        ])
        
        return self.execute_shellcode(shellcode, timeout=1000, debug_name=f"choose_seed_{plant_type}")

    def pick_random_seeds_and_start(self) -> bool:
        """
        Fill remaining seed slots with random seeds and start the game.
        
        Based on AAsm::PickRandomSeeds() from avz_asm.cpp (line 887-897)
        Uses ASaveAllRegister (saves/restores ebp, ebx, esi, edi)
        
        Returns:
            True if successful
        """
        base = self.reader.read_int(Offset.BASE)
        if not base:
            return False
        
        game_ui = self.reader.read_int(base + Offset.GAME_UI)
        if game_ui != 2:  # Not in seed chooser
            return False
        
        seed_chooser = self.reader.read_int(base + Offset.SEED_CHOOSER)
        if not seed_chooser:
            return False
        
        # Based on AAsm::PickRandomSeeds (avz_asm.cpp line 887-897):
        # mov eax, [0x6a9ec0]
        # mov eax, [eax+SEED_CHOOSER]
        # push eax
        # mov ecx, FUNC_PICK_RANDOM (0x4859b0)
        # call ecx
        # With ASaveAllRegister macro (push/pop ebp, ebx, esi, edi in that order)
        
        shellcode = bytes([
            # Save registers (ASaveAllRegister)
            0x55,                           # push ebp
            0x53,                           # push ebx
            0x56,                           # push esi
            0x57,                           # push edi
            
            # push seed_chooser
            0x68, *struct.pack('<I', seed_chooser),
            
            # mov ecx, FUNC_PICK_RANDOM (0x4859b0)
            0xB9, *struct.pack('<I', Offset.FUNC_PICK_RANDOM),
            
            # call ecx
            0xFF, 0xD1,
            
            # Restore registers
            0x5F,                           # pop edi
            0x5E,                           # pop esi
            0x5B,                           # pop ebx
            0x5D,                           # pop ebp
            
            # ret
            0xC3
        ])
        
        return self.execute_shellcode(shellcode, timeout=3000, debug_name="pick_random_seeds")
