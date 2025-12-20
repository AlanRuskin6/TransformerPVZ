"""
Level Control Module
Controls game level restart through memory manipulation and function calls
"""

import time
import ctypes
from typing import Optional
from data.offsets import Offset


class LevelController:
    """
    Controls PVZ game level via memory manipulation.
    
    Based on AsmVsZombies implementation.
    """
    
    # Game UI states
    UI_LOADING = 0
    UI_MAIN_MENU = 1
    UI_LEVEL_INTRO = 2  # 选卡界面
    UI_PLAYING = 3      # 战斗中
    UI_ZOMBIES_WON = 4
    UI_AWARD = 5
    UI_CREDIT = 6
    UI_CHALLENGE = 7
    
    # Game modes (from AAsm::GameMode)
    MODE_ADVENTURE = 0
    MODE_SURVIVAL_DAY = 1
    MODE_SURVIVAL_NIGHT = 2
    MODE_SURVIVAL_POOL = 3
    MODE_SURVIVAL_FOG = 4
    MODE_SURVIVAL_ROOF = 5
    MODE_SURVIVAL_DAY_HARD = 6
    MODE_SURVIVAL_NIGHT_HARD = 7
    MODE_SURVIVAL_POOL_HARD = 8
    MODE_SURVIVAL_FOG_HARD = 9
    MODE_SURVIVAL_ROOF_HARD = 10
    MODE_SURVIVAL_DAY_ENDLESS = 11
    MODE_SURVIVAL_NIGHT_ENDLESS = 12
    MODE_SURVIVAL_POOL_ENDLESS = 13
    MODE_SURVIVAL_FOG_ENDLESS = 14
    MODE_SURVIVAL_ROOF_ENDLESS = 15
    
    # Function addresses (from avz_asm.cpp)
    ADDR_ENTER_GAME = 0x44F560
    ADDR_BACK_TO_MAIN = 0x44FEB0
    ADDR_DELETE_MAIN_MENU = 0x44F9E0
    ADDR_DELETE_LOADING = 0x452CB0
    ADDR_DELETE_OPTIONS = 0x44FD00
    
    def __init__(self, reader, writer, injector):
        """
        Args:
            reader: MemoryReader instance
            writer: MemoryWriter instance  
            injector: MemoryInjector instance (for calling game functions)
        """
        self.reader = reader
        self.writer = writer
        self.injector = injector
    
    def get_game_ui(self) -> int:
        """Get current game UI state"""
        base = self.reader.read_int(Offset.BASE)
        if not base:
            return -1
        return self.reader.read_int(base + Offset.GAME_UI)
    
    def is_in_game(self) -> bool:
        """Check if currently in a game level"""
        return self.get_game_ui() == self.UI_PLAYING
    
    def is_in_select_cards(self) -> bool:
        """Check if in card selection screen"""
        return self.get_game_ui() == self.UI_LEVEL_INTRO
    
    def is_in_main_menu(self) -> bool:
        """Check if in main menu"""
        return self.get_game_ui() == self.UI_MAIN_MENU
    
    def back_to_main(self) -> bool:
        """
        Return to main menu from game.
        
        Returns:
            True if successful
        """
        if not self.is_in_game():
            return False
        
        # Call DoBackToMain (0x44feb0)
        # This is equivalent to the assembly:
        # movl 0x6a9ec0, %eax
        # movl $0x44feb0, %ecx
        # call *%ecx
        return self.injector.call_function(self.ADDR_BACK_TO_MAIN)
    
    def restart_level(self, wait_for_game: bool = True, timeout: float = 30.0) -> bool:
        """
        Restart current level by going back to main menu and re-entering.
        
        This is a simplified approach - it just triggers the game over screen
        to restart the same level the player was on.
        
        Args:
            wait_for_game: Whether to wait for game to be ready
            timeout: Maximum wait time in seconds
            
        Returns:
            True if successfully restarted
        """
        # For now, use a simpler approach: set sun to 0 and wait for player to lose
        # Or we can use keyboard simulation
        
        # The most reliable way is to use keyboard to press menu -> restart
        return self._restart_via_keyboard()
    
    def _restart_via_keyboard(self) -> bool:
        """Restart level using keyboard simulation"""
        import ctypes
        
        user32 = ctypes.windll.user32
        
        # Find PVZ window
        hwnd = user32.FindWindowW(None, "Plants vs. Zombies")
        if not hwnd:
            hwnd = user32.FindWindowW(None, "植物大战僵尸中文版")
        if not hwnd:
            hwnd = user32.FindWindowW(None, "植物大战僵尸")
        
        if not hwnd:
            print("找不到 PVZ 窗口")
            return False
        
        # Bring window to front
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.1)
        
        # Press Escape to open menu
        VK_ESCAPE = 0x1B
        user32.keybd_event(VK_ESCAPE, 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(VK_ESCAPE, 0, 2, 0)  # Key up
        time.sleep(0.3)
        
        # Click "Restart" button (approximate position)
        # This depends on screen resolution, may need adjustment
        # For 800x600 resolution, restart button is around (400, 300)
        
        # Get window rect
        rect = ctypes.wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        
        # Calculate button position (center of window, slightly above middle)
        center_x = (rect.left + rect.right) // 2
        center_y = (rect.top + rect.bottom) // 2 - 50  # Slightly above center
        
        # Move mouse and click
        user32.SetCursorPos(center_x, center_y)
        time.sleep(0.1)
        
        # Click (left button down then up)
        user32.mouse_event(0x0002, 0, 0, 0, 0)  # Left down
        time.sleep(0.05)
        user32.mouse_event(0x0004, 0, 0, 0, 0)  # Left up
        time.sleep(0.5)
        
        # Wait for game to start
        return self._wait_for_game_start(timeout=10.0)
    
    def _wait_for_game_start(self, timeout: float = 10.0) -> bool:
        """Wait for game to enter playing state"""
        start = time.time()
        while time.time() - start < timeout:
            if self.is_in_game():
                return True
            time.sleep(0.2)
        return False
    
    def quick_restart(self) -> bool:
        """
        Quick restart by directly manipulating game state.
        
        This clears all entities and resets the game to wave 0.
        WARNING: This may cause instability!
        """
        if not self.is_in_game():
            return False
        
        base = self.reader.read_int(Offset.BASE)
        if not base:
            return False
        
        board = self.reader.read_int(base + Offset.MAIN_OBJECT)
        if not board:
            return False
        
        # Reset sun to starting amount (50)
        self.writer.write_int(board + Offset.SUN, 50)
        
        # Reset wave to 0
        self.writer.write_int(board + Offset.WAVE, 0)
        
        # Clear all zombies (set dead flag)
        zombie_array = self.reader.read_int(board + Offset.ZOMBIE_ARRAY)
        zombie_max = self.reader.read_int(board + Offset.ZOMBIE_COUNT_MAX)
        
        for i in range(min(zombie_max, 200)):
            addr = zombie_array + i * Offset.ZOMBIE_SIZE
            self.writer.write_byte(addr + Offset.Z_DEAD, 1)
        
        # Clear all plants (set dead flag)
        plant_array = self.reader.read_int(board + Offset.PLANT_ARRAY)
        plant_max = self.reader.read_int(board + Offset.PLANT_COUNT_MAX)
        
        for i in range(min(plant_max, 200)):
            addr = plant_array + i * Offset.PLANT_SIZE
            self.writer.write_byte(addr + Offset.P_DEAD, 1)
        
        # Reset lawnmowers
        lm_array = self.reader.read_int(board + Offset.LAWNMOWER_ARRAY)
        lm_max = self.reader.read_int(board + Offset.LAWNMOWER_COUNT_MAX)
        
        for i in range(min(lm_max, 10)):
            addr = lm_array + i * Offset.LAWNMOWER_SIZE
            self.writer.write_bool(addr + Offset.LM_DEAD, False)
        
        return True
