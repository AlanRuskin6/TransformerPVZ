"""
Process Attachment Module
Handles finding and attaching to the PVZ process
"""

import ctypes
import ctypes.wintypes as wt
from typing import Optional


class ProcessAttacher:
    """Handles attaching to the PVZ process"""
    
    # Windows API constants
    PROCESS_ALL_ACCESS = 0x1F0FFF
    
    def __init__(self):
        self.kernel32 = ctypes.windll.kernel32
        self.user32 = ctypes.windll.user32
        self.process_handle: Optional[int] = None
        self.pid: Optional[int] = None
        
    def find_pvz_window(self) -> Optional[int]:
        """
        Find the PVZ game window
        
        Returns:
            Window handle (HWND) or None if not found
        """
        # Try standard window title first
        hwnd = self.user32.FindWindowW(None, "Plants vs. Zombies")
        if hwnd:
            return hwnd
        
        # Try class name
        hwnd = self.user32.FindWindowW("MainWindow", None)
        if hwnd:
            return hwnd
        
        # Try Chinese title
        hwnd = self.user32.FindWindowW(None, "植物大战僵尸")
        if hwnd:
            return hwnd
        
        return None
    
    def attach(self) -> bool:
        """
        Attach to the PVZ process
        
        Returns:
            True if successfully attached, False otherwise
        """
        hwnd = self.find_pvz_window()
        if not hwnd:
            return False
        
        # Get process ID from window
        pid = wt.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        self.pid = pid.value
        
        if not self.pid:
            return False
        
        # Open process with full access
        self.process_handle = self.kernel32.OpenProcess(
            self.PROCESS_ALL_ACCESS, 
            False, 
            self.pid
        )
        
        return self.process_handle != 0
    
    def detach(self):
        """Detach from the process"""
        if self.process_handle:
            self.kernel32.CloseHandle(self.process_handle)
            self.process_handle = None
            self.pid = None
    
    def is_attached(self) -> bool:
        """Check if attached to process and process is still running"""
        if not self.process_handle:
            return False
            
        exit_code = wt.DWORD()
        # STILL_ACTIVE = 259
        if self.kernel32.GetExitCodeProcess(self.process_handle, ctypes.byref(exit_code)):
            return exit_code.value == 259
        return False
    
    @property
    def handle(self) -> Optional[int]:
        """Get the process handle"""
        return self.process_handle
    
    def __del__(self):
        """Clean up on destruction"""
        self.detach()
