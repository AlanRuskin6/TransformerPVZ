"""
Memory Writer Module
Handles writing values to PVZ process memory

⚠️ DEPRECATED: This module is deprecated and should not be used for new code.
Use the Hook DLL (hook_client) instead for stable and safe memory operations.
This module is kept for backward compatibility only.
"""

import warnings
import ctypes
from typing import Optional

# Issue deprecation warning
warnings.warn(
    "memory.writer is deprecated. Use hook_client for stable memory operations.",
    DeprecationWarning,
    stacklevel=2
)


class MemoryWriter:
    """Writes values to process memory"""
    
    def __init__(self, kernel32, process_handle: int):
        self.kernel32 = kernel32
        self.process = process_handle
    
    def write_int(self, address: int, value: int) -> bool:
        """Write a 4-byte integer to memory"""
        buf = ctypes.c_int(value)
        written = ctypes.c_size_t()
        result = self.kernel32.WriteProcessMemory(
            self.process, address, ctypes.byref(buf), 4, ctypes.byref(written)
        )
        return result != 0
    
    def write_uint(self, address: int, value: int) -> bool:
        """Write a 4-byte unsigned integer to memory"""
        buf = ctypes.c_uint(value)
        written = ctypes.c_size_t()
        result = self.kernel32.WriteProcessMemory(
            self.process, address, ctypes.byref(buf), 4, ctypes.byref(written)
        )
        return result != 0
    
    def write_float(self, address: int, value: float) -> bool:
        """Write a 4-byte float to memory"""
        buf = ctypes.c_float(value)
        written = ctypes.c_size_t()
        result = self.kernel32.WriteProcessMemory(
            self.process, address, ctypes.byref(buf), 4, ctypes.byref(written)
        )
        return result != 0
    
    def write_byte(self, address: int, value: int) -> bool:
        """Write a single byte to memory"""
        buf = ctypes.c_byte(value)
        written = ctypes.c_size_t()
        result = self.kernel32.WriteProcessMemory(
            self.process, address, ctypes.byref(buf), 1, ctypes.byref(written)
        )
        return result != 0
    
    def write_bool(self, address: int, value: bool) -> bool:
        """Write a boolean (single byte) to memory"""
        return self.write_byte(address, 1 if value else 0)
    
    def write_bytes(self, address: int, data: bytes) -> bool:
        """Write multiple bytes to memory"""
        written = ctypes.c_size_t()
        result = self.kernel32.WriteProcessMemory(
            self.process, address, data, len(data), ctypes.byref(written)
        )
        return result != 0
    
    def write_short(self, address: int, value: int) -> bool:
        """Write a 2-byte short to memory"""
        buf = ctypes.c_short(value)
        written = ctypes.c_size_t()
        result = self.kernel32.WriteProcessMemory(
            self.process, address, ctypes.byref(buf), 2, ctypes.byref(written)
        )
        return result != 0
    
    def write_double(self, address: int, value: float) -> bool:
        """Write an 8-byte double to memory"""
        buf = ctypes.c_double(value)
        written = ctypes.c_size_t()
        result = self.kernel32.WriteProcessMemory(
            self.process, address, ctypes.byref(buf), 8, ctypes.byref(written)
        )
        return result != 0
