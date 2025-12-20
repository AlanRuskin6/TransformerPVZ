"""
DLL Injector
使用CreateRemoteThread注入Hook DLL
针对32位目标进程的注入
"""

import os
import ctypes
from ctypes import wintypes
import psutil
import logging
from typing import Optional

# Setup logger
logger = logging.getLogger(__name__)


# Windows API常量
PROCESS_ALL_ACCESS = 0x1F0FFF
PROCESS_CREATE_THREAD = 0x0002
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_WRITE = 0x0020
PROCESS_VM_READ = 0x0010
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04
INFINITE = 0xFFFFFFFF


def find_pvz_process() -> Optional[int]:
    """
    查找PVZ进程
    
    Returns:
        进程ID，未找到返回None
    """
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            name = proc.info['name'].lower()
            if 'plantsvszombies' in name or 'popcapgame1' in name:
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def inject_dll(dll_path: Optional[str] = None, pid: Optional[int] = None) -> bool:
    """
    注入DLL到PVZ进程
    
    Args:
        dll_path: DLL路径，默认为hook/pvz_hook.dll
        pid: 目标进程ID，默认自动查找
        
    Returns:
        True if successful
    """
    # 查找DLL路径
    if dll_path is None:
        # 默认路径
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dll_path = os.path.join(script_dir, 'hook', 'pvz_hook.dll')
    
    if not os.path.exists(dll_path):
        logger.error(f"DLL not found: {dll_path}")
        logger.error("Please build the DLL first using hook/build.bat")
        return False
    
    # 转换为绝对路径
    dll_path = os.path.abspath(dll_path)
    
    # 查找PVZ进程
    if pid is None:
        pid = find_pvz_process()
        if pid is None:
            logger.error("PVZ process not found!")
            logger.error("Please start the game first.")
            return False
    
    logger.info(f"Found PVZ process: PID={pid}")
    
    # Check if DLL is already loaded
    try:
        p = psutil.Process(pid)
        dll_name = os.path.basename(dll_path).lower()
        for module in p.memory_maps():
            if dll_name == os.path.basename(module.path).lower():
                logger.info(f"DLL already injected: {module.path}")
                return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        logger.warning("Could not check loaded modules (AccessDenied), proceeding with injection...")
    except Exception as e:
        logger.warning(f"Error checking loaded modules: {e}")

    logger.info(f"Injecting DLL: {dll_path}")
    
    # 获取kernel32
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    
    # 设置函数原型
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    
    kernel32.VirtualAllocEx.argtypes = [wintypes.HANDLE, wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    kernel32.VirtualAllocEx.restype = wintypes.LPVOID
    
    kernel32.WriteProcessMemory.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.LPCVOID, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    kernel32.WriteProcessMemory.restype = wintypes.BOOL
    
    kernel32.CreateRemoteThread.argtypes = [wintypes.HANDLE, wintypes.LPVOID, ctypes.c_size_t, wintypes.LPVOID, wintypes.LPVOID, wintypes.DWORD, wintypes.LPDWORD]
    kernel32.CreateRemoteThread.restype = wintypes.HANDLE
    
    kernel32.VirtualFreeEx.argtypes = [wintypes.HANDLE, wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD]
    kernel32.VirtualFreeEx.restype = wintypes.BOOL
    
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    
    kernel32.GetExitCodeThread.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
    kernel32.GetExitCodeThread.restype = wintypes.BOOL
    
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    
    # 打开目标进程
    hProcess = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, pid)
    if not hProcess:
        logger.error(f"Failed to open process: {ctypes.get_last_error()}")
        return False
    
    try:
        # 使用 ASCII 路径 - DLL 路径必须是 ASCII 兼容的
        # 对于包含中文的路径，使用短路径名
        try:
            GetShortPathNameW = kernel32.GetShortPathNameW
            GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
            GetShortPathNameW.restype = wintypes.DWORD
            
            # 获取短路径
            buf = ctypes.create_unicode_buffer(512)
            result = GetShortPathNameW(dll_path, buf, 512)
            if result > 0:
                short_path = buf.value
                logger.info(f"Using short path: {short_path}")
                dll_path_to_use = short_path
            else:
                dll_path_to_use = dll_path
        except Exception as e:
            logger.warning(f"Could not get short path: {e}")
            dll_path_to_use = dll_path
        
        # 编码为 ASCII（用于 LoadLibraryA）
        try:
            dll_path_bytes = dll_path_to_use.encode('ascii') + b'\x00'
        except UnicodeEncodeError:
            logger.error("DLL path contains non-ASCII characters and short path failed")
            logger.error("Please move the DLL to a path with only ASCII characters")
            return False
        
        dll_path_len = len(dll_path_bytes)
        
        # 在目标进程中分配内存
        pDllPath = kernel32.VirtualAllocEx(
            hProcess, None, dll_path_len,
            MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
        )
        
        if not pDllPath:
            logger.error(f"Failed to allocate memory: {ctypes.get_last_error()}")
            return False
        
        logger.info(f"Allocated memory at: 0x{pDllPath:08X}")
        
        # 写入DLL路径
        written = ctypes.c_size_t()
        if not kernel32.WriteProcessMemory(
            hProcess, pDllPath, dll_path_bytes, dll_path_len, ctypes.byref(written)
        ):
            logger.error(f"Failed to write memory: {ctypes.get_last_error()}")
            kernel32.VirtualFreeEx(hProcess, pDllPath, 0, MEM_RELEASE)
            return False
        
        logger.info(f"Written {written.value} bytes")
        
        # 获取目标进程中 kernel32.dll 的 LoadLibraryA 地址
        # 注意：在32位进程中，kernel32.dll 的基址和64位不同
        # 但是 LoadLibraryA 在 kernel32.dll 中的相对偏移是固定的
        # 由于 kernel32.dll 总是被加载，我们可以使用一个技巧：
        # 在Windows中，所有进程的 kernel32.dll 加载地址相同（ASLR 例外，但对于同一次启动是相同的）
        
        # 使用 ntdll 来获取32位 kernel32 的 LoadLibraryA
        # 这里我们用一个更可靠的方法：直接读取目标进程的 kernel32
        
        # 方法：枚举目标进程的模块找到 kernel32.dll
        
        # MODULEENTRY32
        class MODULEENTRY32(ctypes.Structure):
            _fields_ = [
                ('dwSize', wintypes.DWORD),
                ('th32ModuleID', wintypes.DWORD),
                ('th32ProcessID', wintypes.DWORD),
                ('GlsblcntUsage', wintypes.DWORD),
                ('ProccntUsage', wintypes.DWORD),
                ('modBaseAddr', ctypes.POINTER(wintypes.BYTE)),
                ('modBaseSize', wintypes.DWORD),
                ('hModule', wintypes.HMODULE),
                ('szModule', ctypes.c_char * 256),
                ('szExePath', ctypes.c_char * 260),
            ]
        
        TH32CS_SNAPMODULE = 0x00000008
        TH32CS_SNAPMODULE32 = 0x00000010
        
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        
        kernel32.Module32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(MODULEENTRY32)]
        kernel32.Module32First.restype = wintypes.BOOL
        
        kernel32.Module32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(MODULEENTRY32)]
        kernel32.Module32Next.restype = wintypes.BOOL
        
        # 创建快照
        hSnapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
        if hSnapshot == -1 or hSnapshot == 0xFFFFFFFF:
            logger.error(f"Failed to create snapshot: {ctypes.get_last_error()}")
            kernel32.VirtualFreeEx(hProcess, pDllPath, 0, MEM_RELEASE)
            return False
        
        kernel32_base = None
        me32 = MODULEENTRY32()
        me32.dwSize = ctypes.sizeof(MODULEENTRY32)
        
        if kernel32.Module32First(hSnapshot, ctypes.byref(me32)):
            while True:
                module_name = me32.szModule.decode('utf-8', errors='ignore').lower()
                if module_name == 'kernel32.dll':
                    kernel32_base = ctypes.cast(me32.modBaseAddr, ctypes.c_void_p).value
                    logger.info(f"Found kernel32.dll at: 0x{kernel32_base:08X}")
                    break
                if not kernel32.Module32Next(hSnapshot, ctypes.byref(me32)):
                    break
        
        kernel32.CloseHandle(hSnapshot)
        
        if kernel32_base is None:
            logger.error("Could not find kernel32.dll in target process")
            kernel32.VirtualFreeEx(hProcess, pDllPath, 0, MEM_RELEASE)
            return False
        
        # 读取 kernel32.dll 的导出表找到 LoadLibraryA
        # 简化方法：LoadLibraryA 在 kernel32 中的偏移通常是固定的
        # 我们从本机的 32 位 kernel32.dll 获取偏移
        
        # 更简单的方法：使用已知的相对地址
        # 在 Windows 10 上，LoadLibraryA 通常在 kernel32 基址 + 某个偏移
        # 但这个偏移会随版本变化
        
        # 最可靠的方法：直接用目标进程中 kernel32.dll 的实际地址
        # 由于我们已经有 kernel32 的基址，我们需要解析 PE 来找到 LoadLibraryA
        
        # 读取 DOS header
        dos_header = (ctypes.c_ubyte * 64)()
        bytes_read = ctypes.c_size_t()
        kernel32.ReadProcessMemory.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.LPVOID, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        kernel32.ReadProcessMemory.restype = wintypes.BOOL
        
        if not kernel32.ReadProcessMemory(hProcess, kernel32_base, dos_header, 64, ctypes.byref(bytes_read)):
            logger.error(f"Failed to read DOS header: {ctypes.get_last_error()}")
            kernel32.VirtualFreeEx(hProcess, pDllPath, 0, MEM_RELEASE)
            return False
        
        # e_lfanew at offset 0x3C
        e_lfanew = int.from_bytes(bytes(dos_header[0x3C:0x40]), 'little')
        
        # 读取 PE header
        pe_header = (ctypes.c_ubyte * 256)()
        if not kernel32.ReadProcessMemory(hProcess, kernel32_base + e_lfanew, pe_header, 256, ctypes.byref(bytes_read)):
            logger.error(f"Failed to read PE header: {ctypes.get_last_error()}")
            kernel32.VirtualFreeEx(hProcess, pDllPath, 0, MEM_RELEASE)
            return False
        
        # Export directory RVA at PE+0x78 (for 32-bit)
        export_dir_rva = int.from_bytes(bytes(pe_header[0x78:0x7C]), 'little')
        
        # 读取导出目录
        export_dir = (ctypes.c_ubyte * 40)()
        if not kernel32.ReadProcessMemory(hProcess, kernel32_base + export_dir_rva, export_dir, 40, ctypes.byref(bytes_read)):
            logger.error(f"Failed to read export directory: {ctypes.get_last_error()}")
            kernel32.VirtualFreeEx(hProcess, pDllPath, 0, MEM_RELEASE)
            return False
        
        num_functions = int.from_bytes(bytes(export_dir[0x14:0x18]), 'little')
        num_names = int.from_bytes(bytes(export_dir[0x18:0x1C]), 'little')
        addr_of_functions = int.from_bytes(bytes(export_dir[0x1C:0x20]), 'little')
        addr_of_names = int.from_bytes(bytes(export_dir[0x20:0x24]), 'little')
        addr_of_ordinals = int.from_bytes(bytes(export_dir[0x24:0x28]), 'little')
        
        # 读取名称表
        names_table = (ctypes.c_ubyte * (num_names * 4))()
        if not kernel32.ReadProcessMemory(hProcess, kernel32_base + addr_of_names, names_table, num_names * 4, ctypes.byref(bytes_read)):
            logger.error(f"Failed to read names table: {ctypes.get_last_error()}")
            kernel32.VirtualFreeEx(hProcess, pDllPath, 0, MEM_RELEASE)
            return False
        
        # 读取序号表
        ordinals_table = (ctypes.c_ubyte * (num_names * 2))()
        if not kernel32.ReadProcessMemory(hProcess, kernel32_base + addr_of_ordinals, ordinals_table, num_names * 2, ctypes.byref(bytes_read)):
            logger.error(f"Failed to read ordinals table: {ctypes.get_last_error()}")
            kernel32.VirtualFreeEx(hProcess, pDllPath, 0, MEM_RELEASE)
            return False
        
        # 读取函数地址表
        functions_table = (ctypes.c_ubyte * (num_functions * 4))()
        if not kernel32.ReadProcessMemory(hProcess, kernel32_base + addr_of_functions, functions_table, num_functions * 4, ctypes.byref(bytes_read)):
            logger.error(f"Failed to read functions table: {ctypes.get_last_error()}")
            kernel32.VirtualFreeEx(hProcess, pDllPath, 0, MEM_RELEASE)
            return False
        
        # 查找 LoadLibraryA
        pLoadLibraryA = None
        for i in range(num_names):
            name_rva = int.from_bytes(bytes(names_table[i*4:(i+1)*4]), 'little')
            
            # 读取函数名
            name_buf = (ctypes.c_ubyte * 32)()
            if kernel32.ReadProcessMemory(hProcess, kernel32_base + name_rva, name_buf, 32, ctypes.byref(bytes_read)):
                name = bytes(name_buf).split(b'\x00')[0].decode('ascii', errors='ignore')
                if name == 'LoadLibraryA':
                    ordinal = int.from_bytes(bytes(ordinals_table[i*2:(i+1)*2]), 'little')
                    func_rva = int.from_bytes(bytes(functions_table[ordinal*4:(ordinal+1)*4]), 'little')
                    pLoadLibraryA = kernel32_base + func_rva
                    logger.info(f"Found LoadLibraryA at: 0x{pLoadLibraryA:08X}")
                    break
        
        if pLoadLibraryA is None:
            logger.error("Could not find LoadLibraryA in kernel32.dll")
            kernel32.VirtualFreeEx(hProcess, pDllPath, 0, MEM_RELEASE)
            return False
        
        # 创建远程线程
        thread_id = wintypes.DWORD()
        hThread = kernel32.CreateRemoteThread(
            hProcess, None, 0, pLoadLibraryA, pDllPath, 0, ctypes.byref(thread_id)
        )
        
        if not hThread:
            logger.error(f"Failed to create remote thread: {ctypes.get_last_error()}")
            kernel32.VirtualFreeEx(hProcess, pDllPath, 0, MEM_RELEASE)
            return False
        
        logger.info(f"Created remote thread: {thread_id.value}")
        
        # 等待线程完成
        wait_result = kernel32.WaitForSingleObject(hThread, 10000)  # 10秒超时
        
        # 获取线程退出码（LoadLibrary 返回的模块句柄）
        exit_code = wintypes.DWORD()
        kernel32.GetExitCodeThread(hThread, ctypes.byref(exit_code))
        
        logger.info(f"Thread exit code (module handle): 0x{exit_code.value:08X}")
        
        if exit_code.value == 0:
            logger.error("LoadLibraryA returned NULL - DLL failed to load")
            logger.error("Possible reasons:")
            logger.error("  - DLL has missing dependencies")
            logger.error("  - DLL path is incorrect")
            logger.error("  - DLL architecture mismatch")
            kernel32.CloseHandle(hThread)
            kernel32.VirtualFreeEx(hProcess, pDllPath, 0, MEM_RELEASE)
            return False
        
        # 清理
        kernel32.CloseHandle(hThread)
        kernel32.VirtualFreeEx(hProcess, pDllPath, 0, MEM_RELEASE)
        
        logger.info("DLL injected successfully!")
        logger.info("Hook DLL should be listening on port 12345")
        return True
        
    finally:
        kernel32.CloseHandle(hProcess)


if __name__ == '__main__':
    # 测试注入
    import time
    
    print("PVZ Hook DLL Injector")
    print("=" * 50)
    
    if inject_dll():
        print("\nWaiting for hook to initialize...")
        time.sleep(1)
        
        # 测试连接
        from .client import HookClient
        client = HookClient()
        if client.connect():
            print("Successfully connected to Hook DLL!")
            state = client.get_state()
            if state:
                print(f"Game state: {state}")
            client.disconnect()
        else:
            print("Failed to connect to Hook DLL")
            print("The DLL may not have loaded correctly")
    else:
        print("\nInjection failed!")
