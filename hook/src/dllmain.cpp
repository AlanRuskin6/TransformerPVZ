#include <Windows.h>
#include <cstdlib>
#include <cstdio>
#include "bridge.h"
#include "game.h"

// 默认端口，优先从配置文件读取，其次环境变量
static int GetHookPort() {
    // 1. 先尝试从游戏目录下的 hook_port.txt 读取
    //    这是最可靠的多实例端口配置方式
    FILE* fp = fopen("hook_port.txt", "r");
    if (fp) {
        int port = 0;
        if (fscanf(fp, "%d", &port) == 1) {
            fclose(fp);
            if (port >= 1024 && port <= 65535) {
                return port;
            }
        }
        fclose(fp);
    }
    
    // 2. 尝试环境变量 (注入后可能无效)
    char* portStr = getenv("PVZ_HOOK_PORT");
    if (portStr) {
        int port = atoi(portStr);
        if (port >= 1024 && port <= 65535) {
            return port;
        }
    }
    return 12345;  // 默认端口
}

// Hook相关 - 使用虚函数表替换方式 (参考 AsmVsZombies)
// 地址 0x667bc0 存储了游戏主循环虚函数指针，原值为 0x452650
static bool g_hooked = false;
static bool g_bridgeInitialized = false;
static constexpr uintptr_t VTABLE_HOOK_ADDR = 0x667bc0;  // 虚函数表地址
static constexpr uintptr_t ORIGINAL_FUNC = 0x452650;     // 原始函数地址

// 内部处理函数 - 在这里做我们的工作
static void DoProcessing() {
    __try {
        // 安全检查：确保游戏基础对象有效
        uintptr_t base = PVZ::GetBase();
        if (!base) return;  // 游戏尚未完全初始化

        // 始终处理 TCP 命令，即使游戏状态不稳定
        // 这样可以避免 Python 客户端超时
        if (g_bridgeInitialized) {
            Bridge::ProcessCommands();
        }

        // 只有在游戏状态稳定时才处理 pending actions
        int gameUI = (int)PVZ::GetGameUI();
        if (gameUI == 3) {  // 在游戏中
            uintptr_t board = PVZ::GetBoard();
            if (!board) return;  // board 正在重建中，跳过 pending actions
        }

        // 只有游戏没暂停时才处理 pending actions
        if (!PVZ::IsGamePaused()) {
            PVZ::ProcessPendingActions();
        }
    }
    __except(EXCEPTION_EXECUTE_HANDLER) {
        // 捕获任何异常，防止游戏崩溃
    }
}

// 导出的Hook函数 - 会被游戏通过虚函数表调用
// 参考 AVZ: __AScriptHook 调用 ScriptHook，ScriptHook 调用 GameTotalLoop
extern "C" __declspec(dllexport) void __cdecl ScriptHook() {
    // 获取帧倍增数
    int mult = PVZ::GetFrameMultiplier();
    
    // 执行 mult 次游戏逻辑
    for (int i = 0; i < mult; i++) {
        // 每帧开始时处理命令
        DoProcessing();
        
        // 执行游戏逻辑
        PVZ::RunGameFrame();
    }
}

bool InstallHook() {
    if (g_hooked) return true;
    
    DWORD oldProtect;
    // 解锁整个代码段的内存保护 (参考 AVZ)
    VirtualProtect((void*)0x400000, 0x35E000, PAGE_EXECUTE_READWRITE, &oldProtect);
    
    // 将虚函数表中的指针替换为我们的Hook函数
    *(uint32_t*)VTABLE_HOOK_ADDR = (uint32_t)&ScriptHook;
    
    g_hooked = true;
    return true;
}

void UninstallHook() {
    if (!g_hooked) return;
    
    DWORD oldProtect;
    VirtualProtect((void*)0x400000, 0x35E000, PAGE_EXECUTE_READWRITE, &oldProtect);
    
    // 恢复虚函数表中的原始指针
    *(uint32_t*)VTABLE_HOOK_ADDR = ORIGINAL_FUNC;
    
    g_hooked = false;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD dwReason, LPVOID lpReserved) {
    switch (dwReason) {
    case DLL_PROCESS_ATTACH:
        // DLL被加载
        DisableThreadLibraryCalls(hModule);
        
        // 先安装Hook（这样游戏就不会卡住）
        InstallHook();
        
        // 然后初始化TCP服务器（端口从环境变量获取）
        g_bridgeInitialized = Bridge::Initialize(GetHookPort());
        break;
        
    case DLL_PROCESS_DETACH:
        // DLL被卸载
        UninstallHook();
        Bridge::Shutdown();
        break;
    }
    return TRUE;
}
