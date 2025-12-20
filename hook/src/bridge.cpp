#include "bridge.h"
#include "game.h"
#include "state.h"
#include <WinSock2.h>
#include <sstream>
#include <vector>
#include <string>

#pragma comment(lib, "ws2_32.lib")

namespace Bridge {

static SOCKET g_serverSocket = INVALID_SOCKET;
static SOCKET g_clientSocket = INVALID_SOCKET;
static bool g_initialized = false;
static std::string g_commandBuffer;

bool Initialize(int port) {
    if (g_initialized) return true;
    
    // 初始化Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return false;
    }
    
    // 创建服务器socket
    g_serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (g_serverSocket == INVALID_SOCKET) {
        WSACleanup();
        return false;
    }
    
    // 设置为非阻塞模式
    u_long mode = 1;
    ioctlsocket(g_serverSocket, FIONBIO, &mode);
    
    // 绑定端口
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(port);
    
    if (bind(g_serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        closesocket(g_serverSocket);
        WSACleanup();
        return false;
    }
    
    // 监听 (增加队列长度防止 10061 错误)
    if (listen(g_serverSocket, 5) == SOCKET_ERROR) {
        closesocket(g_serverSocket);
        WSACleanup();
        return false;
    }
    
    g_initialized = true;
    return true;
}

void Shutdown() {
    if (g_clientSocket != INVALID_SOCKET) {
        closesocket(g_clientSocket);
        g_clientSocket = INVALID_SOCKET;
    }
    if (g_serverSocket != INVALID_SOCKET) {
        closesocket(g_serverSocket);
        g_serverSocket = INVALID_SOCKET;
    }
    if (g_initialized) {
        WSACleanup();
        g_initialized = false;
    }
}

static std::string ProcessCommand(const std::string& cmd) {
    std::istringstream iss(cmd);
    std::string command;
    iss >> command;
    
    if (command == "PLANT") {
        // 直接放置植物 (不扣阳光) - 用于测试/作弊模式
        if (PVZ::IsGamePaused()) {
            return "ERR Game is paused\n";
        }
        int row, col, type;
        if (iss >> row >> col >> type) {
            if (PVZ::PutPlant(row, col, type)) {
                return "OK\n";
            }
        }
        return "ERR Invalid parameters\n";
    }
    else if (command == "PLANT_CARD") {
        // 使用卡槽种植 (扣阳光) - 正常游戏方式
        // 参数: row col cardIndex (cardIndex 是卡槽索引 0-9)
        if (PVZ::IsGamePaused()) {
            return "ERR Game is paused\n";
        }
        int row, col, cardIndex;
        if (iss >> row >> col >> cardIndex) {
            // 转换行列到像素坐标
            int x = PVZ::GridToAbscissa(row, col);
            int y = PVZ::GridToOrdinate(row, col);
            // PlantCard 使用像素坐标和卡槽索引
            if (PVZ::PlantCard(x, y, cardIndex)) {
                return "OK\n";
            }
        }
        return "ERR Invalid parameters (need row col cardIndex)\n";
    }
    else if (command == "CLICK") {
        // 模拟鼠标左键点击 (游戏坐标)
        // 用于收集物品等
        int x, y;
        if (iss >> x >> y) {
            PVZ::LeftClick(x, y);
            return "OK\n";
        }
        return "ERR Invalid parameters (need x y)\n";
    }
    else if (command == "COLLECT") {
        // 收集所有物品(阳光/金币) - 基于AVZ的AItemCollector
        int count = PVZ::CollectAllItems();
        return "OK " + std::to_string(count) + "\n";
    }
    else if (command == "SHOVEL") {
        // 检查游戏是否暂停
        if (PVZ::IsGamePaused()) {
            return "ERR Game is paused\n";
        }
        int row, col;
        if (iss >> row >> col) {
            if (PVZ::Shovel(row, col)) {
                return "OK\n";
            }
        }
        return "ERR Invalid parameters\n";
    }
    else if (command == "FIRE") {
        // 检查游戏是否暂停
        if (PVZ::IsGamePaused()) {
            return "ERR Game is paused\n";
        }
        int x, y, rank;
        if (iss >> x >> y >> rank) {
            if (PVZ::FireCob(x, y, rank)) {
                return "OK\n";
            }
        }
        return "ERR Invalid parameters (need x y rank)\n";
    }
    else if (command == "RESET") {
        if (PVZ::MakeNewBoard()) {
            return "OK\n";
        }
        return "ERR Failed to reset\n";
    }
    else if (command == "CHOOSE") {
        int type;
        if (iss >> type) {
            // 如果 type >= 48，说明是模仿者卡片 (48 + 原植物类型)
            if (type >= 48) {
                if (PVZ::ChooseImitatorCard(type - 48)) {
                    return "OK\n";
                }
            } else {
                if (PVZ::ChooseCard(type)) {
                    return "OK\n";
                }
            }
        }
        return "ERR Invalid parameters\n";
    }
    else if (command == "CARDS") {
        // 批量选卡: CARDS 0 1 2 3 4 5 6 7 8 9
        std::vector<int> cards;
        int type;
        while (iss >> type) {
            cards.push_back(type);
        }
        if (cards.empty()) {
            return "ERR Need card types\n";
        }
        
        // 检查是否有卡片已经被选中但不在我们的列表中
        // 如果上一轮选了不同的卡，需要先清空
        bool needClear = false;
        for (int t = 0; t < 48; t++) {
            int state = PVZ::GetCardMoveState(t);
            if (state == 0 || state == 1) {
                // 这个卡在卡槽中，检查是否是我们想要的
                bool wanted = false;
                for (int wantedCard : cards) {
                    if (wantedCard == t || (wantedCard >= 48 && t == 48)) {
                        wanted = true;
                        break;
                    }
                }
                if (!wanted) {
                    needClear = true;
                    break;
                }
            }
        }
        
        // 如果需要清空，点击取消已选卡片
        if (needClear) {
            // 多次点击清空所有已选卡
            for (int i = 0; i < 15; i++) {
                PVZ::ClearSelectedCards();
                Sleep(80);
            }
        }
        
        // 现在选择我们想要的卡
        for (int t : cards) {
            if (t >= 48) {
                PVZ::ChooseImitatorCard(t - 48);
            } else {
                PVZ::ChooseCard(t);
            }
            Sleep(80);  // 选卡间隔
        }
        return "OK\n";
    }
    else if (command == "ROCK") {
        if (PVZ::Rock()) {
            return "OK\n";
        }
        return "ERR Failed to start\n";
    }
    else if (command == "FLUSH") {
        // 清空命令 - 用于状态转换后确保没有残留命令
        // 这个命令本身不做任何事，只是返回 OK
        return "OK\n";
    }
    else if (command == "BACK") {
        if (PVZ::BackToMain()) {
            return "OK\n";
        }
        return "ERR Failed to back (only works from game UI=3)\n";
    }
    else if (command == "CLOSEOPTS") {
        // Close options/challenge screen (UI=7)
        if (PVZ::CloseOptionsScreen()) {
            return "OK\n";
        }
        return "ERR Failed to close options (only works from UI=7)\n";
    }
    else if (command == "STATE") {
        std::string state = State::GetGameState();
        return state + "\n";
    }
    else if (command == "LEVEL") {
        int level;
        if (iss >> level) {
            // Set level
            if (PVZ::SetLevel(level)) {
                return "OK\n";
            }
            return "ERR Failed to set level\n";
        } else {
            // Get level
            int currentLevel = PVZ::GetLevel();
            return "OK " + std::to_string(currentLevel) + "\n";
        }
    }
    else if (command == "UNLOCK") {
        if (PVZ::UnlockAllPlants()) {
            return "OK\n";
        }
        return "ERR Failed to unlock\n";
    }
    else if (command == "MODE") {
        int mode;
        if (iss >> mode) {
            // Set mode
            if (PVZ::SetGameMode(mode)) {
                return "OK\n";
            }
            return "ERR Failed to set mode\n";
        } else {
            // Get mode
            int currentMode = PVZ::GetGameMode();
            return "OK " + std::to_string(currentMode) + "\n";
        }
    }
    else if (command == "UI") {
        // Get current GameUI state
        // 0=loading, 1=main menu, 2=card selection, 3=playing, 7=options
        int ui = PVZ::GetGameUI();
        return "OK " + std::to_string(ui) + "\n";
    }
    else if (command == "WINSIZE") {
        // Get window size for coordinate scaling
        int width = PVZ::GetWindowWidth();
        int height = PVZ::GetWindowHeight();
        return "OK " + std::to_string(width) + " " + std::to_string(height) + "\n";
    }
    else if (command == "PAUSED") {
        // Check if game is paused (including menu dialogs)
        bool paused = PVZ::IsGamePaused();
        return paused ? "OK 1\n" : "OK 0\n";
    }
    else if (command == "READY") {
        // Check if card selection screen is ready (animation complete)
        bool ready = PVZ::IsCardSelectReady();
        return ready ? "OK 1\n" : "OK 0\n";
    }
    else if (command == "DEBUG_OFFSET") {
        // Debug: Get OrizontalScreenOffset value
        if (PVZ::GetGameUI() != 2) {
            return "ERR not in card select UI\n";
        }
        uintptr_t board = PVZ::GetBoard();
        if (!board) return "ERR no board\n";
        uintptr_t selectCardUi = *(uintptr_t*)(board + 0x15c);
        if (!selectCardUi) return "ERR no selectCardUi\n";
        int offset = *(int*)(selectCardUi + 0x8);
        return "OK " + std::to_string(offset) + "\n";
    }
    else if (command == "START") {
        int mode;
        if (iss >> mode) {
            if (PVZ::StartGame(mode)) {
                return "OK\n";
            }
            return "ERR Failed to start game\n";
        }
        return "ERR Need mode parameter\n";
    }
    else if (command == "SPEED") {
        // 设置/获取游戏速度
        // SPEED -> 返回当前速度
        // SPEED 5.0 -> 设置5倍速
        float speed;
        if (iss >> speed) {
            if (PVZ::SetGameSpeed(speed)) {
                return "OK " + std::to_string(PVZ::GetGameSpeed()) + "\n";
            }
            return "ERR Invalid speed (range: 0.05-10)\n";
        } else {
            return "OK " + std::to_string(PVZ::GetGameSpeed()) + "\n";
        }
    }
    else if (command == "TICKMS") {
        // 设置/获取每帧毫秒数 (更底层的速度控制)
        // TICKMS -> 返回当前值
        // TICKMS 2 -> 设置为2ms (5倍速)
        int ms;
        if (iss >> ms) {
            if (PVZ::SetTickMs(ms)) {
                return "OK " + std::to_string(PVZ::GetTickMs()) + "\n";
            }
            return "ERR Invalid tickMs (range: 1-200)\n";
        } else {
            return "OK " + std::to_string(PVZ::GetTickMs()) + "\n";
        }
    }
    else if (command == "FRAMEMULT") {
        // 设置/获取帧倍增数 (超高速模式)
        // FRAMEMULT -> 返回当前值
        // FRAMEMULT 5 -> 设置为5倍帧 (配合tick_ms=1，可达50x速度)
        int mult;
        if (iss >> mult) {
            if (PVZ::SetFrameMultiplier(mult)) {
                return "OK " + std::to_string(PVZ::GetFrameMultiplier()) + "\n";
            }
            return "ERR Invalid multiplier (range: 1-100)\n";
        } else {
            return "OK " + std::to_string(PVZ::GetFrameMultiplier()) + "\n";
        }
    }
    
    return "ERR Unknown command\n";
}

void ProcessCommands() {
    if (!g_initialized) return;
    
    // 检查是否有新连接 (处理重连)
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(g_serverSocket, &readfds);
    
    timeval timeout = {0, 0}; // 非阻塞
    if (select(0, &readfds, nullptr, nullptr, &timeout) > 0) {
        // 有新连接等待
        if (g_clientSocket != INVALID_SOCKET) {
            closesocket(g_clientSocket); // 关闭旧连接
            g_clientSocket = INVALID_SOCKET;
        }
        
        g_clientSocket = accept(g_serverSocket, nullptr, nullptr);
        if (g_clientSocket != INVALID_SOCKET) {
            // 设置为非阻塞模式
            u_long mode = 1;
            ioctlsocket(g_clientSocket, FIONBIO, &mode);
            g_commandBuffer.clear(); // 清空缓冲区
        }
    }
    
    // 处理客户端命令
    if (g_clientSocket != INVALID_SOCKET) {
        constexpr int BUFFER_SIZE = 4096;
        char buffer[BUFFER_SIZE];
        int bytesRead = recv(g_clientSocket, buffer, BUFFER_SIZE - 1, 0);
        
        if (bytesRead > 0) {
            // Ensure null termination
            buffer[bytesRead] = '\0';
            
            // 追加到缓冲区
            g_commandBuffer.append(buffer);
            
            // 处理缓冲区中的完整命令
            size_t pos = 0;
            while ((pos = g_commandBuffer.find('\n')) != std::string::npos) {
                std::string cmd = g_commandBuffer.substr(0, pos);
                g_commandBuffer.erase(0, pos + 1);
                
                if (!cmd.empty() && cmd[cmd.length()-1] == '\r') {
                    cmd.erase(cmd.length()-1);
                }
                
                if (!cmd.empty()) {
                    std::string response = ProcessCommand(cmd);
                    send(g_clientSocket, response.c_str(), response.length(), 0);
                }
            }
        }
        else if (bytesRead == 0 || WSAGetLastError() != WSAEWOULDBLOCK) {
            // 客户端断开连接
            closesocket(g_clientSocket);
            g_clientSocket = INVALID_SOCKET;
            g_commandBuffer.clear(); // 清空缓冲区
        }
    }
}

}  // namespace Bridge
