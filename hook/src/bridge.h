#pragma once

#include <string>

// Python通信桥接
// TCP服务器，端口12345

namespace Bridge {

// 初始化TCP服务器
bool Initialize(int port = 12345);

// 关闭服务器
void Shutdown();

// 处理一帧的命令（非阻塞）
void ProcessCommands();

// 命令处理结果
struct CommandResult {
    bool success;
    std::string message;
};

}  // namespace Bridge
