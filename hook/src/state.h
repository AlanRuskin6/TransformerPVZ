#pragma once

#include <string>

// 游戏状态序列化
// 收集完整的游戏状态并序列化为JSON格式

namespace State {

// 获取完整游戏状态（JSON格式）
std::string GetGameState();

// 游戏状态结构（内部使用）
struct GameStateInfo {
    int sun;
    int wave;
    int total_waves;
    int scene;
    int game_clock;
    bool in_game;
    int zombie_count;
    int plant_count;
};

}  // namespace State
