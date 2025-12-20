# PVZ Hook DLL

这个DLL通过Hook游戏主循环，在游戏进程内部执行所有写内存操作，彻底消除崩溃风险。

## 编译步骤

### 前置要求
- Visual Studio 2022 (或更高版本)
- CMake 3.15+
- Windows SDK

### 编译命令

1. 使用提供的批处理脚本（推荐）：
```bat
build.bat
```

2. 或手动编译：
```bat
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A Win32 ..
cmake --build . --config Release
copy Release\pvz_hook.dll ..
```

编译成功后，`pvz_hook.dll` 会出现在 `hook/` 目录下。

## 注入方法

### 使用Python自动注入（推荐）
```python
from hook_client import HookClient, inject_dll

# 自动查找PVZ进程并注入
if inject_dll():
    # 连接到Hook DLL
    client = HookClient()
    
    # 发送命令
    client.plant(0, 0, 0)  # 种豌豆射手在0行0列
    client.reset()          # 重置关卡
```

### 手动注入
使用任何DLL注入器（如 `Extreme Injector`, `Xenos Injector` 等）：
1. 启动PVZ游戏
2. 注入 `pvz_hook.dll`
3. DLL会自动在端口12345启动TCP服务器

## 通信协议

Hook DLL在端口12345监听TCP连接，使用文本协议。

### 命令格式
每个命令一行，以`\n`结尾：

```
PLANT row col type\n    # 种植物
SHOVEL row col\n        # 铲植物
FIRE x y\n              # 发射玉米炮
RESET\n                 # 重置关卡
ENTER mode\n            # 进入游戏模式
CHOOSE type\n           # 选卡
ROCK\n                  # 开始游戏
BACK\n                  # 返回主菜单
STATE\n                 # 获取游戏状态
```

### 响应格式
- 成功：`OK\n`
- 失败：`ERR message\n`
- 状态：`{json}\n`

### 示例
```
C: PLANT 0 0 0\n
S: OK\n

C: STATE\n
S: {"sun":150,"wave":1,"total_waves":20,"scene":2,"game_clock":0,"in_game":true,"zombie_count":5,"plant_count":10}\n

C: RESET\n
S: OK\n
```

## Python客户端示例

```python
from hook_client import HookClient

client = HookClient()

# 种植物
client.plant(row=0, col=0, plant_type=0)

# 铲植物
client.shovel(row=0, col=0)

# 重置关卡
client.reset()

# 获取游戏状态
state = client.get_state()
print(f"阳光: {state['sun']}, 波数: {state['wave']}")
```

## 架构说明

### Hook机制
DLL注入后会Hook游戏主循环（地址0x452650），在每帧执行时：
1. 处理Python的TCP命令
2. 调用原始游戏逻辑

这样所有操作都在游戏主线程安全执行，不会有时机问题。

### 核心模块
- `dllmain.cpp` - DLL入口，Hook安装
- `game.cpp` - 游戏函数封装（种植、铲除等）
- `bridge.cpp` - TCP服务器，命令处理
- `state.cpp` - 状态序列化

### 游戏函数地址（基于AVZ）
- PutPlant: 0x40D120
- Shovel: 0x411060
- ChooseCard: 0x486030
- Rock: 0x486D20
- MakeNewBoard: 0x44F5F0
- EnterGame: 0x44F560
- BackToMain: 0x44FEB0

## 故障排除

### DLL无法加载
- 确保是32位DLL（PVZ是32位游戏）
- 检查是否缺少依赖（ws2_32.dll）

### 连接失败
- 确认DLL已成功注入
- 检查防火墙是否阻止端口12345
- 使用`netstat -an | find "12345"`查看端口是否监听

### 游戏崩溃
- 确认游戏版本是1.0.0.1051（中文年度版）
- Hook地址可能不匹配，需要调整

## 相关资源
- [AsmVsZombies](https://github.com/vector-wlc/AsmVsZombies) - 参考实现
- PVZ 1.0.0.1051 - 目标游戏版本
