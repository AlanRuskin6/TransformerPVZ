# Hook Client

Python客户端模块，用于与PVZ Hook DLL通信。

## 安装

```bash
pip install psutil
```

## 快速开始

### 1. 编译Hook DLL

```bash
cd hook
build.bat
```

### 2. 启动游戏并注入DLL

```python
from hook_client import inject_dll

# 自动查找PVZ进程并注入
if inject_dll():
    print("DLL注入成功！")
```

### 3. 连接并使用

```python
from hook_client import HookClient

# 创建客户端
client = HookClient()

# 连接到Hook DLL
if client.connect():
    # 获取游戏状态
    state = client.get_state()
    print(f"阳光: {state['sun']}, 波数: {state['wave']}")
    
    # 种植物
    client.plant(row=0, col=0, plant_type=0)  # 豌豆射手
    
    # 铲植物
    client.shovel(row=0, col=0)
    
    # 重置关卡
    client.reset()
    
    # 断开连接
    client.disconnect()
```

## API参考

### HookClient类

#### 连接管理

```python
client = HookClient(host='127.0.0.1', port=12345, timeout=5.0)
client.connect() -> bool
client.disconnect()
```

#### 游戏操作

```python
# 种植物
client.plant(row, col, plant_type) -> bool

# 铲植物
client.shovel(row, col) -> bool

# 发射玉米炮
client.fire_cob(x, y) -> bool

# 重置关卡
client.reset() -> bool

# 进入游戏模式
client.enter_game(mode) -> bool  # mode=13 为泳池无尽

# 选卡
client.choose_card(plant_type) -> bool

# 开始游戏
client.rock() -> bool

# 返回主菜单
client.back_to_main() -> bool

# 获取游戏状态
client.get_state() -> dict
```

#### 状态字段

`get_state()` 返回的字典包含：

```python
{
    "sun": 150,              # 阳光数量
    "wave": 1,               # 当前波数
    "total_waves": 20,       # 总波数
    "scene": 2,              # 场景类型
    "game_clock": 1234,      # 游戏时钟
    "in_game": True,         # 是否在游戏中
    "zombie_count": 5,       # 僵尸数量
    "plant_count": 10        # 植物数量
}
```

### 辅助函数

```python
from hook_client import find_pvz_process, inject_dll

# 查找PVZ进程
pid = find_pvz_process()  # -> int or None

# 注入DLL
inject_dll(dll_path=None, pid=None)  # -> bool
```

## Context Manager支持

```python
with HookClient() as client:
    state = client.get_state()
    client.plant(0, 0, 0)
    # 自动断开连接
```

## 完整示例

### 自动化训练循环

```python
from hook_client import HookClient, inject_dll
import time

# 注入DLL
if not inject_dll():
    print("注入失败")
    exit(1)

# 等待初始化
time.sleep(2)

# 连接
with HookClient() as client:
    # 主循环
    while True:
        state = client.get_state()
        
        if not state['in_game']:
            # 不在游戏中，等待
            time.sleep(1)
            continue
        
        # 检查阳光
        if state['sun'] >= 50:
            # 种豌豆射手
            client.plant(0, 0, 0)
        
        # 检查失败
        if state['zombie_count'] > 20:
            # 重置关卡
            client.reset()
        
        time.sleep(0.1)
```

### 自动选卡并开始

```python
from hook_client import HookClient

with HookClient() as client:
    # 选择卡片
    cards = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 植物类型
    for card in cards:
        client.choose_card(card)
        time.sleep(0.1)
    
    # 开始游戏
    client.rock()
```

## 故障排除

### 连接失败

- 确认Hook DLL已注入
- 检查端口12345是否被占用
- 使用 `netstat -an | findstr 12345` 查看端口状态

### DLL注入失败

- 确认游戏已启动
- 确认DLL已编译（hook/pvz_hook.dll存在）
- 以管理员权限运行Python脚本
- 确认游戏版本为1.0.0.1051

### 命令执行失败

- 检查游戏状态（是否在正确的界面）
- 查看返回的错误消息
- 确认参数是否正确

## 优势

相比传统的shellcode注入方式，Hook DLL方式有以下优势：

1. **零崩溃** - 所有操作在游戏主线程执行，无时机问题
2. **稳定可靠** - 不会因为界面切换而崩溃
3. **性能更好** - 减少了进程间通信开销
4. **更易调试** - DLL可以输出日志，便于排查问题

## 相关文档

- [Hook DLL文档](../hook/README.md)
- [通信协议](protocol.py)
- [DLL注入器](injector.py)
