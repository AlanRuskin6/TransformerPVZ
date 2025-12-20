"""
Hook Client
TCP客户端，连接到Hook DLL
"""

import socket
import json
import logging
import time
from typing import Optional, Dict
from .protocol import Command, Response

# 安全延迟（秒）- 在发送命令前等待
COMMAND_DELAY = 0.005  # 5ms，极速模式

# Setup logger
logger = logging.getLogger(__name__)


class HookClient:
    """Hook DLL客户端"""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 12345, timeout: float = 5.0):
        """
        初始化客户端
        
        Args:
            host: 服务器地址
            port: 服务器端口
            timeout: 连接超时时间（秒）
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> bool:
        """
        连接到Hook DLL
        
        Returns:
            True if successful
        """
        if self.connected:
            return True
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            return True
        except socket.timeout as e:
            self.logger.error(f"Connection timeout: {e}")
            self.socket = None
            self.connected = False
            return False
        except socket.error as e:
            self.logger.error(f"Connection failed: {e}")
            self.socket = None
            self.connected = False
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during connection: {e}")
            self.socket = None
            self.connected = False
            return False
    
    def disconnect(self):
        """断开连接"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        self.connected = False
    
    def _send_command(self, command: str) -> Optional[str]:
        """
        发送命令并接收响应
        
        Args:
            command: 命令字符串
            
        Returns:
            响应字符串，失败返回None
        """
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            # 安全延迟 - 确保游戏有足够时间处理
            time.sleep(COMMAND_DELAY)
            
            # 发送命令
            self.socket.sendall((command + '\n').encode('utf-8'))
            
            # 接收响应
            response = b''
            while True:
                chunk = self.socket.recv(4096)
                if not chunk:
                    break
                response += chunk
                if b'\n' in response:
                    break
            
            return response.decode('utf-8').strip()
        except socket.timeout as e:
            self.logger.error(f"Command timeout: {e}")
            self.disconnect()
            return None
        except socket.error as e:
            self.logger.error(f"Socket error: {e}")
            self.disconnect()
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.disconnect()
            return None
    
    def plant(self, row: int, col: int, plant_type: int) -> bool:
        """
        直接放置植物（不扣阳光）- 用于测试/作弊模式
        
        Args:
            row: 行（0-5）
            col: 列（0-8）
            plant_type: 植物类型
            
        Returns:
            True if successful
        """
        cmd = f"{Command.PLANT} {row} {col} {plant_type}"
        response = self._send_command(cmd)
        return response and Response.is_success(response)
    
    def plant_card(self, row: int, col: int, card_index: int) -> bool:
        """
        使用卡槽种植（会扣阳光）- 正常游戏方式
        基于AVZ的PlantCard函数
        
        Args:
            row: 行（0-5）
            col: 列（0-8）
            card_index: 卡槽索引（0-9，对应卡槽位置）
            
        Returns:
            True if successful
        """
        cmd = f"PLANT_CARD {row} {col} {card_index}"
        response = self._send_command(cmd)
        return response and Response.is_success(response)
    
    def shovel(self, row: int, col: int) -> bool:
        """
        铲植物
        
        Args:
            row: 行（0-5）
            col: 列（0-8）
            
        Returns:
            True if successful
        """
        cmd = f"{Command.SHOVEL} {row} {col}"
        response = self._send_command(cmd)
        return response and Response.is_success(response)
    
    def fire_cob(self, x: int, y: int) -> bool:
        """
        发射玉米炮
        
        Args:
            x: 目标X坐标（像素）
            y: 目标Y坐标（像素）
            
        Returns:
            True if successful
        """
        cmd = f"{Command.FIRE} {x} {y}"
        response = self._send_command(cmd)
        return response and Response.is_success(response)
    
    def click(self, x: int, y: int) -> bool:
        """
        模拟鼠标左键点击（游戏坐标）
        用于收集物品等
        
        Args:
            x: X坐标（像素）
            y: Y坐标（像素）
            
        Returns:
            True if successful
        """
        cmd = f"CLICK {x} {y}"
        response = self._send_command(cmd)
        return response and Response.is_success(response)
    
    def get_window_size(self) -> tuple:
        """
        获取游戏窗口大小
        
        Returns:
            (width, height) 元组，失败返回 (800, 600) 默认值
        """
        response = self._send_command("WINSIZE")
        if response and Response.is_success(response):
            parts = response.split()
            if len(parts) >= 3:
                try:
                    return (int(parts[1]), int(parts[2]))
                except ValueError:
                    pass
        return (800, 600)
    
    def click_relative(self, rel_x: float, rel_y: float) -> bool:
        """
        使用相对坐标点击（0.0-1.0）
        自动根据窗口大小换算
        
        Args:
            rel_x: 相对X坐标 (0.0-1.0)
            rel_y: 相对Y坐标 (0.0-1.0)
            
        Returns:
            True if successful
        """
        width, height = self.get_window_size()
        x = int(rel_x * width)
        y = int(rel_y * height)
        return self.click(x, y)
    
    def click_scaled(self, x: int, y: int, base_width: int = 800, base_height: int = 600) -> bool:
        """
        使用基于800x600的坐标点击，自动缩放到实际窗口大小
        
        Args:
            x: 基于800x600的X坐标
            y: 基于800x600的Y坐标
            base_width: 基准宽度（默认800）
            base_height: 基准高度（默认600）
            
        Returns:
            True if successful
        """
        width, height = self.get_window_size()
        scaled_x = int(x * width / base_width)
        scaled_y = int(y * height / base_height)
        return self.click(scaled_x, scaled_y)
    
    def collect(self) -> int:
        """
        收集所有物品（阳光/金币）
        基于AVZ的AItemCollector，使用游戏内鼠标模拟
        
        Returns:
            收集的物品数量，失败返回-1
        """
        response = self._send_command("COLLECT")
        if response and Response.is_success(response):
            # Response format: "OK <count>"
            parts = response.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return 0
            return 0
        return -1
    
    def reset(self) -> bool:
        """
        重置关卡 (调用 MakeNewBoard)
        
        Returns:
            True if successful
        """
        response = self._send_command(Command.RESET)
        return response and Response.is_success(response)
    
    def reset_level(self) -> bool:
        """
        重置关卡 (reset 的别名)
        
        Returns:
            True if successful
        """
        return self.reset()
    
    def enter_game(self, mode: int) -> bool:
        """
        进入游戏模式
        
        Args:
            mode: 游戏模式（13=泳池无尽）
            
        Returns:
            True if successful
        """
        cmd = f"{Command.ENTER} {mode}"
        response = self._send_command(cmd)
        return response and Response.is_success(response)
    
    def choose_card(self, plant_type: int) -> bool:
        """
        选卡
        
        Args:
            plant_type: 植物类型 (0-47 普通植物, 48-95 模仿者植物)
                例如: 48 = 模仿豌豆射手 (48 + 0)
            
        Returns:
            True if successful
        """
        cmd = f"{Command.CHOOSE} {plant_type}"
        response = self._send_command(cmd)
        return response and Response.is_success(response)
    
    def select_cards(self, plant_types: list) -> bool:
        """
        批量选卡
        
        Args:
            plant_types: 植物类型列表，最多10张
                例如: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                模仿者: 48 + 原植物类型
            
        Returns:
            True if successful
        """
        if not plant_types:
            return False
        types_str = ' '.join(str(t) for t in plant_types)
        cmd = f"CARDS {types_str}"
        response = self._send_command(cmd)
        return response and Response.is_success(response)
    
    def rock(self) -> bool:
        """
        点击开始游戏按钮
        
        Returns:
            True if successful
        """
        response = self._send_command(Command.ROCK)
        return response and Response.is_success(response)
    
    def back_to_main(self) -> bool:
        """
        返回主菜单
        
        Returns:
            True if successful
        """
        response = self._send_command(Command.BACK)
        return response and Response.is_success(response)
    
    def get_state(self) -> Optional[Dict]:
        """
        获取游戏状态
        
        Returns:
            游戏状态字典，失败返回None
        """
        response = self._send_command(Command.STATE)
        if not response:
            return None
        
        try:
            # 解析JSON响应
            return json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse state: {e}, response: {response}")
            return None
    
    def get_level(self) -> int:
        """
        获取当前冒险模式关卡
        
        Returns:
            关卡号 (1-50)，失败返回0
        """
        response = self._send_command(Command.LEVEL)
        if response and response.startswith("OK"):
            parts = response.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
        return 0
    
    def set_level(self, level: int) -> bool:
        """
        设置冒险模式关卡
        
        Args:
            level: 关卡号 (1-50)
            
        Returns:
            True if successful
        """
        cmd = f"{Command.LEVEL} {level}"
        response = self._send_command(cmd)
        return response and Response.is_success(response)
    
    def unlock_all_plants(self) -> bool:
        """
        解锁所有植物（设置为通关状态）
        
        Returns:
            True if successful
        """
        response = self._send_command(Command.UNLOCK)
        return response and Response.is_success(response)
    
    def get_game_mode(self) -> int:
        """
        获取当前游戏模式
        
        Returns:
            游戏模式ID，失败返回-1
            0 = 冒险模式
            1-5 = 生存普通 (白天/黑夜/泳池/迷雾/屋顶)
            6-10 = 生存困难
            11-15 = 生存无尽 (13 = 泳池无尽)
        """
        response = self._send_command(Command.MODE)
        if response and response.startswith("OK"):
            parts = response.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
        return -1
    
    def set_game_mode(self, mode: int) -> bool:
        """
        设置游戏模式
        
        Args:
            mode: 游戏模式ID
            
        Returns:
            True if successful
        """
        cmd = f"{Command.MODE} {mode}"
        response = self._send_command(cmd)
        return response and Response.is_success(response)
    
    def start_game(self, mode: int) -> bool:
        """
        进入指定游戏模式并开始游戏
        
        Args:
            mode: 游戏模式ID
                0 = 冒险模式
                1-5 = 生存普通 (白天/黑夜/泳池/迷雾/屋顶)
                6-10 = 生存困难 (白天/黑夜/泳池/迷雾/屋顶)
                11-15 = 生存无尽 (白天/黑夜/泳池/迷雾/屋顶)
                13 = 生存无尽泳池 (Pool Endless)
            
        Returns:
            True if successful
        """
        cmd = f"{Command.START} {mode}"
        response = self._send_command(cmd)
        return response and Response.is_success(response)
    
    def start_survival_endless(self, stage: int = 3) -> bool:
        """
        进入生存无尽模式
        
        Args:
            stage: 场景 (1=白天, 2=黑夜, 3=泳池, 4=迷雾, 5=屋顶)
            
        Returns:
            True if successful
        """
        mode = 10 + stage  # GAMEMODE_SURVIVAL_ENDLESS_STAGE_1 = 11
        return self.start_game(mode)
    
    def get_ui(self) -> int:
        """
        获取当前游戏UI状态
        
        Returns:
            UI状态: 0=加载, 1=主菜单, 2=选卡, 3=游戏中, 7=选项
            失败返回-1
        """
        response = self._send_command("UI")
        if response and response.startswith("OK"):
            parts = response.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
        return -1
    
    def is_paused(self) -> bool:
        """
        检查游戏是否暂停（包括菜单打开）
        
        Returns:
            True if game is paused or menu is open
        """
        response = self._send_command("PAUSED")
        if response and response.startswith("OK"):
            parts = response.split()
            if len(parts) >= 2:
                return parts[1] == "1"
        return False
    
    def is_card_select_ready(self) -> bool:
        """
        检查选卡界面是否准备好（动画完成）
        
        Returns:
            True if card selection screen is ready for input
        """
        response = self._send_command("READY")
        if response and response.startswith("OK"):
            parts = response.split()
            if len(parts) >= 2:
                return parts[1] == "1"
        return False
    
    def wait_for_card_select_ready(self, timeout: float = 5.0) -> bool:
        """
        等待选卡界面准备好
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            True if ready within timeout
        """
        import time
        start = time.time()
        while time.time() - start < timeout:
            if self.is_card_select_ready():
                return True
            time.sleep(0.1)
        return False
    
    def wait_for_ui(self, target_ui: int, timeout: float = 10.0) -> bool:
        """
        等待游戏进入指定UI状态
        
        Args:
            target_ui: 目标UI状态 (2=选卡, 3=游戏中)
            timeout: 超时时间（秒）
            
        Returns:
            True if reached target UI
        """
        import time
        start = time.time()
        while time.time() - start < timeout:
            ui = self.get_ui()
            if ui == target_ui:
                return True
            time.sleep(0.2)
        return False
    
    def auto_start_game(self, mode: int, cards: list, timeout: float = 10.0) -> bool:
        """
        自动进入游戏并选卡开始
        
        Args:
            mode: 游戏模式 (13=泳池无尽)
            cards: 卡片列表 (最多10张)
            timeout: 等待选卡界面超时时间
            
        Returns:
            True if successful
        """
        # 1. 进入游戏
        if not self.start_game(mode):
            return False
        
        # 2. 等待选卡界面
        if not self.wait_for_ui(2, timeout):
            return False
        
        # 3. 等待选卡界面动画完成（AVZ检查 OrizontalScreenOffset == 4250）
        if not self.wait_for_card_select_ready(5.0):
            return False
        
        # 4. 选卡
        if not self.select_cards(cards):
            return False
        
        # 5. 开始
        return self.rock()
    
    # =========================================================================
    # 游戏速度控制
    # =========================================================================
    
    def get_game_speed(self) -> float:
        """
        获取当前游戏速度倍率
        
        Returns:
            速度倍率 (1.0 = 正常, 5.0 = 5倍速)
            失败返回 -1.0
        """
        response = self._send_command("SPEED")
        if response and response.startswith("OK"):
            parts = response.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    pass
        return -1.0
    
    def set_game_speed(self, speed: float) -> bool:
        """
        设置游戏速度
        
        Args:
            speed: 速度倍率 (0.05 到 10.0)
                   1.0 = 正常速度
                   5.0 = 5倍速
                   10.0 = 10倍速 (最快)
            
        Returns:
            True if successful
        """
        cmd = f"SPEED {speed}"
        response = self._send_command(cmd)
        return response and response.startswith("OK")
    
    def get_tick_ms(self) -> int:
        """
        获取每帧毫秒数 (底层速度控制)
        
        Returns:
            毫秒数 (默认10, 越小越快)
            失败返回 -1
        """
        response = self._send_command("TICKMS")
        if response and response.startswith("OK"):
            parts = response.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    pass
        return -1
    
    def set_tick_ms(self, ms: int) -> bool:
        """
        设置每帧毫秒数 (底层速度控制)
        
        Args:
            ms: 毫秒数 (1-200)
                1 = 10倍速
                10 = 正常速度
                200 = 0.05倍速
            
        Returns:
            True if successful
        """
        cmd = f"TICKMS {ms}"
        response = self._send_command(cmd)
        return response and response.startswith("OK")

    def __enter__(self):
        """Context manager support"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.disconnect()
