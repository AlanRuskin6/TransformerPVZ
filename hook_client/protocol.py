"""
Protocol definitions for Hook DLL communication
"""


class Command:
    """命令常量"""
    PLANT = "PLANT"
    SHOVEL = "SHOVEL"
    FIRE = "FIRE"
    RESET = "RESET"
    ENTER = "ENTER"
    CHOOSE = "CHOOSE"
    ROCK = "ROCK"
    BACK = "BACK"
    STATE = "STATE"
    LEVEL = "LEVEL"
    UNLOCK = "UNLOCK"
    MODE = "MODE"
    START = "START"


class Response:
    """响应类型"""
    OK = "OK"
    ERR = "ERR"
    
    @staticmethod
    def is_success(response: str) -> bool:
        """检查响应是否成功"""
        return response.startswith(Response.OK)
    
    @staticmethod
    def get_error_message(response: str) -> str:
        """提取错误消息"""
        if response.startswith(Response.ERR):
            return response[4:].strip()
        return ""
