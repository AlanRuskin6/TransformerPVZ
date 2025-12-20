"""
Game Constants from AsmVsZombies
Time constants, game mechanics values
All time values are in centiseconds (cs) = 1/100 second

Reference: re-plants-vs-zombies/ConstEnums.h, Lawn/Plant.h, Lawn/Board.h
"""

from enum import IntEnum

# ============================================================================
# ZombiePhase (僵尸阶段/动作状态)
# Reference: ConstEnums.h - ZombiePhase
# ============================================================================

class ZombiePhase(IntEnum):
    """僵尸阶段状态 (mZombiePhase at +0x28)"""
    NORMAL = 0                    # 正常行走
    DYING = 1                     # 死亡中
    BURNED = 2                    # 被烧死
    MOWERED = 3                   # 被小推车碾死
    BUNGEE_DIVING = 4             # 蹦极下降
    BUNGEE_DIVING_SCREAMING = 5   # 蹦极下降(尖叫)
    BUNGEE_AT_BOTTOM = 6          # 蹦极到底
    BUNGEE_GRABBING = 7           # 蹦极抓取
    BUNGEE_RISING = 8             # 蹦极上升
    BUNGEE_HIT_OUCHY = 9          # 蹦极被击中
    BUNGEE_CUTSCENE = 10          # 蹦极过场
    POLEVAULTER_PRE_VAULT = 11    # 撑杆跳前摇
    POLEVAULTER_IN_VAULT = 12     # 撑杆跳中
    POLEVAULTER_POST_VAULT = 13   # 撑杆跳后
    RISING_FROM_GRAVE = 14        # 从墓碑升起
    JACK_IN_THE_BOX_RUNNING = 15  # 玩偶匣跑动
    JACK_IN_THE_BOX_POPPING = 16  # 玩偶匣爆炸
    BOBSLED_SLIDING = 17          # 雪橇滑行
    BOBSLED_BOARDING = 18         # 雪橇上车
    BOBSLED_CRASHING = 19         # 雪橇撞毁
    POGO_BOUNCING = 20            # 跳跳跳跃
    POGO_HIGH_BOUNCE_1 = 21       # 高跳1
    POGO_HIGH_BOUNCE_2 = 22       # 高跳2
    POGO_HIGH_BOUNCE_3 = 23       # 高跳3
    POGO_HIGH_BOUNCE_4 = 24       # 高跳4
    POGO_HIGH_BOUNCE_5 = 25       # 高跳5
    POGO_HIGH_BOUNCE_6 = 26       # 高跳6
    POGO_FORWARD_BOUNCE_2 = 27    # 前跳2
    POGO_FORWARD_BOUNCE_7 = 28    # 前跳7
    NEWSPAPER_READING = 29        # 读报纸
    NEWSPAPER_MADDENING = 30      # 报纸被打掉(愤怒中)
    NEWSPAPER_MAD = 31            # 报纸愤怒
    DIGGER_TUNNELING = 32         # 矿工挖掘
    DIGGER_RISING = 33            # 矿工升起
    DIGGER_TUNNELING_PAUSE_WITHOUT_AXE = 34  # 矿工无镐暂停
    DIGGER_RISE_WITHOUT_AXE = 35  # 矿工无镐升起
    DIGGER_STUNNED = 36           # 矿工被晕
    DIGGER_WALKING = 37           # 矿工行走
    DIGGER_WALKING_WITHOUT_AXE = 38  # 矿工无镐行走
    DIGGER_CUTSCENE = 39          # 矿工过场
    DANCER_DANCING_IN = 40        # 舞王进场
    DANCER_SNAPPING_FINGERS = 41  # 舞王打响指
    DANCER_SNAPPING_FINGERS_WITH_LIGHT = 42  # 舞王打响指(有光)
    DANCER_SNAPPING_FINGERS_HOLD = 43  # 舞王响指保持
    DANCER_DANCING_LEFT = 44      # 舞王向左跳
    DANCER_WALK_TO_RAISE = 45     # 舞王走去召唤
    DANCER_RAISE_LEFT_1 = 46      # 舞王召唤左1
    DANCER_RAISE_RIGHT_1 = 47     # 舞王召唤右1
    DANCER_RAISE_LEFT_2 = 48      # 舞王召唤左2
    DANCER_RAISE_RIGHT_2 = 49     # 舞王召唤右2
    DANCER_RISING = 50            # 伴舞升起
    DOLPHIN_WALKING = 51          # 海豚行走
    DOLPHIN_INTO_POOL = 52        # 海豚入水
    DOLPHIN_RIDING = 53           # 海豚骑乘
    DOLPHIN_IN_JUMP = 54          # 海豚跳跃
    DOLPHIN_WALKING_IN_POOL = 55  # 海豚水中行走
    DOLPHIN_WALKING_WITHOUT_DOLPHIN = 56  # 无海豚行走
    SNORKEL_WALKING = 57          # 潜水员行走
    SNORKEL_INTO_POOL = 58        # 潜水员入水
    SNORKEL_WALKING_IN_POOL = 59  # 潜水员水中行走
    SNORKEL_UP_TO_EAT = 60        # 潜水员上浮吃
    SNORKEL_EATING_IN_POOL = 61   # 潜水员水中吃
    SNORKEL_DOWN_FROM_EAT = 62    # 潜水员下潜
    ZOMBIQUARIUM_ACCEL = 63       # 僵尸水族馆加速
    ZOMBIQUARIUM_DRIFT = 64       # 僵尸水族馆漂流
    ZOMBIQUARIUM_BACK_AND_FORTH = 65  # 僵尸水族馆来回
    ZOMBIQUARIUM_BITE = 66        # 僵尸水族馆咬
    CATAPULT_LAUNCHING = 67       # 投石车发射
    CATAPULT_RELOADING = 68       # 投石车装填
    GARGANTUAR_THROWING = 69      # 巨人扔小鬼
    GARGANTUAR_SMASHING = 70      # 巨人砸击
    IMP_GETTING_THROWN = 71       # 小鬼被扔
    IMP_LANDING = 72              # 小鬼落地
    BALLOON_FLYING = 73           # 气球飞行
    BALLOON_POPPING = 74          # 气球破裂
    BALLOON_WALKING = 75          # 气球行走
    LADDER_CARRYING = 76          # 扶梯搬运
    LADDER_PLACING = 77           # 扶梯放置
    BOSS_ENTER = 78               # BOSS进场
    BOSS_IDLE = 79                # BOSS待机
    BOSS_SPAWNING = 80            # BOSS生成
    BOSS_STOMPING = 81            # BOSS踩踏
    BOSS_BUNGEES_ENTER = 82       # BOSS蹦极进场
    BOSS_BUNGEES_DROP = 83        # BOSS蹦极放下
    BOSS_BUNGEES_LEAVE = 84       # BOSS蹦极离开
    BOSS_DROP_RV = 85             # BOSS扔房车
    BOSS_HEAD_ENTER = 86          # BOSS头进入
    BOSS_HEAD_IDLE_BEFORE_SPIT = 87  # BOSS头待机(吐前)
    BOSS_HEAD_IDLE_AFTER_SPIT = 88   # BOSS头待机(吐后)
    BOSS_HEAD_SPIT = 89           # BOSS吐火球
    BOSS_HEAD_LEAVE = 90          # BOSS头离开
    YETI_RUNNING = 91             # 雪人逃跑
    SQUASH_PRE_LAUNCH = 92        # 窝瓜预备(被窝瓜压的状态)
    SQUASH_RISING = 93            # 窝瓜上升
    SQUASH_FALLING = 94           # 窝瓜下落
    SQUASH_DONE_FALLING = 95      # 窝瓜落地


# ============================================================================
# PlantState (植物状态)
# Reference: Lawn/Plant.h - PlantState
# ============================================================================

class PlantState(IntEnum):
    """植物状态 (mState)"""
    NOTREADY = 0                   # 未就绪
    READY = 1                      # 就绪
    DOINGSPECIAL = 2               # 执行特殊动作
    SQUASH_LOOK = 3                # 窝瓜看
    SQUASH_PRE_LAUNCH = 4          # 窝瓜预跳
    SQUASH_RISING = 5              # 窝瓜上升
    SQUASH_FALLING = 6             # 窝瓜下落
    SQUASH_DONE_FALLING = 7        # 窝瓜落地
    GRAVEBUSTER_LANDING = 8        # 墓碑吞噬者降落
    GRAVEBUSTER_EATING = 9         # 墓碑吞噬者吃
    CHOMPER_BITING = 10            # 大嘴花咬
    CHOMPER_BITING_GOT_ONE = 11    # 大嘴花咬中
    CHOMPER_BITING_MISSED = 12     # 大嘴花咬空
    CHOMPER_DIGESTING = 13         # 大嘴花消化中
    CHOMPER_SWALLOWING = 14        # 大嘴花吞咽
    POTATO_RISING = 15             # 土豆雷升起
    POTATO_ARMED = 16              # 土豆雷武装
    POTATO_MASHED = 17             # 土豆雷爆炸
    SPIKEWEED_ATTACKING = 18       # 地刺攻击
    SPIKEWEED_ATTACKING_2 = 19     # 地刺攻击2
    SCAREDYSHROOM_LOWERING = 20    # 胆小菇缩下
    SCAREDYSHROOM_SCARED = 21      # 胆小菇害怕
    SCAREDYSHROOM_RAISING = 22     # 胆小菇升起
    SUNSHROOM_SMALL = 23           # 阳光菇(小)
    SUNSHROOM_GROWING = 24         # 阳光菇生长中
    SUNSHROOM_BIG = 25             # 阳光菇(大)
    MAGNETSHROOM_SUCKING = 26      # 磁力菇吸取
    MAGNETSHROOM_CHARGING = 27     # 磁力菇充能
    BOWLING_UP = 28                # 保龄球上
    BOWLING_DOWN = 29              # 保龄球下
    CACTUS_LOW = 30                # 仙人掌(低)
    CACTUS_RISING = 31             # 仙人掌升起
    CACTUS_HIGH = 32               # 仙人掌(高)
    CACTUS_LOWERING = 33           # 仙人掌降低
    TANGLEKELP_GRABBING = 34       # 缠绕海草抓取
    COBCANNON_ARMING = 35          # 玉米炮装填中
    COBCANNON_LOADING = 36         # 玉米炮加载中
    COBCANNON_READY = 37           # 玉米炮就绪
    COBCANNON_FIRING = 38          # 玉米炮发射
    KERNELPULT_BUTTER = 39         # 玉米投手(黄油)
    UMBRELLA_TRIGGERED = 40        # 叶子伞触发
    UMBRELLA_REFLECTING = 41       # 叶子伞反弹
    IMITATER_MORPHING = 42         # 模仿者变身
    ZEN_GARDEN_WATERED = 43        # 禅境花园浇水
    ZEN_GARDEN_NEEDY = 44          # 禅境花园需要
    ZEN_GARDEN_HAPPY = 45          # 禅境花园开心
    MARIGOLD_ENDING = 46           # 金盏花结束
    FLOWERPOT_INVULNERABLE = 47    # 花盆无敌
    LILYPAD_INVULNERABLE = 48      # 荷叶无敌


# ============================================================================
# GridSquareType (场地格子类型)
# Reference: ConstEnums.h - GridSquareType
# ============================================================================

class GridSquareType(IntEnum):
    """场地格子类型"""
    NONE = 0        # 无/不可用
    GRASS = 1       # 草地
    DIRT = 2        # 泥土 (禅境花园)
    POOL = 3        # 水池
    HIGH_GROUND = 4 # 高地 (屋顶)


# ============================================================================
# Cob Cannon (玉米加农炮) Constants
# ============================================================================

COB_FLY_TIME = 373  # 玉米加农炮飞行时间 (cs)
COB_RECOVER_TIME = 3475  # 炮冷却时间 (cs)

# 屋顶炮飞行时间 (根据落点列数不同)
# 索引 0-6 对应 1-7 列
ROOF_COB_FLY_TIME = [359, 362, 364, 367, 369, 372, 373]

# 炮弹爆炸半径 (像素)
COB_EXPLODE_RADIUS = 115

# ============================================================================
# Instant Kill Plants (灰烬植物) Delay
# ============================================================================

CHERRY_DELAY = 100  # 樱桃爆炸延迟 (cs)
JALAPENO_DELAY = 100  # 火爆辣椒延迟 (cs)
DOOM_DELAY = 100  # 毁灭菇爆炸延迟 (cs)
SQUASH_DELAY = 100  # 窝瓜延迟 (cs)

POTATO_MINE_ARM_TIME = 1500  # 土豆雷武装时间 (cs)

# 樱桃爆炸范围 (以列为单位)
CHERRY_EXPLODE_RADIUS = 90  # 像素

# ============================================================================
# Ice Related (冰冻相关)
# ============================================================================

ICE_EFFECT_TIME = 298  # 冰菇生效时间 (cs)
ICE_DURATION = 400  # 冰冻持续时间 (cs)
SLOW_DURATION = 1000  # 减速持续时间 (cs)

# ============================================================================
# Gargantuar (巨人僵尸) Constants
# ============================================================================

GIGA_THROW_IMP_TIME = [105, 210]  # 红眼扔小鬼的时间点 (cs)
HAMMER_CIRCULATION_RATE = 0.644  # 锤击循环概率

# ============================================================================
# 动画常量 (Animation Constants)
# ============================================================================

# 动画循环率偏移 (来自 AVZ judge.h)
ANIM_CIRCULATION_RATE_OFFSET = 0x4

# ============================================================================
# Speed Constants (速度常量)
# ============================================================================

# 僵尸平均速度 (像素/cs)
GIGA_AVG_SPEED = 484 / 3158 * 1.25  # ≈ 0.192 px/cs

# 曾菇每 cs 伤害
GLOOM_DAMAGE_PER_CS = 80 / 200  # = 0.4 hp/cs

# ============================================================================
# Grid Constants (场地常量)
# ============================================================================

GRID_WIDTH = 80  # 每格宽度 (像素)
GRID_HEIGHT = 85  # 每格高度 (像素) - 非屋顶场景
GRID_HEIGHT_ROOF = 85  # 屋顶场景基础高度 (same as non-roof, but roof has slope)

GRID_COLS = 9  # 列数
GRID_ROWS = 6  # 最大行数 (泳池有6行，草地有5行)

# 草地左边界 x 坐标
LAWN_LEFT_X = 40
LAWN_RIGHT_X = LAWN_LEFT_X + GRID_WIDTH * GRID_COLS

# 植物种植的 x 坐标计算: x = LAWN_LEFT_X + col * GRID_WIDTH
# 植物种植的 y 坐标计算: y = 80 + row * GRID_HEIGHT

# ============================================================================
# Scene Types (场景类型)
# ============================================================================

SCENE_DAY = 0  # 白天
SCENE_NIGHT = 1  # 夜晚
SCENE_POOL = 2  # 泳池
SCENE_FOG = 3  # 迷雾
SCENE_ROOF = 4  # 屋顶白天
SCENE_ROOF_NIGHT = 5  # 屋顶夜晚

# ============================================================================
# Game UI States (游戏界面状态)
# ============================================================================

UI_MAIN_MENU = 1
UI_ALMANAC = 2
UI_IN_GAME = 3
UI_PAUSE = 4

# ============================================================================
# Attack Constants (攻击常量)
# ============================================================================

# 豌豆射手攻击间隔 (cs)
PEASHOOTER_ATTACK_INTERVAL = 141

# 各植物伤害值
PEA_DAMAGE = 20
SNOW_PEA_DAMAGE = 20
FIRE_PEA_DAMAGE = 40  # 经过火炬的豌豆
MELON_DAMAGE = 80  # 西瓜投手
WINTER_MELON_DAMAGE = 80  # 冰西瓜

# 范围伤害
CHERRY_DAMAGE = 1800
JALAPENO_DAMAGE = 1800
DOOM_DAMAGE = 1800  # 对普通僵尸
DOOM_DAMAGE_GARG = 900  # 对巨人减半

# ============================================================================
# Zombie States (僵尸状态)
# ============================================================================

Z_STATE_WALKING = 1
Z_STATE_DYING = 2
Z_STATE_DYING_FROM_INSTANT = 3
Z_STATE_DYING_FROM_LAWNMOWER = 4
Z_STATE_BUNGEE_TARGET = 5
Z_STATE_BUNGEE_LANDING = 6
Z_STATE_BUNGEE_RISING = 7
Z_STATE_EATING = 8
Z_STATE_FALLING = 9  # 从蹦极掉下
Z_STATE_POLE_VAULTING = 10
Z_STATE_DANCING = 11
Z_STATE_SNORKEL = 12
Z_STATE_RISING_FROM_GRAVE = 13
Z_STATE_DIGGER_TUNNELING = 14
Z_STATE_DIGGER_RISING = 15
Z_STATE_DIGGER_WALKING = 16
Z_STATE_DOLPHIN_DIVING = 17
Z_STATE_DOLPHIN_JUMPING = 18
Z_STATE_POGO_JUMPING = 19
Z_STATE_LADDER_PLACING = 20
Z_STATE_LADDER_CLIMBING = 21
Z_STATE_YETI_ESCAPING = 22
Z_STATE_BOBSLED_SLIDING = 23
Z_STATE_GARG_THROWING = 24
Z_STATE_BALLOON_FLYING = 25
Z_STATE_DOLPHIN_RIDING = 26
Z_STATE_IMP_LANDING = 27
Z_STATE_ZOMBOSS = 28
