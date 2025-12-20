"""
Memory Offsets from AsmVsZombies (avz_pvz_struct.h)
All memory addresses and structure offsets for PVZ memory reading/writing
"""


class Offset:
    """Memory offsets for PVZ game structures"""
    
    # ========================================================================
    # Base Addresses
    # ========================================================================
    
    BASE = 0x6A9EC0  # PvzBase pointer
    
    # PvzBase offsets
    MAIN_OBJECT = 0x768  # Board/MainObject pointer
    GAME_UI = 0x7FC  # Current UI state (3 = in game)
    PLAYER_INFO = 0x82C  # PlayerInfo pointer
    TICK_MS = 0x454  # int, 每帧时长(ms), 默认10=100fps, 改成1=10倍速
    SEED_CHOOSER = 0x774  # 选卡界面指针 (Seed Chooser Screen)
    
    # ========================================================================
    # PlayerInfo Offsets (玩家存档信息)
    # ========================================================================
    
    PI_LEVEL = 0x24           # int, 当前关卡进度 (用于推算解锁的植物)
    PI_COINS = 0x28           # int, 金币数量
    PI_FINISHED_ADV = 0x2C    # int, 完成冒险模式次数
    PI_PURCHASES = 0x1C0      # long[80], 商店购买记录
    PI_PURCHASE_SIZE = 4      # 每个购买记录4字节
    PI_PURCHASE_COUNT = 80    # 购买记录总数
    
    # ========================================================================
    # MainObject (Board) Offsets
    # ========================================================================
    
    # Zombie array
    ZOMBIE_ARRAY = 0x90
    ZOMBIE_COUNT_MAX = 0x94
    ZOMBIE_COUNT = 0xA0
    
    # Plant array
    PLANT_ARRAY = 0xAC
    PLANT_COUNT_MAX = 0xB0
    PLANT_COUNT = 0xBC
    
    # Seed/Card array
    SEED_ARRAY = 0x144
    
    # Projectile array
    PROJECTILE_ARRAY = 0xC8
    PROJECTILE_COUNT_MAX = 0xCC
    
    # Item (collectible/sun) array
    ITEM_ARRAY = 0xE4
    ITEM_COUNT_MAX = 0xE8
    
    # Lawnmower array
    LAWNMOWER_ARRAY = 0x100
    LAWNMOWER_COUNT_MAX = 0x104
    
    # Game state
    SUN = 0x5560
    GAME_CLOCK = 0x5568
    SCENE = 0x554C
    WAVE = 0x557C
    TOTAL_WAVE = 0x5564
    REFRESH_COUNTDOWN = 0x559C
    HUGE_WAVE_COUNTDOWN = 0x55A4
    LEVEL_END_COUNTDOWN = 0x5604  # 关卡结束倒计时 (>0 表示即将通关)
    
    # Additional game state
    COIN_COUNT = 0x5570
    PAUSED = 0x164
    
    # 场地物品数组 (墓碑、弹坑等)
    PLACE_ITEM_ARRAY = 0x11C
    PLACE_ITEM_COUNT_MAX = 0x120
    
    # 场地格子类型列表 (6行 * 9列 的格子类型)
    GRID_TYPE_LIST = 0x168  # int[6][9], GridSquareType
    
    # 出怪种类列表 (bool数组，标记哪些僵尸会出现)
    ZOMBIE_TYPE_LIST = 0x54D4  # bool[100]
    
    # 全局时钟 (战斗和选卡界面都计时)
    GLOBAL_CLOCK = 0x556C
    
    # 僵尸初始刷新倒计时
    INITIAL_COUNTDOWN = 0x55A0
    
    # 出怪列表 (每波僵尸的具体安排)
    # MAX_ZOMBIE_WAVES = 100, MAX_ZOMBIES_IN_WAVE = 50
    # ZombieType mZombiesInWave[100][50] at +0x6B4
    ZOMBIE_LIST = 0x6B4
    ZOMBIE_LIST_WAVE_SIZE = 50 * 4  # 每波50个僵尸，每个4字节
    ZOMBIE_LIST_MAX_WAVES = 100
    ZOMBIE_LIST_MAX_PER_WAVE = 50
    
    # 点炮倒计时 (游戏30cs防误触机制)
    CLICK_PAO_COUNTDOWN = 0x5754
    
    # 僵尸刷新血量
    ZOMBIE_REFRESH_HP = 0x5594
    
    # ========================================================================
    # Ice Trail (冰道) Data - 冰车留下的冰道
    # Reference: Board.h lines 137-139
    # ========================================================================
    
    # 每行冰道的最小X坐标 (冰道左边界)
    ICE_MIN_X = 0x60C     # int[6], mIceMinX - 每行冰道起点
    
    # 每行冰道剩余时间
    ICE_TIMER = 0x624     # int[6], mIceTimer - 冰道持续时间
    
    # 每行冰道粒子效果ID
    ICE_PARTICLE_ID = 0x63C  # int[6], mIceParticleID
    
    # 全局冰冻陷阱计数器
    ICE_TRAP_COUNTER = 0x5618  # int, mIceTrapCounter
    
    # ========================================================================
    # Zombie Structure (size = 0x15C)
    # ========================================================================
    
    ZOMBIE_SIZE = 0x15C
    
    # Position and movement
    Z_ROW = 0x1C  # int, current row
    Z_X = 0x2C  # float, x position
    Z_Y = 0x30  # float, y position
    Z_SPEED = 0x34  # float, current speed
    Z_HEIGHT = 0x38  # float, height offset (for jumping/flying)
    
    # Type and state
    Z_TYPE = 0x24  # int, zombie type
    Z_STATE = 0x28  # int, current state (walking, eating, etc.)
    
    # Health
    Z_HP = 0xC8  # int, current body HP
    Z_HP_MAX = 0xCC  # int, max body HP
    Z_HELM_TYPE = 0xC4  # int, helmet type (mHelmType)
    Z_ACCESSORY_HP_1 = 0xD0  # int, first accessory HP (cone/bucket/etc.)
    Z_ACCESSORY_HP_2 = 0xDC  # int, second accessory HP (shield door)
    
    # Status effects
    Z_SLOW_COUNTDOWN = 0xAC  # int, slow effect remaining time
    Z_FREEZE_COUNTDOWN = 0xB4  # int, freeze effect remaining time
    Z_BUTTER_COUNTDOWN = 0xB0  # int, butter effect remaining time
    
    # Animation
    Z_ANIMATION = 0x118  # AnimationMain pointer
    Z_ANIMATION_PROGRESS = 0x58  # float, 0.0-1.0
    
    # Status flags
    Z_DEAD = 0xEC  # bool, is dead/removed
    Z_VISIBLE = 0x18  # bool, is visible
    Z_AT_WAVE = 0x6C  # int, which wave spawned this zombie
    
    # Attack
    Z_EATING_COUNTDOWN = 0x88  # int, time until next bite
    
    # 僵尸是否在啃食
    Z_IS_EAT = 0x51  # bool
    
    # 僵尸存在时间
    Z_EXIST_TIME = 0x60  # int
    
    # 僵尸状态倒计时
    Z_STATE_COUNTDOWN = 0x68  # int
    
    # 中弹判定坐标
    Z_BULLET_X = 0x8C  # int, 中弹判定的横坐标
    Z_BULLET_Y = 0x90  # int, 中弹判定的纵坐标
    
    # 受伤判定范围
    Z_HURT_WIDTH = 0x94   # int, 受伤判定宽度
    Z_HURT_HEIGHT = 0x98  # int, 受伤判定高度
    
    # 攻击判定坐标
    Z_ATTACK_X = 0x9C  # int, 攻击判定的横坐标
    Z_ATTACK_Y = 0xA0  # int, 攻击判定的纵坐标
    
    # Special zombie data
    Z_TARGET_ROW = 0x130  # int, for pole vaulter, digger, etc. (mTargetRow)
    Z_HAS_BALLOON = 0xE4  # int, balloon HP (balloon zombie)
    Z_HAS_SHIELD = 0xD8   # int, shield type
    Z_LADDER_PLACED = 0x7C  # int, ladder zombie placed at col (mUseLadderCol)
    Z_HYPNOTIZED = 0xB8   # bool, hypno-shroom effect (mMindControlled)
    
    # Additional zombie status
    Z_HAS_OBJECT = 0xBC   # bool, gargantuar has imp (mHasObject)
    Z_BLOWING_AWAY = 0xB9 # bool, being blown away by blover
    Z_TARGET_COL = 0x80   # int, bungee target column
    
    # Jack-in-box
    Z_BOX_EXPLODED = 0xBA  # bool, jack-in-box exploded
    
    # ========================================================================
    # Plant Structure (size = 0x14C)
    # ========================================================================
    
    PLANT_SIZE = 0x14C
    
    # Position
    P_ROW = 0x1C  # int, row
    P_COL = 0x28  # int, column
    P_X = 0x8  # int, x position (pixel)
    P_Y = 0xC  # int, y position (pixel)
    
    # Type and state
    P_TYPE = 0x24  # int, plant type
    P_STATE = 0x3C  # int, current state
    
    # Health
    P_HP = 0x40  # int, current HP
    P_HP_MAX = 0x44  # int, max HP
    
    # Attack
    P_SHOOT_COUNTDOWN = 0x90  # int, countdown to next shot
    P_EFFECTIVE = 0x48  # int, is awake/effective
    
    # Status
    P_DEAD = 0x141  # bool, is dead/removed
    P_SQUASHED = 0x142  # bool, is being squashed
    P_SLEEPING = 0x143  # bool, is sleeping (mushrooms in day)
    
    # Pumpkin - Note: Same offset as P_DISAPPEAR_COUNTDOWN (0x4C)
    # They share the same memory location but have different meanings
    P_PUMPKIN_HP = 0x4C  # int, pumpkin shield HP if present
    P_DISAPPEAR_COUNTDOWN = 0x4C  # int, countdown to disappear (mDisappearCountdown)
    
    # 受伤判定范围
    P_HURT_WIDTH = 0x10   # int, 受伤判定宽度
    P_HURT_HEIGHT = 0x14  # int, 受伤判定高度
    
    # 植物是否可见
    P_VISIBLE = 0x18  # bool
    
    # 三叶草消失倒计时
    P_BLOVER_COUNTDOWN = 0x4C  # int
    
    # 灰烬/冰菇/三叶草生效倒计时 (重要！)
    P_EXPLODE_COUNTDOWN = 0x50  # int
    
    # 蘑菇倒计时
    P_MUSHROOM_COUNTDOWN = 0x130  # int
    
    # 蹦极抓取状态 (0没被抓住, 1被抓住, 2抱走)
    P_BUNGEE_STATE = 0x134  # int
    
    # 模仿者原始类型 (mImitaterType)
    P_IMITATER_TYPE = 0x138  # int
    
    # Cob Cannon
    P_COB_COUNTDOWN = 0x54  # int, cob cannon reload time
    P_COB_READY = 0x58  # bool, cob cannon ready to fire
    
    # Animation
    P_ANIMATION = 0x94  # AnimationMain pointer
    
    # ========================================================================
    # Seed/Card Structure (size = 0x50)
    # ========================================================================
    # Based on AVZ: ASeed data starts at offset 0x28 within each slot
    # All offsets below are ABSOLUTE (already include the 0x28 base)
    
    SEED_SIZE = 0x50
    
    # Card count (only valid on first seed, offset 0x24 from seed_array)
    S_COUNT = 0x24  # int, number of cards in slot
    
    # Card info (absolute offsets from seed_array + index * SEED_SIZE)
    S_RECHARGE_COUNTDOWN = 0x4C  # int, current recharge remaining (0x24 + 0x28)
    S_RECHARGE_TIME = 0x50       # int, total recharge time (0x28 + 0x28)
    S_TYPE = 0x5C                # int, plant type (0x34 + 0x28)
    S_IMITATOR_TYPE = 0x60       # int, imitator plant type, -1 if not (0x38 + 0x28)
    S_USABLE = 0x70              # bool, is card usable (0x48 + 0x28)
    S_X = 0x30                   # int, x position on screen (0x08 + 0x28)
    S_Y = 0x34                   # int, y position on screen (0x0C + 0x28)
    S_WIDTH = 0x38               # int, card width (0x10 + 0x28)
    S_HEIGHT = 0x3C              # int, card height (0x14 + 0x28)
    
    # ========================================================================
    # Item/Collectible Structure (size = 0xD8)
    # ========================================================================
    
    ITEM_SIZE = 0xD8
    
    # Position
    I_X = 0x24  # float, x position
    I_Y = 0x28  # float, y position
    
    # State
    I_TYPE = 0x58  # int, item type (1=silver, 2=gold, 3=diamond, 4=sun, etc.)
    I_DISAPPEARED = 0x38  # bool, has disappeared
    I_COLLECTED = 0x50  # bool, has been collected
    I_DEAD = 0x20  # bool, is dead/removed
    
    # Note: Sun value is determined by I_TYPE, not a separate field
    # Type 4 = Normal Sun (25), Type 5 = Small Sun (15), Type 6 = Big Sun (50)
    
    # ========================================================================
    # Projectile Structure
    # ========================================================================
    
    PROJECTILE_SIZE = 0x94
    
    PR_X = 0x30  # float, x position
    PR_Y = 0x34  # float, y position
    PR_ROW = 0x1C  # int, row
    PR_TYPE = 0x5C  # int, projectile type
    PR_DEAD = 0x50  # bool, is dead
    
    # 子弹存在时间
    PR_EXIST_TIME = 0x60  # int
    
    # 炮弹落点横坐标 (需要 +87.5 才是实际落点)
    PR_COB_TARGET_X = 0x80  # float
    
    # 炮弹落点行
    PR_COB_TARGET_ROW = 0x84  # int
    
    # ========================================================================
    # Lawnmower Structure (size = 0x48)
    # 参考 re-plants-vs-zombies/Lawn/LawnMower.h
    # ========================================================================

    LAWNMOWER_SIZE = 0x48

    LM_ROW = 0x14    # int, row (mRow)
    LM_X = 0x08      # float, x position (mPosX)
    LM_Y = 0x0C      # float, y position (mPosY)
    LM_STATE = 0x2C  # int, state (mMowerState): 0=rolling_in, 1=ready, 2=triggered, 3=squished
    LM_DEAD = 0x30   # bool, is dead/used (mDead)
    LM_TYPE = 0x34   # int, mower type (mMowerType): 0=lawn, 1=pool, 2=roof, 3=super    # ========================================================================
    # PlaceItem Structure (size = 0xEC) - 场地物品 (墓碑、弹坑等)
    # ========================================================================
    
    PLACE_ITEM_SIZE = 0xEC
    
    PI_TYPE = 0x8      # int, 物品类型
    PI_COL = 0x10      # int, 所在列
    PI_ROW = 0x14      # int, 所在行
    PI_VALUE = 0x18    # int, 数值 (墓碑冒出量/弹坑倒计时/脑子血量)
    PI_DEAD = 0x20     # bool, 是否消失
    
    # ========================================================================
    # Function Addresses (for ASM injection)
    # Verified from AsmVsZombies - compatible with PVZ 1.0.0.1051
    # ========================================================================
    
    FUNC_PLANT = 0x0040D120  # Plant function (verified)
    FUNC_REMOVE_PLANT = 0x004679B0  # Remove/shovel plant (verified)
    FUNC_REFRESH_SEEDS = 0x00488500  # Refresh seed (RefreshSeed in AVZ)
    FUNC_CLICK_SEED = 0x00488590  # SeedPacket::MouseDown (PlantCard in AVZ)
    FUNC_COB_FIRE = 0x00466D50  # Fire cob cannon (verified)
    FUNC_PLANT_READY = 0x0040FD30  # Check plant ready
    FUNC_SHOVEL = 0x00411060  # Shovel plant
    FUNC_CHOOSE_CARD = 0x00486030  # Choose card in seed chooser (AVZ ChooseCard)
    FUNC_ROCK = 0x00486D20  # Rock() 开始游戏 (AVZ "Let's Rock" button)
    FUNC_PICK_RANDOM = 0x004859B0  # PickRandomSeeds() 随机选卡 (AVZ)
    
    # ========================================================================
    # Animation Structure
    # ========================================================================
    
    ANIM_CIRCULATION_RATE = 0x0  # float, animation circulation rate
    ANIM_PROGRESS = 0x8  # float, current progress (0-1)
    ANIM_FRAME = 0x24  # int, current frame


# Item types for collectibles
class ItemType:
    """Item/collectible types"""
    SILVER_COIN = 1
    GOLD_COIN = 2
    DIAMOND = 3
    SUN = 4  # Normal sun (25)
    SMALL_SUN = 5  # Small sun (15)
    BIG_SUN = 6  # Big sun (50)
    FINAL_SUN = 17  # End of level sun


# Scene/Level types
class SceneType:
    """Scene/level types"""
    DAY = 0
    NIGHT = 1
    POOL = 2
    FOG = 3
    ROOF = 4
    ROOF_NIGHT = 5
    
    @staticmethod
    def has_pool(scene: int) -> bool:
        """Check if scene has water/pool"""
        return scene in [SceneType.POOL, SceneType.FOG]
    
    @staticmethod
    def is_day(scene: int) -> bool:
        """Check if scene is daytime (natural sun drops)"""
        return scene in [SceneType.DAY, SceneType.POOL, SceneType.ROOF]
    
    @staticmethod
    def is_night(scene: int) -> bool:
        """Check if scene is night (mushrooms wake)"""
        return scene in [SceneType.NIGHT, SceneType.FOG, SceneType.ROOF_NIGHT]
    
    @staticmethod
    def is_roof(scene: int) -> bool:
        """Check if scene is roof (need flower pots)"""
        return scene in [SceneType.ROOF, SceneType.ROOF_NIGHT]
    
    @staticmethod
    def get_row_count(scene: int) -> int:
        """Get number of playable rows for scene"""
        if scene in [SceneType.POOL, SceneType.FOG]:
            return 6
        else:
            return 5
