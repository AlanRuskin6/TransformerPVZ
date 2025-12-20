#include "game.h"
#include <Windows.h>
#include <atomic>

// 帧倍增数 - 每帧执行多少次游戏逻辑
static std::atomic<int> g_frameMultiplier(1);

// PVZ Memory Addresses - Reference: AsmVsZombies
namespace Addr {
    constexpr uintptr_t BASE = 0x6A9EC0;
    constexpr uintptr_t MAIN_OBJECT = 0x768;
    constexpr uintptr_t GAME_UI = 0x7FC;
    constexpr uintptr_t SEED_CHOOSER = 0x774;
    constexpr uintptr_t MOUSE_WINDOW = 0x320;  // MouseWindow pointer
    constexpr uintptr_t TICK_MS = 0x454;       // 每帧毫秒数 (默认10, 越小越快)
    
    constexpr uintptr_t SUN = 0x5560;
    constexpr uintptr_t WAVE = 0x557C;
    constexpr uintptr_t SCENE = 0x554C;
    constexpr uintptr_t GAME_PAUSED = 0x164;   // bool, game pause state
    
    constexpr uintptr_t PLANT_ARRAY = 0xAC;
    constexpr uintptr_t PLANT_COUNT_MAX = 0xB0;
    constexpr uintptr_t PLANT_SIZE = 0x14C;
    constexpr uintptr_t P_DEAD = 0x141;
    
    constexpr uintptr_t ZOMBIE_ARRAY = 0x90;
    constexpr uintptr_t ZOMBIE_COUNT_MAX = 0x94;
    constexpr uintptr_t ZOMBIE_SIZE = 0x15C;
    constexpr uintptr_t Z_DEAD = 0xEC;
    constexpr uintptr_t Z_X = 0x2C;
}

namespace {
std::atomic<bool> g_startGamePending{ false };
std::atomic<int> g_startGameMode{ 0 };
}

namespace PVZ {

uintptr_t GetBase() {
    return *(uintptr_t*)Addr::BASE;
}

uintptr_t GetBoard() {
    uintptr_t base = GetBase();
    if (!base) return 0;
    return *(uintptr_t*)(base + Addr::MAIN_OBJECT);
}

uintptr_t GetGameUI() {
    uintptr_t base = GetBase();
    if (!base) return 0;
    return *(uintptr_t*)(base + Addr::GAME_UI);
}

int GetSun() {
    uintptr_t board = GetBoard();
    if (!board) return 0;
    return *(int*)(board + Addr::SUN);
}

int GetWave() {
    uintptr_t board = GetBoard();
    if (!board) return 0;
    return *(int*)(board + Addr::WAVE);
}

int GetScene() {
    uintptr_t board = GetBoard();
    if (!board) return 0;
    return *(int*)(board + Addr::SCENE);
}

// GetTickMs - Get milliseconds per game tick (default 10)
// Lower = faster game
int GetTickMs() {
    uintptr_t base = GetBase();
    if (!base) return 10;
    return *(int*)(base + Addr::TICK_MS);
}

// SetTickMs - Set milliseconds per game tick
// Range: 1+ (1=10x speed, 10=normal, lower=faster)
// Removed upper limit for extreme speeds
bool SetTickMs(int ms) {
    if (ms < 1) return false;  // 只限制最小值，不限制最大值
    uintptr_t base = GetBase();
    if (!base) return false;
    *(int*)(base + Addr::TICK_MS) = ms;
    return true;
}

// GetGameSpeed - Get game speed multiplier
float GetGameSpeed() {
    int tickMs = GetTickMs();
    if (tickMs <= 0) return 1.0f;
    return 10.0f / tickMs;
}

// SetGameSpeed - Set game speed (no upper limit)
// Reference: AVZ __AGameSpeedManager::Set
bool SetGameSpeed(float speed) {
    if (speed < 0.05f) return false;  // 只限制最小速度，不限制最高速度
    int ms = (int)(10.0f / speed + 0.5f);
    if (ms < 1) ms = 1;  // 最小1ms = 10x速度
    return SetTickMs(ms);
}

// GetFrameMultiplier - 获取帧倍增数
int GetFrameMultiplier() {
    return g_frameMultiplier.load();
}

// SetFrameMultiplier - 设置帧倍增数
// mult: 1-100, 每帧执行多少次游戏逻辑
// 配合 tick_ms=1 (10x)，mult=10 可以达到 100x 速度
bool SetFrameMultiplier(int mult) {
    if (mult < 1 || mult > 100) return false;
    g_frameMultiplier.store(mult);
    return true;
}

// RunGameFrame - 执行一帧游戏逻辑
// 参考 AVZ: AAsm::GameTotalLoop
void RunGameFrame() {
    __asm {
        mov ecx, dword ptr ds:[0x6A9EC0]
        mov eax, 0x452650
        call eax
    }
}

bool IsInGame() {
    return GetGameUI() == 3;
}

// Check if game is paused (including menu dialogs)
// Reference: AVZ AGameIsPaused() in avz_memory.cpp
// Returns true if: GamePaused flag is set OR a dialog/menu is open
bool IsGamePaused() {
    uintptr_t board = GetBoard();
    if (!board) return false;
    
    // Check GamePaused flag at Board + 0x164
    bool gamePaused = *(bool*)(board + Addr::GAME_PAUSED);
    if (gamePaused) return true;
    
    // Check if a dialog/menu is open (TopWindow != nullptr)
    // MouseWindow at Base + 0x320, TopWindow at MouseWindow + 0x94
    uintptr_t base = GetBase();
    if (!base) return false;
    uintptr_t mouseWindow = *(uintptr_t*)(base + Addr::MOUSE_WINDOW);
    if (!mouseWindow) return false;
    uintptr_t topWindow = *(uintptr_t*)(mouseWindow + 0x94);
    if (topWindow != 0) return true;
    
    return false;
}

// Check if card selection screen is ready for input
// Reference: AVZ avz_card.cpp _ChooseSingleCard
// Must wait for OrizontalScreenOffset == 4250
bool IsCardSelectReady() {
    if (GetGameUI() != 2) return false;
    
    // Check SelectCardUi_p at PvzBase + 0x774 (used by ChooseCard)
    uintptr_t base = GetBase();
    if (!base) return false;
    uintptr_t selectCardUi_p = *(uintptr_t*)(base + 0x774);
    if (!selectCardUi_p) return false;
    
    // Also check SelectCardUi_m at Board + 0x15c
    uintptr_t board = GetBoard();
    if (!board) return false;
    uintptr_t selectCardUi_m = *(uintptr_t*)(board + 0x15c);
    if (!selectCardUi_m) return false;
    
    // OrizontalScreenOffset at SelectCardUi_m + 0x8
    int offset = *(int*)(selectCardUi_m + 0x8);
    
    // Must be 4250 to be ready
    return offset == 4250;
}

bool IsZombieInHouse() {
    uintptr_t board = GetBoard();
    if (!board) return false;
    
    uintptr_t zombieArray = *(uintptr_t*)(board + Addr::ZOMBIE_ARRAY);
    int zombieMax = *(int*)(board + Addr::ZOMBIE_COUNT_MAX);
    
    if (!zombieArray || zombieMax <= 0) return false;
    if (zombieMax > 200) zombieMax = 200;
    
    for (int i = 0; i < zombieMax; i++) {
        uintptr_t addr = zombieArray + i * Addr::ZOMBIE_SIZE;
        bool dead = *(bool*)(addr + Addr::Z_DEAD);
        if (dead) continue;
        
        float x = *(float*)(addr + Addr::Z_X);
        if (x < -50.0f) {
            return true;
        }
    }
    return false;
}

// GridToAbscissa - Convert grid to pixel X
int GridToAbscissa(int row, int col) {
    int result = 0;
    uintptr_t board = GetBoard();
    if (!board) return 0;
    
    __asm {
        mov ecx, board
        mov eax, col
        mov esi, row
        mov edx, 0x41C680
        call edx
        mov result, eax
    }
    return result;
}

// GridToOrdinate - Convert grid to pixel Y
int GridToOrdinate(int row, int col) {
    int result = 0;
    uintptr_t board = GetBoard();
    if (!board) return 0;
    
    __asm {
        mov ebx, board
        mov ecx, col
        mov eax, row
        mov edx, 0x41C740
        call edx
        mov result, eax
    }
    return result;
}

// PutPlant - Place a plant at grid position
// Reference: AsmVsZombies/src/avz_asm.cpp - AAsm::PutPlant
bool PutPlant(int row, int col, int plantType) {
    int imitatorType = -1;
    
    __asm {
        push imitatorType
        push plantType
        mov eax, row
        push col
        mov esi, 0x6A9EC0
        mov esi, [esi]
        mov edi, [esi + 0x768]
        push edi
        mov edx, 0x40D120
        call edx
    }
    
    return true;
}

// RemovePlant - Remove a plant object
void RemovePlant(uintptr_t plant) {
    if (!plant) return;
    
    __asm {
        push plant
        mov edx, 0x4679B0
        call edx
    }
}

// Shovel - Remove plant at grid position
bool Shovel(int row, int col) {
    uintptr_t board = GetBoard();
    if (!board) return false;
    
    int x = GridToAbscissa(row, col);
    int y = GridToOrdinate(row, col);
    
    __asm {
        push 6
        push 1
        mov ecx, y
        mov edx, x
        mov eax, board
        mov ebx, 0x411060
        call ebx
    }
    
    return true;
}

// ReleaseMouse - Release mouse cursor from held item
void ReleaseMouse() {
    uintptr_t board = GetBoard();
    if (!board) return;
    
    __asm {
        mov eax, board
        mov edx, 0x40CD80
        call edx
    }
}

// FireCob - Fire a cob cannon at position
bool FireCob(int x, int y, int cobRank) {
    uintptr_t board = GetBoard();
    if (!board) return false;
    
    uintptr_t plantArray = *(uintptr_t*)(board + Addr::PLANT_ARRAY);
    if (!plantArray) return false;
    
    uintptr_t cobPlant = plantArray + cobRank * Addr::PLANT_SIZE;
    
    __asm {
        mov eax, cobPlant
        push y
        push x
        mov edx, 0x466D50
        call edx
    }
    
    return true;
}

// GetCardMoveState - Get card move state
// Reference: AVZ avz_pvz_struct.h CardMoveState
// 0=moving to slot, 1=in slot, 2=moving down, 3=in choose panel
int GetCardMoveState(int cardType) {
    uintptr_t base = GetBase();
    if (!base) return -1;
    uintptr_t seedChooser = *(uintptr_t*)(base + 0x774);
    if (!seedChooser) return -1;
    
    // CardMoveState at seedChooser + 0xC8 + cardType * 0x3C
    int idx = (cardType > 47) ? 48 : cardType;  // 模仿者用48
    return *(int*)(seedChooser + 0xC8 + idx * 0x3C);
}

// IsCardInSlot - Check if card is already in slot
bool IsCardInSlot(int cardType) {
    int state = GetCardMoveState(cardType);
    return (state == 0 || state == 1);  // 正在移动到卡槽或已在卡槽
}

// ClearSelectedCards - Click to deselect cards from slot
// Reference: AVZ avz_card.cpp click to cancel selection
bool ClearSelectedCards() {
    if (!IsCardSelectReady()) return false;
    
    uintptr_t base = GetBase();
    if (!base) return false;
    uintptr_t mouseWindow = *(uintptr_t*)(base + 0x320);
    if (!mouseWindow) return false;
    
    // 点击选卡区域左上角来取消已选卡片
    // Reference: AVZ Click(mouseWindow, 100, 50, 1)
    __asm {
        push 1            // button = 1 (left click)
        push 50           // y = 50
        push 100          // x = 100  
        push mouseWindow
        mov edx, 0x539390
        call edx
    }
    return true;
}

// ChooseCard - Select a card during card selection screen
// Reference: AVZ AAsm::ChooseCard
// cardType * 15 * 4 = cardType * 60 = cardType * 0x3C
bool ChooseCard(int cardType) {
    // Check if card selection screen is ready (animation complete)
    if (!IsCardSelectReady()) return false;
    
    // Check SeedChooser exists
    uintptr_t base = GetBase();
    if (!base) return false;
    uintptr_t seedChooser = *(uintptr_t*)(base + 0x774);
    if (!seedChooser) return false;
    
    // 检查卡片是否已经在卡槽中，如果是则跳过
    int state = GetCardMoveState(cardType);
    if (state == 0 || state == 1) {
        // 卡片已经在卡槽或正在移动，跳过
        return true;
    }
    
    // AVZ: seedChooser + 0xA4 + cardType * 0x3C
    int ct = cardType;
    __asm {
        mov eax, dword ptr ds:[0x6a9ec0]  // eax = PvzBase
        mov eax, [eax + 0x774]            // eax = SeedChooser (SelectCardUi_p)
        mov edx, ct
        shl edx, 4                         // edx = cardType * 16
        sub edx, ct                        // edx = cardType * 15  
        shl edx, 2                         // edx = cardType * 60
        add edx, 0xa4
        add edx, eax                       // edx = SeedChooser + 0xA4 + cardType * 60
        push edx
        mov ecx, 0x486030
        call ecx
    }
    return true;
}

// ChooseImitatorCard - Select an imitator card
// Reference: AVZ AAsm::ChooseImitatorCard
bool ChooseImitatorCard(int cardType) {
    // Check if card selection screen is ready
    if (!IsCardSelectReady()) return false;
    
    // Check SeedChooser exists
    uintptr_t base = GetBase();
    if (!base) return false;
    uintptr_t seedChooser = *(uintptr_t*)(base + 0x774);
    if (!seedChooser) return false;
    
    // 检查模仿者卡片是否已经在卡槽中（模仿者用48）
    int state = GetCardMoveState(48);  // 模仿者的cardType固定是48
    if (state == 0 || state == 1) {
        // 模仿者卡片已经在卡槽，跳过
        return true;
    }
    
    int ct = cardType;
    __asm {
        mov eax, dword ptr ds:[0x6a9ec0]  // eax = PvzBase
        mov eax, [eax + 0x774]            // eax = SeedChooser
        
        // Set imitator mode
        mov dword ptr [eax + 0x0c08], 3
        mov edx, ct                    // Load cardType into register first
        mov dword ptr [eax + 0x0c18], edx  // Then store
        
        // Get imitator card position
        lea ecx, [eax + 0xbe4]
        mov edx, [eax + 0xa0]
        mov ebx, [edx + 0x8]
        mov [ecx], ebx
        mov ebx, [edx + 0xc]
        mov [ecx + 0x4], ebx
        
        push eax
        push ecx
        mov edx, 0x486030
        call edx
        mov edx, 0x4866e0
        call edx
    }
    return true;
}

// Rock - Confirm card selection and start game (Let's Rock!)
// Reference: AVZ AAsm::PickRandomSeeds (not Rock!)
// AVZ actually calls PickRandomSeeds at 0x4859b0 to start the game
bool Rock() {
    // Check if in card selection screen (GameUI == 2)
    if (GetGameUI() != 2) return false;
    
    // Check SeedChooser exists
    uintptr_t base = GetBase();
    if (!base) return false;
    uintptr_t seedChooser = *(uintptr_t*)(base + 0x774);
    if (!seedChooser) return false;
    
    // AVZ uses PickRandomSeeds (0x4859b0), not Rock (0x486d20)
    // movl 0x6a9ec0, %%eax
    // movl 0x774(%%eax), %%eax
    // pushl %%eax
    // movl $0x4859b0, %%ecx
    // call *%%ecx
    
    __asm {
        mov eax, dword ptr ds:[0x6a9ec0]  // eax = PvzBase
        mov eax, [eax + 0x774]            // eax = SeedChooser
        push eax
        mov ecx, 0x4859b0
        call ecx
    }
    return true;
}

// PlantCard - Use a card from hand at position
bool PlantCard(int x, int y, int index) {
    uintptr_t board = GetBoard();
    if (!board) return false;
    
    uintptr_t seedBank = *(uintptr_t*)(board + 0x144);
    if (!seedBank) return false;
    
    uintptr_t cardAddr = seedBank + index * 0x50 + 0x28;
    
    __asm {
        push y
        push x
        push board
        push cardAddr
        mov ecx, 0x488590
        call ecx
        mov ecx, 1
        mov edx, 0x40FD30
        call edx
    }
    
    return true;
}

// MakeNewBoard - Create a new game board
bool MakeNewBoard() {
    uintptr_t base = GetBase();
    if (!base) return false;
    
    // Save current scene
    uintptr_t board = *(uintptr_t*)(base + Addr::MAIN_OBJECT);
    int scene = board ? *(int*)(board + Addr::SCENE) : 0;
    
    __asm {
        mov ecx, base
        mov eax, 0x44F5F0
        call eax
    }
    
    __asm {
        push base
        mov eax, 0x5518F0
        call eax
    }
    
    // Restore scene
    board = *(uintptr_t*)(base + Addr::MAIN_OBJECT);
    if (board) {
        *(int*)(board + Addr::SCENE) = scene;
    }
    
    return true;
}

// CheckFightExit - Exit from fight/game
bool CheckFightExit() {
    uintptr_t base = GetBase();
    if (!base) return false;
    
    __asm {
        mov eax, base
        mov ecx, 0x4524F0
        call ecx
    }
    
    return true;
}

// BackToMain - Return to main menu
// Reference: AVZ AAsm::DoBackToMain() uses 0x44feb0
bool BackToMain() {
    uintptr_t base = GetBase();
    if (!base) return false;
    
    // Check if in playing state (GameUI == 3)
    int gameUI = (int)GetGameUI();
    if (gameUI != 3) return false;
    
    // Call the correct function: 0x44feb0 (not 0x4524f0)
    __asm {
        mov eax, dword ptr ds:[0x6a9ec0]
        mov ecx, 0x44feb0
        call ecx
    }
    
    return true;
}

// CloseOptionsScreen - Close options/challenge screen (UI=7)
// Reference: AVZ uses 0x44fd00 to delete options screen
bool CloseOptionsScreen() {
    uintptr_t base = GetBase();
    if (!base) return false;
    
    // Check if in options screen (GameUI == 7)
    int gameUI = (int)GetGameUI();
    if (gameUI != 7) return false;
    
    // Call DeleteOptions: 0x44fd00
    __asm {
        mov esi, dword ptr ds:[0x6a9ec0]
        mov eax, 0x44fd00
        call eax
    }
    
    return true;
}

// GetLevel - Get current adventure level (1-50)
int GetLevel() {
    uintptr_t base = GetBase();
    if (!base) return 0;
    
    // PlayerInfo at base + 0x82C
    uintptr_t playerInfo = *(uintptr_t*)(base + 0x82C);
    if (!playerInfo) return 0;
    
    // mLevel at PlayerInfo + 0x24
    return *(int*)(playerInfo + 0x24);
}

// SetLevel - Set adventure level (1-50)
bool SetLevel(int level) {
    if (level < 1 || level > 50) return false;
    
    uintptr_t base = GetBase();
    if (!base) return false;
    
    // PlayerInfo at base + 0x82C
    uintptr_t playerInfo = *(uintptr_t*)(base + 0x82C);
    if (!playerInfo) return 0;
    
    // mLevel at PlayerInfo + 0x24
    *(int*)(playerInfo + 0x24) = level;
    return true;
}

// UnlockAllPlants - Unlock all plants for adventure mode
bool UnlockAllPlants() {
    uintptr_t base = GetBase();
    if (!base) return false;
    
    // PlayerInfo at base + 0x82C
    uintptr_t playerInfo = *(uintptr_t*)(base + 0x82C);
    if (!playerInfo) return false;
    
    // mFinishedAdventure at PlayerInfo + 0x2C
    // Set to 1 to unlock all plants
    *(int*)(playerInfo + 0x2C) = 1;
    
    // Also set level to 50 to ensure all plants unlocked
    *(int*)(playerInfo + 0x24) = 50;
    
    return true;
}

// GetGameMode - Get current game mode
int GetGameMode() {
    uintptr_t base = GetBase();
    if (!base) return -1;
    
    // mGameMode at base + 0x7F8
    return *(int*)(base + 0x7F8);
}

// SetGameMode - Set game mode
bool SetGameMode(int mode) {
    uintptr_t base = GetBase();
    if (!base) return false;
    
    // mGameMode at base + 0x7F8
    *(int*)(base + 0x7F8) = mode;
    return true;
}

// StartGame - Start a new game with specified mode
// Reference: AVZ avz_asm.cpp AAsm::EnterGame
// Must properly clean up existing UI before calling PreNewGame
namespace {
void ExecuteStartGame(int mode) {
    // Check GameUI state - reference AVZ
    int gameUi = PVZ::GetGameUI();
    
    // If already in card selection (2) or playing (3), don't enter
    // AVZ: LEVEL_INTRO = 2, PLAYING = 3
    if (gameUi == 2 || gameUi == 3) return;
    
    // If in loading screen (0) or main menu (1)
    if (gameUi == 0 || gameUi == 1) {
        // If in loading screen (0), delete it first - call 0x452cb0
        if (gameUi == 0) {
            __asm {
                mov ecx, dword ptr ds:[0x6a9ec0]
                mov eax, 0x452cb0
                call eax
            }
        }
        
        // Delete GameSelector - call 0x44f9e0 (KillGameSelector)
        __asm {
            mov esi, dword ptr ds:[0x6a9ec0]
            mov eax, 0x44f9e0
            call eax
        }
    }
    
    // If in options/challenge screen (7), delete it - call 0x44fd00
    if (gameUi == 7) {
        __asm {
            mov esi, dword ptr ds:[0x6a9ec0]
            mov eax, 0x44fd00
            call eax
        }
    }
    
    // NOTE: UI=4 (ZOMBIES_WON) and UI=5 (AWARD) have dialogs that must be
    // closed via mouse click before we can proceed. Handle this in Python.
    if (gameUi == 4 || gameUi == 5) {
        // Cannot handle these states directly - Python should click dialog first
        return;
    }
    
    // Now call PreNewGame(theGameMode, theLookForSavedGame) at 0x44F560
    // AVZ pushes: ok(1), gameMode - stack order is [gameMode, ok] when function reads
    // PreNewGame signature: void PreNewGame(GameMode theGameMode, bool theLookForSavedGame)
    // C++ calling convention pushes right-to-left, so push order: theLookForSavedGame, theGameMode
    int lookForSaved = 0;  // false - don't look for saved game
    int gameMode = mode;
    
    __asm {
        push lookForSaved    // theLookForSavedGame = false
        push gameMode        // theGameMode
        mov esi, dword ptr ds:[0x6a9ec0]  // this pointer in ESI (AVZ style)
        mov eax, 0x44F560
        call eax
    }
}
}

bool StartGame(int mode) {
    uintptr_t base = GetBase();
    if (!base) return false;
    
    *(int*)(base + 0x7F8) = mode;
    g_startGameMode.store(mode, std::memory_order_relaxed);
    g_startGamePending.store(true, std::memory_order_release);
    return true;
}

void ProcessPendingActions() {
    if (!g_startGamePending.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    int mode = g_startGameMode.load(std::memory_order_relaxed);
    ExecuteStartGame(mode);
}

// ============================================================================
// Mouse Simulation Functions - Based on AVZ's AAsm::MouseClick/MouseDown/MouseUp
// These use game's internal mouse handling, not Windows mouse events
// ============================================================================

// Mouse down at position (game coordinates)
// Reference: AVZ AAsm::MouseDown at 0x539390
void MouseDown(int x, int y, int key) {
    uintptr_t base = GetBase();
    if (!base) return;
    
    uintptr_t mouseWindow = *(uintptr_t*)(base + Addr::MOUSE_WINDOW);
    if (!mouseWindow) return;
    
    __asm {
        push x
        mov eax, y
        mov ebx, key
        mov ecx, mouseWindow
        mov edx, 0x539390
        call edx
    }
}

// Mouse up at position (game coordinates)
// Reference: AVZ AAsm::MouseUp at 0x5392e0
void MouseUp(int x, int y, int key) {
    uintptr_t base = GetBase();
    if (!base) return;
    
    uintptr_t mouseWindow = *(uintptr_t*)(base + Addr::MOUSE_WINDOW);
    if (!mouseWindow) return;
    
    __asm {
        push key
        push x
        mov eax, mouseWindow
        mov ebx, y
        mov edx, 0x5392e0
        call edx
    }
}

// Full mouse click (down + up)
// Reference: AVZ AAsm::MouseClick
void MouseClick(int x, int y, int key) {
    MouseDown(x, y, key);
    MouseUp(x, y, key);
}

// Left click at game coordinates
void LeftClick(int x, int y) {
    MouseClick(x, y, 1);  // 1 = left button
}

// ============================================================================
// Window Size Functions
// ============================================================================

// Get window width - Reference: PvzBase + 0xC0
int GetWindowWidth() {
    uintptr_t base = GetBase();
    if (!base) return 800;  // Default
    return *(int*)(base + 0xC0);
}

// Get window height - Reference: PvzBase + 0xC4
int GetWindowHeight() {
    uintptr_t base = GetBase();
    if (!base) return 600;  // Default
    return *(int*)(base + 0xC4);
}

// ============================================================================
// Item Collection - Based on AVZ's AItemCollector
// ============================================================================

// Item structure offsets
namespace ItemAddr {
    constexpr uintptr_t SIZE = 0xD8;
    constexpr uintptr_t DISAPPEARED = 0x38;
    constexpr uintptr_t COLLECTED = 0x50;
    constexpr uintptr_t TYPE = 0x58;
    constexpr uintptr_t X = 0x24;  // Abscissa (float)
    constexpr uintptr_t Y = 0x28;  // Ordinate (float)
}

// Board item array offsets
namespace BoardItemAddr {
    constexpr uintptr_t ARRAY = 0xE4;
    constexpr uintptr_t COUNT_MAX = 0xE8;
}

// Collect all items by simulating clicks
// Returns: number of items collected
int CollectAllItems() {
    uintptr_t board = GetBoard();
    if (!board) return 0;
    
    uintptr_t itemArray = *(uintptr_t*)(board + BoardItemAddr::ARRAY);
    int itemMax = *(int*)(board + BoardItemAddr::COUNT_MAX);
    
    int collected = 0;
    
    for (int i = 0; i < itemMax && i < 100; i++) {
        uintptr_t item = itemArray + i * ItemAddr::SIZE;
        
        // Check if item is alive (AVZ logic: !IsDisappeared && !IsCollected)
        bool isDisappeared = *(uint8_t*)(item + ItemAddr::DISAPPEARED) != 0;
        bool isCollected = *(uint8_t*)(item + ItemAddr::COLLECTED) != 0;
        
        if (isDisappeared || isCollected) continue;
        
        // Get item position
        float itemX = *(float*)(item + ItemAddr::X);
        float itemY = *(float*)(item + ItemAddr::Y);
        
        // Only collect if in valid range (AVZ checks x >= 0 && y >= 70)
        if (itemX >= 0.0f && itemY >= 70.0f) {
            int clickX = static_cast<int>(itemX + 30);
            int clickY = static_cast<int>(itemY + 30);
            LeftClick(clickX, clickY);
            collected++;
        }
    }
    
    return collected;
}

}  // namespace PVZ
