"""
Configuration File
Contains all configurable settings for the PVZ bot

Time units: centiseconds (cs) = 1/100 second, unless otherwise noted.
For config values exposed to users, seconds are used for readability.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class BotConfig:
    """Bot configuration settings"""
    
    # ========================================================================
    # Path Settings
    # ========================================================================
    
    # Path to the game executable
    game_path: str = r"gameobj\PlantsVsZombies.exe"
    
    # Path to load a pre-trained model (None = start from scratch)
    model_load_path: Optional[str] = None
    
    # Path to save the trained model
    model_save_path: str = "models/latest_model.zip"
    
    # ========================================================================
    # General Settings
    # ========================================================================
    
    # Game speed multiplier (e.g., 2.0 for double speed)
    game_speed: float = 1.0
    
    # Whether to automatically plant
    auto_plant: bool = True
    
    # Whether to automatically collect sun
    auto_collect_sun: bool = True
    
    # Interval between actions (seconds, converted to cs internally)
    action_interval: float = 0.15
    
    # Main loop refresh rate (seconds, converted to cs internally)
    refresh_rate: float = 0.05
    
    # ========================================================================
    # Debug Settings
    # ========================================================================
    
    # Enable debug output
    debug: bool = False
    
    # Log level (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
    log_level: int = 1
    
    # Log to file path (None = console only)
    log_file: Optional[str] = None
    
    # ========================================================================
    # Economy Settings
    # ========================================================================
    
    # Target number of sun-producing plant units
    target_sun_plants: int = 8
    
    # Columns to prioritize for sunflowers
    sunflower_columns: List[int] = field(default_factory=lambda: [0, 1])
    
    # ========================================================================
    # Defense Settings
    # ========================================================================
    
    # Column for defensive plants (walls)
    defense_column: int = 4
    
    # X coordinate considered dangerous (pixels)
    danger_x: int = 200
    
    # X coordinate considered critical (pixels)
    critical_x: int = 100
    
    # ========================================================================
    # Optimizer Settings
    # ========================================================================
    
    # Weights for action evaluation
    threat_weight: float = 2.0
    efficiency_weight: float = 1.0
    strategic_weight: float = 1.5
    urgency_weight: float = 3.0
    
    # ========================================================================
    # Strategy Settings
    # ========================================================================
    
    # Rows to manage (depends on scene type)
    row_count: int = 5
    
    # Emergency threat threshold
    emergency_threshold: float = 8.0
    
    # High threat threshold
    high_threat_threshold: float = 5.0
    
    # ========================================================================
    # Time Conversion Helpers
    # ========================================================================
    
    @property
    def action_interval_cs(self) -> int:
        """Get action interval in centiseconds"""
        return int(self.action_interval * 100)
    
    @property
    def refresh_rate_cs(self) -> int:
        """Get refresh rate in centiseconds"""
        return int(self.refresh_rate * 100)


# Default configuration
DEFAULT_CONFIG = BotConfig()


def load_config(config_path: Optional[str] = None) -> BotConfig:
    """
    Load configuration from file
    
    Supports JSON format. If no path provided, looks for config.json
    in the current directory. Falls back to default config if file not found.
    
    Args:
        config_path: Path to config file (JSON)
        
    Returns:
        BotConfig instance
    """
    # Default config path
    if config_path is None:
        config_path = "config.json"
    
    # Check if file exists
    if not os.path.exists(config_path):
        return BotConfig()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create config from loaded data
        # Only use keys that exist in BotConfig
        valid_keys = {f.name for f in BotConfig.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        return BotConfig(**filtered_data)
    except (json.JSONDecodeError, TypeError, IOError) as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return BotConfig()


def save_config(config: BotConfig, config_path: str) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration to save
        config_path: Path to save config file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        data = asdict(config)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"Warning: Failed to save config to {config_path}: {e}")
        return False
