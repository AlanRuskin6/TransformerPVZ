"""
Projectile Data from AsmVsZombies
Contains projectile types, damage values, and speed constants

Reference: re-plants-vs-zombies Projectile.cpp
"""

from enum import IntEnum
from dataclasses import dataclass
from typing import Dict


class ProjectileType(IntEnum):
    """All projectile types in PVZ"""
    PEA = 0
    SNOW_PEA = 1
    CABBAGE = 2
    MELON = 3
    PUFF = 4
    WINTERMELON = 5
    FIRE_PEA = 6
    STAR = 7
    CACTUS = 8
    FUME = 9
    BASKETBALL = 10
    COB = 11
    BUTTER = 12
    KERNEL = 13


@dataclass
class ProjectileData:
    """Complete data for a projectile type"""
    type: ProjectileType
    name: str
    damage: int
    speed: float  # pixels per cs
    is_splash: bool  # has splash damage
    splash_radius: float  # splash damage radius in pixels
    slows: bool  # applies slow effect
    area_effect: bool  # affects multiple targets (fume)


# Projectile damage values
PROJECTILE_DAMAGE: Dict[ProjectileType, int] = {
    ProjectileType.PEA: 20,
    ProjectileType.SNOW_PEA: 20,
    ProjectileType.CABBAGE: 40,
    ProjectileType.MELON: 80,
    ProjectileType.PUFF: 20,
    ProjectileType.WINTERMELON: 80,
    ProjectileType.FIRE_PEA: 40,
    ProjectileType.STAR: 20,
    ProjectileType.CACTUS: 20,
    ProjectileType.FUME: 20,
    ProjectileType.BASKETBALL: 75,
    ProjectileType.COB: 1800,
    ProjectileType.BUTTER: 40,
    ProjectileType.KERNEL: 20,
}


# Projectile speed (pixels per cs)
PROJECTILE_SPEED: Dict[ProjectileType, float] = {
    ProjectileType.PEA: 3.7,
    ProjectileType.SNOW_PEA: 3.7,
    ProjectileType.CABBAGE: 3.0,  # parabolic
    ProjectileType.MELON: 3.0,  # parabolic
    ProjectileType.PUFF: 3.7,
    ProjectileType.WINTERMELON: 3.0,  # parabolic
    ProjectileType.FIRE_PEA: 3.7,
    ProjectileType.STAR: 1.7,
    ProjectileType.CACTUS: 3.7,
    ProjectileType.FUME: 0.0,  # instant (area effect)
    ProjectileType.BASKETBALL: 2.0,
    ProjectileType.COB: 0.0,  # special trajectory
    ProjectileType.BUTTER: 3.0,  # parabolic
    ProjectileType.KERNEL: 3.0,  # parabolic
}


# Splash damage radius for projectiles that have splash
PROJECTILE_SPLASH_RADIUS: Dict[ProjectileType, float] = {
    ProjectileType.MELON: 80.0,
    ProjectileType.WINTERMELON: 80.0,
    ProjectileType.COB: 115.0,
}


# Projectiles that slow zombies
SLOWING_PROJECTILES = {
    ProjectileType.SNOW_PEA,
    ProjectileType.WINTERMELON,
}


# Projectiles that have area/splash effect
SPLASH_PROJECTILES = {
    ProjectileType.MELON,
    ProjectileType.WINTERMELON,
    ProjectileType.COB,
}


# Projectiles that pierce through zombies
PIERCING_PROJECTILES = {
    ProjectileType.FUME,
    ProjectileType.STAR,
}


# Projectiles with parabolic trajectory
PARABOLIC_PROJECTILES = {
    ProjectileType.CABBAGE,
    ProjectileType.MELON,
    ProjectileType.WINTERMELON,
    ProjectileType.BUTTER,
    ProjectileType.KERNEL,
}


def get_projectile_damage(proj_type: ProjectileType) -> int:
    """Get damage value for a projectile type"""
    return PROJECTILE_DAMAGE.get(proj_type, 20)


def get_projectile_speed(proj_type: ProjectileType) -> float:
    """Get speed for a projectile type"""
    return PROJECTILE_SPEED.get(proj_type, 3.7)


def get_splash_radius(proj_type: ProjectileType) -> float:
    """Get splash radius for a projectile type (0 if none)"""
    return PROJECTILE_SPLASH_RADIUS.get(proj_type, 0.0)


def is_slowing_projectile(proj_type: ProjectileType) -> bool:
    """Check if projectile applies slow effect"""
    return proj_type in SLOWING_PROJECTILES


def is_splash_projectile(proj_type: ProjectileType) -> bool:
    """Check if projectile has splash damage"""
    return proj_type in SPLASH_PROJECTILES


def is_piercing_projectile(proj_type: ProjectileType) -> bool:
    """Check if projectile pierces through targets"""
    return proj_type in PIERCING_PROJECTILES
