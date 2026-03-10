"""Lightweight export-only package for single-turn attacks.

Convenience imports are provided to make usage straightforward:
- `deepteam_single.attacks.single_turn` exposes all single-turn attack classes.
- `deepteam_single.attacks.base_attack.BaseAttack` exposes the base attack ABC.
- `deepteam_single.attacks.attack_simulator` exposes `generate`, `a_generate`, and schemas.
"""

# Re-export subpackages for ergonomic imports
from . import attacks as attacks  # noqa: F401

__all__ = ["attacks"]