"""Gym-Macro-Overcooked Environment Registration.

This module registers the Overcooked environment variants with both
gym and gymnasium for compatibility.

Environments:
    - Overcooked-v1: Single-agent Overcooked environment
    - Overcooked-MA-v1: Multi-agent Overcooked environment
"""

# Support both gym and gymnasium
try:
    from gym.envs.registration import register as gym_register
    gym_register(
        id='Overcooked-v1',
        entry_point='gym_macro_overcooked.overcooked_V1:Overcooked_V1',
    )
    gym_register(
        id='Overcooked-MA-v1',
        entry_point='gym_macro_overcooked.overcooked_MA_V1:Overcooked_MA_V1',
    )
except ImportError:
    pass  # gym not available

try:
    from gymnasium.envs.registration import register as gymnasium_register
    gymnasium_register(
        id='Overcooked-v1',
        entry_point='gym_macro_overcooked.overcooked_V1:Overcooked_V1',
    )
    gymnasium_register(
        id='Overcooked-MA-v1',
        entry_point='gym_macro_overcooked.overcooked_MA_V1:Overcooked_MA_V1',
    )
except ImportError:
    pass  # gymnasium not available