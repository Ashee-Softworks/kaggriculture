"""
kaggle_environments — vendored to the single function the environment needs.

Only `resolve_episode_seed` is provided. The real package is Apache-2.0; see NOTICE.
"""

from .utils import resolve_episode_seed

__all__ = ["resolve_episode_seed"]
