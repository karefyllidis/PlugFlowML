"""
Utility modules for HydrAI.
"""

from .plot_style import (
    load_aesthetics,
    apply_style,
    create_figure,
    setup_axes,
    get_profile_style,
    get_color,
    save_figure,
    setup_legend,
    plot_profile
)
from .plot_parallel import plot_parallel_coordinates, plot_parallel_sets
from .plot_nn_architecture import draw_mlp_architecture, write_tikz_mlp
from .run_log import start_run_log, stop_run_log

__all__ = [
    'load_aesthetics',
    'apply_style',
    'create_figure',
    'setup_axes',
    'get_profile_style',
    'get_color',
    'save_figure',
    'setup_legend',
    'plot_profile',
    'plot_parallel_coordinates',
    'plot_parallel_sets',
    'draw_mlp_architecture',
    'write_tikz_mlp',
    'start_run_log',
    'stop_run_log',
]
