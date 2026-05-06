#!/usr/bin/env python3
"""
Plot Style Utilities
====================

Utilities for loading and applying consistent figure aesthetics from JSON configuration.

Author: Nikolas Karefyllidis, PhD
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, Any, Optional, Union


def setup_matplotlib(ax: Union[None, "matplotlib.axes.Axes", np.ndarray] = None) -> None:
    """
    Apply the project's standard matplotlib style globally and, optionally,
    to one or more Axes objects.

    Call once per notebook/script at import time (no arguments) to set rcParams.
    Pass ``ax`` after creating subplots to apply per-axis finishing touches
    (minor ticks, grid style, tick direction).

    Parameters
    ----------
    ax : None | Axes | ndarray of Axes
        If None, only global rcParams are updated.
        If an Axes or array of Axes (e.g. from ``plt.subplots``), per-axis
        styling is also applied.

    Examples
    --------
    >>> from src.utils.plot_style import setup_matplotlib
    >>> setup_matplotlib()                          # global style only
    >>> fig, axes = plt.subplots(1, 3)
    >>> setup_matplotlib(axes)                      # global + per-axis
    """
    plt.rcParams.update({
        # Figure
        "figure.figsize":       (24, 6),
        "figure.dpi":           120,
        "savefig.dpi":          200,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.02,

        # Fonts
        "font.size":            10,
        "axes.labelsize":       10,
        "axes.titlesize":       10,
        "legend.fontsize":      10,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,

        # Lines
        "lines.linewidth":      1.2,
        "lines.markersize":     4,

        # Axes
        "axes.linewidth":       0.8,
        "axes.spines.top":      False,
        "axes.spines.right":    False,

        # Colors
        "text.color":           "black",
        "axes.labelcolor":      "black",
        "xtick.color":          "black",
        "ytick.color":          "black",

        # Ticks
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.major.size":     6,
        "ytick.major.size":     6,
        "xtick.minor.size":     4,
        "ytick.minor.size":     4,
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "xtick.minor.width":    0.5,
        "ytick.minor.width":    0.5,
        "xtick.major.pad":      6,
        "ytick.major.pad":      6,
        "xtick.minor.pad":      4,
        "ytick.minor.pad":      4,

        # Legend
        "legend.frameon":       False,

        # Font export (embed fonts in PDF/PS)
        "pdf.fonttype":         42,
        "ps.fonttype":          42,

        # TeX
        "text.usetex":          False,
    })

    if ax is not None:
        axes_list = list(ax.flat) if hasattr(ax, "flat") else [ax]
        for a in axes_list:
            a.set_axisbelow(True)
            a.grid(False, linestyle="--", linewidth=0.65, color="gray", alpha=0.75)
            a.tick_params(
                which="both", direction="in",
                top=False, bottom=True, left=True, right=False,
            )
            a.minorticks_on()


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def default_figure_aesthetics_path() -> str:
    """Prefer ``configs/style/figure_aesthetics.json``; fall back to flat ``configs/`` or ``styles/``."""
    root = get_project_root()
    preferred = os.path.join(root, 'configs', 'style', 'figure_aesthetics.json')
    legacy_flat = os.path.join(root, 'configs', 'figure_aesthetics.json')
    legacy_styles = os.path.join(root, 'styles', 'figure_aesthetics.json')
    if os.path.isfile(preferred):
        return preferred
    if os.path.isfile(legacy_flat):
        return legacy_flat
    if os.path.isfile(legacy_styles):
        return legacy_styles
    return preferred


def load_aesthetics(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load figure aesthetics from JSON configuration file.
    
    Parameters:
    -----------
    config_file : str, optional
        Path to aesthetics config file. If None, uses ``configs/style/figure_aesthetics.json``,
        then legacy flat ``configs/figure_aesthetics.json`` or ``styles/figure_aesthetics.json``.
    
    Returns:
    --------
    dict
        Aesthetics configuration dictionary
    """
    if config_file is None:
        config_file = default_figure_aesthetics_path()
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Aesthetics config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        aesthetics = json.load(f)
    
    return aesthetics


def apply_style(aesthetics: Optional[Dict[str, Any]] = None, config_file: Optional[str] = None):
    """
    Apply global matplotlib style from aesthetics configuration.
    
    Parameters:
    -----------
    aesthetics : dict, optional
        Aesthetics dictionary. If None, loads from config_file.
    config_file : str, optional
        Path to aesthetics config file. Used if aesthetics is None.
    """
    if aesthetics is None:
        aesthetics = load_aesthetics(config_file)
    
    # Apply font settings
    font_config = aesthetics.get('font', {})
    plt.rcParams['font.family'] = font_config.get('family', 'sans-serif')
    plt.rcParams['font.size'] = font_config.get('size', 12)
    plt.rcParams['font.weight'] = font_config.get('weight', 'normal')
    
    # Apply grid settings
    axes_config = aesthetics.get('axes', {})
    plt.rcParams['axes.grid'] = axes_config.get('grid', True)
    plt.rcParams['grid.alpha'] = axes_config.get('grid_alpha', 0.3)
    plt.rcParams['grid.linestyle'] = axes_config.get('grid_style', '-')
    plt.rcParams['grid.color'] = axes_config.get('grid_color', 'gray')
    
    # Apply figure defaults
    figure_config = aesthetics.get('figure', {})
    plt.rcParams['figure.dpi'] = figure_config.get('dpi', 150)
    plt.rcParams['figure.facecolor'] = figure_config.get('facecolor', 'white')
    plt.rcParams['figure.edgecolor'] = figure_config.get('edgecolor', 'none')


def create_figure(aesthetics: Optional[Dict[str, Any]] = None, 
                  figsize: Optional[tuple] = None,
                  config_file: Optional[str] = None):
    """
    Create a figure with aesthetics applied.
    
    Parameters:
    -----------
    aesthetics : dict, optional
        Aesthetics dictionary. If None, loads from config_file.
    figsize : tuple, optional
        Figure size (width, height). If None, uses aesthetics default.
    config_file : str, optional
        Path to aesthetics config file. Used if aesthetics is None.
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with aesthetics applied
    """
    if aesthetics is None:
        aesthetics = load_aesthetics(config_file)
    
    if figsize is None:
        figsize = tuple(aesthetics.get('figure', {}).get('figsize', [10, 6]))
    
    fig = plt.figure(figsize=figsize)
    
    return fig


def setup_axes(ax, aesthetics: Optional[Dict[str, Any]] = None,
               config_file: Optional[str] = None):
    """
    Setup axes with aesthetics.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes object to configure
    aesthetics : dict, optional
        Aesthetics dictionary. If None, loads from config_file.
    config_file : str, optional
        Path to aesthetics config file. Used if aesthetics is None.
    """
    if aesthetics is None:
        aesthetics = load_aesthetics(config_file)
    
    axes_config = aesthetics.get('axes', {})
    
    # Grid
    if axes_config.get('grid', True):
        ax.grid(True, alpha=axes_config.get('grid_alpha', 0.3))
    
    # Spines
    if axes_config.get('spines_top', False) is False:
        ax.spines['top'].set_visible(False)
    if axes_config.get('spines_right', False) is False:
        ax.spines['right'].set_visible(False)
    
    # Set xlim left to 0 for most plots
    ax.set_xlim(left=0)


def get_profile_style(profile_name: str, 
                     aesthetics: Optional[Dict[str, Any]] = None,
                     config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Get style configuration for a specific profile.
    
    Parameters:
    -----------
    profile_name : str
        Name of the profile (e.g., 'temperature', 'pressure')
    aesthetics : dict, optional
        Aesthetics dictionary. If None, loads from config_file.
    config_file : str, optional
        Path to aesthetics config file. Used if aesthetics is None.
    
    Returns:
    --------
    dict
        Style dictionary for the profile
    """
    if aesthetics is None:
        aesthetics = load_aesthetics(config_file)
    
    profiles = aesthetics.get('profiles', {})
    profile_config = profiles.get(profile_name, {})
    
    # Merge with default line style
    line_config = aesthetics.get('line', {})
    style = {
        'color': profile_config.get('color', line_config.get('color', '#1f77b4')),
        'linewidth': line_config.get('width', 2),
        'linestyle': line_config.get('style', '-'),
        'alpha': line_config.get('alpha', 1.0),
        'marker': line_config.get('marker', 'none'),
        'markersize': line_config.get('markersize', 6),
        'label': profile_config.get('label', profile_name.title()),
        'ylabel': profile_config.get('ylabel', 'Value'),
        'title': profile_config.get('title', f'{profile_name.title()} Profile')
    }
    if 'colors' in profile_config:
        style['colors'] = list(profile_config['colors'])
    
    return style


def get_color(name: str, 
             aesthetics: Optional[Dict[str, Any]] = None,
             config_file: Optional[str] = None) -> str:
    """
    Get color by name from aesthetics configuration.
    
    Parameters:
    -----------
    name : str
        Color name (e.g., 'temperature', 'primary', 'pressure')
    aesthetics : dict, optional
        Aesthetics dictionary. If None, loads from config_file.
    config_file : str, optional
        Path to aesthetics config file. Used if aesthetics is None.
    
    Returns:
    --------
    str
        Color code (hex or name)
    """
    if aesthetics is None:
        aesthetics = load_aesthetics(config_file)
    
    colors = aesthetics.get('colors', {})
    return colors.get(name, '#1f77b4')  # Default to primary blue


def save_figure(fig, filename: str,
               aesthetics: Optional[Dict[str, Any]] = None,
               config_file: Optional[str] = None):
    """
    Save figure with aesthetics settings.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename
    aesthetics : dict, optional
        Aesthetics dictionary. If None, loads from config_file.
    config_file : str, optional
        Path to aesthetics config file. Used if aesthetics is None.
    """
    if aesthetics is None:
        aesthetics = load_aesthetics(config_file)
    
    save_config = aesthetics.get('save', {})
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    fig.savefig(
        filename,
        format=save_config.get('format', 'png'),
        dpi=save_config.get('dpi', 150),
        bbox_inches=save_config.get('bbox_inches', 'tight'),
        facecolor=save_config.get('facecolor', 'white'),
        edgecolor=save_config.get('edgecolor', 'none'),
        transparent=save_config.get('transparent', False)
    )


def setup_legend(ax, aesthetics: Optional[Dict[str, Any]] = None,
                config_file: Optional[str] = None, **kwargs):
    """
    Setup legend with aesthetics.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes object
    aesthetics : dict, optional
        Aesthetics dictionary. If None, loads from config_file.
    config_file : str, optional
        Path to aesthetics config file. Used if aesthetics is None.
    **kwargs
        Additional arguments to pass to legend()
    
    Returns:
    --------
    matplotlib.legend.Legend
        Legend object
    """
    if aesthetics is None:
        aesthetics = load_aesthetics(config_file)
    
    legend_config = aesthetics.get('legend', {})
    
    if not legend_config.get('show', True):
        return None
    
    # Merge kwargs with config
    legend_kwargs = {
        'loc': kwargs.get('loc', legend_config.get('location', 'best')),
        'frameon': kwargs.get('frameon', legend_config.get('frameon', True)),
        'fancybox': kwargs.get('fancybox', legend_config.get('fancybox', True)),
        'shadow': kwargs.get('shadow', legend_config.get('shadow', False)),
        'ncol': kwargs.get('ncol', legend_config.get('ncol', 1)),
        'fontsize': kwargs.get('fontsize', legend_config.get('fontsize', 10)),
        'framealpha': kwargs.get('framealpha', legend_config.get('framealpha', 0.9))
    }
    
    # Override with any provided kwargs
    legend_kwargs.update(kwargs)
    
    return ax.legend(**legend_kwargs)


def plot_profile(x, y, profile_name: str,
                xlabel: str = '$z$ [m]',
                aesthetics: Optional[Dict[str, Any]] = None,
                config_file: Optional[str] = None,
                output_path: Optional[str] = None,
                ylabel: Optional[str] = None,
                title: Optional[str] = None,
                label: Optional[str] = None,
                **plot_kwargs):
    """
    Create a profile plot with aesthetics applied.
    
    Parameters:
    -----------
    x : array-like
        X-axis data
    y : array-like
        Y-axis data
    profile_name : str
        Name of the profile (e.g., 'temperature', 'pressure')
    xlabel : str
        X-axis label
    aesthetics : dict, optional
        Aesthetics dictionary. If None, loads from config_file.
    config_file : str, optional
        Path to aesthetics config file. Used if aesthetics is None.
    output_path : str, optional
        Path to save figure. If None, figure is not saved.
    ylabel, title, label : optional
        Override values from ``profiles`` in the JSON (e.g. species-specific labels).
    **plot_kwargs
        Additional arguments to pass to plot()
    
    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    if aesthetics is None:
        aesthetics = load_aesthetics(config_file)
    
    # Get profile style
    style = get_profile_style(profile_name, aesthetics)
    
    # Create figure
    fig = create_figure(aesthetics)
    ax = fig.add_subplot(111)
    
    # Merge style with plot_kwargs (plot_kwargs take precedence)
    plot_params = {
        'color': style['color'],
        'linewidth': style['linewidth'],
        'linestyle': style['linestyle'],
        'alpha': style['alpha'],
        'marker': style['marker'],
        'markersize': style['markersize'],
        'label': label if label is not None else style['label']
    }
    plot_params.update(plot_kwargs)
    if plot_params.get('marker') in (None, 'none', 'None'):
        plot_params.pop('marker', None)
        plot_params.pop('markersize', None)
    
    # Plot
    ax.plot(x, y, **plot_params)
    
    # Setup axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(style['ylabel'] if ylabel is None else ylabel)
    ax.set_title(style['title'] if title is None else title)
    setup_axes(ax, aesthetics)
    setup_legend(ax, aesthetics)
    
    # Apply layout
    layout_config = aesthetics.get('layout', {})
    if layout_config.get('tight_layout', True):
        plt.tight_layout()
    
    # Save if path provided
    if output_path:
        save_figure(fig, output_path, aesthetics)
    
    return fig, ax
