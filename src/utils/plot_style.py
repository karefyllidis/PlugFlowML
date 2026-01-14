#!/usr/bin/env python3
"""
Plot Style Utilities
====================

Utilities for loading and applying consistent figure aesthetics from JSON configuration.

Author: Nikolas Karefyllidis, PhD
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, Any, Optional


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_aesthetics(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load figure aesthetics from JSON configuration file.
    
    Parameters:
    -----------
    config_file : str, optional
        Path to aesthetics config file. If None, uses default.
    
    Returns:
    --------
    dict
        Aesthetics configuration dictionary
    """
    if config_file is None:
        config_file = os.path.join(get_project_root(), 'styles', 'figure_aesthetics.json')
    
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
        'label': style['label']
    }
    plot_params.update(plot_kwargs)
    
    # Plot
    ax.plot(x, y, **plot_params)
    
    # Setup axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(style['ylabel'])
    ax.set_title(style['title'])
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
