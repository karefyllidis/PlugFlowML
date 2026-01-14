#!/usr/bin/env python3
"""
Example: Using Figure Aesthetics
=================================

Demonstrates how to use the centralized aesthetics system.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from utils.plot_style import (
    load_aesthetics, apply_style, plot_profile, get_color,
    create_figure, get_profile_style, setup_axes, setup_legend, save_figure
)


def example_simple_plot():
    """Example: Simple plot using aesthetics."""
    print("Example 1: Simple plot with aesthetics")
    
    # Generate sample data
    z = np.linspace(0, 5, 200)
    temperature = 925 + 50 * np.sin(z * np.pi / 5)
    
    # Create plot with aesthetics
    fig, ax = plot_profile(
        z, temperature, 'temperature',
        output_path='outputs/figures/example_temperature.png'
    )
    plt.close()
    print("[OK] Saved example_temperature.png")


def example_custom_color():
    """Example: Override color for specific plot."""
    print("\nExample 2: Custom color override")
    
    z = np.linspace(0, 5, 200)
    pressure = 2.0 - 0.1 * z
    
    # Use aesthetics but override color
    fig, ax = plot_profile(
        z, pressure, 'pressure',
        color='#ff0000',  # Override to bright red
        output_path='outputs/figures/example_pressure_custom.png'
    )
    plt.close()
    print("[OK] Saved example_pressure_custom.png")


def example_multiple_profiles():
    """Example: Multiple profiles on one plot."""
    print("\nExample 3: Multiple profiles")
    
    # Load aesthetics
    aesthetics = load_aesthetics()
    apply_style(aesthetics)
    
    # Generate data
    z = np.linspace(0, 5, 200)
    temp = 925 + 50 * np.sin(z * np.pi / 5)
    pressure = 2.0 - 0.1 * z
    
    # Create figure
    fig = create_figure(aesthetics, figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    # Plot multiple profiles
    temp_style = get_profile_style('temperature', aesthetics)
    press_style = get_profile_style('pressure', aesthetics)
    
    ax.plot(z, temp, color=temp_style['color'], linewidth=temp_style['linewidth'],
            label=temp_style['label'])
    ax.plot(z, pressure, color=press_style['color'], linewidth=press_style['linewidth'],
            label=press_style['label'])
    
    ax.set_xlabel('$z$ [m]')
    ax.set_ylabel('Value')
    ax.set_title('Temperature and Pressure Profiles')
    setup_axes(ax, aesthetics)
    setup_legend(ax, aesthetics)
    plt.tight_layout()
    
    save_figure(fig, 'outputs/figures/example_multiple.png', aesthetics)
    plt.close()
    print("[OK] Saved example_multiple.png")


def example_custom_aesthetics():
    """Example: Load custom aesthetics file."""
    print("\nExample 4: Custom aesthetics file")
    
    # You can create your own aesthetics file and load it
    # aesthetics = load_aesthetics('styles/my_custom_aesthetics.json')
    # apply_style(aesthetics)
    
    print("Create your own aesthetics file in styles/ and load it!")


if __name__ == "__main__":
    print("="*60)
    print("Figure Aesthetics Examples")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs('outputs/figures', exist_ok=True)
    
    example_simple_plot()
    example_custom_color()
    example_multiple_profiles()
    example_custom_aesthetics()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Check outputs/figures/ for generated plots")
    print("="*60)
