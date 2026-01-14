# Figure Aesthetics Configuration

This folder contains centralized styling configuration for all generated figures.

## Purpose

Maintain consistent, professional aesthetics across all plots by centralizing:
- Colors
- Fonts and sizes
- Line styles
- Grid settings
- Legend formatting
- Figure dimensions
- Save settings

## Configuration File

**`figure_aesthetics.json`** - Main configuration file

Modify this file to change the appearance of all generated figures.

## Usage

The plotting code automatically loads and applies these settings. No code changes needed - just edit the JSON file!

## Key Sections

### Colors
Define colors for different plot types:
- `temperature`: Red (#d62728)
- `pressure`: Blue (#1f77b4)
- `velocity`: Green (#2ca02c)
- etc.

### Fonts
Control all text appearance:
- Font family, size, weight
- Title and label sizes
- Tick label sizes

### Profiles
Individual settings for each plot type:
- Color
- Label text
- Y-axis label
- Title

## Customization

1. Edit `figure_aesthetics.json`
2. Run your simulation
3. All figures will use the new styling automatically

## Example: Change Temperature Color

```json
"profiles": {
    "temperature": {
        "color": "#ff0000",  // Change to bright red
        ...
    }
}
```

## Example: Change Figure Size

```json
"figure": {
    "figsize": [12, 8],  // Wider and taller
    ...
}
```

## Example: Disable Grid

```json
"axes": {
    "grid": false,
    ...
}
```

## Benefits

- **Consistency**: All plots look the same  
- **Easy Updates**: Change once, apply everywhere  
- **Professional**: Polished, publication-ready figures  
- **Maintainable**: One place to manage all styling  
