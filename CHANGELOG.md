# Changelog

All notable changes to the Generalized PFR Simulation System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-09-20

### Added
- **Comprehensive data export** - 245+ columns of simulation data including all thermodynamic and transport properties
- **Enhanced visualization** - 18+ professional plots including viscosity, thermal conductivity, and heat capacity profiles
- **Export controls** - Optional CSV and plot generation control with `if_csv_out` and `if_plot_out` flags
- **Flexible workflows** - Support for data-only, plots-only, or simulation-only modes

### Changed
- **Export compatibility** - Removed properties not available in current Cantera version to ensure stability
- **Documentation updates** - Updated README and API documentation to reflect comprehensive export capabilities
- **Configuration system** - Added export controls section to configuration template
- **Performance optimization** - Export operations now optional for faster parameter studies

### Fixed
- **Export stability** - Fixed issues with unavailable Cantera properties causing simulation crashes
- **Figure generation** - Ensured all 18 plots are generated successfully
- **CSV export** - Guaranteed 245+ columns of data export without errors

## [2.0.1] - 2025-09-20

### Added
- **Relative position heat flux** - Heat flux profiles now use relative positions (0.0-1.0) for universal applicability
- **Step-wise interpolation** - Option for step-wise heat flux interpolation in addition to linear
- **Automatic heat flux plotting** - Heat flux vs. relative position figure generated automatically
- **Complete species data** - Mass and mole fractions for all species in CSV export
- **Transport properties** - Viscosity and thermal conductivity data export and visualization
- **Thermodynamic completeness** - Internal energy, Gibbs free energy, and heat capacity ratio export

### Changed
- **Heat flux profile format** - Updated to use relative positions (0.0-1.0) instead of absolute meters

## [2.0.0] - 2025-09-18

### Added
- **Generalized multi-reactant support** - System now supports ethane, propane, naphtha, and n-hexane
- **Automatic configuration generation** - Dynamic configuration creation for each reactant type
- **Species name standardization** - Intelligent handling of different naming conventions across mechanisms
- **Professional file naming** - Systematic output file naming with reactant identification
- **Comprehensive documentation** - Professional README, API documentation, and code comments
- **Reactant database system** - JSON-based database for easy reactant management
- **Enhanced error handling** - Robust error handling with informative messages
- **Flexible product tracking** - Configurable target products for each reactant
- **Batch processing support** - Easy execution of multiple simulations
- **Automatic geometry calculations** - Reactor volume and surface area calculated from geometry
- **Realistic heat flux profiles** - Heat flux for high-temperature pyrolysis (150,000 W/m²)
- **Enhanced plotting system** - Individual plots for each variable with expanded product species
- **GitHub repository** - Public repository at https://github.com/karefyllidis/HydrAI

### Changed
- **Mechanism file naming** - Standardized to `[Reactant]_Kinetic-Model.yaml` format
- **Configuration system** - Moved from hardcoded to template-based configuration
- **Output structure** - Enhanced CSV and summary file formats
- **Code organization** - Modular design with clear separation of concerns
- **Heat flux profile** - Updated to realistic values (150,000 W/m²) with 6 data points
- **Reactor geometry** - Volume and surface area now calculated automatically per step
- **Pressure specification** - Clarified as absolute pressure (2.0 bar absolute)

### Fixed
- **Species name mismatches** - Resolved issues with different naming conventions
- **Configuration inconsistencies** - Standardized configuration across all reactants
- **File path issues** - Fixed relative path problems in different environments
- **Numerical stability** - Fixed reactor volume/surface area calculation for solver stability
- **Simulation length** - Fixed simulation to stop at configured reactor length
- **Transport properties** - Added handling for mechanisms without transport properties

### Technical Details
- **New files**: `Main_GeneralizedPFR.py`, `run_simulation.py`, `reactant_database.json`, `config_template.json`
- **Updated files**: All mechanism files renamed for consistency
- **Dependencies**: Updated to require Cantera 3.1.0+, Python 3.8+

## [1.2.0] - 2025-09-16

### Added
- **External environment support** - Refactored to use external virtual environment
- **Enhanced documentation** - Improved project structure and documentation
- **Heat flux profile information** - Added heat flux data to summary files

### Changed
- **Project structure** - Cleaned up and organized project files
- **Documentation** - Removed personal paths and improved clarity

## [1.1.0] - 2025-09-15

### Added
- **Automatic step size calculation** - Step size now calculated from reactor geometry
- **Species count in filenames** - Mechanism files include species count
- **Enhanced project structure** - Improved documentation and organization

### Changed
- **Mechanism file naming** - Standardized naming convention
- **Documentation** - Enhanced project structure documentation

## [1.0.0] - 2025-09-15

### Added
- **Initial PFR simulation system** - Basic plug flow reactor simulation for propane cracking
- **Cantera integration** - Full integration with Cantera chemical kinetics library
- **Basic visualization** - Temperature, pressure, and species profile plots
- **CSV export** - Data export functionality for analysis
- **Heat flux profiles** - External heating simulation support
- **Pressure drop calculations** - Churchill correlation for friction factor

### Technical Details
- **Core files**: `Main_NaphthaPFR.py`, `config.json`
- **Mechanism**: `Propane_KineticModel.yaml`
- **Dependencies**: Cantera 2.6.0+, Python 3.7+

---

## Version History Summary

| Version | Date | Major Features |
|---------|------|----------------|
| 2.0.0 | 2025-01-15 | Multi-reactant support, automatic configuration, professional documentation |
| 1.0.0 | 2024-12-01 | Initial PFR simulation system for propane cracking |

## Future Roadmap

### Planned Features (v2.1.0)
- [ ] **GUI Interface** - Graphical user interface for easier operation
- [ ] **Parameter Studies** - Automated parameter sweep capabilities
- [ ] **Optimization Tools** - Built-in optimization for reactor design
- [ ] **Advanced Visualization** - Interactive plots and 3D visualizations
- [ ] **Export Formats** - Additional export formats (Excel, HDF5, etc.)

### Planned Features (v2.2.0)
- [ ] **Additional Reactants** - Support for more feedstocks (butane, pentane, etc.)
- [ ] **Reactor Types** - Support for CSTR and other reactor types
- [ ] **Heat Integration** - Advanced heat integration modeling
- [ ] **Economic Analysis** - Cost estimation and economic optimization
- [ ] **Uncertainty Analysis** - Monte Carlo and sensitivity analysis tools

### Long-term Goals
- [ ] **Cloud Integration** - Cloud-based simulation capabilities
- [ ] **Machine Learning** - ML-based mechanism reduction and optimization
- [ ] **Real-time Monitoring** - Integration with process control systems
- [ ] **Multi-scale Modeling** - Integration with CFD and other simulation tools

---

**Note**: This changelog follows semantic versioning principles. Breaking changes are indicated by major version increments, new features by minor version increments, and bug fixes by patch version increments.
