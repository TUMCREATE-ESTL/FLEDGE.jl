# Change log

Note that version numbering follows the [Semantic Versioning principle](https://semver.org/).

## [v0.5.0](https://github.com/TUMCREATE-ESTL/fledge/releases/tag/v0.5.0)

### New features

- Overhead line types can now be defined in terms of conductor data and geometric arrangement (Arif Ahmed).
- Added standard form object for generating the standard form of linear optimization problems.
- Added local-approximation variant for linear electric grid model.
- Added linear model set for electric grid model, which enables defining separate linear models for each time step.
- Added power flow solution set, to obtain power flow solutions more conveniently for multiple time steps.
- Added pre-solve method for DER model set, to obtain baseline nominal power time series for flexible DERs.

### Changes

- Improved / simplified `define_optimization...()` methods for most use cases.
- Switched from `multiprocess` to `ray` for parallel processing for performance reasons.

## [v0.4.1](https://github.com/TUMCREATE-ESTL/fledge/releases/tag/v0.4.1)

### Fixes

- Updated `environment.yml`.
- Updated version indicators.

## [v0.4.0](https://github.com/TUMCREATE-ESTL/fledge/releases/tag/v0.4.0)

### New features

- Added problems module with definitions for nominal operation problem (simulation) and optimal operation problem (optimization).
- Added high-level API for executing optimal & nominal operation problems.
- Added various DER models.
- Enabled most DERs for thermal grids (Verena Kleinschmidt).
- Added ability to define electric grid model as single-phase-approximate.
- Added Z-Bus power flow solution method (Arif Ahmed).
- Added plots module (work-in-progress).
- Added ability to set local configuration with `config.yml`.
- Added ability to set base units for apparent power, voltage and thermal power for in scenario definition.

### Changes

- Moved implementation of optimization problems from Pyomo to CVXPY for performance improvements.
- Reformulated optimization constraints to use normalized values for improved numerical performance.
- Improved FLEDGE definition data format documentation.
- Refactored DER model data definition format.
- Refactored price data object.
- Various fixes in linear electric grid model model and DLMP calculations.
- Introduced various error messages for common issues.

## [v0.3.0](https://github.com/TUMCREATE-ESTL/fledge/releases/tag/v0.3.0)

### New features

- Moved to Python as main implementation language.
- Extended linear electric models with methods for defining optimization variables / constraints.
- Added thermal grid model.
- Added linear thermal grid model with methods for defining optimization variables / constraints.
- Added DER models and integrated CoBMo for flexible building models.
- Added methods for defining operation limits and obtaining DLMPs for electric and thermal grids.
- Provided various example scripts for running optimal operation problems for DERs / electric grids / thermal grids / multi-energy grids.

## [v0.2.0](https://github.com/TUMCREATE-ESTL/fledge/releases/tag/v0.2.0)

### Auxiliary Release

- Snapshot before moving to Python as main implementation language.

## [v0.1.0](https://github.com/TUMCREATE-ESTL/fledge/releases/tag/v0.1.0)

### Initial release

- Initial set of modules with Julia as main implementation language.
