# Change log

Note that version numbering follows the [Semantic Versioning principle](https://semver.org/).

## Next release

### New features

- Added problems module with definitions for nominal operation problem (simulation) and optimal operation problem (optimization).
- Added high-level API for executing optimal & nominal operation problems.
- Added various DER models.
- Added ability to define electric grid model as single-phase-approximate.
- Added Z-Bus power flow solution method.
- Added plots module (work-in-progress).
- Added ability to set local configuration with `config.yml`.

### Changes

- Moved implementation of optimization problems from Pyomo to CVXPY for performance improvements.
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
