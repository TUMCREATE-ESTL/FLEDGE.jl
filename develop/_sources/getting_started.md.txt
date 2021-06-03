# Getting Started

## Installation

### Quick installation

1. Check requirements:
   - Python 3.7
   - [Gurobi Optimizer](http://www.gurobi.com/)
2. Clone or download repository. Ensure that the `cobmo` submodule directory is loaded as well.
3. In your Python environment, run:
   1. `pip install -v -e path_to_repository`

Note that the `fledge.plots` module optionally allows adding contextual basemaps to grid plots for orientation. This requires installation of `contextily`, which is an optional dependency, because it needs to be installed through `conda` on Windows. If you need it, please follow the reccomended installation procedure below.

### Recommended installation

The following installation procedure requires additional steps, but can improve performance and includes optional dependencies. For example, the numpy, pandas and cvxpy packages are installed through Anaconda, which ensures the use of more performant math libraries. Additionally, the contextily package is installed, which is required for some geographical plots.

1. Check requirements:
   - [Anaconda Python Distribution](https://www.anaconda.com/distribution/)
   - [Gurobi Optimizer](http://www.gurobi.com/) or [CPLEX Optimizer](https://www.ibm.com/analytics/cplex-optimizer)
2. Clone or download repository. Ensure that the `cobmo` submodule directory is loaded as well.
3. In Anaconda Prompt, run:
   1. `conda create -n fledge python=3.7`
   2. `conda activate fledge`
   3. `conda install -c conda-forge contextily cvxpy numpy pandas`
   4. `pip install -v -e path_to_repository`
4. If you want to use CPLEX:
   1. Install CPLEX Python interface (see latest CPLEX documentation).
   2. Create or modify `config.yml` (see below in "Configuration with `config.yml`").

### Alternative installation

If you are running into errors when installing or running FLEDGE, this may be due to incompatibility with new versions of package dependencies, which have yet to be discovered and fixed. As a workaround, try installing FLEDGE in an tested Anaconda environment via the the provided `environment.yml`, which represents the latest Anaconda Python environment in which FLEDGE was tested and is expected to work.

1. Check requirements:
   - Windows 10
   - [Anaconda Python Distribution](https://www.anaconda.com/distribution/)
   - [Gurobi Optimizer](http://www.gurobi.com/)
2. Clone or download repository. Ensure that the `cobmo` submodule directory is loaded as well.
4. In Anaconda Prompt, run:
   1. `conda env create -f path_to_fledge_repository/environment.yml`
   2. `conda activate fledge`
   3. `pip install -v -e path_to_repository`.

``` important::
    Please create an issue on Github if you run into problems with the normal installation procedure.
```

## Examples

The `examples` directory contains run scripts which demonstrate the usage of FLEDGE.

### API examples

These examples demonstrate the usage of the high-level API to execute predefined problem types.

- `run_api_nominal_operation_problem.py`: Example script for setting up and solving an nominal operation problem. The nominal operation problem (alias: power flow problem, electric grid simulation problem) formulates the steady-state power flow problem for all timesteps of the given scenario subject to the nominal operation schedule of all DERs.
- `run_api_optimal_operation_problem.py`: Example script for setting up and solving an optimal operation problem. The optimal operation problem (alias: optimal dispatch problem, optimal power flow problem) formulates the optimization problem for minimizing the objective functions of DERs and grid operators subject to the model constraints of all DERs and grids.

### Advanced examples

For advanced usage of FLEDGE, the following examples demonstrate in a step-by-step manner how energy system models and optimization problems can be defined and solved with FLEDGE. These example scripts serve as a reference for setting up custom work flows.

- `run_electric_grid_optimal_operation.py`: Example script for setting up and solving an electric grid optimal operation problem.
- `run_thermal_grid_optimal_operation.py`: Example script for setting up and solving a thermal grid optimal operation problem.
- `run_multi_grid_optimal_operation.py`: Example script for setting up and solving a multi-grid optimal operation problem.
- `run_flexible_der_optimal_operation.py`: Example script for setting up and solving a flexible DER optimal operation problem.
- `run_electric_grid_power_flow_single_step.py`: Example script for setting up and solving an single step electric grid power flow problem.

### Validation scripts

Since the model implementations in FLEDGE are not infallible, these validation scripts are provided for model testing.

- `validation_electric_grid_power_flow.py`: Example script for testing / validating the electric grid power flow solution.
- `validation_linear_electric_grid_model.py`: Example script for testing / validating the linear electric grid model.
- `validation_electric_grid_dlmp_solution.py`: Validation script for solving a decentralized DER operation problem based on DLMPs from the centralized problem.

### Other scripts

- The directory `examples/development` contains example scripts which are under development and scripts related to the development of new features for FLEDGE.
- The directory `examples/publications` contains scripts related to publications which based on FLEDGE.

## Configuration with `config.yml`

FLEDGE configuration parameters (e.g. the output format of plots) can be set in `config.yml`. As an initial user, you most likely will not need to modify the configuration.

If you want to change the configuration, you can create or modify `config.yml` in the FLEDGE repository main directory. FLEDGE will automatically create `config.yml` if it does not exist. Initially, `config.yml` will be empty. You can copy configuration parameters from `fledge/config_default.yml` to `config.yml` and modify their value to define your local configuration. To define nested configuration parameters, you need to replicate the nested structure in `config.yml`. For example, to define CPLEX as the optimization solver, use:

```
optimization:
  solver_name: cplex
```

The configuration parameters which are defined in `config.yml` will take precedence over those defined in `fledge/config_default.yml`. If you would like to revert a parameter to its default value, just delete the parameter from `config.yml`. Please do not modify `fledge/config_default.yml` directly.

## Contributing

If you are keen to contribute to this project, please see [Contributing](contributing.md).
