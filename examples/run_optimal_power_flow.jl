"Example script for setting up and solving an optimal power flow problem."

import FLEDGE

import GLPK
import GR
import JuMP
import Logging
import Plots
import Statistics
import TimeSeries

# Settings.
scenario_name = "singapore_6node"
Plots.gr()  # Select plotting backend.

# Get model.
electric_grid_data = (
    FLEDGE.DatabaseInterface.ElectricGridData(scenario_name)
)
electric_grid_index = (
    FLEDGE.ElectricGridModels.ElectricGridIndex(scenario_name)
)
electric_grid_model = (
    FLEDGE.ElectricGridModels.ElectricGridModel(scenario_name)
)
linear_electric_grid_model = (
    FLEDGE.ElectricGridModels.LinearElectricGridModel(scenario_name)
)
nodal_voltage_vector_initial = (
    FLEDGE.PowerFlowSolvers.get_voltage_fixed_point(electric_grid_model)
)
(
    branch_power_vector_1_initial,
    branch_power_vector_2_initial
) = (
    FLEDGE.PowerFlowSolvers.get_branch_power_fixed_point(
        electric_grid_model,
        nodal_voltage_vector_initial
    )
)

# Define derivative model parameters.
load_active_power_vector_nominal = (
    real.(electric_grid_model.load_power_vector_nominal)
)
load_reactive_power_vector_nominal = (
    imag.(electric_grid_model.load_power_vector_nominal)
)

# Instantiate optimization problem.
optimization_problem = (
    JuMP.Model(JuMP.with_optimizer(GLPK.Optimizer, msg_lev=GLPK.MSG_ON))
)

# Define variables.

# Load.
JuMP.@variable(
    optimization_problem,
    load_active_power_vector[electric_grid_index.load_names]
)
JuMP.@variable(
    optimization_problem,
    load_reactive_power_vector[electric_grid_index.load_names]
)

# Power.
JuMP.@variable(
    optimization_problem,
    nodal_power_vector_wye_active_change[electric_grid_index.nodes_phases]
)
JuMP.@variable(
    optimization_problem,
    nodal_power_vector_wye_reactive_change[electric_grid_index.nodes_phases]
)
JuMP.@variable(
    optimization_problem,
    nodal_power_vector_delta_active_change[electric_grid_index.nodes_phases]
)
JuMP.@variable(
    optimization_problem,
    nodal_power_vector_delta_reactive_change[electric_grid_index.nodes_phases]
)

# Voltage.
JuMP.@variable(
    optimization_problem,
    voltage_magnitude_vector[electric_grid_index.nodes_phases]
)
JuMP.@variable(
    optimization_problem,
    voltage_magnitude_per_unit_deviation_vector[electric_grid_index.nodes_phases]
)

# Branch flows.
JuMP.@variable(
    optimization_problem,
    branch_power_vector_1_squared[electric_grid_index.branches_phases]
)
JuMP.@variable(
    optimization_problem,
    branch_power_vector_2_squared[electric_grid_index.branches_phases]
)

# Define constraints.

# Load.
JuMP.@constraint(
    optimization_problem,
    load_active_minimum_maximum,
    (
        0.5 .* load_active_power_vector_nominal
        .<=
        load_active_power_vector.data
        .<=
        1.5 .* load_active_power_vector_nominal
    )
)
JuMP.@constraint(
    optimization_problem,
    load_reactive_minimum_maximum,
    (
        0.5 .* load_reactive_power_vector_nominal
        .<=
        load_reactive_power_vector.data
        .<=
        1.5 .* load_reactive_power_vector_nominal
    )
)

# Power.
JuMP.@constraint(
    optimization_problem,
    nodal_power_vector_wye_active_equation,
    (
        nodal_power_vector_wye_active_change.data
        .==
        0.0
        + electric_grid_model.load_incidence_wye_matrix
        * (
            -1.0
            .* (
                load_active_power_vector.data
                - load_active_power_vector_nominal
            )
        )
    )
)
JuMP.@constraint(
    optimization_problem,
    nodal_power_vector_wye_reactive_equation,
    (
        nodal_power_vector_wye_reactive_change.data
        .==
        0.0
        + electric_grid_model.load_incidence_wye_matrix
        * (
            -1.0
            .* (
                load_reactive_power_vector.data
                - load_reactive_power_vector_nominal
            )
        )
    )
)
JuMP.@constraint(
    optimization_problem,
    nodal_power_vector_delta_active_equation,
    (
        nodal_power_vector_delta_active_change.data
        .==
        0.0
        + electric_grid_model.load_incidence_delta_matrix
        * (
            -1.0
            .* (
                load_active_power_vector.data
                - load_active_power_vector_nominal
            )
        )
    )
)
JuMP.@constraint(
    optimization_problem,
    nodal_power_vector_delta_reactive_equation,
    (
        nodal_power_vector_delta_reactive_change.data
        .==
        0.0
        + electric_grid_model.load_incidence_delta_matrix
        * (
            -1.0
            .* (
                load_reactive_power_vector.data
                - load_reactive_power_vector_nominal
            )
        )
    )
)


# Voltage.
JuMP.@constraint(
    optimization_problem,
    voltage_magnitude_equation,
    (
        voltage_magnitude_vector.data
        .==
        (
            abs.(nodal_voltage_vector_initial)
            + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_wye_active
            * nodal_power_vector_wye_active_change.data
            + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_wye_reactive
            * nodal_power_vector_wye_reactive_change.data
            + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_delta_active
            * nodal_power_vector_delta_active_change.data
            + linear_electric_grid_model.sensitivity_voltage_magnitude_by_power_delta_reactive
            * nodal_power_vector_delta_reactive_change.data
        )
    )
)
JuMP.@constraint(
    optimization_problem,
    voltage_magnitude_equation_0,
    (
        voltage_magnitude_per_unit_deviation_vector.data
        .>=
        0.0
    )
)
JuMP.@constraint(
    optimization_problem,
    voltage_magnitude_equation_1,
    (
        voltage_magnitude_per_unit_deviation_vector.data
        .>=
        (
            voltage_magnitude_vector.data
            ./ abs.(electric_grid_model.nodal_voltage_vector_no_load)
            .- 1.0
        )
    )
)
JuMP.@constraint(
    optimization_problem,
    voltage_magnitude_equation_2,
    (
        voltage_magnitude_per_unit_deviation_vector.data
        .>=
        -1.0
        .* (
            voltage_magnitude_vector.data
            ./ abs.(electric_grid_model.nodal_voltage_vector_no_load)
            .- 1.0
        )
    )
)

# Branch flows.
JuMP.@constraint(
    optimization_problem,
    branch_flow_1_equation,
    (
        branch_power_vector_1_squared.data
        .==
        (
            (abs.(branch_power_vector_1_initial) .^ 2)
            + linear_electric_grid_model.sensitivity_power_branch_from_by_power_wye_active
            * nodal_power_vector_wye_active_change.data
            + linear_electric_grid_model.sensitivity_power_branch_from_by_power_wye_reactive
            * nodal_power_vector_wye_reactive_change.data
            + linear_electric_grid_model.sensitivity_power_branch_from_by_power_delta_active
            * nodal_power_vector_delta_active_change.data
            + linear_electric_grid_model.sensitivity_power_branch_from_by_power_delta_reactive
            * nodal_power_vector_delta_reactive_change.data
        )
    )
)
JuMP.@constraint(
    optimization_problem,
    branch_flow_2_equation,
    (
        branch_power_vector_2_squared.data
        .==
        (
            (abs.(branch_power_vector_2_initial) .^ 2)
            + linear_electric_grid_model.sensitivity_power_branch_to_by_power_wye_active
            * nodal_power_vector_wye_active_change.data
            + linear_electric_grid_model.sensitivity_power_branch_to_by_power_wye_reactive
            * nodal_power_vector_wye_reactive_change.data
            + linear_electric_grid_model.sensitivity_power_branch_to_by_power_delta_active
            * nodal_power_vector_delta_active_change.data
            + linear_electric_grid_model.sensitivity_power_branch_to_by_power_delta_reactive
            * nodal_power_vector_delta_reactive_change.data
        )
    )
)

# Define objective.
JuMP.@objective(
    optimization_problem,
    Min,
    sum(voltage_magnitude_per_unit_deviation_vector.data)
)

# Solve optimization problem.
Logging.@info("", optimization_problem)
JuMP.optimize!(optimization_problem)

# Get results.
optimization_termination_status = JuMP.termination_status(optimization_problem)
Logging.@info("", optimization_termination_status)

voltage_magnitude_vector_per_unit_value = (
    JuMP.value.(voltage_magnitude_vector.data)
    ./ abs.(electric_grid_model.nodal_voltage_vector_no_load)
)
Logging.@info("", Statistics.mean(voltage_magnitude_vector_per_unit_value))
load_active_power_vector_per_unit_value = (
    JuMP.value.(load_active_power_vector.data)
    ./ real.(electric_grid_model.load_power_vector_nominal)
)
Logging.@info("", Statistics.mean(load_active_power_vector_per_unit_value))
branch_power_vector_1_squared_per_unit_value = (
    JuMP.value.(branch_power_vector_1_squared.data)
    ./ (abs.(branch_power_vector_1_initial) .^ 2)
)
Logging.@info("", Statistics.mean(branch_power_vector_1_squared_per_unit_value))
