"""Example script for setting up and solving a multi-grid optimal operation problem."""

import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.database_interface
import fledge.der_models
import fledge.linear_electric_grid_models
import fledge.electric_grid_models
import fledge.power_flow_solvers
import fledge.thermal_grid_models


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'
    results_path = (
        os.path.join(
            fledge.config.results_path,
            f'run_multi_grid_optimal_operation_{fledge.config.timestamp}'
        )
    )

    # Instantiate results directory.
    os.mkdir(results_path)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.database_interface.recreate_database()

    # Obtain data.
    scenario_data = fledge.database_interface.ScenarioData(scenario_name)
    price_data = fledge.database_interface.PriceData(scenario_name)

    # Obtain price timeseries.
    price_name = 'energy'
    price_timeseries = price_data.price_timeseries_dict[price_name]

    # Obtain models.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModel(scenario_name)
    power_flow_solution = fledge.power_flow_solvers.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model = (
        fledge.linear_electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )
    thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
    thermal_power_flow_solution = fledge.thermal_grid_models.ThermalPowerFlowSolution(thermal_grid_model)
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define linear electric grid model constraints.
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define thermal grid model variables.
    thermal_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define thermal grid model constraints.
    thermal_grid_model.define_optimization_constraints(
        optimization_problem,
        thermal_power_flow_solution,
        scenario_data.timesteps
    )

    # Define DER variables.
    der_model_set.define_optimization_variables(
        optimization_problem
    )

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem
    )

    # Define constraints for the connection with the DER power vector of the electric and thermal grids.
    der_model_set.define_optimization_connection_grid(
        optimization_problem,
        power_flow_solution,
        electric_grid_model,
        thermal_power_flow_solution,
        thermal_grid_model
    )

    # Define limit constraints.

    # Electric grid.

    # Voltage.
    voltage_magnitude_vector = (  # Define shorthand.
        lambda node:
        np.abs(power_flow_solution.node_voltage_vector.ravel()[electric_grid_model.nodes.get_loc(node)])
    )
    optimization_problem.voltage_magnitude_vector_minimum_constraint = pyo.Constraint(
        scenario_data.timesteps.to_list(),
        electric_grid_model.nodes.to_list(),
        rule=lambda optimization_problem, timestep, *node: (
            optimization_problem.voltage_magnitude_vector_change[timestep, node]
            + voltage_magnitude_vector(node)
            >=
            0.5 * voltage_magnitude_vector(node)
        )
    )
    optimization_problem.voltage_magnitude_vector_maximum_constraint = pyo.Constraint(
        scenario_data.timesteps.to_list(),
        electric_grid_model.nodes.to_list(),
        rule=lambda optimization_problem, timestep, *node: (
            optimization_problem.voltage_magnitude_vector_change[timestep, node]
            + voltage_magnitude_vector(node)
            <=
            1.5 * voltage_magnitude_vector(node)
        )
    )

    # Branch flows.
    branch_power_vector_1_squared = (  # Define shorthand.
        lambda branch:
        np.abs(power_flow_solution.branch_power_vector_1.ravel()[electric_grid_model.branches.get_loc(branch)] ** 2)
    )
    optimization_problem.branch_power_vector_1_squared_maximum_constraint = pyo.Constraint(
        scenario_data.timesteps.to_list(),
        electric_grid_model.branches.to_list(),
        rule=lambda optimization_problem, timestep, *branch: (
            optimization_problem.branch_power_vector_1_squared_change[timestep, branch]
            + branch_power_vector_1_squared(branch)
            <=
            1.5 * branch_power_vector_1_squared(branch)
        )
    )
    branch_power_vector_2_squared = (  # Define shorthand.
        lambda branch:
        np.abs(power_flow_solution.branch_power_vector_2.ravel()[electric_grid_model.branches.get_loc(branch)] ** 2)
    )
    optimization_problem.branch_power_vector_2_squared_maximum_constraint = pyo.Constraint(
        scenario_data.timesteps.to_list(),
        electric_grid_model.branches.to_list(),
        rule=lambda optimization_problem, timestep, *branch: (
            optimization_problem.branch_power_vector_2_squared_change[timestep, branch]
            + branch_power_vector_2_squared(branch)
            <=
            1.5 * branch_power_vector_2_squared(branch)
        )
    )

    # Thermal grid.

    # Node head.
    node_head_vector = (  # Define shorthand.
        lambda node:
        thermal_power_flow_solution.node_head_vector.ravel()[thermal_grid_model.nodes.get_loc(node)]
    )
    optimization_problem.node_head_vector_minimum_constraint = pyo.Constraint(
        scenario_data.timesteps.to_list(),
        thermal_grid_model.nodes.to_list(),
        rule=lambda optimization_problem, timestep, *node: (
            optimization_problem.node_head_vector[timestep, node]
            # + node_head_vector(node)
            >=
            1.5 * node_head_vector(node)
        )
    )
    # Branch flow.
    branch_flow_vector = (  # Define shorthand.
        lambda branch:
        thermal_power_flow_solution.branch_flow_vector.ravel()[thermal_grid_model.branches.get_loc(branch)]
    )
    optimization_problem.branch_flow_vector_maximum_constraint = pyo.Constraint(
        scenario_data.timesteps.to_list(),
        thermal_grid_model.branches.to_list(),
        rule=lambda optimization_problem, timestep, branch: (  # This will not work if `branches` becomes MultiIndex.
                optimization_problem.branch_flow_vector[timestep, branch]
                # + branch_flow_vector(branch)
                <=
                1.5 * branch_flow_vector(branch)
        )
    )


    # Define electric grid objective.
    # TODO: Not considering loss costs due to unrealiable loss model.
    # if optimization_problem.find_component('objective') is None:
    #     optimization_problem.objective = pyo.Objective(expr=0.0, sense=pyo.minimize)
    # optimization_problem.objective.expr += (
    #     sum(
    #         price_timeseries.at[timestep, 'price_value']
    #         * (
    #             optimization_problem.loss_active_change[timestep]
    #             + np.sum(np.real(power_flow_solution.loss))
    #         )
    #         for timestep in scenario_data.timesteps
    #     )
    # )

    # Define objective.
    thermal_grid_model.define_optimization_objective(
        optimization_problem,
        thermal_power_flow_solution,
        price_timeseries,
        scenario_data.timesteps
    )

    # Define DER objective.
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_timeseries
    )

    # Solve optimization problem.
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    optimization_solver = pyo.SolverFactory(fledge.config.solver_name)
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.solver_output)
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")
    # optimization_problem.display()

    # Obtain results.
    (
        der_active_power_vector,
        der_reactive_power_vector,
        voltage_magnitude_vector,
        branch_power_vector_1_squared,
        branch_power_vector_2_squared,
        loss_active,
        loss_reactive
    ) = linear_electric_grid_model.get_optimization_results(
        optimization_problem,
        power_flow_solution,
        scenario_data.timesteps,
        in_per_unit=True,
        with_mean=True
    )
    (
        der_thermal_power_vector,
        node_head_vector,
        branch_flow_vector,
        pump_power
    ) = thermal_grid_model.get_optimization_results(
        optimization_problem,
        thermal_power_flow_solution,
        scenario_data.timesteps,
        in_per_unit=True,
        with_mean=True
    )

    # Print results.
    print(f"der_active_power_vector = \n{der_active_power_vector.to_string()}")
    print(f"der_reactive_power_vector = \n{der_reactive_power_vector.to_string()}")
    print(f"voltage_magnitude_vector = \n{voltage_magnitude_vector.to_string()}")
    print(f"branch_power_vector_1_squared = \n{branch_power_vector_1_squared.to_string()}")
    print(f"branch_power_vector_2_squared = \n{branch_power_vector_2_squared.to_string()}")
    print(f"loss_active = \n{loss_active.to_string()}")
    print(f"loss_reactive = \n{loss_reactive.to_string()}")
    print(f"der_thermal_power_vector = \n{der_thermal_power_vector.to_string()}")
    print(f"node_head_vector = \n{node_head_vector.to_string()}")
    print(f"branch_flow_vector = \n{branch_flow_vector.to_string()}")
    print(f"pump_power = \n{pump_power.to_string()}")

    # Store results as CSV.
    der_active_power_vector.to_csv(os.path.join(results_path, 'der_active_power_vector.csv'))
    der_reactive_power_vector.to_csv(os.path.join(results_path, 'der_reactive_power_vector.csv'))
    voltage_magnitude_vector.to_csv(os.path.join(results_path, 'voltage_magnitude_vector.csv'))
    branch_power_vector_1_squared.to_csv(os.path.join(results_path, 'branch_power_vector_1_squared.csv'))
    branch_power_vector_2_squared.to_csv(os.path.join(results_path, 'branch_power_vector_2_squared.csv'))
    loss_active.to_csv(os.path.join(results_path, 'loss_active.csv'))
    loss_reactive.to_csv(os.path.join(results_path, 'loss_reactive.csv'))
    der_thermal_power_vector.to_csv(os.path.join(results_path, 'der_thermal_power_vector.csv'))
    node_head_vector.to_csv(os.path.join(results_path, 'node_head_vector.csv'))
    branch_flow_vector.to_csv(os.path.join(results_path, 'branch_flow_vector.csv'))
    pump_power.to_csv(os.path.join(results_path, 'pump_power.csv'))

    # Obtain duals.
    voltage_magnitude_vector_minimum_dual = (
        pd.DataFrame(columns=electric_grid_model.nodes, index=scenario_data.timesteps, dtype=np.float)
    )
    voltage_magnitude_vector_maximum_dual = (
        pd.DataFrame(columns=electric_grid_model.nodes, index=scenario_data.timesteps, dtype=np.float)
    )
    branch_power_vector_1_squared_maximum_dual = (
        pd.DataFrame(columns=electric_grid_model.branches, index=scenario_data.timesteps, dtype=np.float)
    )
    branch_power_vector_2_squared_maximum_dual = (
        pd.DataFrame(columns=electric_grid_model.branches, index=scenario_data.timesteps, dtype=np.float)
    )
    node_head_vector_minimum_dual = (
        pd.DataFrame(columns=thermal_grid_model.nodes, index=scenario_data.timesteps, dtype=np.float)
    )
    branch_flow_vector_maximum_dual = (
        pd.DataFrame(columns=thermal_grid_model.branches, index=scenario_data.timesteps, dtype=np.float)
    )

    for timestep in scenario_data.timesteps:

        for node_index, node in enumerate(electric_grid_model.nodes):
            voltage_magnitude_vector_minimum_dual.at[timestep, node] = (
                optimization_problem.dual[
                    optimization_problem.voltage_magnitude_vector_minimum_constraint[timestep, node]
                ]
            )
            voltage_magnitude_vector_maximum_dual.at[timestep, node] = (
                optimization_problem.dual[
                    optimization_problem.voltage_magnitude_vector_maximum_constraint[timestep, node]
                ]
            )

        for branch_index, branch in enumerate(electric_grid_model.branches):
            branch_power_vector_1_squared_maximum_dual.at[timestep, branch] = (
                optimization_problem.dual[
                    optimization_problem.branch_power_vector_1_squared_maximum_constraint[timestep, branch]
                ]
            )
            branch_power_vector_2_squared_maximum_dual.at[timestep, branch] = (
                optimization_problem.dual[
                    optimization_problem.branch_power_vector_2_squared_maximum_constraint[timestep, branch]
                ]
            )

        for node_index, node in enumerate(thermal_grid_model.nodes):
            node_head_vector_minimum_dual.at[timestep, node] = (
                optimization_problem.dual[
                    optimization_problem.node_head_vector_minimum_constraint[timestep, node]
                ]
            )

        for branch_index, branch in enumerate(thermal_grid_model.branches):
            branch_flow_vector_maximum_dual.at[timestep, branch] = (
                optimization_problem.dual[
                    optimization_problem.branch_flow_vector_maximum_constraint[timestep, branch]
                ]
            )

    # Print duals.
    print(f"voltage_magnitude_vector_minimum_dual = \n{voltage_magnitude_vector_minimum_dual.to_string()}")
    print(f"voltage_magnitude_vector_maximum_dual = \n{voltage_magnitude_vector_maximum_dual.to_string()}")
    print(f"branch_power_vector_1_squared_maximum_dual = \n{branch_power_vector_1_squared_maximum_dual.to_string()}")
    print(f"branch_power_vector_2_squared_maximum_dual = \n{branch_power_vector_2_squared_maximum_dual.to_string()}")
    print(f"node_head_vector_minimum_dual = \n{node_head_vector_minimum_dual.to_string()}")
    print(f"branch_flow_vector_maximum_dual = \n{branch_flow_vector_maximum_dual.to_string()}")


if __name__ == '__main__':
    main()
