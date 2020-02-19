"""Example script for setting up and solving an thermal grid optimal operation problem."""

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.database_interface
import fledge.thermal_grid_models


def main():

    # Settings.
    scenario_name = "singapore_tanjongpagar"

    # Obtain data.
    scenario_data = fledge.database_interface.ScenarioData(scenario_name)

    # Obtain model.
    thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define variables.
    optimization_problem.der_thermal_power_vector = (
        pyo.Var(scenario_data.timesteps.to_list(), thermal_grid_model.ders)
    )
    optimization_problem.branch_flow_vector = (
        pyo.Var(scenario_data.timesteps.to_list(), thermal_grid_model.branches)
    )
    optimization_problem.source_flow = (
        pyo.Var(scenario_data.timesteps.to_list())
    )

    # Define DER constraints.
    # TODO: Arbitrary constraints to demonstrate the functionality.
    optimization_problem.der_constraints = pyo.ConstraintList()
    for timestep in scenario_data.timesteps:
        for der_index, der in enumerate(thermal_grid_model.ders):
            optimization_problem.der_constraints.add(
                optimization_problem.der_thermal_power_vector[timestep, der]
                >=
                0.5 * thermal_grid_model.der_thermal_power_vector_nominal[der_index]
            )
            optimization_problem.der_constraints.add(
                optimization_problem.der_thermal_power_vector[timestep, der]
                <=
                1.0 * thermal_grid_model.der_thermal_power_vector_nominal[der_index]
            )

    # Define thermal grid constraints.
    optimization_problem.thermal_grid_constraints = pyo.ConstraintList()
    for timestep in scenario_data.timesteps:
        for node_index, node in enumerate(thermal_grid_model.nodes):
            if node[1] == 'source':
                optimization_problem.thermal_grid_constraints.add(
                    -1.0 * optimization_problem.source_flow[timestep]
                    ==
                    sum(
                        thermal_grid_model.branch_node_incidence_matrix[node_index, branch_index]
                        * optimization_problem.branch_flow_vector[timestep, branch]
                        for branch_index, branch in enumerate(thermal_grid_model.branches)
                    )
                )
            else:
                optimization_problem.thermal_grid_constraints.add(
                    sum(
                        thermal_grid_model.der_node_incidence_matrix[node_index, der_index]
                        * optimization_problem.der_thermal_power_vector[timestep, der]
                        * thermal_grid_model.enthalpy_difference_distribution_water
                        / fledge.config.water_density
                        for der_index, der in enumerate(thermal_grid_model.ders)
                    )
                    ==
                    sum(
                        thermal_grid_model.branch_node_incidence_matrix[node_index, branch_index]
                        * optimization_problem.branch_flow_vector[timestep, branch]
                        for branch_index, branch in enumerate(thermal_grid_model.branches)
                    )
                )

    # Define objective.
    cost = 0.0
    cost += (
        sum(
            optimization_problem.source_flow[timestep]
            for timestep in scenario_data.timesteps
        )
    )
    optimization_problem.objective = (
        pyo.Objective(
            expr=cost,
            sense=pyo.minimize
        )
    )

    # Solve optimization problem.
    optimization_solver = pyo.SolverFactory(fledge.config.solver_name)
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.solver_output)
    if optimization_result.solver.termination_condition is not pyo.TerminationCondition.optimal:
        raise Exception(f"Invalid solver termination condition: {optimization_result.solver.termination_condition}")
    # optimization_problem.display()

    # Instantiate results variables.
    der_thermal_power_vector = (
        pd.DataFrame(columns=thermal_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    branch_flow_vector = (
        pd.DataFrame(columns=thermal_grid_model.ders, index=scenario_data.timesteps, dtype=np.float)
    )
    source_flow = (
        pd.DataFrame(columns=['total'], index=scenario_data.timesteps, dtype=np.float)
    )

    # Obtain results.
    for timestep in scenario_data.timesteps:

        for der in thermal_grid_model.ders:
            der_thermal_power_vector.at[timestep, der] = (
                optimization_problem.der_thermal_power_vector[timestep, der].value
            )

        for branch in thermal_grid_model.branches:
            branch_flow_vector.at[timestep, branch] = (
                optimization_problem.branch_flow_vector[timestep, branch].value
            )

        source_flow.at[timestep, 'total'] = (
            optimization_problem.source_flow[timestep].value
        )

    # Print some results.
    print(f"der_thermal_power_vector = \n{der_thermal_power_vector.to_string()}")
    print(f"branch_flow_vector = \n{branch_flow_vector.to_string()}")
    print(f"source_flow = \n{source_flow.to_string()}")


if __name__ == "__main__":
    main()