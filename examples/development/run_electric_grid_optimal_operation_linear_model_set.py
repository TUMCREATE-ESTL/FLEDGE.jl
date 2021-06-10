"""Example script for setting up and solving an electric grid optimal operation problem with linear grid model set."""

import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.problems
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_6node'
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    # fledge.data_interface.recreate_database()

    # Obtain data.
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Obtain electric grid model.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)

    # Obtain DER model set & pre-solve.
    # - Pre-solve runs the standalone DER optimal operation problem to obtain a realistic DER power times series
    #   estimate for flexible DERs.
    der_model_set = fledge.der_models.DERModelSet(scenario_name)
    pre_solve_der_results = der_model_set.pre_solve(price_data)

    # Obtain power-flow solution set.
    power_flow_solution_set = (
        fledge.electric_grid_models.PowerFlowSolutionSet(
            electric_grid_model,
            pre_solve_der_results
        )
    )
    pre_solve_power_flow_results = power_flow_solution_set.get_results()

    # Obtain linear electric grid model set.
    linear_electric_grid_model_set = (
        fledge.electric_grid_models.LinearElectricGridModelSet(
            electric_grid_model,
            power_flow_solution_set
        )
    )

    # Obtain objective value based on pre-solve results.
    # - Not needed here, just for demonstration.
    pre_solve_objective_value = 0.0
    pre_solve_objective_value += (
        der_model_set.evaluate_optimization_objective(
            pre_solve_der_results,
            price_data,
            has_electric_grid_objective=True
        )
    )
    pre_solve_objective_value += (
        linear_electric_grid_model_set.evaluate_optimization_objective(
            pre_solve_power_flow_results,
            price_data
        )
    )
    print(f"pre_solve_objective_value = {pre_solve_objective_value}")

    # Instantiate optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define optimization variables.
    linear_electric_grid_model_set.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_variables(optimization_problem)

    # Define constraints.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 10.0 * electric_grid_model.branch_power_vector_magnitude_reference
    linear_electric_grid_model_set.define_optimization_constraints(
        optimization_problem,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
    )
    der_model_set.define_optimization_constraints(optimization_problem)

    # Define objective.
    linear_electric_grid_model_set.define_optimization_objective(
        optimization_problem,
        price_data,
    )
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_data
    )

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = fledge.problems.Results()
    results.update(linear_electric_grid_model_set.get_optimization_results(optimization_problem))
    results.update(der_model_set.get_optimization_results(optimization_problem))

    # Print results.
    print(results)

    # Store results to CSV.
    results.save(results_path)

    # Obtain DLMPs.
    dlmps = (
        linear_electric_grid_model_set.get_optimization_dlmps(
            optimization_problem,
            price_data
        )
    )

    # Print DLMPs.
    print(dlmps)

    # Store DLMPs to CSV.
    dlmps.save(results_path)

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
