"""Validation script for solving a decentralized DER operation problem based on DLMPs from the centralized problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge


def main():

    # TODO: Currently not working. Review limits below.

    # Settings.
    scenario_name = fledge.config.config['tests']['scenario_name']
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain data.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name, price_type='singapore_wholesale')

    # Obtain models.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Instantiate centralized optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define optimization variables.
    linear_electric_grid_model.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_variables(optimization_problem)

    # Define constraints.
    # TODO: Review limits.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    # node_voltage_magnitude_vector_minimum[
    #     fledge.utils.get_index(electric_grid_model.nodes, node_name='4')
    # ] *= 0.965 / 0.5
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 10.0 * electric_grid_model.branch_power_vector_magnitude_reference
    # branch_power_magnitude_vector_maximum[
    #     fledge.utils.get_index(electric_grid_model.branches, branch_type='line', branch_name='2')
    # ] *= 1.2 / 10.0
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
    )
    der_model_set.define_optimization_constraints(optimization_problem)

    # Define objective.
    linear_electric_grid_model.define_optimization_objective(
        optimization_problem,
        price_data
    )
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_data
    )

    # Solve centralized optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = fledge.problems.Results()
    results.update(linear_electric_grid_model.get_optimization_results(optimization_problem))
    results.update(der_model_set.get_optimization_results(optimization_problem))

    # Print results.
    print(results)

    # Store results to CSV.
    results.save(results_path)

    # Obtain DLMPs.
    dlmps = (
        linear_electric_grid_model.get_optimization_dlmps(
            optimization_problem,
            price_data
        )
    )

    # Print DLMPs.
    print(dlmps)

    # Store DLMPs as CSV.
    dlmps.save(results_path)

    # Validate DLMPs.
    der_name = '4_2'
    price_data_dlmps = price_data.copy()
    price_data_dlmps.price_timeseries = dlmps['electric_grid_total_dlmp_price_timeseries']

    # Instantiate decentralized DER optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define optimization variables.
    der_model_set.der_models[der_name].define_optimization_variables(
        optimization_problem
    )

    # Define constraints.
    der_model_set.der_models[der_name].define_optimization_constraints(
        optimization_problem
    )

    # Define objective.
    der_model_set.der_models[der_name].define_optimization_objective(
        optimization_problem,
        price_data_dlmps
    )

    # Solve decentralized DER optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results_validation = (
        der_model_set.der_models[der_name].get_optimization_results(
            optimization_problem
        )
    )

    # Print results.
    print(results_validation)

    # Plot: Price comparison.
    values_1 = (
        1e6 / scenario_data.scenario.at['base_apparent_power']
        * price_data.price_timeseries.loc[:, ('active_power', slice(None), der_name)].iloc[:, 0]
    )
    values_2 = (
        1e6 / scenario_data.scenario.at['base_apparent_power']
        * price_data_dlmps.price_timeseries.loc[:, ('active_power', slice(None), der_name)].iloc[:, 0]
    )
    values_3 = (
        1e6 / scenario_data.scenario.at['base_apparent_power']
        * price_data_dlmps.price_timeseries.loc[:, ('reactive_power', slice(None), der_name)].iloc[:, 0]
    )

    title = 'Price comparison'
    filename = 'price_timeseries_comparison'
    y_label = 'Price'
    value_unit = 'S$/MWh'

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=values_1.index,
        y=values_1.values,
        name='Wholesale price',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values_2.index,
        y=values_2.values,
        name='DLMP (active power)',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values_3.index,
        y=values_3.values,
        name='DLMP (reactive power)',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.update_layout(
        title=title,
        yaxis_title=f'{y_label} [{value_unit}]',
        xaxis=go.layout.XAxis(tickformat='%H:%M'),
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.5, yanchor='auto')
    )
    # figure.show()
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, filename))

    # Plot: Active power comparison.
    values_1 = (
        1e-6 * scenario_data.scenario.at['base_apparent_power']
        * results['output_vector'].loc[:, (der_name, 'active_power')].abs()
    )
    values_2 = (
        1e-6 * scenario_data.scenario.at['base_apparent_power']
        * results_validation['output_vector'].loc[:, 'active_power'].abs()
    )

    title = 'Active power comparison'
    filename = 'active_power_comparison'
    y_label = 'Active power'
    value_unit = 'MW'

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=values_1.index,
        y=values_1.values,
        name='Centralized solution',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=values_2.index,
        y=values_2.values,
        name='DER (decentralized) solution',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.update_layout(
        title=title,
        yaxis_title=f'{y_label} [{value_unit}]',
        xaxis=go.layout.XAxis(tickformat='%H:%M'),
        legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.01, yanchor='auto')
    )
    # figure.show()
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, filename))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
