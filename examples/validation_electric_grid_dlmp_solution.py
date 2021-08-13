"""Validation script for solving a decentralized DER operation problem based on DLMPs from the centralized problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mesmo


def main():

    # TODO: Currently not working. Review limits below.

    # Settings.
    scenario_name = mesmo.config.config['tests']['scenario_name']
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    price_data = mesmo.data_interface.PriceData(scenario_name, price_type='singapore_wholesale')
    price_data.price_sensitivity_coefficient = 1e-6

    # Obtain models.
    electric_grid_model = mesmo.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = mesmo.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model_set = (
        mesmo.electric_grid_models.LinearElectricGridModelSet(
            electric_grid_model,
            power_flow_solution
        )
    )
    der_model_set = mesmo.der_models.DERModelSet(scenario_name)

    # Instantiate centralized optimization problem.
    optimization_centralized = mesmo.utils.OptimizationProblem()

    # Define electric grid problem.
    # TODO: Review limits.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    # node_voltage_magnitude_vector_minimum[
    #     mesmo.utils.get_index(electric_grid_model.nodes, node_name='4')
    # ] *= 0.95 / 0.5
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 10.0 * electric_grid_model.branch_power_vector_magnitude_reference
    # branch_power_magnitude_vector_maximum[
    #     mesmo.utils.get_index(electric_grid_model.branches, branch_type='line', branch_name='2')
    # ] *= 1.2 / 10.0
    linear_electric_grid_model_set.define_optimization_problem(
        optimization_centralized,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
    )

    # Define DER problem.
    der_model_set.define_optimization_problem(optimization_centralized, price_data)

    # Solve centralized optimization problem.
    optimization_centralized.solve()

    # Obtain results.
    results_centralized = mesmo.problems.Results()
    results_centralized.update(linear_electric_grid_model_set.get_optimization_results(optimization_centralized))
    results_centralized.update(der_model_set.get_optimization_results(optimization_centralized))

    # Print results.
    print(results_centralized)

    # Store results to CSV.
    results_centralized.save(results_path)

    # Obtain DLMPs.
    dlmps = linear_electric_grid_model_set.get_optimization_dlmps(optimization_centralized, price_data)

    # Print DLMPs.
    print(dlmps)

    # Store DLMPs as CSV.
    dlmps.save(results_path)

    # Validate DLMPs.
    der_name = '4_2'
    price_data_dlmps = price_data.copy()
    price_data_dlmps.price_timeseries = dlmps['electric_grid_total_dlmp_price_timeseries']

    # Instantiate decentralized DER optimization problem.
    optimization_decentralized = mesmo.utils.OptimizationProblem()

    # Define DER problem.
    der_model_set.define_optimization_problem(optimization_decentralized, price_data_dlmps)

    # Solve decentralized DER optimization problem.
    optimization_decentralized.solve()

    # Obtain results.
    results_decentralized = der_model_set.get_optimization_results(optimization_decentralized)

    # Print results.
    print(results_decentralized)

    # Plot: Price comparison.
    price_active_wholesale = (
        1e6 / scenario_data.scenario.at['base_apparent_power']
        * price_data.price_timeseries.loc[:, ('active_power', slice(None), der_name)].iloc[:, 0]
    )
    price_active_dlmp = (
        1e6 / scenario_data.scenario.at['base_apparent_power']
        * price_data_dlmps.price_timeseries.loc[:, ('active_power', slice(None), der_name)].iloc[:, 0]
    )
    price_reactive_dlmp = (
        1e6 / scenario_data.scenario.at['base_apparent_power']
        * price_data_dlmps.price_timeseries.loc[:, ('reactive_power', slice(None), der_name)].iloc[:, 0]
    )

    title = 'Price comparison'
    filename = 'price_timeseries_comparison'
    y_label = 'Price'
    value_unit = 'S$/MWh'

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=price_active_wholesale.index,
        y=price_active_wholesale.values,
        name='Wholesale price',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=price_active_dlmp.index,
        y=price_active_dlmp.values,
        name='DLMP (active power)',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=price_reactive_dlmp.index,
        y=price_reactive_dlmp.values,
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
    mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, filename))

    # Plot: Active power comparison.
    active_power_centralized = (
        1e-6 * scenario_data.scenario.at['base_apparent_power']
        * results_centralized['der_active_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs()
    )
    active_power_decentralized = (
        1e-6 * scenario_data.scenario.at['base_apparent_power']
        * results_decentralized['der_active_power_vector'].loc[:, (slice(None), der_name)].iloc[:, 0].abs()
    )

    title = 'Active power comparison'
    filename = 'active_power_comparison'
    y_label = 'Active power'
    value_unit = 'MW'

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=active_power_centralized.index,
        y=active_power_centralized.values,
        name='Centralized solution',
        fill='tozeroy',
        line=go.scatter.Line(shape='hv')
    ))
    figure.add_trace(go.Scatter(
        x=active_power_decentralized.index,
        y=active_power_decentralized.values,
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
    mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, filename))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
