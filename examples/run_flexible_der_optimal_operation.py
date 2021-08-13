"""Example script for setting up and solving a flexible DER optimal operation problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mesmo


def main():

    # Settings.
    scenario_name = 'singapore_6node'
    der_name = '4_2'  # Must be valid flexible DER from given scenario.
    results_path = mesmo.utils.get_results_path(__file__, f'{scenario_name}_der_{der_name}')

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data.
    price_data = mesmo.data_interface.PriceData(scenario_name)

    # Obtain model.
    # - The following creates a DER model set with a single item.
    # - DER model set is obtained rather than single DER model, because the optimization problem definitions are only
    #   implemented in the DER model set.
    der_model_set = mesmo.der_models.DERModelSet(scenario_name, der_name=der_name)

    # Define optimization problem.
    optimization_problem = mesmo.utils.OptimizationProblem()
    der_model_set.define_optimization_problem(optimization_problem, price_data)

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = der_model_set.get_optimization_results(optimization_problem)

    # Store results to CSV.
    results.save(results_path)

    # Plot results.
    for output in der_model_set.flexible_der_models[der_name].outputs:

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=der_model_set.flexible_der_models[der_name].output_maximum_timeseries.index,
            y=der_model_set.flexible_der_models[der_name].output_maximum_timeseries.loc[:, output].values,
            name='Maximum',
            line=go.scatter.Line(shape='hv')
        ))
        figure.add_trace(go.Scatter(
            x=der_model_set.flexible_der_models[der_name].output_minimum_timeseries.index,
            y=der_model_set.flexible_der_models[der_name].output_minimum_timeseries.loc[:, output].values,
            name='Minimum',
            line=go.scatter.Line(shape='hv')
        ))
        figure.add_trace(go.Scatter(
            x=results['output_vector'].index,
            y=results['output_vector'].loc[:, (der_name, output)].values,
            name='Optimal',
            line=go.scatter.Line(shape='hv')
        ))
        figure.update_layout(
            title=f'Output: {output}',
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
        )
        # figure.show()
        mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, output))

    for disturbance in der_model_set.flexible_der_models[der_name].disturbances:

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=der_model_set.flexible_der_models[der_name].disturbance_timeseries.index,
            y=der_model_set.flexible_der_models[der_name].disturbance_timeseries.loc[:, disturbance].values,
            line=go.scatter.Line(shape='hv')
        ))
        figure.update_layout(
            title=f'Disturbance: {disturbance}',
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            showlegend=False
        )
        # figure.show()
        mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, disturbance))

    for commodity_type in ['active_power', 'reactive_power', 'thermal_power']:

        if commodity_type in price_data.price_timeseries.columns.get_level_values('commodity_type'):
            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=price_data.price_timeseries.index,
                y=price_data.price_timeseries.loc[:, (commodity_type, 'source', 'source')].values,
                line=go.scatter.Line(shape='hv')
            ))
            figure.update_layout(
                title=f'Price: {commodity_type}',
                xaxis=go.layout.XAxis(tickformat='%H:%M')
            )
            # figure.show()
            mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, f'price_{commodity_type}'))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
