"""Example script for setting up and solving a standard form flexible DER optimal operation problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge


def main():

    # Settings.
    scenario_name = 'singapore_6node'
    der_name = '4_2'  # Must be valid flexible DER from given scenario.
    results_path = fledge.utils.get_results_path(__file__, f'{scenario_name}_der_{der_name}')

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain data.
    der_data = fledge.data_interface.DERData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Obtain model.
    der_model: fledge.der_models.FlexibleDERModel = fledge.der_models.make_der_model(der_name, der_data)

    # Obtain standard form.
    standard_form = fledge.utils.StandardForm()

    # Define variables.
    standard_form.define_variable('state_vector', timestep=der_model.timesteps, state=der_model.states)
    standard_form.define_variable('control_vector', timestep=der_model.timesteps, control=der_model.controls)
    standard_form.define_variable('output_vector', timestep=der_model.timesteps, output=der_model.outputs)

    # Define constraints.

    # Initial state.
    # - For states which represent storage state of charge, initial state of charge is final state of charge.
    if any(~der_model.states.isin(der_model.storage_states)):
        standard_form.define_constraint(
            ('constant', der_model.state_vector_initial.values[~der_model.states.isin(der_model.storage_states)]),
            '==',
            ('variable', 1.0, dict(
                name='state_vector', timestep=der_model.timesteps[0],
                state=der_model.states[~der_model.states.isin(der_model.storage_states)]
            ))
        )
    # - For other states, set initial state according to the initial state vector.
    if any(der_model.states.isin(der_model.storage_states)):
        standard_form.define_constraint(
            ('variable', 1.0, dict(
                name='state_vector', timestep=der_model.timesteps[0],
                state=der_model.states[der_model.states.isin(der_model.storage_states)]
            )),
            '==',
            ('variable', 1.0, dict(
                name='state_vector', timestep=der_model.timesteps[-1],
                state=der_model.states[der_model.states.isin(der_model.storage_states)]
            ))
        )

    # State equation.
    for timestep, timestep_previous in zip(der_model.timesteps[1:], der_model.timesteps[:-1]):
        standard_form.define_constraint(
            ('variable', 1.0, dict(name='state_vector', timestep=timestep)),
            '==',
            ('variable', der_model.state_matrix.values, dict(name='state_vector', timestep=timestep_previous)),
            ('variable', der_model.control_matrix.values, dict(name='control_vector', timestep=timestep_previous)),
            ('constant', der_model.disturbance_matrix.values @ der_model.disturbance_timeseries.loc[timestep_previous, :].values)
        )

    # Output equation.
    for timestep in der_model.timesteps:
        standard_form.define_constraint(
            ('variable', 1.0, dict(name='output_vector', timestep=timestep)),
            '==',
            ('variable', der_model.state_output_matrix.values, dict(name='state_vector', timestep=timestep)),
            ('variable', der_model.control_output_matrix.values, dict(name='control_vector', timestep=timestep)),
            ('constant', der_model.disturbance_output_matrix.values @ der_model.disturbance_timeseries.loc[timestep, :].values)
        )

    # Output limits.
    for timestep in der_model.timesteps:
        standard_form.define_constraint(
            ('variable', 1.0, dict(name='output_vector', timestep=timestep)),
            '>=',
            ('constant', der_model.output_minimum_timeseries.loc[timestep, :].values),
        )
        standard_form.define_constraint(
            ('variable', 1.0, dict(name='output_vector', timestep=timestep)),
            '<=',
            ('constant', der_model.output_maximum_timeseries.loc[timestep, :].values),
        )

    a_matrix = standard_form.get_a_matrix()
    b_vector = standard_form.get_b_vector()

    # Instantiate optimization problem.
    optimization_problem_1 = fledge.utils.OptimizationProblem()
    optimization_problem_1.x_vector = cp.Variable((len(standard_form.variables), 1))
    optimization_problem_1.constraints.append(
        a_matrix.toarray() @ optimization_problem_1.x_vector <= b_vector
    )

    # Obtain timestep interval in hours, for conversion of power to energy.
    timestep_interval_hours = (der_model.timesteps[1] - der_model.timesteps[0]) / pd.Timedelta('1h')

    # Define objective.
    # Active power cost / revenue.
    # - Cost for load / demand, revenue for generation / supply.
    for timestep in der_model.timesteps:
        x_index = fledge.utils.get_index(standard_form.variables, name='output_vector', timestep=timestep)
        optimization_problem_1.objective += (
            (
                price_data.price_timeseries.loc[timestep, ('active_power', slice(None), der_model.der_name)].values
                * -1.0 * timestep_interval_hours  # In Wh.
                @ (
                    der_model.mapping_active_power_by_output.values
                    @ optimization_problem_1.x_vector[x_index, :]
                )
            )
            + ((
                price_data.price_sensitivity_coefficient
                * timestep_interval_hours  # In Wh.
                * cp.sum((
                    der_model.mapping_active_power_by_output.values
                    @ optimization_problem_1.x_vector[x_index, :]
                ) ** 2)
            ) if price_data.price_sensitivity_coefficient != 0.0 else 0.0)
        )

    # Solve optimization problem.
    optimization_problem_1.solve()

    # Obtain results.
    results_1 = standard_form.get_results(optimization_problem_1.x_vector)

    # Instantiate optimization problem.
    optimization_problem_2 = fledge.utils.OptimizationProblem()

    # Define variables.
    optimization_problem_2.state_vector = {der_model.der_name: cp.Variable((len(der_model.timesteps), len(der_model.states)))}
    optimization_problem_2.control_vector = {der_model.der_name: cp.Variable((len(der_model.timesteps), len(der_model.controls)))}
    optimization_problem_2.output_vector = {der_model.der_name: cp.Variable((len(der_model.timesteps), len(der_model.outputs)))}

    # Define constraints.
    # Initial state.
    # - For states which represent storage state of charge, initial state of charge is final state of charge.
    if any(der_model.states.isin(der_model.storage_states)):
        optimization_problem_2.constraints.append(
            optimization_problem_2.state_vector[der_model.der_name][0, der_model.states.isin(der_model.storage_states)]
            ==
            optimization_problem_2.state_vector[der_model.der_name][-1, der_model.states.isin(der_model.storage_states)]
        )
    # - For other states, set initial state according to the initial state vector.
    if any(~der_model.states.isin(der_model.storage_states)):
        optimization_problem_2.constraints.append(
            optimization_problem_2.state_vector[der_model.der_name][0, ~der_model.states.isin(der_model.storage_states)]
            ==
            der_model.state_vector_initial.loc[~der_model.states.isin(der_model.storage_states)].values
        )

    # State equation.
    optimization_problem_2.constraints.append(
        optimization_problem_2.state_vector[der_model.der_name][1:, :]
        ==
        cp.transpose(
            der_model.state_matrix.values
            @ cp.transpose(optimization_problem_2.state_vector[der_model.der_name][:-1, :])
            + der_model.control_matrix.values
            @ cp.transpose(optimization_problem_2.control_vector[der_model.der_name][:-1, :])
            + der_model.disturbance_matrix.values
            @ np.transpose(der_model.disturbance_timeseries.iloc[:-1, :].values)
        )
    )

    # Output equation.
    optimization_problem_2.constraints.append(
        optimization_problem_2.output_vector[der_model.der_name]
        ==
        cp.transpose(
            der_model.state_output_matrix.values
            @ cp.transpose(optimization_problem_2.state_vector[der_model.der_name])
            + der_model.control_output_matrix.values
            @ cp.transpose(optimization_problem_2.control_vector[der_model.der_name])
            + der_model.disturbance_output_matrix.values
            @ np.transpose(der_model.disturbance_timeseries.values)
        )
    )

    # Output limits.
    outputs_minimum_infinite = (
        (der_model.output_minimum_timeseries == -np.inf).all()
    )
    optimization_problem_2.constraints.append(
        optimization_problem_2.output_vector[der_model.der_name][:, ~outputs_minimum_infinite]
        >=
        der_model.output_minimum_timeseries.loc[:, ~outputs_minimum_infinite].values
    )
    outputs_maximum_infinite = (
        (der_model.output_maximum_timeseries == np.inf).all()
    )
    optimization_problem_2.constraints.append(
        optimization_problem_2.output_vector[der_model.der_name][:, ~outputs_maximum_infinite]
        <=
        der_model.output_maximum_timeseries.loc[:, ~outputs_maximum_infinite].values
    )

    # Obtain timestep interval in hours, for conversion of power to energy.
    timestep_interval_hours = (der_model.timesteps[1] - der_model.timesteps[0]) / pd.Timedelta('1h')

    # Define objective.
    # Active power cost / revenue.
    # - Cost for load / demand, revenue for generation / supply.
    optimization_problem_2.objective += (
        (
            price_data.price_timeseries.loc[:, ('active_power', slice(None), der_model.der_name)].values.T
            * -1.0 * timestep_interval_hours  # In Wh.
            @ cp.transpose(
                der_model.mapping_active_power_by_output.values
                @ cp.transpose(optimization_problem_2.output_vector[der_model.der_name])
            )
        )
        + ((
            price_data.price_sensitivity_coefficient
            * timestep_interval_hours  # In Wh.
            * cp.sum((
                der_model.mapping_active_power_by_output.values
                @ cp.transpose(optimization_problem_2.output_vector[der_model.der_name])
            ) ** 2)
        ) if price_data.price_sensitivity_coefficient != 0.0 else 0.0)
    )

    # Solve optimization problem.
    optimization_problem_2.solve()

    # Obtain results.
    results_2 = der_model.get_optimization_results(optimization_problem_2)

    # Store results to CSV.
    results_2.save(results_path)

    # Plot results.
    for output in der_model.outputs:

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=der_model.output_maximum_timeseries.index,
            y=der_model.output_maximum_timeseries.loc[:, output].values,
            name='Maximum',
            line=go.scatter.Line(shape='hv')
        ))
        figure.add_trace(go.Scatter(
            x=der_model.output_minimum_timeseries.index,
            y=der_model.output_minimum_timeseries.loc[:, output].values,
            name='Minimum',
            line=go.scatter.Line(shape='hv')
        ))
        figure.add_trace(go.Scatter(
            x=results_1['output_vector'].index,
            y=results_1['output_vector'].loc[:, output].values.ravel(),
            name='Optimal (standard form)',
            line=go.scatter.Line(shape='hv', width=4)
        ))
        figure.add_trace(go.Scatter(
            x=results_2['output_vector'].index,
            y=results_2['output_vector'].loc[:, output].values,
            name='Optimal (traditional form)',
            line=go.scatter.Line(shape='hv', width=2)
        ))
        figure.update_layout(
            title=f'Output: {output}',
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
        )
        # figure.show()
        fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'output_{output}'))

    for disturbance in der_model.disturbances:

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=der_model.disturbance_timeseries.index,
            y=der_model.disturbance_timeseries.loc[:, disturbance].values,
            line=go.scatter.Line(shape='hv')
        ))
        figure.update_layout(
            title=f'Disturbance: {disturbance}',
            xaxis=go.layout.XAxis(tickformat='%H:%M'),
            showlegend=False
        )
        # figure.show()
        fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'disturbance_{disturbance}'))

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
            fledge.utils.write_figure_plotly(figure, os.path.join(results_path, f'price_{commodity_type}'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
