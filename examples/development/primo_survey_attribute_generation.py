"""Script for generating attribute values for the choice experiment in the PRIMO survey."""

import cvxpy as cp
import numpy as np
from multimethod import multimethod
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mesmo


def main():

    # Settings.
    scenario_name = 'primo_survey'
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data.
    der_data = mesmo.data_interface.DERData(scenario_name)
    price_data = mesmo.data_interface.PriceData(scenario_name)
    timesteps_off_peak = (der_data.scenario_data.timesteps.hour <= 7) | (der_data.scenario_data.timesteps.hour >= 23)
    price_data.price_timeseries.loc[timesteps_off_peak, ('active_power', slice(None), slice(None))] += (
        0.0476 / 1e3 * der_data.scenario_data.scenario.at['base_apparent_power']
    )
    price_data.price_timeseries.loc[~timesteps_off_peak, ('active_power', slice(None), slice(None))] += (
        0.0617 / 1e3 * der_data.scenario_data.scenario.at['base_apparent_power']
    )

    # Obtain air-conditioning models.
    ac_1 = mesmo.der_models.FlexibleBuildingModel(der_data, 'flexible_building')
    ac_1.disturbance_output_matrix *= 0.0
    outputs_temperature = ac_1.outputs.str.contains('temperature')
    outputs_heat = ac_1.outputs.str.contains('_heat_')
    timestep_reset = ac_1.timesteps.hour == 00
    ac_1.output_maximum_timeseries.loc[:, outputs_heat] = 0.0
    ac_1.output_minimum_timeseries.loc[timestep_reset, outputs_temperature] = (
        ac_1.output_maximum_timeseries.loc[timestep_reset, outputs_temperature].values - 0.01
    )
    ac_1.output_minimum_timeseries.loc[:, outputs_temperature] = (
        ac_1.output_maximum_timeseries.loc[:, outputs_temperature].values - 0.01
    )
    ac_2 = mesmo.der_models.FlexibleBuildingModel(der_data, 'flexible_building')
    ac_2.disturbance_output_matrix *= 0.0
    ac_2.output_maximum_timeseries.loc[:, outputs_heat] = 0.0
    ac_2.output_minimum_timeseries.loc[timestep_reset, outputs_temperature] = (
        ac_2.output_maximum_timeseries.loc[timestep_reset, outputs_temperature].values - 0.01
    )
    timesteps_nonsmart = (
        np.convolve(np.random.rand(len(ac_2.timesteps) + 3), np.ones(4)/4, mode='valid') > 0.5
    )
    ac_2.output_minimum_timeseries.loc[timesteps_nonsmart, outputs_temperature] = (
        ac_2.output_maximum_timeseries.loc[timesteps_nonsmart, outputs_temperature].values - 0.01
    )
    ac_3 = mesmo.der_models.FlexibleBuildingModel(der_data, 'flexible_building')
    ac_3.disturbance_output_matrix *= 0.0
    ac_3.output_maximum_timeseries.loc[:, outputs_heat] = 0.0
    ac_3.output_minimum_timeseries.loc[timestep_reset, outputs_temperature] = (
        ac_3.output_maximum_timeseries.loc[timestep_reset, outputs_temperature].values - 0.01
    )

    # Obtain EV charger models.
    ev_1 = mesmo.der_models.FlexibleEVChargerModel(der_data, 'flexible_ev_charger')
    # timesteps_urgent_depart = (ev_1.timesteps.hour > 20) | (ev_1.timesteps.hour < 17)
    timesteps_urgent_depart = ev_1.timesteps.hour > 11
    ev_1.output_maximum_timeseries.loc[timesteps_urgent_depart, 'active_power_charge'] = 0.0
    ev_2 = mesmo.der_models.FlexibleEVChargerModel(der_data, 'flexible_ev_charger')
    ev_3 = mesmo.der_models.FlexibleEVChargerModel(der_data, 'flexible_ev_charger')
    ev_3.output_maximum_timeseries.loc[:, 'active_power_discharge'] = (
        ev_3.output_maximum_timeseries.loc[:, 'active_power_charge'].values
    )

    # Obtain solutions.
    results = {
        ('ac', 'fixed', 'yes'): solve_problem(ac_1, price_data),
        ('ac', 'upper/lower', 'yes'): solve_problem(ac_2, price_data),
        ('ac', 'upper/lower', 'no'): solve_problem(ac_3, price_data),
        ('ev', '3h', 'no'): solve_problem(ev_1, price_data),
        ('ev', '8h', 'no'): solve_problem(ev_2, price_data),
        ('ev', '8h', 'yes'): solve_problem(ev_3, price_data)
    }

    # Obtain cost distribution.
    costs_daily = (
        pd.DataFrame(
            index=price_data.price_timeseries.index.strftime('%Y-%m-%d').unique()[:-1].rename(None),
            columns=pd.MultiIndex.from_tuples(results.keys(), names=['type', 'attribute_1', 'attribute_2'])
        )
    )
    for label, result in results.items():
        results[label].cost_daily = (
            result.cost_timeseries.groupby(result.cost_timeseries.index.strftime('%Y-%m-%d')).sum().iloc[:-1]
        )
        results[label].cost_mean = results[label].cost_daily.mean()
        results[label].cost_var = results[label].cost_daily.var()
        costs_daily.loc[results[label].cost_daily.index, [label]] = results[label].cost_daily.values
    # Filter valid cost values.
    ac_invalid = (
        (costs_daily.loc[:, ('ac', 'fixed', 'yes')] < costs_daily.loc[:, ('ac', 'upper/lower', 'yes')])
        | (costs_daily.loc[:, ('ac', 'fixed', 'yes')] < costs_daily.loc[:, ('ac', 'upper/lower', 'no')])
        | (costs_daily.loc[:, ('ac', 'upper/lower', 'yes')] < costs_daily.loc[:, ('ac', 'upper/lower', 'no')])
    )
    ev_invalid = (
        (costs_daily.loc[:, ('ev', '3h', 'no')] < costs_daily.loc[:, ('ev', '8h', 'no')])
        | (costs_daily.loc[:, ('ev', '3h', 'no')] < costs_daily.loc[:, ('ev', '8h', 'yes')])
        | (costs_daily.loc[:, ('ev', '8h', 'no')] < costs_daily.loc[:, ('ev', '8h', 'yes')])
    )
    costs_daily.loc[ac_invalid, ('ac', slice(None), slice(None))] = np.nan
    costs_daily.loc[ev_invalid, ('ev', slice(None), slice(None))] = np.nan
    # Generate cost overview.
    costs_overview = pd.DataFrame(columns=costs_daily.columns)
    costs_overview.loc['Daily mean [$/d]', costs_daily.columns] = round(costs_daily.mean(), 2)
    costs_overview.loc['Monthly mean [$/m]', costs_daily.columns] = round(costs_daily.mean() * 30, 2)
    costs_overview.loc['Hourly mean [¢/h]', costs_daily.columns] = round(costs_daily.mean() * 100 / 24, 2)
    print(f"costs_overview = \n{costs_overview}")

    # Save / plot results.
    costs_daily.to_csv(os.path.join(results_path, 'costs_daily.csv'))
    costs_overview.to_csv(os.path.join(results_path, 'costs_overview.csv'))
    save_results(results, results_path)
    plot_results(results, results_path)

    # Plot prices.
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


def solve_problem(
        flexible_der_model: mesmo.der_models.FlexibleDERModel,
        price_data: mesmo.data_interface.PriceData
) -> mesmo.der_models.DERModelOperationResults:

    # Enforce storage states, initial state is linked to final state.
    flexible_der_model.storage_states = flexible_der_model.states

    # Instantiate optimization problem.
    optimization_problem = mesmo.utils.OptimizationProblem()

    # Define / solve optimization problem.
    flexible_der_model.define_optimization_variables(optimization_problem)
    flexible_der_model.define_optimization_constraints(optimization_problem)
    flexible_der_model.define_optimization_objective(optimization_problem, price_data)
    optimization_problem.solve()

    # Obtain results.
    results = flexible_der_model.get_optimization_results(optimization_problem)
    results.objective = optimization_problem.objective.value
    # Obtain timestep interval in hours, for conversion of power to energy.
    timestep_interval_hours = (flexible_der_model.timesteps[1] - flexible_der_model.timesteps[0]) / pd.Timedelta('1h')
    results.cost_timeseries = (
        -1.0 * (flexible_der_model.mapping_active_power_by_output @ results.output_vector.T).T
        * price_data.price_timeseries.loc[:, [('active_power', 'source', 'source')]].values
        * timestep_interval_hours
    )

    return results


def save_results(
        results: dict,
        results_path: str
):

    for label, result in results.items():

        # Parse label.
        label = mesmo.utils.get_alphanumeric_string(str(label))

        # Create folder.
        try:
            os.mkdir(os.path.join(results_path, label))
        except Exception:
            pass

        result.save(os.path.join(results_path, label))


@multimethod
def plot_results(
        results: dict,
        results_path: str
):

    for label, result in results.items():
        plot_results(result, results_path, label)


@multimethod
def plot_results(
        results: mesmo.der_models.DERModelOperationResults,
        results_path: str,
        label: tuple
):

    # Parse label.
    label = mesmo.utils.get_alphanumeric_string(str(label))

    # Create folder.
    try:
        os.mkdir(os.path.join(results_path, label))
    except Exception:
        pass

    # Plot outputs.
    for output in results.der_model.outputs:

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=results.der_model.output_maximum_timeseries.index,
            y=results.der_model.output_maximum_timeseries.loc[:, output].values,
            name='Maximum',
            line=go.scatter.Line(shape='hv')
        ))
        figure.add_trace(go.Scatter(
            x=results.der_model.output_minimum_timeseries.index,
            y=results.der_model.output_minimum_timeseries.loc[:, output].values,
            name='Minimum',
            line=go.scatter.Line(shape='hv')
        ))
        figure.add_trace(go.Scatter(
            x=results['output_vector'].index,
            y=results['output_vector'].loc[:, output].values,
            name='Optimal',
            line=go.scatter.Line(shape='hv')
        ))
        figure.update_layout(
            title=f'Output: {output}',
            # xaxis=go.layout.XAxis(tickformat='%H:%M'),
            legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
        )
        # figure.show()
        mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, label, output))

    # Plot disturbances.
    for disturbance in results.der_model.disturbances:

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=results.der_model.disturbance_timeseries.index,
            y=results.der_model.disturbance_timeseries.loc[:, disturbance].values,
            line=go.scatter.Line(shape='hv')
        ))
        figure.update_layout(
            title=f'Disturbance: {disturbance}',
            # xaxis=go.layout.XAxis(tickformat='%H:%M'),
            showlegend=False
        )
        # figure.show()
        mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, label, disturbance))


if __name__ == '__main__':
    main()
