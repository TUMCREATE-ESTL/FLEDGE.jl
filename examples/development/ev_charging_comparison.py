"""Example script for EV charging study."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.sparse as sp

import mesmo


def main():

    # Settings.
    scenario_basename = 'singapore_geylang'
    results_path = mesmo.utils.get_results_path(__file__, scenario_basename)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.config.config['paths']['additional_data'].append(
        os.path.join(
            os.path.dirname(os.path.normpath(__file__)),
            '..', '..', '..', 'data_processing', 'singapore_synthetic_grid_to_mesmo', 'output'
        )
    )
    mesmo.data_interface.recreate_database()

    # Obtain problems.
    problem_dict = (
        mesmo.problems.ProblemDict({
            'Baseline scenario': mesmo.problems.NominalOperationProblem(f'{scenario_basename}_no_charging'),
            '25% EVs (uncontrolled)': mesmo.problems.OptimalOperationProblem(f'{scenario_basename}_25percent'),
            '75% EVs (uncontrolled)': mesmo.problems.OptimalOperationProblem(f'{scenario_basename}_75percent'),
            '100% EVs (uncontrolled)': mesmo.problems.OptimalOperationProblem(f'{scenario_basename}_100percent'),
            '25% EVs (smart charging)': mesmo.problems.OptimalOperationProblem(f'{scenario_basename}_25percent'),
            '75% EVs (smart charging)': mesmo.problems.OptimalOperationProblem(f'{scenario_basename}_75percent'),
            '100% EVs (smart charging)': mesmo.problems.OptimalOperationProblem(f'{scenario_basename}_100percent'),
            '100% EVs (peak shaving)': mesmo.problems.OptimalOperationProblem(f'{scenario_basename}_100percent'),
            '100% EVs (price based)': mesmo.problems.OptimalOperationProblem(f'{scenario_basename}_100percent'),
        })
    )

    # Modify problems.
    for key in problem_dict:
        if key in ['25% EVs (smart charging)', '75% EVs (smart charging)', '100% EVs (smart charging)']:
            problem_dict[key].optimization_problem.define_variable(
                'branch_peak_power', branch=problem_dict[key].electric_grid_model.branches
            )
            for branch_index in range(len(problem_dict[key].electric_grid_model.branches)):
                problem_dict[key].optimization_problem.define_constraint(
                    ('variable', np.ones((len(problem_dict[key].timesteps), 1)), dict(
                        name='branch_peak_power', branch=problem_dict[key].electric_grid_model.branches[branch_index]
                    )),
                    '>=',
                    ('variable', sp.eye(len(problem_dict[key].timesteps)), dict(
                        name='branch_power_magnitude_vector_1', timestep=problem_dict[key].timesteps,
                        branch=problem_dict[key].electric_grid_model.branches[branch_index]
                    )),
                    ('variable', sp.eye(len(problem_dict[key].timesteps)), dict(
                        name='branch_power_magnitude_vector_2', timestep=problem_dict[key].timesteps,
                        branch=problem_dict[key].electric_grid_model.branches[branch_index]
                    ))
                )
            problem_dict[key].optimization_problem.define_objective(
                ('variable', 1e9, dict(name='branch_peak_power'))
            )
        elif key in ['100% EVs (peak shaving)']:
            # TODO: Label optimization variables with 'per_unit' for better understanding?
            problem_dict[key].optimization_problem.define_variable('der_peak_power')
            problem_dict[key].optimization_problem.define_constraint(
                ('variable', np.ones((len(problem_dict[key].timesteps), 1)), dict(name='der_peak_power')),
                '<=',
                ('variable', (
                    sp.block_diag(
                        [np.array([np.real(problem_dict[key].electric_grid_model.der_power_vector_reference)])]
                        * len(problem_dict[key].timesteps)
                    )
                ), dict(
                    name='der_active_power_vector', timestep=problem_dict[key].timesteps,
                    der=problem_dict[key].electric_grid_model.ders
                ))
            )
            problem_dict[key].optimization_problem.define_objective(
                ('variable', -1e9, dict(name='der_peak_power'))
            )
        elif key in ['100% EVs (price based)']:
            price_data = mesmo.data_interface.PriceData(
                f'{scenario_basename}_100percent',
                price_type='singapore_wholesale'
            )
            problem_dict[key].linear_electric_grid_model_set.define_optimization_parameters(
                problem_dict[key].optimization_problem,
                price_data
            )
            problem_dict[key].der_model_set.define_optimization_parameters(
                problem_dict[key].optimization_problem,
                price_data
            )

    # Solve problems.
    for problem in problem_dict.values():
        problem.solve()

    # Obtain results.
    results_dict = (
        mesmo.problems.ResultsDict({
            label: problem.get_results()
            for label, problem in problem_dict.items()
        })
    )
    results_sets = dict()
    results_sets['_nominal'] = {
        key: value
        for key, value in results_dict.items()
        if key in [
            'Baseline scenario',
            '25% EVs (uncontrolled)',
            '75% EVs (uncontrolled)',
            '100% EVs (uncontrolled)'
        ]
    }
    results_sets['_nominal_ev_only'] = {
        key: value
        for key, value in results_dict.items()
        if key in [
            '25% EVs (uncontrolled)',
            '75% EVs (uncontrolled)',
            '100% EVs (uncontrolled)'
        ]
    }
    results_sets['_optimal25'] = {
        key: value
        for key, value in results_dict.items()
        if key in [
            'Baseline scenario',
            '25% EVs (uncontrolled)',
            '25% EVs (smart charging)'
        ]
    }
    results_sets['_optimal75'] = {
        key: value
        for key, value in results_dict.items()
        if key in [
            'Baseline scenario',
            '75% EVs (uncontrolled)',
            '75% EVs (smart charging)'
        ]
    }
    results_sets['_optimal100'] = {
        key: value
        for key, value in results_dict.items()
        if key in [
            'Baseline scenario',
            '100% EVs (uncontrolled)',
            '100% EVs (smart charging)'
        ]
    }
    results_sets['_peak_shaving'] = {
        key: value
        for key, value in results_dict.items()
        if key in [
            '100% EVs (uncontrolled)',
            '100% EVs (peak shaving)',
            '100% EVs (smart charging)'
        ]
    }
    results_sets['_price_based'] = {
        key: value
        for key, value in results_dict.items()
        if key in [
            '100% EVs (uncontrolled)',
            '100% EVs (price based)',
            '100% EVs (smart charging)'
        ]
    }

    # Make plots.
    for results_label, results_set in results_sets.items():
        if len(results_set) > 0:
            mesmo.plots.plot_histogram_cumulative_branch_utilization(
                results_set,
                results_path,
                filename_suffix=results_label,
                branch_type='transformer',
                vertical_line=0.9
            )
            mesmo.plots.plot_histogram_cumulative_branch_utilization(
                results_set,
                results_path,
                filename_suffix=results_label,
                branch_type='line',
                vertical_line=0.9
            )
            mesmo.plots.plot_histogram_node_utilization(
                results_set,
                results_path,
                filename_suffix=results_label,
            )
            mesmo.plots.plot_aggregate_timeseries_der_power(
                results_set,
                results_path,
                filename_suffix=results_label,
                value_unit_label='MW',
                der_type_labels={'fixed_load': 'Base load', 'flexible_ev_charger': 'EV charging'}
            )

    # # Plot individual DER results.
    # for results_label in ['100% EVs (smart charging)']:
    #     for der_name, der_model in results_dict[results_label].der_model_set.flexible_der_models.items():
    #         for output in der_model.outputs:
    #             figure = go.Figure()
    #             figure.add_trace(go.Scatter(
    #                 x=der_model.output_maximum_timeseries.index,
    #                 y=der_model.output_maximum_timeseries.loc[:, output].values,
    #                 name='Maximum',
    #                 line=go.scatter.Line(shape='hv')
    #             ))
    #             figure.add_trace(go.Scatter(
    #                 x=der_model.output_minimum_timeseries.index,
    #                 y=der_model.output_minimum_timeseries.loc[:, output].values,
    #                 name='Minimum',
    #                 line=go.scatter.Line(shape='hv')
    #             ))
    #             figure.add_trace(go.Scatter(
    #                 x=results_dict[results_label]['output_vector'].index,
    #                 y=results_dict[results_label]['output_vector'].loc[:, (der_name, output)].values,
    #                 name='Optimal',
    #                 line=go.scatter.Line(shape='hv', width=4)
    #             ))
    #             figure.update_layout(
    #                 title=f'DER: {der_name} / Output: {output}',
    #                 xaxis=go.layout.XAxis(tickformat='%H:%M'),
    #                 legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
    #             )
    #             # figure.show()
    #             mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, f'output_{der_name}_{output}'))
    #         for disturbance in der_model.disturbances:
    #             figure = go.Figure()
    #             figure.add_trace(go.Scatter(
    #                 x=der_model.disturbance_timeseries.index,
    #                 y=der_model.disturbance_timeseries.loc[:, disturbance].values,
    #                 line=go.scatter.Line(shape='hv')
    #             ))
    #             figure.update_layout(
    #                 title=f'DER: {der_name} / Disturbance: {disturbance}',
    #                 xaxis=go.layout.XAxis(tickformat='%H:%M'),
    #                 showlegend=False
    #             )
    #             # figure.show()
    #             mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, f'disturbance_{der_name}_{disturbance}'))

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
