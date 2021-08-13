"""Example script for setting up and solving a standard form flexible DER optimal operation problem."""

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
    scenario_name = 'singapore_tanjongpagar'
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)
    os.mkdir(os.path.join(results_path, 'standard_form'))
    os.mkdir(os.path.join(results_path, 'traditional_form'))

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    # mesmo.data_interface.recreate_database()

    # Obtain data.
    der_data = mesmo.data_interface.DERData(scenario_name)
    price_data = mesmo.data_interface.PriceData(scenario_name, der_data)
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    has_thermal_grid = pd.notnull(scenario_data.scenario.at['thermal_grid_name'])

    # Obtain model.
    der_model_set = mesmo.der_models.DERModelSet(der_data)
    electric_grid_model = mesmo.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = mesmo.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model = (
        mesmo.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )
    linear_electric_grid_model_set = (
        mesmo.electric_grid_models.LinearElectricGridModelSet(
            electric_grid_model,
            power_flow_solution,
            linear_electric_grid_model_method=mesmo.electric_grid_models.LinearElectricGridModelGlobal
        )
    )
    if has_thermal_grid:
        thermal_grid_model = mesmo.thermal_grid_models.ThermalGridModel(scenario_name)
        thermal_power_flow_solution = mesmo.thermal_grid_models.ThermalPowerFlowSolution(thermal_grid_model)
        linear_thermal_grid_model = (
            mesmo.thermal_grid_models.LinearThermalGridModelGlobal(
                thermal_grid_model,
                thermal_power_flow_solution
            )
        )
        linear_thermal_grid_model_set = (
            mesmo.thermal_grid_models.LinearThermalGridModelSet(
                thermal_grid_model,
                thermal_power_flow_solution,
                linear_thermal_grid_model_method=mesmo.thermal_grid_models.LinearThermalGridModelGlobal
            )
        )

    # Define grid limits.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 10.0 * electric_grid_model.branch_power_vector_magnitude_reference

    # Instantiate optimization problem.
    mesmo.utils.log_time('standard-form interface')
    mesmo.utils.log_time('standard-form problem')
    optimization_problem = mesmo.utils.OptimizationProblem()

    # Define linear electric grid model set problem.
    linear_electric_grid_model_set.define_optimization_variables(optimization_problem)
    linear_electric_grid_model_set.define_optimization_parameters(
        optimization_problem,
        price_data,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
    )
    linear_electric_grid_model_set.define_optimization_constraints(optimization_problem)
    linear_electric_grid_model_set.define_optimization_objective(optimization_problem)

    # Define linear thermal grid model set problem.
    if has_thermal_grid:
        linear_thermal_grid_model_set.define_optimization_variables(optimization_problem)
        linear_thermal_grid_model_set.define_optimization_parameters(
            optimization_problem,
            price_data
        )
        linear_thermal_grid_model_set.define_optimization_constraints(optimization_problem)
        linear_thermal_grid_model_set.define_optimization_objective(optimization_problem)

    # Define DER model set problem.
    der_model_set.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_parameters(optimization_problem, price_data)
    der_model_set.define_optimization_constraints(optimization_problem)
    optimization_problem.flags['has_thermal_grid_objective'] = True
    der_model_set.define_optimization_objective(optimization_problem)
    mesmo.utils.log_time('standard-form problem')

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results_1 = mesmo.problems.Results()
    objective_1 = optimization_problem.evaluate_objective(optimization_problem.x_vector)
    results_1.update(linear_electric_grid_model_set.get_optimization_results(optimization_problem))
    results_1.update(linear_electric_grid_model_set.get_optimization_dlmps(optimization_problem, price_data))
    objective_1_electric_grid = linear_electric_grid_model_set.evaluate_optimization_objective(results_1, price_data)
    if has_thermal_grid:
        results_1.update(linear_thermal_grid_model_set.get_optimization_results(optimization_problem))
        results_1.update(linear_thermal_grid_model_set.get_optimization_dlmps(optimization_problem, price_data))
        objective_1_thermal_grid = linear_thermal_grid_model_set.evaluate_optimization_objective(results_1, price_data)
    results_1.update(der_model_set.get_optimization_results(optimization_problem))
    objective_1_der = der_model_set.evaluate_optimization_objective(results_1, price_data, has_electric_grid_objective=True)
    results_1.save(os.path.join(results_path, 'standard_form'))
    mesmo.utils.log_time('standard-form interface')
    der_model_set.pre_solve(price_data)

    # Instantiate optimization problem.
    mesmo.utils.log_time('cvxpy interface')
    mesmo.utils.log_time('cvxpy problem')
    optimization_problem_cvxpy = OptimizationProblemCVXPY()

    # Define electric grid problem.
    linear_electric_grid_model.define_optimization_variables(optimization_problem_cvxpy)
    linear_electric_grid_model.define_optimization_constraints(optimization_problem_cvxpy)
    linear_electric_grid_model.define_optimization_objective(optimization_problem_cvxpy, price_data)

    # Define thermal grid problem.
    if has_thermal_grid:
        linear_thermal_grid_model.define_optimization_variables(optimization_problem_cvxpy)
        linear_thermal_grid_model.define_optimization_constraints(optimization_problem_cvxpy)
        linear_thermal_grid_model.define_optimization_objective(optimization_problem_cvxpy, price_data)

    # Define flexible DER state space variables.
    optimization_problem_cvxpy.state_vector = dict.fromkeys(der_model_set.flexible_der_names)
    optimization_problem_cvxpy.control_vector = dict.fromkeys(der_model_set.flexible_der_names)
    optimization_problem_cvxpy.output_vector = dict.fromkeys(der_model_set.flexible_der_names)
    for der_name in der_model_set.flexible_der_names:
        optimization_problem_cvxpy.state_vector[der_name] = (
            cp.Variable((
                len(der_model_set.flexible_der_models[der_name].timesteps),
                len(der_model_set.flexible_der_models[der_name].states)
            ))
        )
        optimization_problem_cvxpy.control_vector[der_name] = (
            cp.Variable((
                len(der_model_set.flexible_der_models[der_name].timesteps),
                len(der_model_set.flexible_der_models[der_name].controls)
            ))
        )
        optimization_problem_cvxpy.output_vector[der_name] = (
            cp.Variable((
                len(der_model_set.flexible_der_models[der_name].timesteps),
                len(der_model_set.flexible_der_models[der_name].outputs)
            ))
        )

    # Define DER constraints for each DER.
    for der_name in der_model_set.der_names:
        der_model_set.der_models[der_name].define_optimization_constraints(optimization_problem_cvxpy)

    # Define objective for each DER.
    for der_name in der_model_set.der_names:
        der_model_set.der_models[der_name].define_optimization_objective(optimization_problem_cvxpy, price_data)

    mesmo.utils.log_time('cvxpy problem')

    # Solve optimization problem.
    mesmo.utils.log_time('cvxpy solve')
    optimization_problem_cvxpy.solve()
    mesmo.utils.log_time('cvxpy solve')

    # Obtain results.
    mesmo.utils.log_time('cvxpy get results')
    # Instantiate results variables.
    state_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.states)
    control_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.controls)
    output_vector = pd.DataFrame(0.0, index=der_model_set.timesteps, columns=der_model_set.outputs)
    # Obtain results.
    for der_name in der_model_set.flexible_der_names:
        state_vector.loc[:, (der_name, slice(None))] = (
            optimization_problem_cvxpy.state_vector[der_name].value
        )
        control_vector.loc[:, (der_name, slice(None))] = (
            optimization_problem_cvxpy.control_vector[der_name].value
        )
        output_vector.loc[:, (der_name, slice(None))] = (
            optimization_problem_cvxpy.output_vector[der_name].value
        )
    results_2 = mesmo.problems.Results(
        state_vector=state_vector,
        control_vector=control_vector,
        output_vector=output_vector,
    )
    results_2.update(linear_electric_grid_model.get_optimization_results(optimization_problem_cvxpy))
    results_2.update(linear_electric_grid_model.get_optimization_dlmps(optimization_problem_cvxpy, price_data))
    if has_thermal_grid:
        results_2.update(linear_thermal_grid_model.get_optimization_results(optimization_problem_cvxpy))
        results_2.update(linear_thermal_grid_model.get_optimization_dlmps(optimization_problem_cvxpy, price_data))
    objective_2 = float(optimization_problem_cvxpy.objective.value)
    results_2.save(os.path.join(results_path, 'traditional_form'))
    mesmo.utils.log_time('cvxpy get results')
    mesmo.utils.log_time('cvxpy interface')

    # Plot results.
    for der_name, der_model in der_model_set.flexible_der_models.items():
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
                y=results_1['output_vector'].loc[:, (der_name, output)].values,
                name='Optimal (standard form)',
                line=go.scatter.Line(shape='hv', width=4)
            ))
            figure.add_trace(go.Scatter(
                x=results_2['output_vector'].index,
                y=results_2['output_vector'].loc[:, (der_name, output)].values,
                name='Optimal (traditional form)',
                line=go.scatter.Line(shape='hv', width=2)
            ))
            figure.update_layout(
                title=f'DER: {der_name} / Output: {output}',
                xaxis=go.layout.XAxis(tickformat='%H:%M'),
                legend=go.layout.Legend(x=0.99, xanchor='auto', y=0.99, yanchor='auto')
            )
            # figure.show()
            mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, f'output_{der_name}_{output}'))

    for der_name, der_model in der_model_set.flexible_der_models.items():
        for disturbance in der_model.disturbances:

            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=der_model.disturbance_timeseries.index,
                y=der_model.disturbance_timeseries.loc[:, disturbance].values,
                line=go.scatter.Line(shape='hv')
            ))
            figure.update_layout(
                title=f'DER: {der_name} / Disturbance: {disturbance}',
                xaxis=go.layout.XAxis(tickformat='%H:%M'),
                showlegend=False
            )
            # figure.show()
            mesmo.utils.write_figure_plotly(figure, os.path.join(results_path, f'disturbance_{der_name}_{disturbance}'))

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


class OptimizationProblemCVXPY(object):
    """Optimization problem object for use with CVXPY."""

    constraints: list
    objective: cp.Expression
    has_der_objective: bool = False
    has_electric_grid_objective: bool = False
    has_thermal_grid_objective: bool = False
    cvxpy_problem: cp.Problem

    def __init__(self):

        self.constraints = []
        self.objective = cp.Constant(value=0.0)

    def solve(
            self,
            keep_problem=False,
            **kwargs
    ):

        # Instantiate CVXPY problem object.
        if hasattr(self, 'cvxpy_problem') and keep_problem:
            pass
        else:
            self.cvxpy_problem = cp.Problem(cp.Minimize(self.objective), self.constraints)

        # Solve optimization problem.
        self.cvxpy_problem.solve(
            solver=(
                mesmo.config.config['optimization']['solver_name'].upper()
                if mesmo.config.config['optimization']['solver_name'] is not None
                else None
            ),
            verbose=mesmo.config.config['optimization']['show_solver_output'],
            **kwargs,
            **mesmo.config.solver_parameters
        )

        # Assert that solver exited with an optimal solution. If not, raise an error.
        if not (self.cvxpy_problem.status == cp.OPTIMAL):
            raise cp.SolverError(f"Solver termination status: {self.cvxpy_problem.status}")


if __name__ == '__main__':
    main()
