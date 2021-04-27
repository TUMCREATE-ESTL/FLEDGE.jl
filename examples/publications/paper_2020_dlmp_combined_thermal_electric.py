"""Run script for reproducing results of the Paper: 'Distribution Locational Marginal Pricing for Combined Thermal
and Electric Grid Operation', available at: <https://doi.org/10.36227/techrxiv.11918712.v1>.
"""

import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.problems
import fledge.thermal_grid_models
import fledge.utils


def main():

    # Settings.
    scenario_name = 'singapore_tanjongpagar'
    scenario = 1  # Choices: 1 (unconstrained operation), 2 (constrained branch flow), 3 (constrained pressure head).
    results_path = fledge.utils.get_results_path(__file__, f'scenario{scenario}_{scenario_name}')

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain data.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Obtain models.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )
    thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
    thermal_grid_model.energy_transfer_station_head_loss = 0.0  # TODO: Document modifications for Thermal Electric DLMP paper
    thermal_grid_model.cooling_plant_efficiency = 10.0  # TODO: Document modifications for Thermal Electric DLMP paper
    thermal_power_flow_solution = fledge.thermal_grid_models.ThermalPowerFlowSolution(thermal_grid_model)
    linear_thermal_grid_model = (
        fledge.thermal_grid_models.LinearThermalGridModel(
            thermal_grid_model,
            thermal_power_flow_solution
        )
    )
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Instantiate optimization problem.
    optimization_problem = fledge.utils.OptimizationProblem()

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define linear electric grid model constraints.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(power_flow_solution.node_voltage_vector)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(power_flow_solution.node_voltage_vector)
    branch_power_magnitude_vector_maximum = 1.5 * np.abs(power_flow_solution.branch_power_vector_1)
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps,
        node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
        node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
        branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
    )

    # Define thermal grid model variables.
    linear_thermal_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define thermal grid model constraints.
    node_head_vector_minimum = 1.5 * thermal_power_flow_solution.node_head_vector
    branch_flow_vector_maximum = 1.5 * thermal_power_flow_solution.branch_flow_vector
    # Modify limits for scenarios.
    if scenario == 1:
        pass
    elif scenario == 2:
        branch_flow_vector_maximum[thermal_grid_model.branches.get_loc('4')] *= 0.1 / 1.5
    elif scenario == 3:
        node_head_vector_minimum[thermal_grid_model.nodes.get_loc(('no_source', '15'))] *= 0.1 / 1.5
    else:
        ValueError(f"Invalid scenario: {scenario}")
    linear_thermal_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps,
        node_head_vector_minimum=node_head_vector_minimum,
        branch_flow_vector_maximum=branch_flow_vector_maximum
    )

    # Define DER variables.
    der_model_set.define_optimization_variables(
        optimization_problem
    )

    # Define DER constraints.
    der_model_set.define_optimization_constraints(
        optimization_problem,
        electric_grid_model=electric_grid_model,
        thermal_grid_model=thermal_grid_model
    )

    # Define objective.
    linear_thermal_grid_model.define_optimization_objective(
        optimization_problem,
        price_data,
        scenario_data.timesteps
    )

    # Define DER objective.
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_data
    )

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    in_per_unit = False
    results = fledge.problems.Results()
    results.update(
        linear_electric_grid_model.get_optimization_results(
            optimization_problem,
            power_flow_solution,
            scenario_data.timesteps
        )
    )
    results.update(
        linear_thermal_grid_model.get_optimization_results(
            optimization_problem,
            scenario_data.timesteps
        )
    )
    results.update(
        der_model_set.get_optimization_results(
            optimization_problem
        )
    )

    # Print results.
    print(results)

    # Store results as CSV.
    results.save(results_path)

    # Obtain DLMPs.
    dlmps = fledge.problems.Results()
    dlmps.update(
        linear_electric_grid_model.get_optimization_dlmps(
            optimization_problem,
            price_data,
            scenario_data.timesteps
        )
    )
    dlmps.update(
        linear_thermal_grid_model.get_optimization_dlmps(
            optimization_problem,
            price_data,
            scenario_data.timesteps
        )
    )

    # Print DLMPs.
    print(dlmps)

    # Store DLMPs as CSV.
    dlmps.save(results_path)

    # Plot thermal grid DLMPs.
    thermal_grid_dlmp = (
        pd.concat(
            [
                dlmps['thermal_grid_energy_dlmp_node_thermal_power'],
                dlmps['thermal_grid_pump_dlmp_node_thermal_power'],
                dlmps['thermal_grid_head_dlmp_node_thermal_power'],
                dlmps['thermal_grid_congestion_dlmp_node_thermal_power']
            ],
            axis='columns',
            keys=['energy', 'pump', 'head', 'congestion'],
            names=['dlmp_type']
        )
    )
    colors = list(color['color'] for color in matplotlib.rcParams['axes.prop_cycle'])
    for der in thermal_grid_model.ders:

        # Obtain corresponding node.
        node = (
            thermal_grid_model.nodes[
                thermal_grid_model.der_node_incidence_matrix[
                :,
                thermal_grid_model.ders.get_loc(der)
                ].toarray().ravel() == 1
                ][0]
        )

        # Create plot.
        fig, (ax1, lax) = plt.subplots(ncols=2, figsize=[7.8, 2.6], gridspec_kw={"width_ratios": [100, 1]})
        ax1.set_title(f'{der}')
        ax1.stackplot(
            scenario_data.timesteps,
            (
                thermal_grid_dlmp.loc[:, (slice(None), *node)].droplevel(['node_type', 'node_name'], axis='columns').T
                * 1.0e3
            ),
            labels=['Energy', 'Pumping', 'Head', 'Congest.'],
            colors=[colors[0], colors[1], colors[2], colors[3]],
            step='post'
        )
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price [S$/MWh]')
        # ax1.set_ylim((0.0, 10.0))
        ax2 = plt.twinx(ax1)
        ax2.plot(
            results['der_thermal_power_vector'].loc[:, der].abs() / (1 if in_per_unit else 1e6),
            label='Thrm. pw.',
            drawstyle='steps-post',
            color='darkgrey',
            linewidth=3
        )
        ax2.plot(
            results['der_active_power_vector'].loc[:, der].abs() / (1 if in_per_unit else 1e6),
            label='Active pw.',
            drawstyle='steps-post',
            color='black',
            linewidth=1.5
        )
        ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_xlim((scenario_data.timesteps[0], scenario_data.timesteps[-1]))
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Power [p.u.]') if in_per_unit else ax2.set_ylabel('Power [MW]')
        ax2.set_ylim((0.0, 1.0)) if in_per_unit else ax2.set_ylim((0.0, 30.0))
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        lax.legend((*h1, *h2), (*l1, *l2), borderaxespad=0)
        lax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'thermal_grid_dlmp_{der}.pdf'))
        plt.close()

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
