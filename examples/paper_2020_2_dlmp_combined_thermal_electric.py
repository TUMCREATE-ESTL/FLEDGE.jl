"""Run script for reproducing results of the Paper XXX."""

import matplotlib.dates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pyomo.environ as pyo

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.plots
import fledge.thermal_grid_models
import fledge.utils


def main(
        scenario_number=None
):

    # Settings.
    scenario_number = 1 if scenario_number is None else scenario_number
    # Choices:
    # 1 - unconstrained operation,
    # 2 - constrained thermal grid branch flow,
    # 3 - constrained thermal grid pressure head,
    # 4 - constrained electric grid branch power,
    # 5 - constrained electric grid voltage,
    # 6 - added cooling plant (cheaper than source),
    # 7 - added cooling plant (more expensive than source),
    # 8 - added cooling plant (more expensive than source) + constrained thermal grid branch flow,
    # 9 - added very large cooling plant (cheaper than source) + constrained thermal grid branch flow,
    # 10 - added PV plant (cheaper than source),
    # 11 - added PV plant (more expensive than source),
    # 12 - added PV plant (more expensive than source) + constrained electric grid branch power,
    # 13 - added very large PV plant (cheaper than source) + constrained electric grid branch power,
    # 14 - added cooling plant (more expensive than source) + added very large PV plant (cheaper than source)
    #      + constrained electric grid branch power.
    # 15 - added cooling plant (more expensive than source) + added very large PV plant (cheaper than source)
    #      + constrained electric grid branch power + constrained thermal grid branch flow.
    if scenario_number in [1, 2, 3, 4, 5]:
        scenario_name = 'paper_2020_2_scenario_1_2_3_4_5'
    elif scenario_number in [6, 7, 8]:
        scenario_name = 'paper_2020_2_scenario_6_7_8'
    elif scenario_number in [9]:
        scenario_name = 'paper_2020_2_scenario_9'
    elif scenario_number in [10, 11, 12]:
        scenario_name = 'paper_2020_2_scenario_10_11_12'
    elif scenario_number in [13]:
        scenario_name = 'paper_2020_2_scenario_13'
    elif scenario_number in [14]:
        scenario_name = 'paper_2020_2_scenario_14'
    elif scenario_number in [15]:
        scenario_name = 'paper_2020_2_scenario_15'
    else:
        scenario_name = 'singapore_tanjongpagar_modified'

    # Obtain results path.
    results_path = (
        fledge.utils.get_results_path(f'paper_2020_2_dlmp_combined_thermal_electric_scenario_{scenario_number}', scenario_name)
    )

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain data.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    price_data = fledge.data_interface.PriceData(scenario_name)

    # Obtain price timeseries.
    price_type = 'singapore_wholesale'
    price_timeseries = price_data.price_timeseries_dict[price_type]

    # Obtain models.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
    # Use base scenario power flow for consistent linear model behavior and per unit values.
    power_flow_solution = fledge.electric_grid_models.PowerFlowSolutionFixedPoint('singapore_tanjongpagar_modified')
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelGlobal(
            electric_grid_model,
            power_flow_solution
        )
    )
    thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
    # Use base scenario power flow for consistent linear model behavior and per unit values.
    thermal_power_flow_solution = fledge.thermal_grid_models.ThermalPowerFlowSolution('singapore_tanjongpagar_modified')
    linear_thermal_grid_model = (
        fledge.thermal_grid_models.LinearThermalGridModel(
            thermal_grid_model,
            thermal_power_flow_solution
        )
    )
    der_model_set = fledge.der_models.DERModelSet(scenario_name)

    # Modify DER models depending on scenario.
    # Cooling plant.
    if scenario_number in [6, 9]:
        der_model_set.flexible_der_models['23'].control_output_matrix.at['thermal_power', 'active_power'] *= 2.0
    elif scenario_number in [7, 8, 14, 15]:
        # Cooling plant COP must remain larger than building cooling COP, otherwise cooling plant never dispatched.
        der_model_set.flexible_der_models['23'].control_output_matrix.at['thermal_power', 'active_power'] *= 0.8
    # PV plant.
    if scenario_number in [11, 12]:
        der_model_set.flexible_der_models['24'].levelized_cost_of_energy = 0.1
    if scenario_number in [15]:
        der_model_set.flexible_der_models['24'].levelized_cost_of_energy = 0.04

    # Instantiate optimization problem.
    optimization_problem = pyo.ConcreteModel()

    # Define linear electric grid model variables.
    linear_electric_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define linear electric grid model constraints.
    voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_vector_squared_maximum = 100.0 * (electric_grid_model.branch_power_vector_magnitude_reference ** 2)
    # Modify limits for scenarios.
    if scenario_number in [4, 12, 13, 14, 15]:
        branch_power_vector_squared_maximum[
            fledge.utils.get_index(electric_grid_model.branches, branch_name='4')
        ] *= 8.5 / 100.0
    elif scenario_number in [5]:
        voltage_magnitude_vector_minimum[
            fledge.utils.get_index(electric_grid_model.nodes, node_name='15')
        ] *= 0.9985 / 0.5
    else:
        pass
    linear_electric_grid_model.define_optimization_constraints(
        optimization_problem,
        scenario_data.timesteps,
        voltage_magnitude_vector_minimum=voltage_magnitude_vector_minimum,
        voltage_magnitude_vector_maximum=voltage_magnitude_vector_maximum,
        branch_power_vector_squared_maximum=branch_power_vector_squared_maximum
    )

    # Define thermal grid model variables.
    linear_thermal_grid_model.define_optimization_variables(
        optimization_problem,
        scenario_data.timesteps
    )

    # Define thermal grid model constraints.
    # TODO: Rename branch_flow_vector_maximum to branch_flow_vector_magnitude_maximum.
    # TODO: Revise node_head_vector constraint formulation.
    node_head_vector_minimum = 100.0 * thermal_power_flow_solution.node_head_vector
    branch_flow_vector_maximum = 100.0 * np.abs(thermal_power_flow_solution.branch_flow_vector)
    # Modify limits for scenarios.
    if scenario_number in [2, 8, 9, 15]:
        branch_flow_vector_maximum[
            fledge.utils.get_index(thermal_grid_model.branches, branch_name='4')
        ] *= 0.2 / 100.0
    elif scenario_number in [3]:
        node_head_vector_minimum[
            fledge.utils.get_index(thermal_grid_model.nodes, node_name='15')
        ] *= 0.2 / 100.0
    else:
        pass
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
        power_flow_solution=power_flow_solution,
        thermal_grid_model=thermal_grid_model,
        thermal_power_flow_solution=thermal_power_flow_solution
    )

    # Connect thermal grid source.
    # TODO: Incorporate this workaround in model definition.
    der_index = fledge.utils.get_index(electric_grid_model.ders, der_type='thermal_grid_source')[0]
    der = electric_grid_model.ders[der_index]
    for timestep in scenario_data.timesteps:
        optimization_problem.der_model_constraints.add(
            optimization_problem.der_active_power_vector_change[timestep, der]
            ==
            sum(
                optimization_problem.der_thermal_power_vector[timestep, thermal_der]
                for thermal_der in thermal_grid_model.ders
            )
            / thermal_grid_model.cooling_plant_efficiency
            + optimization_problem.pump_power[timestep]
            - np.real(power_flow_solution.der_power_vector[der_index])
        )
        optimization_problem.der_model_constraints.add(
            optimization_problem.der_reactive_power_vector_change[timestep, der]
            ==
            0.5  # Constant power factor.
            * optimization_problem.der_active_power_vector_change[timestep, der]
        )

    # Define electric grid objective.
    linear_electric_grid_model.define_optimization_objective(
        optimization_problem,
        price_timeseries=price_timeseries,
        timesteps=scenario_data.timesteps
    )

    # Define thermal grid objective.
    # linear_thermal_grid_model.define_optimization_objective(
    #     optimization_problem,
    #     price_timeseries=price_timeseries,
    #     timesteps=scenario_data.timesteps
    # )

    # Define DER objective.
    der_model_set.define_optimization_objective(
        optimization_problem,
        price_timeseries,
        electric_grid_model=electric_grid_model,
        thermal_grid_model=thermal_grid_model
    )

    # Solve optimization problem.
    optimization_problem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    optimization_solver = pyo.SolverFactory(fledge.config.config['optimization']['solver_name'])
    optimization_result = optimization_solver.solve(optimization_problem, tee=fledge.config.config['optimization']['show_solver_output'])
    try:
        assert optimization_result.solver.termination_condition is pyo.TerminationCondition.optimal
    except AssertionError:
        raise AssertionError(f"Solver termination condition: {optimization_result.solver.termination_condition}")
    # optimization_problem.display()

    # Obtain results.
    in_per_unit = True
    results = (
        linear_electric_grid_model.get_optimization_results(
            optimization_problem,
            power_flow_solution,
            scenario_data.timesteps,
            in_per_unit=in_per_unit
        )
    )
    results.update(
        linear_thermal_grid_model.get_optimization_results(
            optimization_problem,
            scenario_data.timesteps,
            in_per_unit=in_per_unit
        )
    )
    results.update(
        der_model_set.get_optimization_results(
            optimization_problem
        )
    )

    # Obtain additional results.
    branch_power_vector_magnitude_per_unit = (
        (
            np.sqrt(np.abs(results['branch_power_vector_1_squared']))
            + np.sqrt(np.abs(results['branch_power_vector_2_squared']))
        ) / 2
        # / electric_grid_model.branch_power_vector_magnitude_reference
    )
    branch_power_vector_magnitude_per_unit.loc['maximum', :] = branch_power_vector_magnitude_per_unit.max(axis='rows')
    node_voltage_vector_magnitude_per_unit = (
        np.abs(results['voltage_magnitude_vector'])
        # / np.abs(electric_grid_model.node_voltage_vector_reference)
    )
    node_voltage_vector_magnitude_per_unit.loc['maximum', :] = node_voltage_vector_magnitude_per_unit.max(axis='rows')
    node_voltage_vector_magnitude_per_unit.loc['minimum', :] = node_voltage_vector_magnitude_per_unit.min(axis='rows')
    results.update({
        'branch_power_vector_magnitude_per_unit': branch_power_vector_magnitude_per_unit,
        'node_voltage_vector_magnitude_per_unit': node_voltage_vector_magnitude_per_unit
    })

    # Print results.
    print(results)

    # Store results as CSV.
    results.to_csv(results_path)

    # Obtain DLMPs.
    dlmps = (
        linear_electric_grid_model.get_optimization_dlmps(
            optimization_problem,
            price_timeseries,
            scenario_data.timesteps
        )
    )
    dlmps.update(
        linear_thermal_grid_model.get_optimization_dlmps(
            optimization_problem,
            price_timeseries,
            scenario_data.timesteps
        )
    )

    # Print DLMPs.
    print(dlmps)

    # Store DLMPs as CSV.
    dlmps.to_csv(results_path)

    # Plot thermal grid DLMPs.
    thermal_grid_dlmp = (
        pd.concat(
            [
                dlmps['thermal_grid_energy_dlmp'],
                dlmps['thermal_grid_pump_dlmp'],
                dlmps['thermal_grid_head_dlmp'],
                dlmps['thermal_grid_congestion_dlmp']
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
                ].toarray().ravel() != 0
            ]
        )

        # Create plot.
        fig, (ax1, lax) = plt.subplots(ncols=2, figsize=[7.8, 2.6], gridspec_kw={"width_ratios": [100, 1]})
        ax1.set_title(f"DER {der[1]} ({der[0].replace('_', ' ').capitalize()})")
        ax1.stackplot(
            scenario_data.timesteps,
            (
                thermal_grid_dlmp.loc[:, (slice(None), *zip(*node))].groupby('dlmp_type', axis='columns').mean().T
                * 1.0e3
            ),
            labels=['Energy', 'Pumping', 'Head', 'Congest.'],
            colors=[colors[0], colors[1], colors[2], colors[3]],
            step='post'
        )
        ax1.plot(
            (
                thermal_grid_dlmp.loc[:, (slice(None), *zip(*node))].sum(axis='columns')
                * 1.0e3
            ),
            label='Total DLMP',
            drawstyle='steps-post',
            color='red',
            linewidth=1.0
        )
        ax1.grid(True)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price [S$/MWh]')
        # ax1.set_ylim((0.0, 10.0))
        ax2 = plt.twinx(ax1)
        if der in thermal_grid_model.ders:
            ax2.plot(
                results['der_thermal_power_vector'].loc[:, der].abs() / (1 if in_per_unit else 1e6),
                label='Thrm. pw.',
                drawstyle='steps-post',
                color='darkgrey',
                linewidth=3
            )
        if der in electric_grid_model.ders:
            ax2.plot(
                results['der_active_power_vector'].loc[:, der].abs() / (1 if in_per_unit else 1e6),
                label='Active pw.',
                drawstyle='steps-post',
                color='black',
                linewidth=1.5
            )
        ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_xlim((scenario_data.timesteps[0].toordinal(), scenario_data.timesteps[-1].toordinal()))
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Power [p.u.]') if in_per_unit else ax2.set_ylabel('Power [MW]')
        # ax2.set_ylim((0.0, 1.0)) if in_per_unit else ax2.set_ylim((0.0, 30.0))
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        lax.legend((*h1, *h2), (*l1, *l2), borderaxespad=0)
        lax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'thermal_grid_der_dlmp_{der}.png'))
        # plt.show()
        plt.close()

    # Plot electric grid DLMPs.
    electric_grid_dlmp = (
        pd.concat(
            [
                dlmps['electric_grid_energy_dlmp'],
                dlmps['electric_grid_loss_dlmp'],
                dlmps['electric_grid_voltage_dlmp'],
                dlmps['electric_grid_congestion_dlmp']
            ],
            axis='columns',
            keys=['energy', 'loss', 'voltage', 'congestion'],
            names=['dlmp_type']
        )
    )
    colors = list(color['color'] for color in matplotlib.rcParams['axes.prop_cycle'])
    for der in electric_grid_model.ders:

        # Obtain corresponding node.
        # TODO: Consider delta connected DERs.
        node = (
            electric_grid_model.nodes[
                electric_grid_model.der_incidence_wye_matrix[
                    :, electric_grid_model.ders.get_loc(der)
                ].toarray().ravel() > 0
            ]
        )

        # Create plot.
        fig, (ax1, lax) = plt.subplots(ncols=2, figsize=[7.8, 2.6], gridspec_kw={"width_ratios": [100, 1]})
        ax1.set_title(f"DER {der[1]} ({der[0].replace('_', ' ').capitalize()})")
        ax1.stackplot(
            scenario_data.timesteps,
            (
                electric_grid_dlmp.loc[:, (slice(None), *zip(*node))].groupby('dlmp_type', axis='columns').mean().T
                * 1.0e3
            ),
            labels=['Energy', 'Loss', 'Voltage', 'Congest.'],
            colors=[colors[0], colors[1], colors[2], colors[3]],
            step='post'
        )
        ax1.plot(
            (
                electric_grid_dlmp.loc[
                    :, (slice(None), *zip(*node))
                ].groupby('dlmp_type', axis='columns').mean().sum(axis='columns')
                * 1.0e3
            ),
            label='Total DLMP',
            drawstyle='steps-post',
            color='red',
            linewidth=1.0
        )
        ax1.grid(True)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price [S$/MWh]')
        # ax1.set_ylim((0.0, 10.0))
        ax2 = plt.twinx(ax1)
        if der in thermal_grid_model.ders:
            ax2.plot(
                results['der_thermal_power_vector'].loc[:, der].abs() / (1 if in_per_unit else 1e6),
                label='Thrm. pw.',
                drawstyle='steps-post',
                color='darkgrey',
                linewidth=3
            )
        if der in electric_grid_model.ders:
            ax2.plot(
                results['der_active_power_vector'].loc[:, der].abs() / (1 if in_per_unit else 1e6),
                label='Active pw.',
                drawstyle='steps-post',
                color='black',
                linewidth=1.5
            )
        ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_xlim((scenario_data.timesteps[0].toordinal(), scenario_data.timesteps[-1].toordinal()))
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Power [p.u.]') if in_per_unit else ax2.set_ylabel('Power [MW]')
        # ax2.set_ylim((0.0, 1.0)) if in_per_unit else ax2.set_ylim((0.0, 30.0))
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        lax.legend((*h1, *h2), (*l1, *l2), borderaxespad=0)
        lax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f'electric_grid_der_dlmp_{der}.png'))
        # plt.show()
        plt.close()

    # Obtain graphs.
    electric_grid_graph = fledge.plots.ElectricGridGraph(scenario_name)
    thermal_grid_graph = fledge.plots.ThermalGridGraph(scenario_name)

    # Plot thermal grid DLMPs in grid.
    dlmp_types = [
        'thermal_grid_energy_dlmp',
        'thermal_grid_pump_dlmp',
        'thermal_grid_head_dlmp',
        'thermal_grid_congestion_dlmp'
    ]
    for timestep in scenario_data.timesteps:
        for dlmp_type in dlmp_types:
            node_color = (
                dlmps[dlmp_type].loc[timestep, :].groupby('node_name').mean().reindex(thermal_grid_graph.nodes).values
                * 1.0e3
            )
            plt.title(
                f"{dlmp_type.replace('_', ' ').capitalize().replace('dlmp', 'DLMP')}"
                f" at {timestep.strftime('%H:%M:%S')}"
            )
            nx.draw_networkx_nodes(
                thermal_grid_graph,
                pos=thermal_grid_graph.node_positions,
                nodelist=(
                    thermal_grid_model.nodes[
                        fledge.utils.get_index(thermal_grid_model.nodes, node_type='source')
                    ].get_level_values('node_name')[:1].to_list()
                ),
                node_size=150.0,
                node_color='red',
                with_labels=False
            )
            nx.draw_networkx(
                thermal_grid_graph,
                pos=thermal_grid_graph.node_positions,
                arrows=False,
                node_size=100.0,
                node_color=node_color,
                edgecolors='black',  # Make node border visible.
                with_labels=False
            )
            sm = (
                plt.cm.ScalarMappable(
                    norm=plt.Normalize(
                        vmin=np.min(node_color),
                        vmax=np.max(node_color)
                    )
                )
            )
            cb = plt.colorbar(sm, shrink=0.9)
            cb.set_label('Price [S$/MWh]')
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, f'{dlmp_type}_{timestep.strftime("%H-%M-%S")}.png'))
            # plt.show()
            plt.close()

    # Plot electric grid DLMPs in grid.
    dlmp_types = [
        'electric_grid_energy_dlmp',
        'electric_grid_voltage_dlmp',
        'electric_grid_congestion_dlmp',
        'electric_grid_loss_dlmp'
    ]
    for timestep in scenario_data.timesteps:
        for dlmp_type in dlmp_types:
            node_color = (
                dlmps[dlmp_type].loc[timestep, :].groupby('node_name').mean().reindex(electric_grid_graph.nodes).values
                * 1.0e3
            )
            plt.title(
                f"{dlmp_type.replace('_', ' ').capitalize().replace('dlmp', 'DLMP')}"
                f" at {timestep.strftime('%H:%M:%S')}"
            )
            nx.draw_networkx_nodes(
                electric_grid_graph,
                pos=electric_grid_graph.node_positions,
                nodelist=(
                    electric_grid_model.nodes[
                        fledge.utils.get_index(electric_grid_model.nodes, node_type='source')
                    ].get_level_values('node_name')[:1].to_list()
                ),
                node_size=150.0,
                node_color='red',
                with_labels=False
            )
            nx.draw_networkx(
                electric_grid_graph,
                pos=electric_grid_graph.node_positions,
                arrows=False,
                node_size=100.0,
                node_color=node_color,
                edgecolors='black',  # Make node border visible.
                with_labels=False
            )
            sm = (
                plt.cm.ScalarMappable(
                    norm=plt.Normalize(
                        vmin=np.min(node_color),
                        vmax=np.max(node_color)
                    )
                )
            )
            cb = plt.colorbar(sm, shrink=0.9)
            cb.set_label('Price [S$/MWh]')
            plt.tight_layout()
            plt.savefig(os.path.join(results_path, f'{dlmp_type}_{timestep.strftime("%H-%M-%S")}.png'))
            # plt.show()
            plt.close()

    # Plot electric grid line utilization.
    fledge.plots.plot_electric_grid_line_utilization(
        electric_grid_model,
        electric_grid_graph,
        branch_power_vector_magnitude_per_unit,
        results_path
    )

    # Plot electric grid nodes voltage drop.
    fledge.plots.plot_electric_grid_node_voltage_drop(
        electric_grid_model,
        electric_grid_graph,
        node_voltage_vector_magnitude_per_unit,
        results_path
    )

    # Print results path.
    os.startfile(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':

    run_all = True

    if run_all:
        for scenario_number in range(1, 16):
            main(scenario_number)
    else:
        main()
