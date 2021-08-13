"""Run script for reproducing results of the Paper XXX."""

import matplotlib.dates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd

import mesmo


def main(
        scenario_number=None
):

    # TODO: To be updated for new optimization problem interface.

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
        scenario_name = 'paper_2021_troitzsch_dlmp_scenario_1_2_3_4_5'
    elif scenario_number in [6, 7, 8]:
        scenario_name = 'paper_2021_troitzsch_dlmp_scenario_6_7_8'
    elif scenario_number in [9]:
        scenario_name = 'paper_2021_troitzsch_dlmp_scenario_9'
    elif scenario_number in [10, 11, 12]:
        scenario_name = 'paper_2021_troitzsch_dlmp_scenario_10_11_12'
    elif scenario_number in [13]:
        scenario_name = 'paper_2021_troitzsch_dlmp_scenario_13'
    elif scenario_number in [14]:
        scenario_name = 'paper_2021_troitzsch_dlmp_scenario_14'
    elif scenario_number in [15]:
        scenario_name = 'paper_2021_troitzsch_dlmp_scenario_15'
    else:
        scenario_name = 'paper_2021_troitzsch_dlmp_scenario_1_2_3_4_5'

    # Obtain results path.
    results_path = mesmo.utils.get_results_path(__file__, f'scenario{scenario_number}_{scenario_name}')

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain data.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    price_data = mesmo.data_interface.PriceData(scenario_name)

    # Obtain models.
    electric_grid_model = mesmo.electric_grid_models.ElectricGridModelDefault(scenario_name)
    # Use base scenario power flow for consistent linear model behavior and per unit values.
    # TODO: Fix reliance on default scenario power flow.
    power_flow_solution = mesmo.electric_grid_models.PowerFlowSolutionFixedPoint('paper_2021_troitzsch_dlmp_scenario_1_2_3_4_5')
    linear_electric_grid_model_set = (
        mesmo.electric_grid_models.LinearElectricGridModelSet(
            electric_grid_model,
            power_flow_solution,
            linear_electric_grid_model_method=mesmo.electric_grid_models.LinearElectricGridModelGlobal
        )
    )
    thermal_grid_model = mesmo.thermal_grid_models.ThermalGridModel(scenario_name)
    # Use base scenario power flow for consistent linear model behavior and per unit values.
    # TODO: Fix reliance on default scenario power flow.
    thermal_power_flow_solution = mesmo.thermal_grid_models.ThermalPowerFlowSolution('paper_2021_troitzsch_dlmp_scenario_1_2_3_4_5')
    linear_thermal_grid_model_set = (
        mesmo.thermal_grid_models.LinearThermalGridModelSet(
            thermal_grid_model,
            thermal_power_flow_solution
        )
    )
    der_model_set = mesmo.der_models.DERModelSet(scenario_name)

    # Modify DER models depending on scenario.
    # Cooling plant.
    if scenario_number in [6, 9]:
        der_model_set.flexible_der_models['23'].control_output_matrix.at['thermal_power', 'active_power'] *= 2.0
    elif scenario_number in [7, 8, 14, 15]:
        # Cooling plant COP must remain larger than building cooling COP, otherwise cooling plant never dispatched.
        der_model_set.flexible_der_models['23'].control_output_matrix.at['thermal_power', 'active_power'] *= 0.8
    # PV plant.
    if scenario_number in [11, 12]:
        der_model_set.flexible_der_models['24'].marginal_cost = 0.1
    if scenario_number in [15]:
        der_model_set.flexible_der_models['24'].marginal_cost = 0.04

    # Instantiate optimization problem.
    optimization_problem = mesmo.utils.OptimizationProblem()

    # Define electric grid problem.
    node_voltage_magnitude_vector_minimum = 0.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    node_voltage_magnitude_vector_maximum = 1.5 * np.abs(electric_grid_model.node_voltage_vector_reference)
    branch_power_magnitude_vector_maximum = 100.0 * electric_grid_model.branch_power_vector_magnitude_reference
    # Modify limits for scenarios.
    if scenario_number in [4, 12, 13, 14, 15]:
        branch_power_magnitude_vector_maximum[
            mesmo.utils.get_index(electric_grid_model.branches, branch_name='4')
        ] *= 8.5 / 100.0
    elif scenario_number in [5]:
        node_voltage_magnitude_vector_minimum[
            mesmo.utils.get_index(electric_grid_model.nodes, node_name='15')
        ] *= 0.9985 / 0.5
    else:
        pass
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

    # Define thermal grid problem.
    # TODO: Rename branch_flow_vector_maximum to branch_flow_vector_magnitude_maximum.
    # TODO: Revise node_head_vector constraint formulation.
    node_head_vector_minimum = 100.0 * thermal_power_flow_solution.node_head_vector
    branch_flow_vector_maximum = 100.0 * np.abs(thermal_power_flow_solution.branch_flow_vector)
    # Modify limits for scenarios.
    if scenario_number in [2, 8, 9, 15]:
        branch_flow_vector_maximum[
            mesmo.utils.get_index(thermal_grid_model.branches, branch_name='4')
        ] *= 0.2 / 100.0
    elif scenario_number in [3]:
        node_head_vector_minimum[
            mesmo.utils.get_index(thermal_grid_model.nodes, node_name='15')
        ] *= 0.2 / 100.0
    else:
        pass
    linear_thermal_grid_model_set.define_optimization_variables(optimization_problem)
    linear_thermal_grid_model_set.define_optimization_parameters(
        optimization_problem,
        price_data,
        node_head_vector_minimum=node_head_vector_minimum,
        branch_flow_vector_maximum=branch_flow_vector_maximum
    )
    linear_thermal_grid_model_set.define_optimization_constraints(optimization_problem)
    linear_thermal_grid_model_set.define_optimization_objective(optimization_problem)

    # Define DER problem.
    der_model_set.define_optimization_variables(optimization_problem)
    der_model_set.define_optimization_parameters(optimization_problem, price_data)
    der_model_set.define_optimization_constraints(optimization_problem)
    der_model_set.define_optimization_objective(optimization_problem)

    # Solve optimization problem.
    optimization_problem.solve()

    # Obtain results.
    results = mesmo.problems.Results()
    results.update(linear_electric_grid_model_set.get_optimization_results(optimization_problem))
    results.update(linear_thermal_grid_model_set.get_optimization_results(optimization_problem))
    results.update(der_model_set.get_optimization_results(optimization_problem))
    results.save(results_path)

    # Obtain DLMPs.
    dlmps = mesmo.problems.Results()
    dlmps.update(linear_electric_grid_model_set.get_optimization_dlmps(optimization_problem, price_data))
    dlmps.update(linear_thermal_grid_model_set.get_optimization_dlmps(optimization_problem, price_data))
    dlmps.save(results_path)

    # Plot results.
    in_per_unit = False
    results_suffix = '_per_unit' if in_per_unit else ''

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
                results[f'der_thermal_power_vector{results_suffix}'].loc[:, der].abs() / (1 if in_per_unit else 1e6),
                label='Thrm. pw.',
                drawstyle='steps-post',
                color='darkgrey',
                linewidth=3
            )
        if der in electric_grid_model.ders:
            ax2.plot(
                results[f'der_active_power_vector{results_suffix}'].loc[:, der].abs() / (1 if in_per_unit else 1e6),
                label='Active pw.',
                drawstyle='steps-post',
                color='black',
                linewidth=1.5
            )
        ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_xlim((scenario_data.timesteps[0], scenario_data.timesteps[-1]))
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
                dlmps['electric_grid_energy_dlmp_node_active_power'],
                dlmps['electric_grid_loss_dlmp_node_active_power'],
                dlmps['electric_grid_voltage_dlmp_node_active_power'],
                dlmps['electric_grid_congestion_dlmp_node_active_power']
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
                results[f'der_thermal_power_vector{results_suffix}'].loc[:, der].abs() / (1 if in_per_unit else 1e6),
                label='Thrm. pw.',
                drawstyle='steps-post',
                color='darkgrey',
                linewidth=3
            )
        if der in electric_grid_model.ders:
            ax2.plot(
                results[f'der_active_power_vector{results_suffix}'].loc[:, der].abs() / (1 if in_per_unit else 1e6),
                label='Active pw.',
                drawstyle='steps-post',
                color='black',
                linewidth=1.5
            )
        ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_xlim((scenario_data.timesteps[0], scenario_data.timesteps[-1]))
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
    electric_grid_graph = mesmo.plots.ElectricGridGraph(scenario_name)
    thermal_grid_graph = mesmo.plots.ThermalGridGraph(scenario_name)

    # Plot thermal grid DLMPs in grid.
    dlmp_types = [
        'thermal_grid_energy_dlmp_node_thermal_power',
        'thermal_grid_pump_dlmp_node_thermal_power',
        'thermal_grid_head_dlmp_node_thermal_power',
        'thermal_grid_congestion_dlmp_node_thermal_power'
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
            nx.draw(
                thermal_grid_graph,
                pos=thermal_grid_graph.node_positions,
                nodelist=(
                    thermal_grid_model.nodes[
                        mesmo.utils.get_index(thermal_grid_model.nodes, node_type='source')
                    ].get_level_values('node_name')[:1].to_list()
                ),
                edgelist=[],
                node_size=150.0,
                node_color='red'
            )
            nx.draw(
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
        'electric_grid_energy_dlmp_node_active_power',
        'electric_grid_voltage_dlmp_node_active_power',
        'electric_grid_congestion_dlmp_node_active_power',
        'electric_grid_loss_dlmp_node_active_power'
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
            nx.draw(
                electric_grid_graph,
                pos=electric_grid_graph.node_positions,
                nodelist=(
                    electric_grid_model.nodes[
                        mesmo.utils.get_index(electric_grid_model.nodes, node_type='source')
                    ].get_level_values('node_name')[:1].to_list()
                ),
                edgelist=[],
                node_size=150.0,
                node_color='red'
            )
            nx.draw(
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
    mesmo.plots.plot_grid_line_utilization(
        electric_grid_model,
        electric_grid_graph,
        results[f'branch_power_magnitude_vector_1{results_suffix}'] * (100.0 if in_per_unit else 1.0e-3),
        results_path,
        value_unit='%' if in_per_unit else 'kW',
    )
    mesmo.plots.plot_grid_line_utilization(
        thermal_grid_model,
        thermal_grid_graph,
        results[f'branch_flow_vector{results_suffix}'] * (100.0 if in_per_unit else 1.0e-3),
        results_path,
        value_unit='%' if in_per_unit else 'kW',
    )

    # Plot electric grid nodes voltage drop.
    mesmo.plots.plot_grid_node_utilization(
        electric_grid_model,
        electric_grid_graph,
        results['node_voltage_magnitude_vector_per_unit'],
        results_path
    )

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':

    run_all = True

    if run_all:
        for scenario_number in range(1, 16):
            main(scenario_number)
    else:
        main()
