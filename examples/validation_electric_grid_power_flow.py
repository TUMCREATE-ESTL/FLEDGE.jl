"""Example script for testing / validating the electric grid power flow solution."""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt  # TODO: Remove matplotlib dependency.
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import mesmo


def main():

    # Settings.
    scenario_name = mesmo.config.config['tests']['scenario_name']
    results_path = mesmo.utils.get_results_path(__file__, scenario_name)
    power_multipliers = np.arange(-0.2, 1.2, 0.1)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    mesmo.data_interface.recreate_database()

    # Obtain base scaling parameters.
    scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
    base_power = scenario_data.scenario.at['base_apparent_power']
    base_voltage = scenario_data.scenario.at['base_voltage']

    # Obtain electric grid models.
    electric_grid_model_default = mesmo.electric_grid_models.ElectricGridModelDefault(scenario_name)
    electric_grid_model_opendss = mesmo.electric_grid_models.ElectricGridModelOpenDSS(scenario_name)

    # Obtain nominal power flow solutions.
    power_flow_solution_mesmo_nominal = mesmo.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model_default)
    power_flow_solution_opendss_nominal = mesmo.electric_grid_models.PowerFlowSolutionOpenDSS(electric_grid_model_opendss)

    # Instantiate results variables.
    der_power_vector = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.ders, dtype=float)
    )
    node_voltage_vector_mesmo = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.nodes, dtype=complex)
    )
    node_voltage_vector_opendss = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.nodes, dtype=complex)
    )
    node_voltage_vector_magnitude_mesmo = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.nodes, dtype=float)
    )
    node_voltage_vector_magnitude_opendss = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.nodes, dtype=float)
    )
    branch_power_vector_1_magnitude_mesmo = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.branches, dtype=float)
    )
    branch_power_vector_1_magnitude_opendss = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.branches, dtype=float)
    )
    branch_power_vector_2_magnitude_mesmo = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.branches, dtype=float)
    )
    branch_power_vector_2_magnitude_opendss = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model_default.branches, dtype=float)
    )
    loss_active_mesmo = (
        pd.Series(index=power_multipliers, dtype=float)
    )
    loss_active_opendss = (
        pd.Series(index=power_multipliers, dtype=float)
    )
    loss_reactive_mesmo = (
        pd.Series(index=power_multipliers, dtype=float)
    )
    loss_reactive_opendss = (
        pd.Series(index=power_multipliers, dtype=float)
    )

    # Obtain DER power / change.
    der_power_vector.loc[:, :] = (
        np.transpose([power_multipliers])
        @ np.array([np.real(electric_grid_model_default.der_power_vector_reference)])
    )

    # Obtain OpenDSS solutions.
    power_flow_solutions_mesmo = (
        mesmo.utils.starmap(
            mesmo.electric_grid_models.PowerFlowSolutionFixedPoint,
            [(electric_grid_model_default, row) for row in der_power_vector.values]
        )
    )
    power_flow_solutions_mesmo = dict(zip(power_multipliers, power_flow_solutions_mesmo))
    for power_multiplier in power_multipliers:
        power_flow_solution = power_flow_solutions_mesmo[power_multiplier]
        node_voltage_vector_mesmo.loc[power_multiplier, :] = power_flow_solution.node_voltage_vector
        node_voltage_vector_magnitude_mesmo.loc[power_multiplier, :] = np.abs(power_flow_solution.node_voltage_vector)
        branch_power_vector_1_magnitude_mesmo.loc[power_multiplier, :] = np.abs(power_flow_solution.branch_power_vector_1)
        branch_power_vector_2_magnitude_mesmo.loc[power_multiplier, :] = np.abs(power_flow_solution.branch_power_vector_2)
        loss_active_mesmo.loc[power_multiplier] = np.real(power_flow_solution.loss)
        loss_reactive_mesmo.loc[power_multiplier] = np.imag(power_flow_solution.loss)

    power_flow_solutions_opendss = (
        mesmo.utils.starmap(
            mesmo.electric_grid_models.PowerFlowSolutionOpenDSS,
            [(electric_grid_model_opendss, row) for row in der_power_vector.values]
        )
    )
    power_flow_solutions_opendss = dict(zip(power_multipliers, power_flow_solutions_opendss))
    for power_multiplier in power_multipliers:
        power_flow_solution = power_flow_solutions_opendss[power_multiplier]
        node_voltage_vector_opendss.loc[power_multiplier, :] = power_flow_solution.node_voltage_vector
        node_voltage_vector_magnitude_opendss.loc[power_multiplier, :] = np.abs(power_flow_solution.node_voltage_vector)
        branch_power_vector_1_magnitude_opendss.loc[power_multiplier, :] = np.abs(power_flow_solution.branch_power_vector_1)
        branch_power_vector_2_magnitude_opendss.loc[power_multiplier, :] = np.abs(power_flow_solution.branch_power_vector_2)
        loss_active_opendss.loc[power_multiplier] = np.real(power_flow_solution.loss)
        loss_reactive_opendss.loc[power_multiplier] = np.imag(power_flow_solution.loss)

    # Obtain error values.
    node_voltage_vector_error = (
        100.0 * (
            (node_voltage_vector_mesmo - node_voltage_vector_opendss)
            / node_voltage_vector_opendss
        ).abs().mean(axis='columns')
    )
    node_voltage_vector_magnitude_error = (
        100.0 * (
            (node_voltage_vector_magnitude_mesmo - node_voltage_vector_magnitude_opendss)
            / node_voltage_vector_magnitude_opendss
        ).mean(axis='columns')
    )
    branch_power_vector_1_magnitude_error = (
        100.0 * (
            (branch_power_vector_1_magnitude_mesmo - branch_power_vector_1_magnitude_opendss)
            / branch_power_vector_1_magnitude_opendss
        ).mean(axis='columns')
    )
    branch_power_vector_2_magnitude_error = (
        100.0 * (
            (branch_power_vector_2_magnitude_mesmo - branch_power_vector_2_magnitude_opendss)
            / branch_power_vector_2_magnitude_opendss
        ).mean(axis='columns')
    )
    loss_active_error = (
        100.0 * (
            (loss_active_mesmo - loss_active_opendss)
            / loss_active_opendss
        )
    )
    loss_reactive_error = (
        100.0 * (
            (loss_reactive_mesmo - loss_reactive_opendss)
            / loss_reactive_opendss
        )
    )

    # Obtain error table.
    power_flow_solution_error = (
        pd.DataFrame(
            [
                node_voltage_vector_error,
                node_voltage_vector_magnitude_error,
                branch_power_vector_1_magnitude_error,
                branch_power_vector_2_magnitude_error,
                loss_active_error,
                loss_reactive_error
            ],
            index=[
                'node_voltage_vector_error',
                'node_voltage_vector_magnitude_error',
                'branch_power_vector_1_magnitude_error',
                'branch_power_vector_2_magnitude_error',
                'loss_active_error',
                'loss_reactive_error'
            ]
        )
    )
    power_flow_solution_error = power_flow_solution_error.round(2)

    # Print results.
    print(f"der_power_vector =\n{der_power_vector}")
    print(f"node_voltage_vector_mesmo =\n{node_voltage_vector_mesmo}")
    print(f"node_voltage_vector_opendss =\n{node_voltage_vector_opendss}")
    print(f"node_voltage_vector_magnitude_mesmo =\n{node_voltage_vector_magnitude_mesmo}")
    print(f"node_voltage_vector_magnitude_opendss =\n{node_voltage_vector_magnitude_opendss}")
    print(f"branch_power_vector_1_magnitude_mesmo =\n{branch_power_vector_1_magnitude_mesmo}")
    print(f"branch_power_vector_1_magnitude_opendss =\n{branch_power_vector_1_magnitude_opendss}")
    print(f"branch_power_vector_2_magnitude_mesmo =\n{branch_power_vector_2_magnitude_mesmo}")
    print(f"branch_power_vector_2_magnitude_opendss =\n{branch_power_vector_2_magnitude_opendss}")
    print(f"loss_active_mesmo =\n{loss_active_mesmo}")
    print(f"loss_active_opendss =\n{loss_active_opendss}")
    print(f"loss_reactive_mesmo =\n{loss_reactive_mesmo}")
    print(f"loss_reactive_opendss =\n{loss_reactive_opendss}")
    print(f"power_flow_solution_error =\n{power_flow_solution_error}")

    # Store results as CSV.
    der_power_vector.to_csv(os.path.join(results_path, 'der_power_vector.csv'))
    node_voltage_vector_mesmo.to_csv(os.path.join(results_path, 'node_voltage_vector_mesmo.csv'))
    node_voltage_vector_opendss.to_csv(os.path.join(results_path, 'node_voltage_vector_opendss.csv'))
    node_voltage_vector_magnitude_mesmo.to_csv(os.path.join(results_path, 'node_voltage_vector_magnitude_mesmo.csv'))
    node_voltage_vector_magnitude_opendss.to_csv(os.path.join(results_path, 'node_voltage_vector_magnitude_opendss.csv'))
    branch_power_vector_1_magnitude_mesmo.to_csv(os.path.join(results_path, 'branch_power_vector_1_magnitude_mesmo.csv'))
    branch_power_vector_1_magnitude_opendss.to_csv(os.path.join(results_path, 'branch_power_vector_1_magnitude_opendss.csv'))
    branch_power_vector_2_magnitude_mesmo.to_csv(os.path.join(results_path, 'branch_power_vector_2_magnitude_mesmo.csv'))
    branch_power_vector_2_magnitude_opendss.to_csv(os.path.join(results_path, 'branch_power_vector_2_magnitude_opendss.csv'))
    loss_active_mesmo.to_csv(os.path.join(results_path, 'loss_active_mesmo.csv'))
    loss_active_opendss.to_csv(os.path.join(results_path, 'loss_active_opendss.csv'))
    loss_reactive_mesmo.to_csv(os.path.join(results_path, 'loss_reactive_mesmo.csv'))
    loss_reactive_opendss.to_csv(os.path.join(results_path, 'loss_reactive_opendss.csv'))
    power_flow_solution_error.to_csv(os.path.join(results_path, 'power_flow_solution_error.csv'))

    # Plot results.

    # Nominal OpenDSS solution comparison.
    plt.title('Node voltage magnitude [p.u.]')
    plt.scatter(
        range(len(electric_grid_model_default.nodes)),
        (
            np.abs(power_flow_solution_mesmo_nominal.node_voltage_vector)
            / np.abs(electric_grid_model_default.node_voltage_vector_reference)
        ),
        marker=4
    )
    plt.scatter(
        range(len(electric_grid_model_default.nodes)),
        (
            np.abs(power_flow_solution_opendss_nominal.node_voltage_vector)
            / np.abs(electric_grid_model_default.node_voltage_vector_reference)
        ),
        marker=5
    )
    plt.xticks(
        range(len(electric_grid_model_default.nodes)),
        electric_grid_model_default.nodes,
        rotation=45,
        ha='right'
    )
    plt.legend(['MESMO', 'OpenDSS'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'nominal_voltage_magnitude.png'))
    # plt.show()
    plt.close()

    plt.title('Branch power (direction 1) magnitude [p.u.]')
    plt.scatter(
        range(len(electric_grid_model_default.branches)),
        (
            np.abs(power_flow_solution_mesmo_nominal.branch_power_vector_1)
            / electric_grid_model_default.branch_power_vector_magnitude_reference
        ),
        marker=4
    )
    plt.scatter(
        range(len(electric_grid_model_default.branches)),
        (
            np.abs(power_flow_solution_opendss_nominal.branch_power_vector_1)
            / electric_grid_model_default.branch_power_vector_magnitude_reference
        ),
        marker=5
    )
    plt.xticks(
        range(len(electric_grid_model_default.branches)),
        electric_grid_model_default.branches,
        rotation=45,
        ha='right'
    )
    plt.legend(['MESMO', 'OpenDSS'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'nominal_branch_power_1_magnitude.png'))
    # plt.show()
    plt.close()

    plt.title('Branch power (direction 2) magnitude [p.u.]')
    plt.scatter(
        range(len(electric_grid_model_default.branches)),
        (
            np.abs(power_flow_solution_mesmo_nominal.branch_power_vector_2)
            / electric_grid_model_default.branch_power_vector_magnitude_reference
        ),
        marker=4
    )
    plt.scatter(
        range(len(electric_grid_model_default.branches)),
        (
            np.abs(power_flow_solution_opendss_nominal.branch_power_vector_2)
            / electric_grid_model_default.branch_power_vector_magnitude_reference
        ),
        marker=5
    )
    plt.xticks(
        range(len(electric_grid_model_default.branches)),
        electric_grid_model_default.branches,
        rotation=45,
        ha='right'
    )
    plt.legend(['MESMO', 'OpenDSS'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'nominal_branch_power_2_magnitude.png'))
    # plt.show()
    plt.close()

    plt.title('Total power loss [VA]')
    plt.scatter(
        [1, 2],
        [
            np.real(base_power * power_flow_solution_mesmo_nominal.loss ),
            np.imag(base_power * power_flow_solution_mesmo_nominal.loss )
        ],
        marker=4
    )
    plt.scatter(
        [1, 2],
        [
            np.real(base_power * power_flow_solution_opendss_nominal.loss),
            np.imag(base_power * power_flow_solution_opendss_nominal.loss)
        ],
        marker=5
    )
    plt.xticks(
        [0.5, 1, 2, 2.5],
        ['', 'active', 'reactive', ''],
        rotation=45,
        ha='right'
    )
    plt.legend(['MESMO', 'OpenDSS'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'nominal_loss.png'))
    # plt.show()
    plt.close()

    # Voltage magnitude.
    for node_index, node in enumerate(electric_grid_model_default.nodes):
        plt.plot(power_multipliers, base_voltage * node_voltage_vector_magnitude_opendss.loc[:, node], label='OpenDSS')
        plt.plot(power_multipliers, base_voltage * node_voltage_vector_magnitude_mesmo.loc[:, node], label='MESMO')
        plt.legend()
        plt.title(f"Voltage magnitude [V] for\n (node_type, node_name, phase): {node}")
        plt.savefig(os.path.join(results_path, f'voltage_magnitude_{node}.png'))
        # plt.show()
        plt.close()

    # Branch flow.
    for branch_index, branch in enumerate(electric_grid_model_default.branches):
        plt.plot(power_multipliers, base_power * branch_power_vector_1_magnitude_opendss.loc[:, branch], label='OpenDSS')
        plt.plot(power_multipliers, base_power * branch_power_vector_1_magnitude_mesmo.loc[:, branch], label='MESMO')
        plt.legend()
        plt.title(f"Branch power 1 magnitude [VA] for\n (branch_type, branch_name, phase): {branch}")
        plt.savefig(os.path.join(results_path, f'branch_power_1_{branch}.png'))
        # plt.show()
        plt.close()

        plt.plot(power_multipliers, base_power * branch_power_vector_2_magnitude_opendss.loc[:, branch], label='OpenDSS')
        plt.plot(power_multipliers, base_power * branch_power_vector_2_magnitude_mesmo.loc[:, branch], label='MESMO')
        plt.legend()
        plt.title(f"Branch power 2 magnitude [VA] for\n (branch_type, branch_name, phase): {branch}")
        plt.savefig(os.path.join(results_path, f'branch_power_2_{branch}.png'))
        # plt.show()
        plt.close()

    # Loss.
    plt.plot(power_multipliers, base_power * loss_active_opendss, label='OpenDSS')
    plt.plot(power_multipliers, base_power * loss_active_mesmo, label='MESMO')
    plt.legend()
    plt.title("Total loss active [W]")
    plt.savefig(os.path.join(results_path, f'loss_active.png'))
    # plt.show()
    plt.close()

    plt.plot(power_multipliers, base_power * loss_reactive_opendss, label='OpenDSS')
    plt.plot(power_multipliers, base_power * loss_reactive_mesmo, label='MESMO')
    plt.legend()
    plt.title("Total loss reactive [VAr]")
    plt.savefig(os.path.join(results_path, f'loss_reactive.png'))
    # plt.show()
    plt.close()

    # Print results path.
    mesmo.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
