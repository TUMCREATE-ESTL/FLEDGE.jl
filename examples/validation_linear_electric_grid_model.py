"""Example script for testing / validating the linear electric grid model."""

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt  # TODO: Remove matplotlib dependency.
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge


def main():

    # Settings.
    scenario_name = fledge.config.config['tests']['scenario_name']
    results_path = fledge.utils.get_results_path(__file__, scenario_name)
    power_multipliers = np.arange(-0.2, 1.8, 0.1)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain base scaling parameters.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)
    base_power = scenario_data.scenario.at['base_apparent_power']
    base_voltage = scenario_data.scenario.at['base_voltage']

    # Obtain electric grid model.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)

    # Obtain power flow solution for nominal power conditions.
    power_flow_solution_initial = fledge.electric_grid_models.PowerFlowSolutionFixedPoint(electric_grid_model)

    # Obtain linear electric grid model for nominal power conditions.
    linear_electric_grid_model = (
        fledge.electric_grid_models.LinearElectricGridModelLocal(
            electric_grid_model,
            power_flow_solution_initial
        )
    )

    # Instantiate results variables.
    der_power_vector_active = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.ders, dtype=float)
    )
    der_power_vector_reactive = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.ders, dtype=float)
    )
    der_power_vector_active_change = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.ders, dtype=float)
    )
    der_power_vector_reactive_change = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.ders, dtype=float)
    )
    node_voltage_vector_power_flow = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.nodes, dtype=complex)
    )
    node_voltage_vector_linear_model = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.nodes, dtype=complex)
    )
    node_voltage_vector_magnitude_power_flow = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.nodes, dtype=float)
    )
    node_voltage_vector_magnitude_linear_model = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.nodes, dtype=float)
    )
    branch_power_vector_1_squared_power_flow = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.branches, dtype=float)
    )
    branch_power_vector_1_squared_linear_model = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.branches, dtype=float)
    )
    branch_power_vector_2_squared_power_flow = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.branches, dtype=float)
    )
    branch_power_vector_2_squared_linear_model = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.branches, dtype=float)
    )
    branch_power_vector_1_magnitude_power_flow = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.branches, dtype=float)
    )
    branch_power_vector_1_magnitude_linear_model = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.branches, dtype=float)
    )
    branch_power_vector_2_magnitude_power_flow = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.branches, dtype=float)
    )
    branch_power_vector_2_magnitude_linear_model = (
        pd.DataFrame(index=power_multipliers, columns=electric_grid_model.branches, dtype=float)
    )
    loss_active_power_flow = (
        pd.Series(index=power_multipliers, dtype=float)
    )
    loss_active_linear_model = (
        pd.Series(index=power_multipliers, dtype=float)
    )
    loss_reactive_power_flow = (
        pd.Series(index=power_multipliers, dtype=float)
    )
    loss_reactive_linear_model = (
        pd.Series(index=power_multipliers, dtype=float)
    )

    # Obtain DER power / change.
    der_power_vector_active.loc[:, :] = (
        np.transpose([power_multipliers])
        @ np.array([np.real(power_flow_solution_initial.der_power_vector)])
    )
    der_power_vector_reactive.loc[:, :] = (
        np.transpose([power_multipliers])
        @ np.array([np.imag(power_flow_solution_initial.der_power_vector)])
    )
    der_power_vector_active_change.loc[:, :] = (
        np.transpose([power_multipliers - 1])
        @ np.array([np.real(power_flow_solution_initial.der_power_vector)])
    )
    der_power_vector_reactive_change.loc[:, :] = (
        np.transpose([power_multipliers - 1])
        @ np.array([np.imag(power_flow_solution_initial.der_power_vector)])
    )

    # Obtain power flow solutions.
    power_flow_solutions = (
        fledge.utils.starmap(
            fledge.electric_grid_models.PowerFlowSolutionFixedPoint,
            [(electric_grid_model, row) for row in (der_power_vector_active + 1.0j * der_power_vector_reactive).values]
        )
    )
    power_flow_solutions = dict(zip(power_multipliers, power_flow_solutions))
    for power_multiplier in power_multipliers:
        power_flow_solution = power_flow_solutions[power_multiplier]
        node_voltage_vector_power_flow.loc[power_multiplier, :] = power_flow_solution.node_voltage_vector
        node_voltage_vector_magnitude_power_flow.loc[power_multiplier, :] = np.abs(power_flow_solution.node_voltage_vector)
        branch_power_vector_1_magnitude_power_flow.loc[power_multiplier, :] = np.abs(power_flow_solution.branch_power_vector_1)
        branch_power_vector_2_magnitude_power_flow.loc[power_multiplier, :] = np.abs(power_flow_solution.branch_power_vector_2)
        branch_power_vector_1_squared_power_flow.loc[power_multiplier, :] = np.abs(power_flow_solution.branch_power_vector_1) ** 2
        branch_power_vector_2_squared_power_flow.loc[power_multiplier, :] = np.abs(power_flow_solution.branch_power_vector_2) ** 2
        loss_active_power_flow.loc[power_multiplier] = np.real(power_flow_solution.loss)
        loss_reactive_power_flow.loc[power_multiplier] = np.imag(power_flow_solution.loss)

    # Obtain linear model solutions.
    node_voltage_vector_linear_model.loc[:, :] = (
        np.transpose([power_flow_solution_initial.node_voltage_vector] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_voltage_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_voltage_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).transpose()
    node_voltage_vector_magnitude_linear_model.loc[:, :] = (
        np.transpose([np.abs(power_flow_solution_initial.node_voltage_vector)] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_voltage_magnitude_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).transpose()
    branch_power_vector_1_magnitude_linear_model.loc[:, :] = (
        np.transpose([np.abs(power_flow_solution_initial.branch_power_vector_1)] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_branch_power_1_magnitude_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).transpose()
    branch_power_vector_2_magnitude_linear_model.loc[:, :] = (
        np.transpose([np.abs(power_flow_solution_initial.branch_power_vector_2)] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_branch_power_2_magnitude_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).transpose()
    branch_power_vector_1_squared_linear_model.loc[:, :] = (
        np.transpose([np.abs(power_flow_solution_initial.branch_power_vector_1 ** 2)] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_branch_power_1_squared_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_branch_power_1_squared_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).transpose()
    branch_power_vector_2_squared_linear_model.loc[:, :] = (
        np.transpose([np.abs(power_flow_solution_initial.branch_power_vector_2 ** 2)] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_branch_power_2_squared_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_branch_power_2_squared_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).transpose()
    loss_active_linear_model.loc[:] = (
        np.transpose([np.real(power_flow_solution_initial.loss)] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_loss_active_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_loss_active_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).ravel()
    loss_reactive_linear_model.loc[:] = (
        np.transpose([np.imag(power_flow_solution_initial.loss)] * len(power_multipliers))
        + linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_active
        @ np.transpose(der_power_vector_active_change.values)
        + linear_electric_grid_model.sensitivity_loss_reactive_by_der_power_reactive
        @ np.transpose(der_power_vector_reactive_change.values)
    ).ravel()

    # Obtain error values.
    node_voltage_vector_error = (
        100.0 * (
            (node_voltage_vector_linear_model - node_voltage_vector_power_flow)
            / node_voltage_vector_power_flow
        ).abs().mean(axis='columns')
    )
    node_voltage_vector_error_real = (
        100.0 * (
            (node_voltage_vector_linear_model.apply(np.real) - node_voltage_vector_power_flow.apply(np.real))
            / node_voltage_vector_power_flow.apply(np.real)
        ).mean(axis='columns')
    )
    node_voltage_vector_error_imag = (
        100.0 * (
            (node_voltage_vector_linear_model.apply(np.imag) - node_voltage_vector_power_flow.apply(np.imag))
            / node_voltage_vector_power_flow.apply(np.imag)
        ).mean(axis='columns')
    )
    node_voltage_vector_magnitude_error = (
        100.0 * (
            (node_voltage_vector_magnitude_linear_model - node_voltage_vector_magnitude_power_flow)
            / node_voltage_vector_magnitude_power_flow
        ).mean(axis='columns')
    )
    branch_power_vector_1_magnitude_error = (
        100.0 * (
            (branch_power_vector_1_magnitude_linear_model - branch_power_vector_1_magnitude_power_flow)
            / branch_power_vector_1_magnitude_power_flow
        ).mean(axis='columns')
    )
    branch_power_vector_2_magnitude_error = (
        100.0 * (
            (branch_power_vector_2_magnitude_linear_model - branch_power_vector_2_magnitude_power_flow)
            / branch_power_vector_2_magnitude_power_flow
        ).mean(axis='columns')
    )
    branch_power_vector_1_squared_error = (
        100.0 * (
            (branch_power_vector_1_squared_linear_model - branch_power_vector_1_squared_power_flow)
            / branch_power_vector_1_squared_power_flow
        ).mean(axis='columns')
    )
    branch_power_vector_2_squared_error = (
        100.0 * (
            (branch_power_vector_2_squared_linear_model - branch_power_vector_2_squared_power_flow)
            / branch_power_vector_2_squared_power_flow
        ).mean(axis='columns')
    )
    loss_active_error = (
        100.0 * (
            (loss_active_linear_model - loss_active_power_flow)
            / loss_active_power_flow
        )
    )
    loss_reactive_error = (
        100.0 * (
            (loss_reactive_linear_model - loss_reactive_power_flow)
            / loss_reactive_power_flow
        )
    )

    # Obtain error table.
    linear_electric_grid_model_error = (
        pd.DataFrame(
            [
                node_voltage_vector_error,
                node_voltage_vector_error_real,
                node_voltage_vector_error_imag,
                node_voltage_vector_magnitude_error,
                branch_power_vector_1_magnitude_error,
                branch_power_vector_2_magnitude_error,
                branch_power_vector_1_squared_error,
                branch_power_vector_2_squared_error,
                loss_active_error,
                loss_reactive_error
            ],
            index=[
                'node_voltage_vector_error',
                'node_voltage_vector_error_real',
                'node_voltage_vector_error_imag',
                'node_voltage_vector_magnitude_error',
                'branch_power_vector_1_magnitude_error',
                'branch_power_vector_2_magnitude_error',
                'branch_power_vector_1_squared_error',
                'branch_power_vector_2_squared_error',
                'loss_active_error',
                'loss_reactive_error'
            ]
        )
    )
    linear_electric_grid_model_error = linear_electric_grid_model_error.round(2)

    # Print results.
    print(f"linear_electric_grid_model_error =\n{linear_electric_grid_model_error}")

    # Store results as CSV.
    der_power_vector_active.to_csv(os.path.join(results_path, 'der_power_vector_active.csv'))
    der_power_vector_reactive.to_csv(os.path.join(results_path, 'der_power_vector_reactive.csv'))
    der_power_vector_active_change.to_csv(os.path.join(results_path, 'der_power_vector_active_change.csv'))
    der_power_vector_reactive_change.to_csv(os.path.join(results_path, 'der_power_vector_reactive_change.csv'))
    node_voltage_vector_power_flow.to_csv(os.path.join(results_path, 'node_voltage_vector_power_flow.csv'))
    node_voltage_vector_linear_model.to_csv(os.path.join(results_path, 'node_voltage_vector_linear_model.csv'))
    node_voltage_vector_magnitude_power_flow.to_csv(os.path.join(results_path, 'node_voltage_vector_magnitude_power_flow.csv'))
    node_voltage_vector_magnitude_linear_model.to_csv(os.path.join(results_path, 'node_voltage_vector_magnitude_linear_model.csv'))
    branch_power_vector_1_squared_power_flow.to_csv(os.path.join(results_path, 'branch_power_vector_1_squared_power_flow.csv'))
    branch_power_vector_1_squared_linear_model.to_csv(os.path.join(results_path, 'branch_power_vector_1_squared_linear_model.csv'))
    branch_power_vector_2_squared_power_flow.to_csv(os.path.join(results_path, 'branch_power_vector_2_squared_power_flow.csv'))
    branch_power_vector_2_squared_linear_model.to_csv(os.path.join(results_path, 'branch_power_vector_2_squared_linear_model.csv'))
    branch_power_vector_1_magnitude_power_flow.to_csv(os.path.join(results_path, 'branch_power_vector_1_magnitude_power_flow.csv'))
    branch_power_vector_1_magnitude_linear_model.to_csv(os.path.join(results_path, 'branch_power_vector_1_magnitude_linear_model.csv'))
    branch_power_vector_2_magnitude_power_flow.to_csv(os.path.join(results_path, 'branch_power_vector_2_magnitude_power_flow.csv'))
    branch_power_vector_2_magnitude_linear_model.to_csv(os.path.join(results_path, 'branch_power_vector_2_magnitude_linear_model.csv'))
    loss_active_power_flow.to_csv(os.path.join(results_path, 'loss_active_power_flow.csv'))
    loss_active_linear_model.to_csv(os.path.join(results_path, 'loss_active_linear_model.csv'))
    loss_reactive_power_flow.to_csv(os.path.join(results_path, 'loss_reactive_power_flow.csv'))
    loss_reactive_linear_model.to_csv(os.path.join(results_path, 'loss_reactive_linear_model.csv'))
    linear_electric_grid_model_error.to_csv(os.path.join(results_path, 'linear_electric_grid_model_error.csv'))

    # Plot results.

    # Voltage.
    for node_index, node in enumerate(electric_grid_model.nodes):
        plt.plot(power_multipliers, base_voltage * node_voltage_vector_magnitude_power_flow.loc[:, node], label='Power flow')
        plt.plot(power_multipliers, base_voltage * node_voltage_vector_magnitude_linear_model.loc[:, node], label='Linear model')
        plt.scatter([0.0], [base_voltage * abs(electric_grid_model.node_voltage_vector_reference[node_index])], label='No load')
        plt.scatter([1.0], [base_voltage * abs(power_flow_solution_initial.node_voltage_vector[node_index])], label='Initial point')
        plt.legend()
        plt.title(f"Voltage magnitude [V] for\n (node_type, node_name, phase): {node}")
        plt.savefig(os.path.join(results_path, f'voltage_magnitude_{node}.png'))
        # plt.show()
        plt.close()

        plt.plot(power_multipliers, np.real(node_voltage_vector_power_flow.loc[:, node]), label='Power flow')
        plt.plot(power_multipliers, np.real(node_voltage_vector_linear_model.loc[:, node]), label='Linear model')
        plt.scatter([0.0], [np.real(electric_grid_model.node_voltage_vector_reference[node_index])], label='No load')
        plt.scatter([1.0], [np.real(power_flow_solution_initial.node_voltage_vector[node_index])], label='Initial point')
        plt.legend()
        plt.title(f"Voltage (real component) [V] for\n (node_type, node_name, phase): {node}")
        plt.savefig(os.path.join(results_path, f'voltage_real_{node}.png'))
        # plt.show()
        plt.close()

        plt.plot(power_multipliers, np.imag(node_voltage_vector_power_flow.loc[:, node]), label='Power flow')
        plt.plot(power_multipliers, np.imag(node_voltage_vector_linear_model.loc[:, node]), label='Linear model')
        plt.scatter([0.0], [np.imag(electric_grid_model.node_voltage_vector_reference[node_index])], label='No load')
        plt.scatter([1.0], [np.imag(power_flow_solution_initial.node_voltage_vector[node_index])], label='Initial point')
        plt.legend()
        plt.title(f"Voltage (imaginary component) [V] for\n (node_type, node_name, phase): {node}")
        plt.savefig(os.path.join(results_path, f'voltage_imag_{node}.png'))
        # plt.show()
        plt.close()

    # Branch flow.
    for branch_index, branch in enumerate(electric_grid_model.branches):
        plt.plot(power_multipliers, base_power * branch_power_vector_1_magnitude_power_flow.loc[:, branch], label='Power flow')
        plt.plot(power_multipliers, base_power * branch_power_vector_1_magnitude_linear_model.loc[:, branch], label='Linear model')
        plt.scatter([0.0], [0.0], label='No load')
        plt.scatter([1.0], [base_power * abs(power_flow_solution_initial.branch_power_vector_1[branch_index])], label='Initial point')
        plt.legend()
        plt.title(f"Branch power 1 magnitude [VA] for\n (branch_type, branch_name, phase): {branch}")
        plt.savefig(os.path.join(results_path, f'branch_power_1_magnitude_{branch}.png'))
        # plt.show()
        plt.close()

        plt.plot(power_multipliers, base_power * branch_power_vector_2_magnitude_power_flow.loc[:, branch], label='Power flow')
        plt.plot(power_multipliers, base_power * branch_power_vector_2_magnitude_linear_model.loc[:, branch], label='Linear model')
        plt.scatter([0.0], [0.0], label='No load')
        plt.scatter([1.0], [base_power * abs(power_flow_solution_initial.branch_power_vector_2[branch_index])], label='Initial point')
        plt.legend()
        plt.title(f"Branch power 2 magnitude [VA] for\n (branch_type, branch_name, phase): {branch}")
        plt.savefig(os.path.join(results_path, f'branch_power_2_magnitude_{branch}.png'))
        # plt.show()
        plt.close()

        plt.plot(power_multipliers, (base_power ** 2) * branch_power_vector_1_squared_power_flow.loc[:, branch], label='Power flow')
        plt.plot(power_multipliers, (base_power ** 2) * branch_power_vector_1_squared_linear_model.loc[:, branch], label='Linear model')
        plt.scatter([0.0], [0.0], label='No load')
        plt.scatter([1.0], [(base_power ** 2) * abs(power_flow_solution_initial.branch_power_vector_1[branch_index] ** 2)], label='Initial point')
        plt.legend()
        plt.title(f"Branch power 1 squared [VA²] for\n (branch_type, branch_name, phase): {branch}")
        plt.savefig(os.path.join(results_path, f'branch_power_1_squared_{branch}.png'))
        # plt.show()
        plt.close()

        plt.plot(power_multipliers, (base_power ** 2) * branch_power_vector_2_squared_power_flow.loc[:, branch], label='Power flow')
        plt.plot(power_multipliers, (base_power ** 2) * branch_power_vector_2_squared_linear_model.loc[:, branch], label='Linear model')
        plt.scatter([0.0], [0.0], label='No load')
        plt.scatter([1.0], [(base_power ** 2) * abs(power_flow_solution_initial.branch_power_vector_2[branch_index] ** 2)], label='Initial point')
        plt.legend()
        plt.title(f"Branch power 2 squared [VA²] for\n (branch_type, branch_name, phase): {branch}")
        plt.savefig(os.path.join(results_path, f'branch_power_2_squared_{branch}.png'))
        # plt.show()
        plt.close()

    # Loss.
    plt.plot(power_multipliers, base_power * loss_active_power_flow, label='Power flow')
    plt.plot(power_multipliers, base_power * loss_active_linear_model, label='Linear model')
    plt.scatter([0.0], [0.0], label='No load')
    plt.scatter([1.0], [base_power * np.real([power_flow_solution_initial.loss])], label='Initial point')
    plt.legend()
    plt.title("Total loss active [W]")
    plt.savefig(os.path.join(results_path, f'loss_active.png'))
    # plt.show()
    plt.close()

    plt.plot(power_multipliers, base_power * loss_reactive_power_flow, label='Power flow')
    plt.plot(power_multipliers, base_power * loss_reactive_linear_model, label='Linear model')
    plt.scatter([1.0], [base_power * np.imag([power_flow_solution_initial.loss])], label='Initial point')
    plt.legend()
    plt.title("Total loss reactive [VAr]")
    plt.savefig(os.path.join(results_path, f'loss_reactive.png'))
    # plt.show()
    plt.close()

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
