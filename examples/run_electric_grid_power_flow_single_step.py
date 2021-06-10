"""Example script for setting up and solving an single step electric grid power flow problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import fledge

def main():

    # Settings.
    scenario_name = 'test_2node'
    results_path = fledge.utils.get_results_path(__file__, scenario_name)

    # Recreate / overwrite database, to incorporate changes in the CSV files.
    fledge.data_interface.recreate_database()

    # Obtain scenario data.
    # - This contains general information, e.g. the base power values which are needed for the plots below.
    scenario_data = fledge.data_interface.ScenarioData(scenario_name)

    # Obtain electric grid model.
    # - The ElectricGridModelDefault object defines index sets for node names / branch names / der names / phases /
    #   node types / branch types, the nodal admittance / transformation matrices, branch admittance /
    #   incidence matrices, DER incidence matrices and no load voltage vector as well as nominal power vector.
    # - The model is created for the electric grid which is defined for the given scenario in the data directory.
    electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)

    # Obtain the nominal DER power vector.
    der_power_vector_nominal = electric_grid_model.der_power_vector_reference

    # Obtain power flow solution.
    # - The power flow solution object obtains the solution for nodal voltage vector / branch power vector
    #   and total loss (all complex valued) for the given DER power vector.
    # - There are different power flow solution objects depending on the solution algorithm / method:
    #   `PowerFlowSolutionFixedPoint`, `PowerFlowSolutionZBus`.
    #   (`PowerFlowSolutionOpenDSS` requires a `ElectricGridModelOpenDSS` instead of `ElectricGridModelDefault`.)
    power_flow_solution = (
        fledge.electric_grid_models.PowerFlowSolutionZBus(
            electric_grid_model,
            der_power_vector_nominal
        )
    )

    # Obtain results (as numpy arrays).
    der_power_vector = power_flow_solution.der_power_vector  # The DER power vector is also stored in the solution.
    node_voltage_vector = power_flow_solution.node_voltage_vector
    node_voltage_vector_magnitude = np.abs(power_flow_solution.node_voltage_vector)
    branch_power_vector_1 = power_flow_solution.branch_power_vector_1
    branch_power_vector_2 = power_flow_solution.branch_power_vector_2
    loss = power_flow_solution.loss

    # Print results.
    print(f"der_power_vector = \n{der_power_vector}")
    print(f"node_voltage_vector = \n{node_voltage_vector}")
    print(f"node_voltage_vector_magnitude = \n{node_voltage_vector_magnitude}")
    print(f"branch_power_vector_1 = \n{branch_power_vector_1}")
    print(f"branch_power_vector_2 = \n{branch_power_vector_2}")
    print(f"loss = {loss}")

    # Save results to CSV.
    np.savetxt(os.path.join(results_path, f'der_power_vector.csv'), der_power_vector, delimiter=',')
    np.savetxt(os.path.join(results_path, f'node_voltage_vector.csv'), node_voltage_vector, delimiter=',')
    np.savetxt(os.path.join(results_path, f'node_voltage_vector_magnitude.csv'), node_voltage_vector_magnitude, delimiter=',')
    np.savetxt(os.path.join(results_path, f'branch_power_vector_1.csv'), branch_power_vector_1, delimiter=',')
    np.savetxt(os.path.join(results_path, f'branch_power_vector_2.csv'), branch_power_vector_2, delimiter=',')
    np.savetxt(os.path.join(results_path, f'loss.csv'), loss, delimiter=',')

    # Plot some results.

    figure = go.Figure()
    figure.add_bar(
        x=electric_grid_model.ders.to_series().astype(str),  # Labels.
        y=(np.real(der_power_vector) * scenario_data.scenario.at['base_apparent_power'] / 1e3),  # Values in kW.
    )
    figure.update_layout(
        title="DER active power",
        yaxis_title="Active power [kW]",
        xaxis=dict(tickangle=-45),
        showlegend=False,
        margin=dict(b=150)
    )
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, figure.layout.title.text))

    figure = go.Figure()
    figure.add_bar(
        x=electric_grid_model.ders.to_series().astype(str),  # Labels.
        y=(np.imag(der_power_vector) * scenario_data.scenario.at['base_apparent_power'] / 1e3),  # Values in kW.
    )
    figure.update_layout(
        title="DER reactive power",
        yaxis_title="Reactive power [kVA<sub>r</sub>]",
        xaxis=dict(tickangle=-45),
        showlegend=False,
        margin=dict(b=150)
    )
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, figure.layout.title.text))

    figure = go.Figure()
    figure.add_bar(
        x=electric_grid_model.nodes.to_series().astype(str),  # Labels.
        y=(node_voltage_vector_magnitude * scenario_data.scenario.at['base_voltage'] / 1e3),  # Values in kV.
    )
    figure.update_layout(
        title="Node voltage magnitude",
        yaxis_title="Voltage magnitude [kV]",
        xaxis=dict(tickangle=-45),
        showlegend=False,
        margin=dict(b=150)
    )
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, figure.layout.title.text))

    figure = go.Figure()
    figure.add_bar(
        x=electric_grid_model.nodes.to_series().astype(str),  # Labels.
        y=(node_voltage_vector_magnitude / np.abs(electric_grid_model.node_voltage_vector_reference)),  # Values in p.u.
    )
    figure.update_layout(
        title="Node voltage per-unit magnitude",
        yaxis_title="Voltage magnitude [p.u.]",
        xaxis=dict(tickangle=-45),
        showlegend=False,
        margin=dict(b=150)
    )
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, figure.layout.title.text))

    figure = go.Figure()
    figure.add_bar(
        x=electric_grid_model.branches.to_series().astype(str),  # Labels.
        y=(np.abs(branch_power_vector_1) * scenario_data.scenario.at['base_apparent_power'] / 1e3),  # Values in kW.
    )
    figure.update_layout(
        title="Branch apparent power flow (direction 1)",
        yaxis_title="Apparent power [kVA]",
        xaxis=dict(tickangle=-45),
        showlegend=False,
        margin=dict(b=150)
    )
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, figure.layout.title.text))

    figure = go.Figure()
    figure.add_bar(
        x=electric_grid_model.branches.to_series().astype(str),  # Labels.
        y=(np.abs(branch_power_vector_2) * scenario_data.scenario.at['base_apparent_power'] / 1e3),  # Values in kW.
    )
    figure.update_layout(
        title="Branch apparent power flow (direction 2)",
        yaxis_title="Apparent power [kVA]",
        xaxis=dict(tickangle=-45),
        showlegend=False,
        margin=dict(b=150)
    )
    fledge.utils.write_figure_plotly(figure, os.path.join(results_path, figure.layout.title.text))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
