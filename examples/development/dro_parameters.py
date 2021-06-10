"""Example script for DRO problem."""

import cvxpy as cp
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import cobmo
import fledge


def main():

    # Settings.
    scenario_name = 'singapore_6node'
    der_name = '4_4'  # 4_4 is a flexible building.

    # Get results path.
    results_path = fledge.utils.get_results_path(__file__, f'{scenario_name}_der_{der_name}')

    # Recreate / overwrite database, to incorporate changes in the CSV definition files.
    fledge.data_interface.recreate_database()

    # Obtain data object.
    der_data = fledge.data_interface.DERData(scenario_name)

    # Obtain model object.
    building_model = (
        cobmo.building_model.BuildingModel(
            der_data.ders.at[der_name, 'der_model_name'],
            timestep_start='2017-01-01T00:00:00',
            timestep_end='2018-01-01T00:00:00',
            timestep_interval='00:30:00'
        )
    )

    # Calculate uncertainty parameters.
    disturbance_timeseries_copy = building_model.disturbance_timeseries.copy()
    disturbance_timeseries_copy.index = disturbance_timeseries_copy.index.strftime('%H:%M:%S')
    disturbance_timeseries_copy = disturbance_timeseries_copy.groupby(disturbance_timeseries_copy.index)
    disturbance_timeseries_mean = disturbance_timeseries_copy.mean()
    disturbance_timeseries_var = disturbance_timeseries_copy.var()

    # Print results.
    print(f"disturbance_timeseries_mean = {disturbance_timeseries_mean}")
    print(f"disturbance_timeseries_var = {disturbance_timeseries_var}")

    # Store results.
    building_model.disturbance_timeseries.to_csv(os.path.join(results_path, 'disturbance_timeseries_raw.csv'))
    disturbance_timeseries_mean.to_csv(os.path.join(results_path, 'disturbance_timeseries_mean.csv'))
    disturbance_timeseries_var.to_csv(os.path.join(results_path, 'disturbance_timeseries_var.csv'))

    # Print results path.
    fledge.utils.launch(results_path)
    print(f"Results are stored in: {results_path}")


if __name__ == '__main__':
    main()
