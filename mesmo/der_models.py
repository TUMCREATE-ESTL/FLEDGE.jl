"""Distributed energy resource (DER) models."""

import cvxpy as cp
import inspect
import itertools
from multimethod import multimethod
import numpy as np
import pandas as pd
import scipy.constants
import scipy.sparse as sp
import sys
import typing

import mesmo.config
import mesmo.data_interface
import mesmo.electric_grid_models
import mesmo.utils

logger = mesmo.config.get_logger(__name__)


class DERModel(object):
    """DER model object."""

    der_type: str = None
    # TODO: Revise marginal cost implementation to split active / reactive / thermal power cost.
    marginal_cost: float
    der_name: str
    is_standalone: bool
    is_electric_grid_connected: bool
    is_thermal_grid_connected: bool
    electric_grid_der_index: typing.List[int]
    thermal_grid_der_index: typing.List[int]
    timesteps: pd.Index
    active_power_nominal: float
    reactive_power_nominal: float
    thermal_power_nominal: float
    active_power_nominal_timeseries: pd.Series
    reactive_power_nominal_timeseries: pd.Series
    thermal_power_nominal_timeseries: pd.Series

    def __init__(
            self,
            der_data: mesmo.data_interface.DERData,
            der_name: str,
            is_standalone=False
    ):

        # Get shorthand for DER data.
        der = der_data.ders.loc[der_name, :]

        # Store DER name.
        self.der_name = der_name

        # Obtain grid connection flags.
        self.is_standalone = is_standalone
        self.is_electric_grid_connected = pd.notnull(der.at['electric_grid_name'])
        self.is_thermal_grid_connected = pd.notnull(der.at['thermal_grid_name'])

        # Obtain DER grid indexes.
        self.electric_grid_der_index = (
            [der_data.ders.loc[der_data.ders.loc[:, 'electric_grid_name'].notnull(), :].index.get_loc(der_name)]
            if self.is_electric_grid_connected
            else []
        )
        self.thermal_grid_der_index = (
            [der_data.ders.loc[der_data.ders.loc[:, 'thermal_grid_name'].notnull(), :].index.get_loc(der_name)]
            if self.is_thermal_grid_connected
            else []
        )

        # Obtain timesteps index.
        self.timesteps = der_data.scenario_data.timesteps

        # Obtain nominal power values.
        self.active_power_nominal = (
            der.at['active_power_nominal'] if pd.notnull(der.at['active_power_nominal']) else 0.0
        )
        self.reactive_power_nominal = (
            der.at['reactive_power_nominal'] if pd.notnull(der.at['reactive_power_nominal']) else 0.0
        )
        self.thermal_power_nominal = (
            der.at['thermal_power_nominal'] if pd.notnull(der.at['thermal_power_nominal']) else 0.0
        )

        # Construct nominal active and reactive power timeseries.
        if (
                pd.notnull(der.at['definition_type'])
                and (('schedule' in der.at['definition_type']) or ('timeseries' in der.at['definition_type']))
                and self.is_electric_grid_connected
        ):
            self.active_power_nominal_timeseries = (
                der_data.der_definitions[
                    der.at['definition_index']
                ].loc[:, 'value'].copy().abs().rename('active_power')
            )
            self.reactive_power_nominal_timeseries = (
                der_data.der_definitions[
                    der.at['definition_index']
                ].loc[:, 'value'].copy().abs().rename('reactive_power')
            )
            if 'per_unit' in der.at['definition_type']:
                # If per unit definition, multiply nominal active / reactive power.
                self.active_power_nominal_timeseries *= self.active_power_nominal
                self.reactive_power_nominal_timeseries *= self.reactive_power_nominal
            else:
                self.active_power_nominal_timeseries *= (
                    np.sign(self.active_power_nominal)
                    / der_data.scenario_data.scenario.at['base_apparent_power']
                )
                self.reactive_power_nominal_timeseries *= (
                    np.sign(self.reactive_power_nominal)
                    * (
                        self.reactive_power_nominal / self.active_power_nominal
                        if self.active_power_nominal != 0.0
                        else 1.0
                    )
                    / der_data.scenario_data.scenario.at['base_apparent_power']
                )
        else:
            self.active_power_nominal_timeseries = (
                pd.Series(0.0, index=self.timesteps, name='active_power')
            )
            self.reactive_power_nominal_timeseries = (
                pd.Series(0.0, index=self.timesteps, name='reactive_power')
            )

        # Construct nominal thermal power timeseries.
        if (
                pd.notnull(der.at['definition_type'])
                and (('schedule' in der.at['definition_type']) or ('timeseries' in der.at['definition_type']))
                and self.is_thermal_grid_connected
        ):

            # Construct nominal thermal power timeseries.
            self.thermal_power_nominal_timeseries = (
                der_data.der_definitions[
                    der.at['definition_index']
                ].loc[:, 'value'].copy().abs().rename('thermal_power')
            )
            if 'per_unit' in der.at['definition_type']:
                # If per unit definition, multiply nominal thermal power.
                self.thermal_power_nominal_timeseries *= self.thermal_power_nominal
            else:
                self.thermal_power_nominal_timeseries *= (
                    np.sign(self.thermal_power_nominal)
                    / der_data.scenario_data.scenario.at['base_thermal_power']
                )
        else:
            self.thermal_power_nominal_timeseries = (
                pd.Series(0.0, index=self.timesteps, name='thermal_power')
            )

        # Obtain marginal cost.
        self.marginal_cost = der.at['marginal_cost'] if pd.notnull(der.at['marginal_cost']) else 0.0


class DERModelOperationResults(mesmo.utils.ResultsBase):

    der_model: DERModel
    state_vector: pd.DataFrame
    control_vector: pd.DataFrame
    output_vector: pd.DataFrame


class FixedDERModel(DERModel):
    """Fixed DER model object."""


class FixedLoadModel(FixedDERModel):
    """Fixed load model object."""

    der_type = 'fixed_load'

    def __init__(
            self,
            der_data: mesmo.data_interface.DERData,
            der_name: str,
            **kwargs
    ):
        """Construct fixed load model object by `der_data` and `der_name`."""

        # Common initializations are implemented in parent class.
        super().__init__(der_data, der_name, **kwargs)

        # If connected to both electric and thermal grid, raise error.
        if self.is_electric_grid_connected and self.is_thermal_grid_connected:
            raise AssertionError(
                f"Fixed load '{self.der_name}' can only be connected to either electric grid or thermal grid."
            )


class FixedEVChargerModel(FixedDERModel):
    """EV charger model object."""

    der_type = 'fixed_ev_charger'

    def __init__(
            self,
            der_data: mesmo.data_interface.DERData,
            der_name: str,
            **kwargs
    ):
        """Construct EV charger model object by `der_data` and `der_name`."""

        # Common initializations are implemented in parent class.
        super().__init__(der_data, der_name, **kwargs)

        # If connected to thermal grid, raise error.
        if self.is_electric_grid_connected and self.is_thermal_grid_connected:
            raise AssertionError(
                f"Fixed EV charger '{self.der_name}' can only be connected to electric grid."
            )


class FixedGeneratorModel(FixedDERModel):
    """Fixed generator model object, representing a generic generator with fixed nominal output."""

    der_type = 'fixed_generator'
    marginal_cost: float

    def __init__(
            self,
            der_data: mesmo.data_interface.DERData,
            der_name: str,
            **kwargs
    ):

        # Common initializations are implemented in parent class.
        super().__init__(der_data, der_name, **kwargs)

        # Get shorthand for DER data.
        der = der_data.ders.loc[self.der_name, :]

        # If connected to both electric and thermal grid, raise error.
        if self.is_electric_grid_connected and self.is_thermal_grid_connected:
            raise AssertionError(
                f"Fixed generator '{self.der_name}' can only be connected to either electric grid or thermal grid."
            )


class FlexibleDERModel(DERModel):
    """Flexible DER model, e.g., flexible load, object."""

    states: pd.Index
    storage_states: pd.Index = pd.Index([])
    controls: pd.Index
    disturbances: pd.Index
    outputs: pd.Index
    mapping_active_power_by_output: pd.DataFrame
    mapping_reactive_power_by_output: pd.DataFrame
    mapping_thermal_power_by_output: pd.DataFrame
    state_vector_initial: pd.Series
    state_matrix: pd.DataFrame
    control_matrix: pd.DataFrame
    disturbance_matrix: pd.DataFrame
    state_output_matrix: pd.DataFrame
    control_output_matrix: pd.DataFrame
    disturbance_output_matrix: pd.DataFrame
    disturbance_timeseries: pd.DataFrame
    output_maximum_timeseries: pd.DataFrame
    output_minimum_timeseries: pd.DataFrame


class FlexibleLoadModel(FlexibleDERModel):
    """Flexible load model object."""

    der_type = 'flexible_load'

    def __init__(
            self,
            der_data: mesmo.data_interface.DERData,
            der_name: str,
            **kwargs
    ):
        """Construct flexible load model object by `der_data` and `der_name`."""

        # Common initializations are implemented in parent class.
        super().__init__(der_data, der_name, **kwargs)

        # Get shorthand for DER data.
        der = der_data.ders.loc[self.der_name, :]

        # If connected to both electric and thermal grid, raise error.
        if self.is_electric_grid_connected and self.is_thermal_grid_connected:
            raise AssertionError(
                f"Flexible load '{self.der_name}' can only be connected to either electric grid or thermal grid."
            )

        if self.is_electric_grid_connected:

            # Instantiate indexes.
            self.states = pd.Index(['state_of_charge'])
            self.storage_states = pd.Index(['state_of_charge'])
            self.controls = pd.Index(['active_power'])
            self.disturbances = pd.Index(['active_power'])
            self.outputs = pd.Index(['state_of_charge', 'active_power', 'reactive_power'])

            # Instantiate initial state.
            # - Note that this is not used for `storage_states`, whose initial state is coupled with their final state.
            self.state_vector_initial = (
                pd.Series(0.0, index=self.states)
            )

            # Instantiate state space matrices.
            # TODO: Add shifting losses / self discharge.
            self.state_matrix = (
                pd.DataFrame(0.0, index=self.states, columns=self.states)
            )
            self.state_matrix.at['state_of_charge', 'state_of_charge'] = 1.0
            self.control_matrix = (
                pd.DataFrame(0.0, index=self.states, columns=self.controls)
            )
            self.control_matrix.at['state_of_charge', 'active_power'] = (
                -1.0
            )
            self.disturbance_matrix = (
                pd.DataFrame(0.0, index=self.states, columns=self.disturbances)
            )
            self.disturbance_matrix.at['state_of_charge', 'active_power'] = (
                1.0
            )
            self.state_output_matrix = (
                pd.DataFrame(0.0, index=self.outputs, columns=self.states)
            )
            self.state_output_matrix.at['state_of_charge', 'state_of_charge'] = 1.0
            self.control_output_matrix = (
                pd.DataFrame(0.0, index=self.outputs, columns=self.controls)
            )
            self.control_output_matrix.at['active_power', 'active_power'] = 1.0
            self.control_output_matrix.at['reactive_power', 'active_power'] = (
                self.reactive_power_nominal / self.active_power_nominal
                if self.active_power_nominal != 0.0
                else 0.0
            )
            self.disturbance_output_matrix = (
                pd.DataFrame(0.0, index=self.outputs, columns=self.disturbances)
            )

            # Instantiate disturbance timeseries.
            self.disturbance_timeseries = (
                self.active_power_nominal_timeseries.to_frame()
            )

            # Construct output constraint timeseries
            self.output_maximum_timeseries = (
                pd.concat([
                    pd.Series((
                        np.abs(der['active_power_nominal'] if der['active_power_nominal'] != 0.0 else 1.0)
                        * der['energy_storage_capacity_per_unit']
                        * (pd.Timedelta('1h') / der_data.scenario_data.scenario.at['timestep_interval'])
                    ), index=self.active_power_nominal_timeseries.index, name='state_of_charge'),
                    (
                        der['power_per_unit_minimum']  # Take minimum, because load is negative power.
                        * self.active_power_nominal_timeseries
                    ),
                    (
                        der['power_per_unit_minimum']  # Take minimum, because load is negative power.
                        * self.reactive_power_nominal_timeseries
                    )
                ], axis='columns')
            )
            self.output_minimum_timeseries = (
                pd.concat([
                    pd.Series(0.0, index=self.active_power_nominal_timeseries.index, name='state_of_charge'),
                    (
                        der['power_per_unit_maximum']  # Take maximum, because load is negative power.
                        * self.active_power_nominal_timeseries
                    ),
                    (
                        der['power_per_unit_maximum']  # Take maximum, because load is negative power.
                        * self.reactive_power_nominal_timeseries
                    )
                ], axis='columns')
            )

        if self.is_thermal_grid_connected:

            # Instantiate indexes.
            self.states = pd.Index(['state_of_charge'])
            self.storage_states = pd.Index(['state_of_charge'])
            self.controls = pd.Index(['thermal_power'])
            self.disturbances = pd.Index(['thermal_power'])
            self.outputs = pd.Index(['state_of_charge', 'thermal_power'])

            # Instantiate initial state.
            # - Note that this is not used for `storage_states`, whose initial state is coupled with their final state.
            self.state_vector_initial = (
                pd.Series(0.0, index=self.states)
            )

            # Instantiate state space matrices.
            # TODO: Add shifting losses / self discharge.
            self.state_matrix = (
                pd.DataFrame(0.0, index=self.states, columns=self.states)
            )
            self.state_matrix.at['state_of_charge', 'state_of_charge'] = 1.0
            self.control_matrix = (
                pd.DataFrame(0.0, index=self.states, columns=self.controls)
            )
            self.control_matrix.at['state_of_charge', 'thermal_power'] = (
                -1.0
            )
            self.disturbance_matrix = (
                pd.DataFrame(0.0, index=self.states, columns=self.disturbances)
            )
            self.disturbance_matrix.at['state_of_charge', 'thermal_power'] = (
                1.0
            )
            self.state_output_matrix = (
                pd.DataFrame(0.0, index=self.outputs, columns=self.states)
            )
            self.state_output_matrix.at['state_of_charge', 'state_of_charge'] = 1.0
            self.control_output_matrix = (
                pd.DataFrame(0.0, index=self.outputs, columns=self.controls)
            )
            self.control_output_matrix.at['thermal_power', 'thermal_power'] = 1.0
            self.disturbance_output_matrix = (
                pd.DataFrame(0.0, index=self.outputs, columns=self.disturbances)
            )

            # Instantiate disturbance timeseries.
            self.disturbance_timeseries = (
                self.thermal_power_nominal_timeseries.to_frame()
            )

            # Construct output constraint timeseries
            self.output_maximum_timeseries = (
                pd.concat([
                    pd.Series((
                        np.abs(der['thermal_power_nominal'] if der['thermal_power_nominal'] != 0.0 else 1.0)
                        * der['energy_storage_capacity_per_unit']
                        * (pd.Timedelta('1h') / der_data.scenario_data.scenario.at['timestep_interval'])
                    ), index=self.thermal_power_nominal_timeseries.index, name='state_of_charge'),
                    (
                        der['power_per_unit_minimum']  # Take minimum, because load is negative power.
                        * self.thermal_power_nominal_timeseries
                    )
                ], axis='columns')
            )
            self.output_minimum_timeseries = (
                pd.concat([
                    pd.Series(0.0, index=self.thermal_power_nominal_timeseries.index, name='state_of_charge'),
                    (
                        der['power_per_unit_maximum']  # Take maximum, because load is negative power.
                        * self.thermal_power_nominal_timeseries
                    )
                ], axis='columns')
            )

        # Define power mapping matrices.
        self.mapping_active_power_by_output = pd.DataFrame(0.0, index=['active_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_active_power_by_output.at['active_power', 'active_power'] = 1.0
        self.mapping_reactive_power_by_output = pd.DataFrame(0.0, index=['reactive_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_reactive_power_by_output.at['reactive_power', 'reactive_power'] = 1.0
        self.mapping_thermal_power_by_output = pd.DataFrame(0.0, index=['thermal_power'], columns=self.outputs)
        if self.is_thermal_grid_connected:
            self.mapping_thermal_power_by_output.at['thermal_power', 'thermal_power'] = 1.0


class FlexibleEVChargerModel(FlexibleDERModel):
    """Flexible EV charger model object."""

    der_type = 'flexible_ev_charger'

    def __init__(
            self,
            der_data: mesmo.data_interface.DERData,
            der_name: str,
            **kwargs
    ):
        """Construct flexible load model object by `der_data` and `der_name`."""

        # Common initializations are implemented in parent class.
        super().__init__(der_data, der_name, **kwargs)

        # Get shorthand for DER data.
        der = der_data.ders.loc[self.der_name, :]
        der = pd.concat([der, der_data.der_definitions[der.at['definition_index']]])

        # If connected to thermal grid, raise error.
        if self.is_electric_grid_connected and self.is_thermal_grid_connected:
            raise AssertionError(
                f"Fixed EV charger '{self.der_name}' can only be connected to electric grid."
            )

        # Construct nominal active and reactive power timeseries.
        if (
                pd.notnull(der.at['nominal_charging_definition_type'])
                and (
                    ('schedule' in der.at['nominal_charging_definition_type'])
                    or ('timeseries' in der.at['nominal_charging_definition_type'])
                )
                and self.is_electric_grid_connected
        ):
            self.active_power_nominal_timeseries = (
                der_data.der_definitions[
                    der.at['nominal_charging_definition_index']
                ].loc[:, 'value'].copy().abs().rename('active_power')
            )
            self.reactive_power_nominal_timeseries = (
                der_data.der_definitions[
                    der.at['nominal_charging_definition_index']
                ].loc[:, 'value'].copy().abs().rename('reactive_power')
            )
            if 'per_unit' in der.at['nominal_charging_definition_type']:
                # If per unit definition, multiply nominal active / reactive power.
                self.active_power_nominal_timeseries *= self.active_power_nominal
                self.reactive_power_nominal_timeseries *= self.reactive_power_nominal
            else:
                self.active_power_nominal_timeseries *= (
                    np.sign(self.active_power_nominal)
                    / der_data.scenario_data.scenario.at['base_apparent_power']
                )
                self.reactive_power_nominal_timeseries *= (
                    np.sign(self.reactive_power_nominal)
                    * (
                        self.reactive_power_nominal / self.active_power_nominal
                        if self.active_power_nominal != 0.0
                        else 1.0
                    )
                    / der_data.scenario_data.scenario.at['base_apparent_power']
                )
        else:
            self.active_power_nominal_timeseries = (
                pd.Series(0.0, index=self.timesteps, name='active_power')
            )
            self.reactive_power_nominal_timeseries = (
                pd.Series(0.0, index=self.timesteps, name='reactive_power')
            )

        # Instantiate indexes.
        self.states = pd.Index(['charged_energy'])
        self.storage_states = pd.Index(['charged_energy'])
        self.controls = pd.Index(['active_power_charge', 'active_power_discharge'])
        self.disturbances = pd.Index(['departing_energy'])
        self.outputs = (
            pd.Index([
                'charged_energy', 'active_power_charge', 'active_power_discharge',
                'active_power', 'reactive_power'
            ])
        )

        # Define power mapping matrices.
        self.mapping_active_power_by_output = pd.DataFrame(0.0, index=['active_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_active_power_by_output.at['active_power', 'active_power'] = 1.0
        self.mapping_reactive_power_by_output = pd.DataFrame(0.0, index=['reactive_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_reactive_power_by_output.at['reactive_power', 'reactive_power'] = 1.0
        self.mapping_thermal_power_by_output = pd.DataFrame(0.0, index=['thermal_power'], columns=self.outputs)

        # Instantiate initial state.
        # - Note that this is not used for `storage_states`, whose initial state is coupled with their final state.
        self.state_vector_initial = (
            pd.Series(0.0, index=self.states)
        )

        # Instantiate state space matrices.
        # TODO: Add shifting losses / self discharge.
        self.state_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.states)
        )
        self.state_matrix.at['charged_energy', 'charged_energy'] = 1.0
        self.control_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.controls)
        )
        self.control_matrix.at['charged_energy', 'active_power_charge'] = (
            der['charging_efficiency']
            * (der_data.scenario_data.scenario.at['timestep_interval'] / pd.Timedelta('1h'))
        )
        self.control_matrix.at['charged_energy', 'active_power_discharge'] = (
            -1.0
            * (der_data.scenario_data.scenario.at['timestep_interval'] / pd.Timedelta('1h'))
        )
        self.disturbance_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.disturbances)
        )
        self.disturbance_matrix.at['charged_energy', 'departing_energy'] = -1.0
        self.state_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.states)
        )
        self.state_output_matrix.at['charged_energy', 'charged_energy'] = 1.0
        self.control_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.controls)
        )
        self.control_output_matrix.at['active_power_charge', 'active_power_charge'] = 1.0
        self.control_output_matrix.at['active_power_discharge', 'active_power_discharge'] = 1.0
        self.control_output_matrix.at['active_power', 'active_power_charge'] = -1.0
        self.control_output_matrix.at['active_power', 'active_power_discharge'] = 1.0
        self.control_output_matrix.at['reactive_power', 'active_power_charge'] = (
            -1.0 * self.reactive_power_nominal / self.active_power_nominal
            if self.active_power_nominal != 0.0
            else 0.0
        )
        self.control_output_matrix.at['reactive_power', 'active_power_discharge'] = (
            self.reactive_power_nominal / self.active_power_nominal
            if self.active_power_nominal != 0.0
            else 0.0
        )
        self.disturbance_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.disturbances)
        )

        # Instantiate disturbance timeseries.
        self.disturbance_timeseries = (
            pd.concat([
                pd.Series((
                    der_data.der_definitions[der.at['departing_energy_definition_index']].loc[:, 'value'].copy()
                    / der_data.scenario_data.scenario.at['base_apparent_power']
                ), index=self.timesteps, name='departing_energy')
            ], axis='columns')
        )

        # Construct output constraint timeseries.
        self.output_maximum_timeseries = (
            pd.concat([
                pd.Series((
                    der_data.der_definitions[der.at['maximum_energy_definition_index']].loc[:, 'value'].copy()
                    / der_data.scenario_data.scenario.at['base_apparent_power']
                ), index=self.timesteps, name='charged_energy'),
                pd.Series((
                    der_data.der_definitions[der.at['maximum_charging_definition_index']].loc[:, 'value'].copy()
                    / der_data.scenario_data.scenario.at['base_apparent_power']
                ), index=self.timesteps, name='active_power_charge'),
                pd.Series((
                    der_data.der_definitions[der.at['maximum_discharging_definition_index']].loc[:, 'value'].copy()
                    / der_data.scenario_data.scenario.at['base_apparent_power']
                ), index=self.timesteps, name='active_power_discharge'),
                pd.Series(+np.inf, index=self.active_power_nominal_timeseries.index, name='active_power'),
                pd.Series(+np.inf, index=self.timesteps, name='reactive_power')
            ], axis='columns')
        )
        self.output_minimum_timeseries = (
            pd.concat([
                pd.Series(0.0, index=self.timesteps, name='charged_energy'),
                pd.Series(0.0, index=self.active_power_nominal_timeseries.index, name='active_power_charge'),
                pd.Series(0.0, index=self.active_power_nominal_timeseries.index, name='active_power_discharge'),
                pd.Series(-np.inf, index=self.active_power_nominal_timeseries.index, name='active_power'),
                pd.Series(-np.inf, index=self.timesteps, name='reactive_power')
            ], axis='columns')
        )


class FlexibleGeneratorModel(FlexibleDERModel):
    """Fixed generator model object, representing a generic generator with fixed nominal output."""

    der_type = 'flexible_generator'
    marginal_cost: float

    def __init__(
            self,
            der_data: mesmo.data_interface.DERData,
            der_name: str,
            **kwargs
    ):

        # Common initializations are implemented in parent class.
        super().__init__(der_data, der_name, **kwargs)

        # Get shorthand for DER data.
        der = der_data.ders.loc[self.der_name, :]

        # If connected to both electric and thermal grid, raise error.
        if self.is_electric_grid_connected and self.is_thermal_grid_connected:
            raise AssertionError(
                f"Flexible load '{self.der_name}' can only be connected to either electric grid or thermal grid."
            )

        if self.is_electric_grid_connected:

            # Instantiate indexes.
            self.states = pd.Index(['_'])  # Define placeholder '_' to avoid issues in the optimization problem definition.
            self.controls = pd.Index(['active_power'])
            self.disturbances = pd.Index([])
            self.outputs = pd.Index(['active_power', 'reactive_power'])

            # Instantiate initial state.
            self.state_vector_initial = (
                pd.Series(0.0, index=self.states)
            )

            # Instantiate state space matrices.
            self.state_matrix = (
                pd.DataFrame(0.0, index=self.states, columns=self.states)
            )
            self.control_matrix = (
                pd.DataFrame(0.0, index=self.states, columns=self.controls)
            )
            self.disturbance_matrix = (
                pd.DataFrame(0.0, index=self.states, columns=self.disturbances)
            )
            self.state_output_matrix = (
                pd.DataFrame(0.0, index=self.outputs, columns=self.states)
            )
            self.control_output_matrix = (
                pd.DataFrame(0.0, index=self.outputs, columns=self.controls)
            )
            self.control_output_matrix.at['active_power', 'active_power'] = 1.0
            self.control_output_matrix.at['reactive_power', 'active_power'] = (
                self.reactive_power_nominal / self.active_power_nominal
                if self.active_power_nominal != 0.0
                else 0.0
            )
            self.disturbance_output_matrix = (
                pd.DataFrame(0.0, index=self.outputs, columns=self.disturbances)
            )

            # Instantiate disturbance timeseries.
            self.disturbance_timeseries = (
                pd.DataFrame(0.0, index=self.active_power_nominal_timeseries.index, columns=self.disturbances)
            )

            # Construct output constraint timeseries
            # TODO: Revise constraints with pu limits.
            self.output_maximum_timeseries = (
                pd.concat([
                    self.active_power_nominal_timeseries,
                    self.reactive_power_nominal_timeseries
                ], axis='columns')
            )
            self.output_minimum_timeseries = (
                pd.concat([
                    0.0 * self.active_power_nominal_timeseries,
                    0.0 * self.reactive_power_nominal_timeseries
                ], axis='columns')
            )

        if self.is_thermal_grid_connected:

            # Instantiate indexes.
            self.states = pd.Index(['_'])  # Define placeholder '_' to avoid issues in the optimization problem definition.
            self.controls = pd.Index(['thermal_power'])
            self.disturbances = pd.Index([])
            self.outputs = pd.Index(['thermal_power'])

            # Instantiate initial state.
            self.state_vector_initial = (
                pd.Series(0.0, index=self.states)
            )

            # Instantiate state space matrices.
            self.state_matrix = (
                pd.DataFrame(0.0, index=self.states, columns=self.states)
            )
            self.control_matrix = (
                pd.DataFrame(0.0, index=self.states, columns=self.controls)
            )
            self.disturbance_matrix = (
                pd.DataFrame(0.0, index=self.states, columns=self.disturbances)
            )
            self.state_output_matrix = (
                pd.DataFrame(0.0, index=self.outputs, columns=self.states)
            )
            self.control_output_matrix = (
                pd.DataFrame(0.0, index=self.outputs, columns=self.controls)
            )
            self.control_output_matrix.at['thermal_power', 'thermal_power'] = 1.0

            self.disturbance_output_matrix = (
                pd.DataFrame(0.0, index=self.outputs, columns=self.disturbances)
            )

            # Instantiate disturbance timeseries.
            self.disturbance_timeseries = (
                pd.DataFrame(0.0, index=self.thermal_power_nominal_timeseries.index, columns=self.disturbances)
            )

            # Construct output constraint timeseries
            # TODO: Revise constraints with pu limits.
            self.output_maximum_timeseries = (
                pd.DataFrame(self.thermal_power_nominal_timeseries, index=self.timesteps)
            )

            self.output_minimum_timeseries = (
                pd.DataFrame(0.0 * self.thermal_power_nominal_timeseries, index=self.timesteps)
            )

        # Define power mapping matrices.
        self.mapping_active_power_by_output = pd.DataFrame(0.0, index=['active_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_active_power_by_output.at['active_power', 'active_power'] = 1.0
        self.mapping_reactive_power_by_output = pd.DataFrame(0.0, index=['reactive_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_reactive_power_by_output.at['reactive_power', 'reactive_power'] = 1.0
        self.mapping_thermal_power_by_output = pd.DataFrame(0.0, index=['thermal_power'], columns=self.outputs)
        if self.is_thermal_grid_connected:
            self.mapping_thermal_power_by_output.at['thermal_power', 'thermal_power'] = 1.0


class StorageModel(FlexibleDERModel):
    """Energy storage model object."""

    der_type = 'storage'

    def __init__(
            self,
            der_data: mesmo.data_interface.DERData,
            der_name: str,
            **kwargs
    ):

        # Common initializations are implemented in parent class.
        super().__init__(der_data, der_name, **kwargs)

        # Get shorthand for DER data.
        der = der_data.ders.loc[self.der_name, :]

        # Obtain grid connection flags.
        # - Currently only implemented for electric grids.
        self.is_thermal_grid_connected = False

        # Instantiate indexes.
        self.states = pd.Index(['state_of_charge'])
        self.storage_states = pd.Index(['state_of_charge'])
        self.controls = pd.Index(['active_power_charge', 'active_power_discharge'])
        self.disturbances = pd.Index([])
        self.outputs = (
            pd.Index([
                'state_of_charge', 'active_power_charge', 'active_power_discharge',
                'active_power', 'reactive_power'
            ])
        )

        # Define power mapping matrices.
        self.mapping_active_power_by_output = pd.DataFrame(0.0, index=['active_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_active_power_by_output.at['active_power', 'active_power'] = 1.0
        self.mapping_reactive_power_by_output = pd.DataFrame(0.0, index=['reactive_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_reactive_power_by_output.at['reactive_power', 'reactive_power'] = 1.0
        self.mapping_thermal_power_by_output = pd.DataFrame(0.0, index=['thermal_power'], columns=self.outputs)

        # Instantiate initial state.
        # - Note that this is not used for `storage_states`, whose initial state is coupled with their final state.
        self.state_vector_initial = (
            pd.Series(0.0, index=self.states)
        )

        # Instantiate state space matrices.
        self.state_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.states)
        )
        self.state_matrix.at['state_of_charge', 'state_of_charge'] = (
            1.0
            - der['self_discharge_rate']
            * (der_data.scenario_data.scenario.at['timestep_interval'].seconds / 3600.0)
        )
        self.control_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.controls)
        )
        self.control_matrix.at['state_of_charge', 'active_power_charge'] = (
            der['charging_efficiency']
            * der_data.scenario_data.scenario.at['timestep_interval']
            / (der['active_power_nominal'] if der['active_power_nominal'] != 0.0 else 1.0)
            / (der['energy_storage_capacity_per_unit'] * pd.Timedelta('1h'))
        )
        self.control_matrix.at['state_of_charge', 'active_power_discharge'] = (
            -1.0
            * der_data.scenario_data.scenario.at['timestep_interval']
            / (der['active_power_nominal'] if der['active_power_nominal'] != 0.0 else 1.0)
            / (der['energy_storage_capacity_per_unit'] * pd.Timedelta('1h'))
        )
        self.disturbance_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.disturbances)
        )
        self.state_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.states)
        )
        self.state_output_matrix.at['state_of_charge', 'state_of_charge'] = 1.0
        self.control_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.controls)
        )
        self.control_output_matrix.at['active_power_charge', 'active_power_charge'] = 1.0
        self.control_output_matrix.at['active_power_discharge', 'active_power_discharge'] = 1.0
        self.control_output_matrix.at['active_power', 'active_power_charge'] = -1.0
        self.control_output_matrix.at['active_power', 'active_power_discharge'] = 1.0
        self.control_output_matrix.at['reactive_power', 'active_power_charge'] = (
            -1.0 * self.reactive_power_nominal / self.active_power_nominal
            if self.active_power_nominal != 0.0
            else 0.0
        )
        self.control_output_matrix.at['reactive_power', 'active_power_discharge'] = (
            self.reactive_power_nominal / self.active_power_nominal
            if self.active_power_nominal != 0.0
            else 0.0
        )
        self.disturbance_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.disturbances)
        )

        # Instantiate disturbance timeseries.
        self.disturbance_timeseries = (
            pd.DataFrame(0.0, index=self.active_power_nominal_timeseries.index, columns=self.disturbances)
        )

        # Construct output constraint timeseries
        self.output_maximum_timeseries = (
            pd.concat([
                pd.Series(1.0, index=self.active_power_nominal_timeseries.index, name='state_of_charge'),
                (
                    der['power_per_unit_maximum']
                    * der['active_power_nominal']
                    * pd.Series(1.0, index=self.active_power_nominal_timeseries.index, name='active_power_charge')
                ),
                (
                    der['power_per_unit_maximum']
                    * der['active_power_nominal']
                    * pd.Series(1.0, index=self.active_power_nominal_timeseries.index, name='active_power_discharge')
                ),
                pd.Series(np.inf, index=self.active_power_nominal_timeseries.index, name='active_power'),
                pd.Series(np.inf, index=self.active_power_nominal_timeseries.index, name='reactive_power')
            ], axis='columns')
        )
        self.output_minimum_timeseries = (
            pd.concat([
                pd.Series(0.0, index=self.active_power_nominal_timeseries.index, name='state_of_charge'),
                pd.Series(0.0, index=self.active_power_nominal_timeseries.index, name='active_power_charge'),
                pd.Series(0.0, index=self.active_power_nominal_timeseries.index, name='active_power_discharge'),
                pd.Series(-np.inf, index=self.active_power_nominal_timeseries.index, name='active_power'),
                pd.Series(-np.inf, index=self.active_power_nominal_timeseries.index, name='reactive_power')
            ], axis='columns')
        )


class FlexibleBuildingModel(FlexibleDERModel):
    """Flexible load model object."""

    der_type = 'flexible_building'

    power_factor_nominal: float
    is_electric_grid_connected: bool
    is_thermal_grid_connected: bool

    def __init__(
            self,
            der_data: mesmo.data_interface.DERData,
            der_name: str,
            **kwargs
    ):
        """Construct flexible building model object by `der_data` and `der_name`."""

        # Common initializations are implemented in parent class.
        super().__init__(der_data, der_name, **kwargs)

        # Get shorthand for DER data.
        der = der_data.ders.loc[self.der_name, :]

        # Obtain CoBMo building model.
        flexible_building_model = (
            mesmo.utils.get_building_model(
                der.at['der_model_name'],
                timestep_start=der_data.scenario_data.scenario.at['timestep_start'],
                timestep_end=der_data.scenario_data.scenario.at['timestep_end'],
                timestep_interval=der_data.scenario_data.scenario.at['timestep_interval'],
                connect_electric_grid=self.is_electric_grid_connected,
                connect_thermal_grid_cooling=self.is_thermal_grid_connected
            )
        )

        # Obtain nominal power factor.
        if self.is_electric_grid_connected:
            power_factor_nominal = (
                np.cos(np.arctan(
                    self.reactive_power_nominal
                    / self.active_power_nominal
                ))
                if ((self.active_power_nominal != 0.0) and (self.reactive_power_nominal != 0.0))
                else 1.0
            )

        # TODO: Obtain proper nominal power timseries for CoBMo models.

        # Obtain indexes.
        self.states = flexible_building_model.states
        self.controls = flexible_building_model.controls
        self.disturbances = flexible_building_model.disturbances
        self.outputs = flexible_building_model.outputs

        # Define power mapping matrices.
        self.mapping_active_power_by_output = pd.DataFrame(0.0, index=['active_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_active_power_by_output.at['active_power', 'grid_electric_power'] = (
                -1.0
                * flexible_building_model.zone_area_total
                / der_data.scenario_data.scenario.at['base_apparent_power']
            )
        self.mapping_reactive_power_by_output = pd.DataFrame(0.0, index=['reactive_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_reactive_power_by_output.at['reactive_power', 'grid_electric_power'] = (
                -1.0
                * np.tan(np.arccos(power_factor_nominal))
                * flexible_building_model.zone_area_total
                / der_data.scenario_data.scenario.at['base_apparent_power']
            )
        self.mapping_thermal_power_by_output = pd.DataFrame(0.0, index=['thermal_power'], columns=self.outputs)
        if self.is_thermal_grid_connected:
            self.mapping_thermal_power_by_output.at['thermal_power', 'grid_thermal_power_cooling'] = (
                -1.0
                * flexible_building_model.zone_area_total
                / der_data.scenario_data.scenario.at['base_thermal_power']
            )

        # Obtain initial state.
        self.state_vector_initial = flexible_building_model.state_vector_initial

        # Obtain state space matrices.
        self.state_matrix = flexible_building_model.state_matrix
        self.control_matrix = flexible_building_model.control_matrix
        self.disturbance_matrix = flexible_building_model.disturbance_matrix
        self.state_output_matrix = flexible_building_model.state_output_matrix
        self.control_output_matrix = flexible_building_model.control_output_matrix
        self.disturbance_output_matrix = flexible_building_model.disturbance_output_matrix

        # Instantiate disturbance timeseries.
        self.disturbance_timeseries = flexible_building_model.disturbance_timeseries

        # Obtain output constraint timeseries.
        self.output_minimum_timeseries = flexible_building_model.output_minimum_timeseries
        self.output_maximum_timeseries = flexible_building_model.output_maximum_timeseries


class CoolingPlantModel(FlexibleDERModel):
    """Cooling plant model object."""

    der_type = 'cooling_plant'
    cooling_plant_efficiency: float

    def __init__(
            self,
            der_data: mesmo.data_interface.DERData,
            der_name: str,
            **kwargs
    ):
        """Construct flexible load model object by `der_data` and `der_name`."""

        # Common initializations are implemented in parent class.
        super().__init__(der_data, der_name, **kwargs)

        # Get shorthand for DER data.
        der = der_data.ders.loc[self.der_name, :].copy()
        der = pd.concat([der, der_data.der_definitions[der.at['definition_index']]])

        # Cooling plant must be connected to both thermal grid and electric grid.
        if not (self.is_standalone or (self.is_electric_grid_connected and self.is_thermal_grid_connected)):
            raise ValueError(f"Cooling plant '{self.der_name}' must be connected to both thermal grid and electric grid")

        # Obtain cooling plant efficiency.
        # TODO: Enable consideration for dynamic wet bulb temperature.
        ambient_air_wet_bulb_temperature = (
            der.at['cooling_tower_set_reference_temperature_wet_bulb']
        )
        condensation_temperature = (
            der.at['cooling_tower_set_reference_temperature_condenser_water']
            + (
                der.at['cooling_tower_set_reference_temperature_slope']
                * (
                    ambient_air_wet_bulb_temperature
                    - der.at['cooling_tower_set_reference_temperature_wet_bulb']
                )
            )
            + der.at['condenser_water_temperature_difference']
            + der.at['chiller_set_condenser_minimum_temperature_difference']
            + 273.15
        )
        chiller_inverse_coefficient_of_performance = (
            (
                (
                    condensation_temperature
                    / der.at['chiller_set_evaporation_temperature']
                )
                - 1.0
            )
            * (
                der.at['chiller_set_beta']
                + 1.0
            )
        )
        evaporator_pump_specific_electric_power = (
            (1.0 / der.at['plant_pump_efficiency'])
            * scipy.constants.value('standard acceleration of gravity')
            * der.at['water_density']
            * der.at['evaporator_pump_head']
            / (
                der.at['water_density']
                * der.at['enthalpy_difference_distribution_water']
            )
        )
        condenser_specific_thermal_power = (
            1.0 + chiller_inverse_coefficient_of_performance
        )
        condenser_pump_specific_electric_power = (
            (1.0 / der.at['plant_pump_efficiency'])
            * scipy.constants.value('standard acceleration of gravity')
            * der.at['water_density']
            * der.at['condenser_pump_head']
            * condenser_specific_thermal_power
            / (
                der.at['water_density']
                * der.at['condenser_water_enthalpy_difference']
            )
        )
        cooling_tower_ventilation_specific_electric_power = (
            der.at['cooling_tower_set_ventilation_factor']
            * condenser_specific_thermal_power
        )
        self.cooling_plant_efficiency = (
            1.0 / sum([
                chiller_inverse_coefficient_of_performance,
                evaporator_pump_specific_electric_power,
                condenser_pump_specific_electric_power,
                cooling_tower_ventilation_specific_electric_power
            ])
        )

        # Store timesteps index.
        self.timesteps = der_data.scenario_data.timesteps

        # Instantiate indexes.
        self.states = pd.Index(['_'])  # Define placeholder '_' to avoid issues in the optimization problem definition.
        self.controls = pd.Index(['active_power'])
        self.disturbances = pd.Index([])
        self.outputs = pd.Index(['active_power', 'reactive_power', 'thermal_power'])

        # Define power mapping matrices.
        self.mapping_active_power_by_output = pd.DataFrame(0.0, index=['active_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_active_power_by_output.at['active_power', 'active_power'] = 1.0
        self.mapping_reactive_power_by_output = pd.DataFrame(0.0, index=['reactive_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_reactive_power_by_output.at['reactive_power', 'reactive_power'] = 1.0
        self.mapping_thermal_power_by_output = pd.DataFrame(0.0, index=['thermal_power'], columns=self.outputs)
        if self.is_thermal_grid_connected:
            self.mapping_thermal_power_by_output.at['thermal_power', 'thermal_power'] = 1.0

        # Instantiate initial state.
        self.state_vector_initial = (
            pd.Series(0.0, index=self.states)
        )

        # Instantiate state space matrices.
        self.state_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.states)
        )
        self.control_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.controls)
        )
        self.disturbance_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.disturbances)
        )
        self.state_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.states)
        )
        self.control_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.controls)
        )
        self.control_output_matrix.at['active_power', 'active_power'] = 1.0
        self.control_output_matrix.at['reactive_power', 'active_power'] = (
            self.reactive_power_nominal / self.active_power_nominal
            if self.active_power_nominal != 0.0
            else 0.0
        )
        self.control_output_matrix.at['thermal_power', 'active_power'] = (
            -1.0
            * self.cooling_plant_efficiency
            * der_data.scenario_data.scenario.at['base_apparent_power']
            / der_data.scenario_data.scenario.at['base_thermal_power']
        )
        self.disturbance_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.disturbances)
        )

        # Instantiate disturbance timeseries.
        self.disturbance_timeseries = (
            pd.DataFrame(0.0, index=self.timesteps, columns=self.disturbances)
        )

        # Construct output constraint timeseries
        self.output_maximum_timeseries = (
            pd.DataFrame(
                [[0.0, 0.0, self.thermal_power_nominal]],
                index=self.timesteps,
                columns=self.outputs
            )
        )
        self.output_minimum_timeseries = (
            pd.DataFrame(
                [[self.active_power_nominal, self.reactive_power_nominal, 0.0]],
                index=self.timesteps,
                columns=self.outputs
            )
        )


class HeatPumpModel(FlexibleDERModel):
    """Heat pump model object."""

    der_type = 'heat_pump'
    heat_pump_efficiency: float

    def __init__(
            self,
            der_data: mesmo.data_interface.DERData,
            der_name: str,
            **kwargs
    ):

        # Common initializations are implemented in parent class.
        super().__init__(der_data, der_name, **kwargs)

        # Get shorthand for DER data.
        der = der_data.ders.loc[self.der_name, :]

        # If not connected to both thermal grid and electric grid, raise error.
        if not (self.is_standalone or (self.is_electric_grid_connected and self.is_thermal_grid_connected)):
            raise AssertionError(
                f"Heat pump '{self.der_name}' must be connected to both thermal grid and electric grid."
            )

        # Obtain heat pump efficiency.
        self.heat_pump_efficiency = der.at['heat_pump_efficiency']

        # Store timesteps index.
        self.timesteps = der_data.scenario_data.timesteps

        # Manipulate thermal power timeseries.
        self.thermal_power_nominal_timeseries *= self.heat_pump_efficiency

        # Instantiate indexes.
        self.states = pd.Index(['_'])  # Define placeholder '_' to avoid issues in the optimization problem definition.
        self.controls = pd.Index(['active_power'])
        self.disturbances = pd.Index([])
        self.outputs = pd.Index(['active_power', 'reactive_power', 'thermal_power'])

        # Define power mapping matrices.
        self.mapping_active_power_by_output = pd.DataFrame(0.0, index=['active_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_active_power_by_output.at['active_power', 'active_power'] = 1.0
        self.mapping_reactive_power_by_output = pd.DataFrame(0.0, index=['reactive_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_reactive_power_by_output.at['reactive_power', 'reactive_power'] = 1.0
        self.mapping_thermal_power_by_output = pd.DataFrame(0.0, index=['thermal_power'], columns=self.outputs)
        if self.is_thermal_grid_connected:
            self.mapping_thermal_power_by_output.at['thermal_power', 'thermal_power'] = 1.0

        # Instantiate initial state.
        self.state_vector_initial = (
            pd.Series(0.0, index=self.states)
        )

        # Instantiate state space matrices.
        self.state_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.states)
        )
        self.control_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.controls)
        )
        self.disturbance_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.disturbances)
        )
        self.state_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.states)
        )
        self.control_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.controls)
        )
        self.control_output_matrix.at['active_power', 'active_power'] = 1.0
        self.control_output_matrix.at['reactive_power', 'active_power'] = (
            self.reactive_power_nominal / self.active_power_nominal
            if self.active_power_nominal != 0.0
            else 0.0
        )
        self.control_output_matrix.at['thermal_power', 'active_power'] = -1.0 * self.heat_pump_efficiency
        self.disturbance_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.disturbances)
        )

        # Instantiate disturbance timeseries.
        self.disturbance_timeseries = (
            pd.DataFrame(0.0, index=self.timesteps, columns=self.disturbances)
        )

        # Construct output constraint timeseries.
        # TODO: Confirm the maximum / minimum definitions.
        self.output_maximum_timeseries = (
            pd.concat([
                (
                    der['power_per_unit_minimum']  # Take minimum, because load is negative power.
                    * self.active_power_nominal_timeseries
                ),
                (
                    der['power_per_unit_minimum']  # Take minimum, because load is negative power.
                    * self.reactive_power_nominal_timeseries
                ),
                (
                    self.thermal_power_nominal
                    * self.thermal_power_nominal_timeseries
                )
            ], axis ='columns')
        )
        self.output_minimum_timeseries = (
            pd.concat([
                (
                    der['power_per_unit_maximum']  # Take maximum, because load is negative power.
                    * self.active_power_nominal_timeseries
                ),
                (
                    der['power_per_unit_maximum']  # Take maximum, because load is negative power.
                    * self.reactive_power_nominal_timeseries
                ),
                (
                    0.0
                    * self.thermal_power_nominal_timeseries
                )
            ], axis='columns')
        )


class FlexibleCHP(FlexibleDERModel):

    der_type = 'flexible_chp'
    marginal_cost: float
    thermal_efficiency: float
    electric_efficiency: float

    def __init__(
            self,
            der_data: mesmo.data_interface.DERData,
            der_name: str,
            **kwargs
    ):

        # Common initializations are implemented in parent class.
        super().__init__(der_data, der_name, **kwargs)

        # Get shorthand for DER data.
        der = der_data.ders.loc[self.der_name, :]

        # If not connected to both thermal grid and electric grid, raise error.
        if not (self.is_standalone or (self.is_electric_grid_connected and self.is_thermal_grid_connected)):
            raise AssertionError(
                f"CHP '{self.der_name}' must be connected to both thermal grid and electric grid."
            )

        # Store timesteps index.
        self.timesteps = der_data.scenario_data.timesteps

        # Obtain thermal and electrical efficiency
        self.thermal_efficiency = der.at['thermal_efficiency']
        self.electric_efficiency = der.at['electric_efficiency']

        # Manipulate nominal active and reactive power timeseries.
        self.active_power_nominal_timeseries *= (self.electric_efficiency / self.thermal_efficiency)
        self.reactive_power_nominal_timeseries *= (self.electric_efficiency / self.thermal_efficiency)

        # Instantiate indexes.
        self.states = pd.Index(['_'])  # Define placeholder '_' to avoid issues in the optimization problem definition.
        self.controls = pd.Index(['active_power'])
        self.disturbances = pd.Index([])
        self.outputs = pd.Index(['active_power', 'reactive_power', 'thermal_power'])

        # Define power mapping matrices.
        self.mapping_active_power_by_output = pd.DataFrame(0.0, index=['active_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_active_power_by_output.at['active_power', 'active_power'] = 1.0
        self.mapping_reactive_power_by_output = pd.DataFrame(0.0, index=['reactive_power'], columns=self.outputs)
        if self.is_electric_grid_connected:
            self.mapping_reactive_power_by_output.at['reactive_power', 'reactive_power'] = 1.0
        self.mapping_thermal_power_by_output = pd.DataFrame(0.0, index=['thermal_power'], columns=self.outputs)
        if self.is_thermal_grid_connected:
            self.mapping_thermal_power_by_output.at['thermal_power', 'thermal_power'] = 1.0

        # Instantiate initial state.
        self.state_vector_initial = (
            pd.Series(0.0, index=self.states)
        )

        # Instantiate state space matrices.
        self.state_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.states)
        )
        self.control_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.controls)
        )
        self.disturbance_matrix = (
            pd.DataFrame(0.0, index=self.states, columns=self.disturbances)
        )
        self.state_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.states)
        )
        self.control_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.controls)
        )
        self.control_output_matrix.at['active_power', 'active_power'] = 1.0
        self.control_output_matrix.at['reactive_power', 'active_power'] = (
            self.reactive_power_nominal / self.active_power_nominal
            if self.active_power_nominal != 0.0
            else 0.0
        )
        self.control_output_matrix.at['thermal_power', 'active_power'] = (
            1.0 * (self.thermal_efficiency / self.electric_efficiency)
        )
        self.disturbance_output_matrix = (
            pd.DataFrame(0.0, index=self.outputs, columns=self.disturbances)
        )

        # Instantiate disturbance timeseries.
        self.disturbance_timeseries = (
            pd.DataFrame(0.0, index=self.timesteps, columns=self.disturbances)
        )

        # Construct output constraint timeseries.
        # TODO: Confirm the maximum / minimum definitions.
        self.output_maximum_timeseries = (
            pd.concat([
                self.active_power_nominal_timeseries,
                self.reactive_power_nominal_timeseries,
                self.thermal_power_nominal_timeseries
            ], axis='columns')
        )
        self.output_minimum_timeseries = (
            pd.concat([
                0.0 * self.active_power_nominal_timeseries,
                0.0 * self.reactive_power_nominal_timeseries,
                0.0 * self.thermal_power_nominal_timeseries
            ], axis='columns')
        )


class DERModelSetBase:

    timesteps: pd.Index
    ders: pd.Index
    electric_ders: pd.Index
    thermal_ders: pd.Index
    der_names: pd.Index
    fixed_der_names: pd.Index
    flexible_der_names: pd.Index
    der_models: typing.Dict[str, DERModel]
    fixed_der_models: typing.Dict[str, FixedDERModel]
    flexible_der_models: typing.Dict[str, FlexibleDERModel]
    states: pd.Index
    controls: pd.Index
    outputs: pd.Index
    storage_states: pd.Index
    der_active_power_vector_reference: np.array
    der_reactive_power_vector_reference: np.array
    der_thermal_power_vector_reference: np.array


class DERModelSetOperationResults(mesmo.electric_grid_models.ElectricGridDEROperationResults):

    der_model_set: DERModelSetBase
    state_vector: pd.DataFrame
    control_vector: pd.DataFrame
    output_vector: pd.DataFrame
    # TODO: Add output constraint and disturbance timeseries.
    der_thermal_power_vector: pd.DataFrame
    der_thermal_power_vector_per_unit: pd.DataFrame


class DERModelSet(DERModelSetBase):
    """DER model set object."""

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            **kwargs
    ):

        # Obtain data.
        der_data = mesmo.data_interface.DERData(scenario_name)

        self.__init__(
            der_data,
            **kwargs
        )

    @multimethod
    def __init__(
            self,
            der_data: mesmo.data_interface.DERData,
            der_name: str = None
    ):

        # Filter DER data, if passing `der_name` to select specific DER.
        if der_name is not None:
            if der_name not in der_data.ders.index:
                raise ValueError(f"DER '{der_name}' not found in DER data.")
            else:
                ders = der_data.ders.loc[[der_name], :]
        else:
            ders = der_data.ders

        # Obtain timesteps.
        self.timesteps = der_data.scenario_data.timesteps

        # Obtain DER index sets.
        # - Note: Implementation changes to `ders`, `electric_ders` and `thermal_ders` index sets must be aligned
        #   with `ElectricGridModel.ders` and `ThermalGridModel.ders`.
        self.ders = pd.MultiIndex.from_frame(ders.loc[:, ['der_type', 'der_name']])
        self.electric_ders = self.ders[pd.notnull(ders.loc[:, 'electric_grid_name'])]
        self.thermal_ders = self.ders[pd.notnull(ders.loc[:, 'thermal_grid_name'])]
        self.der_names = ders.index

        # Obtain DER models.
        mesmo.utils.log_time("DER model setup")
        der_models = (
            mesmo.utils.starmap(
                make_der_models,
                zip(
                    mesmo.utils.chunk_list(self.der_names.to_list())
                ),
                dict(
                    der_data=der_data
                )
            )
        )
        self.der_models = dict()
        for chunk in der_models:
            self.der_models.update(chunk)
        mesmo.utils.log_time("DER model setup")

        # Obtain fixed / flexible DER name / models.
        self.fixed_der_names = list()
        self.flexible_der_names = list()
        self.fixed_der_models = dict()
        self.flexible_der_models = dict()
        for der_name in self.der_names:
            if isinstance(self.der_models[der_name], FixedDERModel):
                self.fixed_der_names.append(der_name)
                self.fixed_der_models[der_name] = self.der_models[der_name]
            elif isinstance(self.der_models[der_name], FlexibleDERModel):
                self.flexible_der_names.append(der_name)
                self.flexible_der_models[der_name] = self.der_models[der_name]
            else:
                # Raise error, if DER model object is neither fixed nor flexible DER model.
                raise TypeError(
                    f"DER model class `{type(self.der_models[der_name])}` for DER '{der_name}' "
                    f"is not a subclass of `FixedDERModel` or `FlexibleDERModel`."
                )
        self.fixed_der_names = pd.Index(self.fixed_der_names)
        self.flexible_der_names = pd.Index(self.flexible_der_names)

        # Update model data, i.e. parameters which are aggregated from individual DER models.
        self.update_data()

    def update_data(self):

        # Obtain flexible DER state space indexes.
        self.states = (
            pd.MultiIndex.from_tuples([
                (der_name, state)
                for der_name in self.flexible_der_names
                for state in self.flexible_der_models[der_name].states
            ])
            if len(self.flexible_der_names) > 0 else pd.Index([])
        )
        self.controls = (
            pd.MultiIndex.from_tuples([
                (der_name, control)
                for der_name in self.flexible_der_names
                for control in self.flexible_der_models[der_name].controls
            ])
            if len(self.flexible_der_names) > 0 else pd.Index([])
        )
        self.outputs = (
            pd.MultiIndex.from_tuples([
                (der_name, output)
                for der_name in self.flexible_der_names
                for output in self.flexible_der_models[der_name].outputs
            ])
            if len(self.flexible_der_names) > 0 else pd.Index([])
        )
        self.storage_states = (
            pd.MultiIndex.from_tuples([
                (der_name, state)
                for der_name in self.flexible_der_names
                for state in self.flexible_der_models[der_name].storage_states
            ], names=['der_name', 'state'])
            if len(self.flexible_der_names) > 0 else pd.Index([])
        )

        # Obtain nominal power vectors.
        if len(self.electric_ders) > 0:
            self.der_active_power_vector_reference = (
                np.array([
                    self.der_models[der_name].active_power_nominal
                    for der_type, der_name in self.electric_ders
                ])
            )
            self.der_reactive_power_vector_reference = (
                np.array([
                    self.der_models[der_name].reactive_power_nominal
                    for der_type, der_name in self.electric_ders
                ])
            )
        if len(self.thermal_ders) > 0:
            self.der_thermal_power_vector_reference = (
                np.array([
                    self.der_models[der_name].thermal_power_nominal
                    for der_type, der_name in self.thermal_ders
                ])
            )

    def define_optimization_problem(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            price_data: mesmo.data_interface.PriceData,
            scenarios: typing.Union[list, pd.Index] = None
    ):

        # Define optimization problem definitions through respective sub-methods.
        self.define_optimization_variables(optimization_problem, scenarios=scenarios)
        self.define_optimization_parameters(optimization_problem, price_data, scenarios=scenarios)
        self.define_optimization_constraints(optimization_problem, scenarios=scenarios)
        self.define_optimization_objective(optimization_problem, scenarios=scenarios)

    def define_optimization_variables(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            scenarios: typing.Union[list, pd.Index] = None
    ):

        # If no scenarios given, obtain default value.
        if scenarios is None:
            scenarios = [None]

        # Define state space variables.
        optimization_problem.define_variable(
            'state_vector', scenario=scenarios, timestep=self.timesteps, state=self.states
        )
        optimization_problem.define_variable(
            'control_vector', scenario=scenarios, timestep=self.timesteps, control=self.controls
        )
        optimization_problem.define_variable(
            'output_vector', scenario=scenarios, timestep=self.timesteps, output=self.outputs
        )

        # Define DER power vector variables.
        # - Only if these have not yet been defined within `LinearElectricGridModel` or `LinearThermalGridModel`.
        if (
                ('der_active_power_vector' not in optimization_problem.variables.loc[:, 'name'].values)
                and (len(self.electric_ders) > 0)
        ):
            optimization_problem.define_variable(
                'der_active_power_vector', scenario=scenarios, timestep=self.timesteps, der=self.electric_ders
            )
        if (
                ('der_reactive_power_vector' not in optimization_problem.variables.loc[:, 'name'].values)
                and (len(self.electric_ders) > 0)
        ):
            optimization_problem.define_variable(
                'der_reactive_power_vector', scenario=scenarios, timestep=self.timesteps, der=self.electric_ders
            )
        if (
                ('der_thermal_power_vector' not in optimization_problem.variables.loc[:, 'name'].values)
                and (len(self.thermal_ders) > 0)
        ):
            optimization_problem.define_variable(
                'der_thermal_power_vector', scenario=scenarios, timestep=self.timesteps, der=self.thermal_ders
            )

    def define_optimization_parameters(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            price_data: mesmo.data_interface.PriceData,
            scenarios: typing.Union[list, pd.Index] = None
    ):

        # If no scenarios given, obtain default value.
        if scenarios is None:
            scenarios = [None]

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (self.timesteps[1] - self.timesteps[0]) / pd.Timedelta('1h')

        # Define parameters.
        optimization_problem.define_parameter(
            'state_vector_initial',
            np.concatenate([
                self.flexible_der_models[der_name].state_vector_initial.values
                for der_name in self.flexible_der_names
            ])[~self.states.isin(self.storage_states)]
        )
        optimization_problem.define_parameter(
            'state_matrix',
            sp.block_diag([
                self.flexible_der_models[der_name].state_matrix.values
                for der_name in self.flexible_der_names
            ])
        )
        optimization_problem.define_parameter(
            'control_matrix',
            sp.block_diag([
                self.flexible_der_models[der_name].control_matrix.values
                for der_name in self.flexible_der_names
            ])
        )
        optimization_problem.define_parameter(
            'disturbance_state_equation',
            (
                sp.block_diag([
                    self.flexible_der_models[der_name].disturbance_matrix.values
                    for der_name in self.flexible_der_names
                ])
                @ pd.concat([
                    self.flexible_der_models[der_name].disturbance_timeseries
                    for der_name in self.flexible_der_names
                ], axis='columns').iloc[:-1, :].T.values
            ).T.ravel()
        )
        optimization_problem.define_parameter(
            'state_output_matrix',
            sp.block_diag([
                self.flexible_der_models[der_name].state_output_matrix.values
                for der_name in self.flexible_der_names
            ])
        )
        optimization_problem.define_parameter(
            'control_output_matrix',
            sp.block_diag([
                self.flexible_der_models[der_name].control_output_matrix.values
                for der_name in self.flexible_der_names
            ])
        )
        optimization_problem.define_parameter(
            'disturbance_output_equation',
            (
                sp.block_diag([
                    self.flexible_der_models[der_name].disturbance_output_matrix.values
                    for der_name in self.flexible_der_names
                ])
                @ pd.concat([
                    self.flexible_der_models[der_name].disturbance_timeseries
                    for der_name in self.flexible_der_names
                ], axis='columns').T.values
            ).T.ravel()
        )
        if len(self.electric_ders) > 0:
            optimization_problem.define_parameter(
                'active_power_constant',
                np.concatenate([
                    np.transpose([
                        self.fixed_der_models[der_name].active_power_nominal_timeseries.values
                        / (
                            self.fixed_der_models[der_name].active_power_nominal
                            if self.fixed_der_models[der_name].active_power_nominal != 0.0
                            else 1.0
                        )
                        if self.fixed_der_models[der_name].is_electric_grid_connected
                        else 0.0 * self.fixed_der_models[der_name].active_power_nominal_timeseries.values
                    ])
                    if der_name in self.fixed_der_names
                    else np.zeros((len(self.timesteps), 1))
                    for der_type, der_name in self.electric_ders
                ], axis=1).ravel()
            )
            optimization_problem.define_parameter(
                'mapping_active_power_by_output',
                sp.block_diag([
                    (
                        self.flexible_der_models[der_name].mapping_active_power_by_output.values
                        / (
                            self.flexible_der_models[der_name].active_power_nominal
                            if self.flexible_der_models[der_name].active_power_nominal != 0.0
                            else 1.0
                        )
                        if self.flexible_der_models[der_name].is_electric_grid_connected
                        else np.zeros((0, len(self.flexible_der_models[der_name].outputs)))
                    )
                    if der_name in self.flexible_der_names
                    else (
                        np.zeros((1, 0))
                        if self.der_models[der_name].is_electric_grid_connected
                        else np.zeros((0, 0))
                    )
                    for der_type, der_name in self.ders
                ])
            )
            optimization_problem.define_parameter(
                'reactive_power_constant',
                np.concatenate([
                    np.transpose([
                        self.fixed_der_models[der_name].reactive_power_nominal_timeseries.values
                        / (
                            self.fixed_der_models[der_name].reactive_power_nominal
                            if self.fixed_der_models[der_name].reactive_power_nominal != 0.0
                            else 1.0
                        )
                        if self.fixed_der_models[der_name].is_electric_grid_connected
                        else 0.0 * self.fixed_der_models[der_name].reactive_power_nominal_timeseries.values
                    ])
                    if der_name in self.fixed_der_names
                    else np.zeros((len(self.timesteps), 1))
                    for der_type, der_name in self.electric_ders
                ], axis=1).ravel()
            )
            optimization_problem.define_parameter(
                'mapping_reactive_power_by_output',
                sp.block_diag([
                    (
                        self.flexible_der_models[der_name].mapping_reactive_power_by_output.values
                        / (
                            self.flexible_der_models[der_name].reactive_power_nominal
                            if self.flexible_der_models[der_name].reactive_power_nominal != 0.0
                            else 1.0
                        )
                        if self.flexible_der_models[der_name].is_electric_grid_connected
                        else np.zeros((0, len(self.flexible_der_models[der_name].outputs)))
                    )
                    if der_name in self.flexible_der_names
                    else (
                        np.zeros((1, 0))
                        if self.der_models[der_name].is_electric_grid_connected
                        else np.zeros((0, 0))
                    )
                    for der_type, der_name in self.ders
                ])
            )
        if len(self.thermal_ders) > 0:
            optimization_problem.define_parameter(
                'thermal_power_constant',
                np.concatenate([
                    np.transpose([
                        self.fixed_der_models[der_name].thermal_power_nominal_timeseries.values
                        / (
                            self.fixed_der_models[der_name].thermal_power_nominal
                            if self.fixed_der_models[der_name].thermal_power_nominal != 0.0
                            else 1.0
                        )
                        if self.fixed_der_models[der_name].is_thermal_grid_connected
                        else 0.0 * self.fixed_der_models[der_name].thermal_power_nominal_timeseries.values
                    ])
                    if der_name in self.fixed_der_names
                    else np.zeros((len(self.timesteps), 1))
                    for der_type, der_name in self.thermal_ders
                ], axis=1).ravel()
            )
            optimization_problem.define_parameter(
                'mapping_thermal_power_by_output',
                sp.block_diag([
                    (
                        self.flexible_der_models[der_name].mapping_thermal_power_by_output.values
                        / (
                            self.flexible_der_models[der_name].thermal_power_nominal
                            if self.flexible_der_models[der_name].thermal_power_nominal != 0.0
                            else 1.0
                        )
                        if self.flexible_der_models[der_name].is_thermal_grid_connected
                        else np.zeros((0, len(self.flexible_der_models[der_name].outputs)))
                    )
                    if der_name in self.flexible_der_names
                    else (
                        np.zeros((1, 0))
                        if self.der_models[der_name].is_thermal_grid_connected
                        else np.zeros((0, 0))
                    )
                    for der_type, der_name in self.ders
                ])
            )
        optimization_problem.define_parameter(
            'output_minimum_timeseries',
            pd.concat([
                self.flexible_der_models[der_name].output_minimum_timeseries
                for der_name in self.flexible_der_names
            ], axis='columns').values.ravel()
        )
        optimization_problem.define_parameter(
            'output_maximum_timeseries',
            pd.concat([
                self.flexible_der_models[der_name].output_maximum_timeseries
                for der_name in self.flexible_der_names
            ], axis='columns').values.ravel()
        )

        # Define objective parameters.
        if len(self.electric_ders) > 0:
            optimization_problem.define_parameter(
                'der_active_power_cost',
                np.array([(
                    (
                        price_data.price_timeseries.iloc[:, mesmo.utils.get_index(
                            price_data.price_timeseries.columns,
                            commodity_type='active_power',
                            der_name=self.electric_ders.get_level_values('der_name')
                        )].values
                    )
                    * -1.0 * timestep_interval_hours  # In Wh.
                    @ sp.block_diag(self.der_active_power_vector_reference)
                ).ravel()])
            )
            optimization_problem.define_parameter(
                'der_active_power_cost_sensitivity',
                price_data.price_sensitivity_coefficient
                * timestep_interval_hours  # In Wh.
                * np.concatenate(
                    [np.array([self.der_active_power_vector_reference ** 2])] * len(self.timesteps),
                    axis=1
                )
            )
            optimization_problem.define_parameter(
                'der_reactive_power_cost',
                np.array([(
                    (
                        price_data.price_timeseries.iloc[:, mesmo.utils.get_index(
                            price_data.price_timeseries.columns,
                            commodity_type='reactive_power',
                            der_name=self.electric_ders.get_level_values('der_name')
                        )].values
                    )
                    * -1.0 * timestep_interval_hours  # In Wh.
                    @ sp.block_diag(self.der_reactive_power_vector_reference)
                ).ravel()])
            )
            optimization_problem.define_parameter(
                'der_reactive_power_cost_sensitivity',
                price_data.price_sensitivity_coefficient
                * timestep_interval_hours  # In Wh.
                * np.concatenate(
                    [np.array([self.der_reactive_power_vector_reference ** 2])] * len(self.timesteps),
                    axis=1
                )
            )
        if len(self.thermal_ders) > 0:
            optimization_problem.define_parameter(
                'der_thermal_power_cost',
                np.array([(
                    (
                        price_data.price_timeseries.iloc[:, mesmo.utils.get_index(
                            price_data.price_timeseries.columns,
                            commodity_type='thermal_power',
                            der_name=self.thermal_ders.get_level_values('der_name')
                        )].values
                    )
                    * -1.0 * timestep_interval_hours  # In Wh.
                    @ sp.block_diag(self.der_thermal_power_vector_reference)
                ).ravel()])
            )
            optimization_problem.define_parameter(
                'der_thermal_power_cost_sensitivity',
                price_data.price_sensitivity_coefficient
                * timestep_interval_hours  # In Wh.
                * np.concatenate(
                    [np.array([self.der_thermal_power_vector_reference ** 2])] * len(self.timesteps),
                    axis=1
                )
            )
        # TODO: Revise marginal cost implementation to split active / reactive / thermal power cost.
        # TODO: Related: Cost for CHP defined twice.
        if len(self.electric_ders) > 0:
            optimization_problem.define_parameter(
                'der_active_power_marginal_cost',
                np.concatenate([[[
                    self.der_models[der_name].marginal_cost
                    * timestep_interval_hours  # In Wh.
                    * self.der_models[der_name].active_power_nominal
                    for der_type, der_name in self.electric_ders
                ] * len(self.timesteps)]], axis=1)
            )
            optimization_problem.define_parameter(
                'der_reactive_power_marginal_cost',
                np.concatenate([[[
                    0.0
                    # self.der_models[der_name].marginal_cost
                    # * timestep_interval_hours  # In Wh.
                    # * self.der_models[der_name].reactive_power_nominal
                    for der_type, der_name in self.electric_ders
                ] * len(self.timesteps)]], axis=1)
            )
        if len(self.thermal_ders) > 0:
            optimization_problem.define_parameter(
                'der_thermal_power_marginal_cost',
                np.concatenate([[[
                    self.der_models[der_name].marginal_cost
                    * timestep_interval_hours  # In Wh.
                    * self.der_models[der_name].thermal_power_nominal
                    for der_type, der_name in self.thermal_ders
                ] * len(self.timesteps)]], axis=1)
            )

    def define_optimization_constraints(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            scenarios: typing.Union[list, pd.Index] = None
    ):

        # If no scenarios given, obtain default value.
        if scenarios is None:
            scenarios = [None]

        # Define DER model constraints.
        # Initial state.
        # - For states which represent storage state of charge, initial state of charge is final state of charge.
        if any(self.states.isin(self.storage_states)):
            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='state_vector', scenario=scenarios, timestep=self.timesteps[0],
                    state=self.states[self.states.isin(self.storage_states)]
                )),
                '==',
                ('variable', 1.0, dict(
                    name='state_vector', scenario=scenarios, timestep=self.timesteps[-1],
                    state=self.states[self.states.isin(self.storage_states)]
                )),
                broadcast='scenario'
            )
        # - For other states, set initial state according to the initial state vector.
        if any(~self.states.isin(self.storage_states)):
            optimization_problem.define_constraint(
                ('constant', 'state_vector_initial', dict(scenario=scenarios)),
                '==',
                ('variable', 1.0, dict(
                    name='state_vector', scenario=scenarios, timestep=self.timesteps[0],
                    state=self.states[~self.states.isin(self.storage_states)]
                )),
                broadcast='scenario'
            )

        # State equation.
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(name='state_vector', scenario=scenarios, timestep=self.timesteps[1:])),
            '==',
            ('variable', 'state_matrix', dict(
                name='state_vector', scenario=scenarios, timestep=self.timesteps[:-1]
            )),
            ('variable', 'control_matrix', dict(
                name='control_vector', scenario=scenarios, timestep=self.timesteps[:-1]
            )),
            ('constant', 'disturbance_state_equation', dict(scenario=scenarios)),
            broadcast=['timestep', 'scenario']
        )

        # Output equation.
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(name='output_vector', scenario=scenarios, timestep=self.timesteps)),
            '==',
            ('variable', 'state_output_matrix', dict(
                name='state_vector', scenario=scenarios, timestep=self.timesteps
            )),
            ('variable', 'control_output_matrix', dict(
                name='control_vector', scenario=scenarios, timestep=self.timesteps
            )),
            ('constant', 'disturbance_output_equation', dict(scenario=scenarios)),
            broadcast=['timestep', 'scenario']
        )

        # Output limits.
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(name='output_vector', scenario=scenarios, timestep=self.timesteps)),
            '>=',
            ('constant', 'output_minimum_timeseries', dict(scenario=scenarios)),
            broadcast=['timestep', 'scenario']
        )
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(name='output_vector', scenario=scenarios, timestep=self.timesteps)),
            '<=',
            ('constant', 'output_maximum_timeseries', dict(scenario=scenarios)),
            broadcast=['timestep', 'scenario']
        )

        # Define connection constraints.
        if len(self.electric_ders) > 0:
            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='der_active_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.electric_ders
                )),
                '==',
                ('constant', 'active_power_constant', dict(scenario=scenarios)),
                ('variable', 'mapping_active_power_by_output', dict(
                    name='output_vector', scenario=scenarios, timestep=self.timesteps
                )),
                broadcast=['timestep', 'scenario']
            )
            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='der_reactive_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.electric_ders
                )),
                '==',
                ('constant', 'reactive_power_constant', dict(scenario=scenarios)),
                ('variable', 'mapping_reactive_power_by_output', dict(
                    name='output_vector', scenario=scenarios, timestep=self.timesteps
                )),
                broadcast=['timestep', 'scenario']
            )
        if len(self.thermal_ders) > 0:
            optimization_problem.define_constraint(
                ('variable', 1.0, dict(
                    name='der_thermal_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.thermal_ders
                )),
                '==',
                ('constant', 'thermal_power_constant', dict(scenario=scenarios)),
                ('variable', 'mapping_thermal_power_by_output', dict(
                    name='output_vector', scenario=scenarios, timestep=self.timesteps
                )),
                broadcast=['timestep', 'scenario']
            )

    def define_optimization_objective(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            scenarios: typing.Union[list, pd.Index] = None
    ):

        # If no scenarios given, obtain default value.
        if scenarios is None:
            scenarios = [None]

        # Set objective flag.
        optimization_problem.flags['has_der_objective'] = True

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (self.timesteps[1] - self.timesteps[0]) / pd.Timedelta('1h')

        # Define objective for electric loads.
        # - Defined as cost of electric power supply at the DER node.
        # - Cost for load / demand, revenue for generation / supply.
        # - Only defined here, if not yet defined as cost of electric supply at electric grid source node
        #   in `mesmo.electric_grid_models.LinearElectricGridModelSet.define_optimization_objective`.
        if (len(self.electric_ders) > 0) and not optimization_problem.flags.get('has_electric_grid_objective'):
            optimization_problem.define_objective(
                ('variable', 'der_active_power_cost', dict(
                    name='der_active_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.electric_ders
                )),
                ('variable', 'der_active_power_cost_sensitivity', dict(
                    name='der_active_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.electric_ders
                ), dict(
                    name='der_active_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.electric_ders
                )),
                ('variable', 'der_reactive_power_cost', dict(
                    name='der_reactive_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.electric_ders
                )),
                ('variable', 'der_reactive_power_cost_sensitivity', dict(
                    name='der_reactive_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.electric_ders
                ), dict(
                name='der_reactive_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.electric_ders
                )),
                broadcast='scenario'
            )

        # Define objective for thermal loads.
        # - Defined as cost of thermal power supply at the DER node.
        # - Only defined here, if not yet defined as cost of thermal supply at thermal grid source node
        #   in `mesmo.electric_grid_models.LinearThermalGridModel.define_optimization_objective`.
        if (len(self.thermal_ders) > 0) and not optimization_problem.flags.get('has_thermal_grid_objective'):
            optimization_problem.define_objective(
                ('variable', 'der_thermal_power_cost', dict(
                    name='der_thermal_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.thermal_ders
                )),
                ('variable', 'der_thermal_power_cost_sensitivity', dict(
                    name='der_thermal_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.thermal_ders
                ), dict(
                    name='der_thermal_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.thermal_ders
                )),
                broadcast='scenario'
            )

        # Define objective for electric generators.
        # - That is: Active power generation cost.
        # - Always defined here as the cost of electric power generation at the DER node.
        if len(self.electric_ders) > 0:
            optimization_problem.define_objective(
                ('variable', 'der_active_power_marginal_cost', dict(
                    name='der_active_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.electric_ders
                )),
                broadcast='scenario'
            )
            optimization_problem.define_objective(
                ('variable', 'der_reactive_power_marginal_cost', dict(
                    name='der_reactive_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.electric_ders
                )),
                broadcast='scenario'
            )

        # Define objective for thermal generators.
        # - That is: Thermal power generation cost.
        # - Always defined here as the cost of thermal power generation at the DER node.
        if len(self.thermal_ders) > 0:
            optimization_problem.define_objective(
                ('variable', 'der_thermal_power_marginal_cost', dict(
                    name='der_thermal_power_vector', scenario=scenarios, timestep=self.timesteps,
                    der=self.thermal_ders
                )),
                broadcast='scenario'
            )

    def evaluate_optimization_objective(
            self,
            results: DERModelSetOperationResults,
            price_data: mesmo.data_interface.PriceData,
            has_electric_grid_objective: bool = False,
            has_thermal_grid_objective: bool = False
    ) -> float:

        # Instantiate optimization problem.
        optimization_problem = mesmo.utils.OptimizationProblem()
        optimization_problem.flags['has_electric_grid_objective'] = has_electric_grid_objective
        optimization_problem.flags['has_thermal_grid_objective'] = has_thermal_grid_objective
        self.define_optimization_variables(optimization_problem)
        self.define_optimization_parameters(optimization_problem, price_data)
        self.define_optimization_objective(optimization_problem)

        # Instantiate variable vector.
        x_vector = np.zeros((len(optimization_problem.variables), 1))

        # Set variable vector values.
        objective_variable_names = list()
        if len(self.electric_ders) > 0:
            objective_variable_names.extend([
                'der_active_power_vector_per_unit',
                'der_reactive_power_vector_per_unit'
            ])
        if len(self.thermal_ders) > 0:
            objective_variable_names.extend([
                'der_thermal_power_vector_per_unit'
            ])
        for variable_name in objective_variable_names:
            index = mesmo.utils.get_index(optimization_problem.variables, name=variable_name.replace('_per_unit', ''))
            x_vector[index, 0] = results[variable_name].values.ravel()

        # Obtain objective value.
        objective = optimization_problem.evaluate_objective(x_vector)

        return objective

    def get_optimization_results(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            scenarios: typing.Union[list, pd.Index] = None
    ) -> DERModelSetOperationResults:

        # If no scenarios given, obtain default value.
        # TODO: Validate behavior when passing scenarios.
        if scenarios is None:
            scenarios = [None]

        # Obtain results.
        state_vector = (
            optimization_problem.results['state_vector'].loc[self.timesteps, self.states]
            if len(self.states) > 0 else None
        )
        control_vector = (
            optimization_problem.results['control_vector'].loc[self.timesteps, self.controls]
            if len(self.controls) > 0 else None
        )
        output_vector = (
            optimization_problem.results['output_vector'].loc[self.timesteps, self.outputs]
            if len(self.outputs) > 0 else None
        )
        der_active_power_vector_per_unit = (
            optimization_problem.results['der_active_power_vector'].loc[self.timesteps, self.electric_ders]
            if len(self.electric_ders) > 0 else None
        )
        der_active_power_vector = (
            der_active_power_vector_per_unit
            * self.der_active_power_vector_reference
            if len(self.electric_ders) > 0 else None
        )
        der_reactive_power_vector_per_unit = (
            optimization_problem.results['der_reactive_power_vector'].loc[self.timesteps, self.electric_ders]
            if len(self.electric_ders) > 0 else None
        )
        der_reactive_power_vector = (
            der_reactive_power_vector_per_unit
            * self.der_reactive_power_vector_reference
            if len(self.electric_ders) > 0 else None
        )
        der_thermal_power_vector_per_unit = (
            optimization_problem.results['der_thermal_power_vector'].loc[self.timesteps, self.thermal_ders]
            if len(self.thermal_ders) > 0 else None
        )
        der_thermal_power_vector = (
            der_thermal_power_vector_per_unit
            * self.der_thermal_power_vector_reference
            if len(self.thermal_ders) > 0 else None
        )

        return DERModelSetOperationResults(
            der_model_set=self,
            state_vector=state_vector,
            control_vector=control_vector,
            output_vector=output_vector,
            der_active_power_vector=der_active_power_vector,
            der_active_power_vector_per_unit=der_active_power_vector_per_unit,
            der_reactive_power_vector=der_reactive_power_vector,
            der_reactive_power_vector_per_unit=der_reactive_power_vector_per_unit,
            der_thermal_power_vector=der_thermal_power_vector,
            der_thermal_power_vector_per_unit=der_thermal_power_vector_per_unit
        )

    def pre_solve(
            self,
            price_data: mesmo.data_interface.PriceData
    ) -> DERModelSetOperationResults:

        # Instantiate optimization problem.
        optimization_problem = mesmo.utils.OptimizationProblem()
        self.define_optimization_variables(optimization_problem)
        self.define_optimization_parameters(optimization_problem, price_data)
        self.define_optimization_constraints(optimization_problem)
        self.define_optimization_objective(optimization_problem)

        # Solve optimization problem and obtain results.
        optimization_problem.solve()
        results = self.get_optimization_results(optimization_problem)

        # Update nominal DER power time series.
        for der_name in self.der_names:
            if self.der_models[der_name].is_electric_grid_connected:
                self.der_models[der_name].active_power_nominal_timeseries.loc[:] = (
                    results.der_active_power_vector.loc[:, (slice(None), der_name)].values[:, 0]
                )
                self.der_models[der_name].reactive_power_nominal_timeseries.loc[:] = (
                    results.der_reactive_power_vector.loc[:, (slice(None), der_name)].values[:, 0]
                )
            if self.der_models[der_name].is_thermal_grid_connected:
                self.der_models[der_name].thermal_power_nominal_timeseries.loc[:] = (
                    results.der_thermal_power_vector.loc[:, (slice(None), der_name)].values[:, 0]
                )
        self.update_data()

        return results


def make_der_models(
        der_names: typing.List[str],
        der_data: mesmo.data_interface.DERData
) -> typing.Dict[str, DERModel]:

    der_models = dict.fromkeys(der_names)

    for der_name in der_names:
        der_models[der_name] = make_der_model(der_name, der_data)

    return der_models


def make_der_model(
        der_name: str,
        der_data: mesmo.data_interface.DERData,
        is_standalone=False
) -> DERModel:
    """Factory method for DER models, makes appropriate DER model type for given `der_name`."""

    # Obtain DER type.
    der_type = der_data.ders.at[der_name, 'der_type']

    # Obtain DER model classes.
    der_model_classes = (
        inspect.getmembers(sys.modules[__name__], lambda cls: inspect.isclass(cls) and issubclass(cls, DERModel))
    )

    # Obtain DER model for given `der_type`.
    for der_model_class_name, der_model_class in der_model_classes:
        if der_type == der_model_class.der_type:
            return der_model_class(der_data, der_name, is_standalone=is_standalone)

    # Raise error, if no DER model class found for given `der_type`.
    raise ValueError(
        f"Can't find DER model class for DER '{der_name}' of type '{der_type}'. "
        f"Please check if valid `der_type` is defined."
    )
