"""Thermal grid models module."""

import cvxpy as cp
from multimethod import multimethod
import numpy as np
import pandas as pd
import scipy.constants
import scipy.sparse
import scipy.sparse.linalg

import fledge.config
import fledge.data_interface
import fledge.utils

logger = fledge.config.get_logger(__name__)


class ThermalGridModel(object):
    """Thermal grid model object."""

    timesteps: pd.Index
    node_names: pd.Index
    line_names: pd.Index
    der_names: pd.Index
    der_types: pd.Index
    nodes: pd.Index
    branches: pd.Index
    ders = pd.Index
    branch_node_incidence_matrix: scipy.sparse.spmatrix
    der_node_incidence_matrix: scipy.sparse.spmatrix
    der_thermal_power_vector_reference: np.ndarray
    branch_flow_vector_reference: np.ndarray
    node_head_vector_reference: np.ndarray

    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain thermal grid data.
        thermal_grid_data = fledge.data_interface.ThermalGridData(scenario_name)

        # Obtain index set for time steps.
        # - This is needed for optimization problem definitions within linear thermal grid models.
        self.timesteps = thermal_grid_data.scenario_data.timesteps

        # Obtain node / line / DER names.
        self.node_names = pd.Index(thermal_grid_data.thermal_grid_nodes['node_name'])
        self.line_names = pd.Index(thermal_grid_data.thermal_grid_lines['line_name'])
        self.der_names = pd.Index(thermal_grid_data.thermal_grid_ders['der_name'])
        self.der_types = pd.Index(thermal_grid_data.thermal_grid_ders['der_type']).unique()

        # Obtain node / branch / DER index set.
        nodes = (
            pd.concat([
                thermal_grid_data.thermal_grid_nodes.loc[:, 'node_name'].apply(
                    # Obtain `node_type` column.
                    lambda value:
                    'source' if value == thermal_grid_data.thermal_grid.at['source_node_name']
                    else 'no_source'
                ).rename('node_type'),
                thermal_grid_data.thermal_grid_nodes.loc[:, 'node_name']
            ], axis='columns')
        )
        self.nodes = pd.MultiIndex.from_frame(nodes)
        self.branches = self.line_names.rename('branch_name')
        self.ders = pd.MultiIndex.from_frame(thermal_grid_data.thermal_grid_ders[['der_type', 'der_name']])

        # Define branch to node incidence matrix.
        self.branch_node_incidence_matrix = (
            scipy.sparse.dok_matrix((len(self.nodes), len(self.branches)), dtype=int)
        )
        for node_index, node_name in enumerate(self.nodes.get_level_values('node_name')):
            for branch_index, branch in enumerate(self.branches):
                if node_name == thermal_grid_data.thermal_grid_lines.at[branch, 'node_1_name']:
                    self.branch_node_incidence_matrix[node_index, branch_index] += +1.0
                elif node_name == thermal_grid_data.thermal_grid_lines.at[branch, 'node_2_name']:
                    self.branch_node_incidence_matrix[node_index, branch_index] += -1.0
        self.branch_node_incidence_matrix = self.branch_node_incidence_matrix.tocsr()

        # Define DER to node incidence matrix.
        self.der_node_incidence_matrix = (
            scipy.sparse.dok_matrix((len(self.nodes), len(self.ders)), dtype=int)
        )
        for node_index, node_name in enumerate(self.nodes.get_level_values('node_name')):
            for der_index, der_name in enumerate(self.der_names):
                if node_name == thermal_grid_data.thermal_grid_ders.at[der_name, 'node_name']:
                    self.der_node_incidence_matrix[node_index, der_index] = 1.0
        self.der_node_incidence_matrix = self.der_node_incidence_matrix.tocsr()

        # Obtain DER nominal thermal power vector.
        self.der_thermal_power_vector_reference = (
            thermal_grid_data.thermal_grid_ders.loc[:, 'thermal_power_nominal'].values
        )

        # Obtain nominal branch flow vector.
        self.branch_flow_vector_reference = (
            np.pi
            * (thermal_grid_data.thermal_grid_lines.loc[:, 'diameter'].values / 2) ** 2
            * thermal_grid_data.thermal_grid_lines.loc[:, 'maximum_velocity'].values
        )

        # Obtain nominal branch flow vector.
        # TODO: Define proper node head reference vector.
        self.node_head_vector_reference = (
            np.ones(len(self.nodes))
        )

        # Obtain line parameters.
        self.line_length_vector = thermal_grid_data.thermal_grid_lines['length'].values
        self.line_diameter_vector = thermal_grid_data.thermal_grid_lines['diameter'].values
        self.line_roughness_vector = thermal_grid_data.thermal_grid_lines['absolute_roughness'].values

        # Obtain other system parameters.
        self.energy_transfer_station_head_loss = (
            float(thermal_grid_data.thermal_grid['energy_transfer_station_head_loss'])
        )
        self.enthalpy_difference_distribution_water = (
            float(thermal_grid_data.thermal_grid['enthalpy_difference_distribution_water'])
        )
        self.distribution_pump_efficiency = (
            float(thermal_grid_data.thermal_grid['distribution_pump_efficiency'])
        )

        # Obtain cooling plant efficiency.
        # TODO: Enable consideration for dynamic wet bulb temperature.
        ambient_air_wet_bulb_temperature = (
            thermal_grid_data.thermal_grid.at['cooling_tower_set_reference_temperature_wet_bulb']
        )
        condensation_temperature = (
            thermal_grid_data.thermal_grid.at['cooling_tower_set_reference_temperature_condenser_water']
            + (
                thermal_grid_data.thermal_grid.at['cooling_tower_set_reference_temperature_slope']
                * (
                    ambient_air_wet_bulb_temperature
                    - thermal_grid_data.thermal_grid.at['cooling_tower_set_reference_temperature_wet_bulb']
                )
            )
            + thermal_grid_data.thermal_grid.at['condenser_water_temperature_difference']
            + thermal_grid_data.thermal_grid.at['chiller_set_condenser_minimum_temperature_difference']
            + 273.15
        )
        chiller_inverse_coefficient_of_performance = (
            (
                (
                    condensation_temperature
                    / thermal_grid_data.thermal_grid.at['chiller_set_evaporation_temperature']
                )
                - 1.0
            )
            * (
                thermal_grid_data.thermal_grid.at['chiller_set_beta']
                + 1.0
            )
        )
        evaporator_pump_specific_electric_power = (
            (1.0 / thermal_grid_data.thermal_grid.at['plant_pump_efficiency'])
            * scipy.constants.value('standard acceleration of gravity')
            * thermal_grid_data.thermal_grid.at['water_density']
            * thermal_grid_data.thermal_grid.at['evaporator_pump_head']
            / (
                thermal_grid_data.thermal_grid.at['water_density']
                * thermal_grid_data.thermal_grid.at['enthalpy_difference_distribution_water']
            )
        )
        condenser_specific_thermal_power = (
            1.0 + chiller_inverse_coefficient_of_performance
        )
        condenser_pump_specific_electric_power = (
            (1.0 / thermal_grid_data.thermal_grid.at['plant_pump_efficiency'])
            * scipy.constants.value('standard acceleration of gravity')
            * thermal_grid_data.thermal_grid.at['water_density']
            * thermal_grid_data.thermal_grid.at['condenser_pump_head']
            * condenser_specific_thermal_power
            / (
                thermal_grid_data.thermal_grid.at['water_density']
                * thermal_grid_data.thermal_grid.at['condenser_water_enthalpy_difference']
            )
        )
        cooling_tower_ventilation_specific_electric_power = (
            thermal_grid_data.thermal_grid.at['cooling_tower_set_ventilation_factor']
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


class ThermalPowerFlowSolution(object):
    """Thermal grid power flow solution object."""

    der_thermal_power_vector: np.ndarray
    der_flow_vector: np.ndarray
    source_flow: float
    branch_flow_vector: np.ndarray
    branch_velocity_vector: np.ndarray
    branch_reynold_vector: np.ndarray
    branch_friction_factor_vector: np.ndarray
    branch_head_vector: np.ndarray
    source_head: float
    node_head_vector: np.ndarray
    pump_power: float
    source_electric_power_cooling_plant: float

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain thermal grid model.
        thermal_grid_model = ThermalGridModel(scenario_name)

        self.__init__(
            thermal_grid_model
        )

    @multimethod
    def __init__(
            self,
            thermal_grid_model: ThermalGridModel
    ):

        # Obtain DER thermal power vector.
        der_thermal_power_vector = thermal_grid_model.der_thermal_power_vector_reference

        self.__init__(
            thermal_grid_model,
            der_thermal_power_vector
        )

    @multimethod
    def __init__(
            self,
            thermal_grid_model: ThermalGridModel,
            der_thermal_power_vector: np.ndarray
    ):

        # Obtain DER thermal power vector.
        self.der_thermal_power_vector = der_thermal_power_vector.ravel()

        # Obtain DER / source volume flow vector.
        self.der_flow_vector = (
            self.der_thermal_power_vector
            / fledge.config.water_density
            / thermal_grid_model.enthalpy_difference_distribution_water
        )
        self.source_flow = (
            -1.0 * np.sum(self.der_flow_vector)
        )

        # Obtain branch volume flow vector.
        self.branch_flow_vector = (
            scipy.sparse.linalg.spsolve(
                thermal_grid_model.branch_node_incidence_matrix[
                    fledge.utils.get_index(thermal_grid_model.nodes, node_type='no_source'),
                    :
                ],
                thermal_grid_model.der_node_incidence_matrix[
                    fledge.utils.get_index(thermal_grid_model.nodes, node_type='no_source'),
                    :
                ]
                @ np.transpose([self.der_flow_vector])
            )
        ).ravel()

        # Obtain branch velocity vector.
        self.branch_velocity_vector = (
            4.0 * self.branch_flow_vector
            / (np.pi * thermal_grid_model.line_diameter_vector ** 2)
        )

        # Obtain branch Reynolds coefficient vector.
        self.branch_reynold_vector = (
            np.abs(self.branch_velocity_vector)
            * thermal_grid_model.line_diameter_vector
            / fledge.config.water_kinematic_viscosity
        )

        # Obtain branch friction factor vector.
        @np.vectorize
        def get_friction_factor(
                reynold,
                roughness,
                diameter
        ):

            # No flow.
            if reynold == 0:
                friction_factor = 0

            # Laminar Flow, based on Hagen-Poiseuille velocity profile, analytical correlation.
            elif 0 < reynold < 4000:
                friction_factor = 64 / reynold

            # Turbulent flow, Swamee-Jain formula, approximating correlation of Colebrook-White equation.
            elif 4000 <= reynold:
                if not (reynold <= 100000000 and 0.000001 <= ((roughness / 1000) / diameter) <= 0.01):
                    logger.warn("Exceeding validity range of Swamee-Jain formula for calculation of friction factor.")
                friction_factor = 1.325 / (
                    np.log(
                        (roughness / 1000) / (3.7 * diameter) + 5.74 / (reynold ** 0.9)
                    )
                ) ** 2

            else:
                raise ValueError(f"Invalid Reynolds coefficient: {reynold}")

            # Convert from 1/m to 1/km.
            friction_factor *= 1.0e3

            return friction_factor

        self.branch_friction_factor_vector = (
            get_friction_factor(
                self.branch_reynold_vector,
                thermal_grid_model.line_roughness_vector,
                thermal_grid_model.line_diameter_vector
            )
        )

        # Obtain branch head loss vector.
        # - Darcy-Weisbach Equation.
        self.branch_head_vector = (
            self.branch_friction_factor_vector
            * self.branch_flow_vector
            * np.abs(self.branch_flow_vector)  # TODO: Check if absolute value needed.
            * 8.0 * thermal_grid_model.line_length_vector
            / (
                fledge.config.gravitational_acceleration
                * thermal_grid_model.line_diameter_vector ** 5
                * np.pi ** 2
            )
        )

        # Obtain node / source head vector.
        node_head_vector_no_source = (
            scipy.sparse.linalg.spsolve(
                np.transpose(
                    thermal_grid_model.branch_node_incidence_matrix[
                        fledge.utils.get_index(thermal_grid_model.nodes, node_type='no_source'),
                        :
                    ]
                ),
                self.branch_head_vector
            )
        )
        self.source_head = (
            np.max(np.abs(node_head_vector_no_source))
        )
        self.node_head_vector = np.zeros(len(thermal_grid_model.nodes), dtype=float)
        self.node_head_vector[fledge.utils.get_index(thermal_grid_model.nodes, node_type='no_source')] = (
            node_head_vector_no_source
        )

        # Obtain secondary pump / cooling plant electric power.
        self.pump_power = (
            (
                2.0 * self.source_head
                + thermal_grid_model.energy_transfer_station_head_loss
            )
            * self.source_flow
            * fledge.config.water_density
            * fledge.config.gravitational_acceleration
            / thermal_grid_model.distribution_pump_efficiency
        )
        self.source_electric_power_cooling_plant = (
            self.source_flow
            * thermal_grid_model.enthalpy_difference_distribution_water
            * fledge.config.water_density
            / thermal_grid_model.cooling_plant_efficiency
        )


class ThermalGridOperationResults(fledge.utils.ResultsBase):

    thermal_grid_model: ThermalGridModel
    der_thermal_power_vector: pd.DataFrame
    der_thermal_power_vector_per_unit: pd.DataFrame
    node_head_vector: pd.DataFrame
    node_head_vector_per_unit: pd.DataFrame
    branch_flow_vector: pd.DataFrame
    branch_flow_vector_per_unit: pd.DataFrame
    pump_power: pd.DataFrame


class ThermalGridDLMPResults(fledge.utils.ResultsBase):

    thermal_grid_energy_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_head_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_congestion_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_pump_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_total_dlmp_node_thermal_power: pd.DataFrame
    thermal_grid_energy_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_head_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_congestion_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_pump_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_total_dlmp_der_thermal_power: pd.DataFrame
    thermal_grid_total_dlmp_price_timeseries: pd.DataFrame


class LinearThermalGridModel(object):
    """Linear thermal grid model object."""

    thermal_grid_model: ThermalGridModel
    thermal_power_flow_solution: ThermalPowerFlowSolution
    sensitivity_branch_flow_by_der_power: scipy.sparse.spmatrix
    sensitivity_node_head_by_der_power: scipy.sparse.spmatrix
    sensitivity_pump_power_by_der_power: np.array

    def __init__(
            self,
            thermal_grid_model: ThermalGridModel,
            thermal_power_flow_solution: ThermalPowerFlowSolution,
    ):

        # Store thermal grid model.
        self.thermal_grid_model = thermal_grid_model

        # Store thermal power flow solution.
        self.thermal_power_flow_solution = thermal_power_flow_solution

        # Obtain inverse / transpose incidence matrices.
        node_index_no_source = (
            fledge.utils.get_index(self.thermal_grid_model.nodes, node_type='no_source')  # Define shorthand.
        )
        branch_node_incidence_matrix_inverse = (
            scipy.sparse.dok_matrix(
                (len(self.thermal_grid_model.branches), len(self.thermal_grid_model.nodes)),
                dtype=float
            )
        )
        branch_node_incidence_matrix_inverse[np.ix_(
            range(len(self.thermal_grid_model.branches)),
            node_index_no_source
        )] = (
            scipy.sparse.linalg.inv(
                self.thermal_grid_model.branch_node_incidence_matrix[node_index_no_source, :].tocsc()
            )
        )
        branch_node_incidence_matrix_inverse = branch_node_incidence_matrix_inverse.tocsr()
        branch_node_incidence_matrix_transpose_inverse = (
            scipy.sparse.dok_matrix(
                (len(self.thermal_grid_model.nodes), len(self.thermal_grid_model.branches)),
                dtype=float
            )
        )
        branch_node_incidence_matrix_transpose_inverse[np.ix_(
            node_index_no_source,
            range(len(self.thermal_grid_model.branches))
        )] = (
            scipy.sparse.linalg.inv(
                self.thermal_grid_model.branch_node_incidence_matrix[node_index_no_source, :].transpose()
            )
        )
        branch_node_incidence_matrix_transpose_inverse = branch_node_incidence_matrix_transpose_inverse.tocsr()
        der_node_incidence_matrix_transpose = np.transpose(self.thermal_grid_model.der_node_incidence_matrix)

        # Obtain sensitivity matrices.
        self.sensitivity_node_power_by_der_power = (
            self.thermal_grid_model.der_node_incidence_matrix
        )
        self.sensitivity_branch_flow_by_node_power = (
            branch_node_incidence_matrix_inverse
            / fledge.config.water_density
            / self.thermal_grid_model.enthalpy_difference_distribution_water
        )
        self.sensitivity_branch_flow_by_der_power = (
            self.sensitivity_branch_flow_by_node_power
            @ self.sensitivity_node_power_by_der_power
        )
        self.sensitivity_node_head_by_node_power = (
            branch_node_incidence_matrix_transpose_inverse
            @ scipy.sparse.diags(
                np.abs(thermal_power_flow_solution.branch_flow_vector)
                * thermal_power_flow_solution.branch_friction_factor_vector
                * 8.0 * self.thermal_grid_model.line_length_vector
                / (
                    fledge.config.gravitational_acceleration
                    * self.thermal_grid_model.line_diameter_vector ** 5
                    * np.pi ** 2
                )
            )
            @ self.sensitivity_branch_flow_by_node_power
        )
        self.sensitivity_node_head_by_der_power = (
            self.sensitivity_node_head_by_node_power
            @ self.sensitivity_node_power_by_der_power
        )
        self.sensitivity_pump_power_by_node_power = (
            (
                (-1.0 * thermal_power_flow_solution.der_flow_vector)
                @ (-2.0 * der_node_incidence_matrix_transpose)
                @ self.sensitivity_node_head_by_node_power
                * fledge.config.water_density
                * fledge.config.gravitational_acceleration
                / self.thermal_grid_model.distribution_pump_efficiency
            )
            + (
                -1.0
                * self.thermal_grid_model.energy_transfer_station_head_loss
                * fledge.config.gravitational_acceleration
                / self.thermal_grid_model.enthalpy_difference_distribution_water
                / self.thermal_grid_model.distribution_pump_efficiency
            )
        )
        self.sensitivity_pump_power_by_der_power = (
            np.array([
                self.sensitivity_pump_power_by_node_power
                @ self.sensitivity_node_power_by_der_power
            ])
        )

    def define_optimization_variables(
            self,
            optimization_problem: fledge.utils.OptimizationProblem
    ):
        """Define decision variables for given `optimization_problem`."""

        # Define DER power vector variable.
        # - Only if this has not yet been defined within `DERModelSet`.
        if not hasattr(optimization_problem, 'der_thermal_power_vector'):
            optimization_problem.der_thermal_power_vector = (
                cp.Variable((len(self.thermal_grid_model.timesteps), len(self.thermal_grid_model.ders)))
            )

        # Define node head, branch flow and pump power variables.
        optimization_problem.node_head_vector = (
            cp.Variable((len(self.thermal_grid_model.timesteps), len(self.thermal_grid_model.nodes)))
        )
        optimization_problem.branch_flow_vector = (
            cp.Variable((len(self.thermal_grid_model.timesteps), len(self.thermal_grid_model.branches)))
        )
        optimization_problem.pump_power = (
            cp.Variable((len(self.thermal_grid_model.timesteps), 1))
        )
        # # TODO: Pump power not non-negative?
        # optimization_problem.pump_power = (
        #     cp.Variable((len(self.thermal_grid_model.timesteps), 1))
        # )

    def define_optimization_constraints(
            self,
            optimization_problem: fledge.utils.OptimizationProblem,
            timestep_index=slice(None),  # TODO: Enable passing as time step / list of time steps.
            node_head_vector_minimum: np.ndarray = None,
            branch_flow_vector_maximum: np.ndarray = None
    ):
        """Define constraints to express the linear thermal grid model equations for given `optimization_problem`."""

        # Define node head equation.
        optimization_problem.constraints.append(
            optimization_problem.node_head_vector[timestep_index, :]
            ==
            cp.transpose(
                self.sensitivity_node_head_by_der_power
                @ cp.transpose(cp.multiply(
                    optimization_problem.der_thermal_power_vector[timestep_index, :],
                    np.array([self.thermal_grid_model.der_thermal_power_vector_reference])
                ))
            )
            / np.array([self.thermal_grid_model.node_head_vector_reference])
        )

        # Define branch flow equation.
        optimization_problem.constraints.append(
            optimization_problem.branch_flow_vector[timestep_index, :]
            ==
            cp.transpose(
                self.sensitivity_branch_flow_by_der_power
                @ cp.transpose(cp.multiply(
                    optimization_problem.der_thermal_power_vector[timestep_index, :],
                    np.array([self.thermal_grid_model.der_thermal_power_vector_reference])
                ))
            )
            / np.array([self.thermal_grid_model.branch_flow_vector_reference])
        )

        # Define pump power equation.
        optimization_problem.constraints.append(
            optimization_problem.pump_power[timestep_index, :]
            ==
            cp.transpose(
                self.sensitivity_pump_power_by_der_power
                @ cp.transpose(cp.multiply(
                    optimization_problem.der_thermal_power_vector[timestep_index, :],
                    np.array([self.thermal_grid_model.der_thermal_power_vector_reference])
                ))
            )
        )

        # Define node head limits.
        # - Add dedicated constraints variables to enable retrieving dual variables.
        # - When using `LinearElectricGridModelSet`, this will be defined only once with the first model, because it
        #   does not depend on the sensitivity matrices.
        if (
                (node_head_vector_minimum is not None)
                and not hasattr(optimization_problem, 'node_head_vector_minimum_constraint')
        ):
            optimization_problem.node_head_vector_minimum_constraint = (
                optimization_problem.node_head_vector
                - np.array([node_head_vector_minimum.ravel()])
                / np.array([self.thermal_grid_model.node_head_vector_reference])
                >=
                0.0
            )
            optimization_problem.constraints.append(optimization_problem.node_head_vector_minimum_constraint)

        # Define branch flow limits.
        # - Add dedicated constraints variables to enable retrieving dual variables.
        # - When using `LinearElectricGridModelSet`, this will be defined only once with the first model, because it
        #   does not depend on the sensitivity matrices.
        if (
                (branch_flow_vector_maximum is not None)
                and not hasattr(optimization_problem, 'branch_flow_vector_minimum_constraint')
        ):
            optimization_problem.branch_flow_vector_minimum_constraint = (
                optimization_problem.branch_flow_vector
                + np.array([branch_flow_vector_maximum.ravel()])
                / np.array([self.thermal_grid_model.branch_flow_vector_reference])
                >=
                0.0
            )
            optimization_problem.constraints.append(optimization_problem.branch_flow_vector_minimum_constraint)
            optimization_problem.branch_flow_vector_maximum_constraint = (
                optimization_problem.branch_flow_vector
                - np.array([branch_flow_vector_maximum.ravel()])
                / np.array([self.thermal_grid_model.branch_flow_vector_reference])
                <=
                0.0
            )
            optimization_problem.constraints.append(optimization_problem.branch_flow_vector_maximum_constraint)

    def define_optimization_objective(
            self,
            optimization_problem: fledge.utils.OptimizationProblem,
            price_data: fledge.data_interface.PriceData,
            timestep_index=slice(None)  # TODO: Enable passing as time step / list of time steps.
    ):

        # Set objective flag.
        optimization_problem.has_thermal_grid_objective = True

        # Obtain timestep interval in hours, for conversion of power to energy.
        if len(self.thermal_grid_model.timesteps) > 1:
            timestep_interval_hours = (
                (self.thermal_grid_model.timesteps[1] - self.thermal_grid_model.timesteps[0])
                / pd.Timedelta('1h')
            )
        else:
            timestep_interval_hours = 1.0

        # Define objective for thermal loads.
        # - Defined as cost of thermal supply at electric grid source node.
        # - Only defined here, if not yet defined as cost of thermal power supply at the DER node
        #   in `fledge.der_models.DERModel.define_optimization_objective`.
        if not optimization_problem.has_der_objective:

            # Thermal power cost / revenue.
            # - Cost for load / demand, revenue for generation / supply.
            optimization_problem.objective += (
                (
                    np.array([
                        price_data.price_timeseries.loc[:, ('thermal_power', 'source', 'source')].values[timestep_index]
                    ])
                    * timestep_interval_hours  # In Wh.
                    @ cp.sum(-1.0 * (
                        cp.multiply(
                            optimization_problem.der_thermal_power_vector[timestep_index, :],
                            np.array([self.thermal_grid_model.der_thermal_power_vector_reference])
                        )
                    ), axis=1, keepdims=True)  # Sum along DERs, i.e. sum for each timestep.
                )
                + ((
                    price_data.price_sensitivity_coefficient
                    * timestep_interval_hours  # In Wh.
                    * cp.sum((
                        cp.multiply(
                            optimization_problem.der_thermal_power_vector[timestep_index, :],
                            np.array([self.thermal_grid_model.der_thermal_power_vector_reference])
                        )
                    ) ** 2)
                ) if price_data.price_sensitivity_coefficient != 0.0 else 0.0)
            )

        # Define pump cost.
        optimization_problem.objective += (
            (
                # TODO: Use active power instead of thermal power price.
                np.array([
                    price_data.price_timeseries.loc[:, ('thermal_power', 'source', 'source')].values[timestep_index]
                ])
                * timestep_interval_hours  # In Wh.
                @ optimization_problem.pump_power[timestep_index, :]
            )
            + ((
                price_data.price_sensitivity_coefficient
                * timestep_interval_hours  # In Wh.
                * cp.sum(optimization_problem.pump_power[timestep_index, :] ** 2)
            ) if price_data.price_sensitivity_coefficient != 0.0 else 0.0)
        )

    def get_optimization_dlmps(
            self,
            optimization_problem: fledge.utils.OptimizationProblem,
            price_data: fledge.data_interface.PriceData
    ) -> ThermalGridDLMPResults:

        # Obtain duals.
        node_head_vector_minimum_dual = (
            pd.DataFrame(
                (
                    optimization_problem.node_head_vector_minimum_constraint.dual_value
                    / np.array([self.thermal_grid_model.node_head_vector_reference])
                    if hasattr(optimization_problem, 'node_head_vector_minimum_constraint')
                    else 0.0
                ),
                columns=self.thermal_grid_model.nodes,
                index=self.thermal_grid_model.timesteps
            )
        )
        branch_flow_vector_minimum_dual = (
            pd.DataFrame(
                (
                    optimization_problem.branch_flow_vector_minimum_constraint.dual_value
                    / np.array([self.thermal_grid_model.branch_flow_vector_reference])
                    if hasattr(optimization_problem, 'branch_flow_vector_minimum_constraint')
                    else 0.0
                ),
                columns=self.thermal_grid_model.branches,
                index=self.thermal_grid_model.timesteps
            )
        )
        branch_flow_vector_maximum_dual = (
            pd.DataFrame(
                (
                    -1.0 * optimization_problem.branch_flow_vector_maximum_constraint.dual_value
                    / np.array([self.thermal_grid_model.branch_flow_vector_reference])
                    if hasattr(optimization_problem, 'branch_flow_vector_maximum_constraint')
                    else 0.0
                ),
                columns=self.thermal_grid_model.branches,
                index=self.thermal_grid_model.timesteps
            )
        )

        # Instantiate DLMP variables.
        thermal_grid_energy_dlmp_node_thermal_power = (
            pd.DataFrame(columns=self.thermal_grid_model.nodes, index=self.thermal_grid_model.timesteps, dtype=float)
        )
        thermal_grid_head_dlmp_node_thermal_power = (
            pd.DataFrame(columns=self.thermal_grid_model.nodes, index=self.thermal_grid_model.timesteps, dtype=float)
        )
        thermal_grid_congestion_dlmp_node_thermal_power = (
            pd.DataFrame(columns=self.thermal_grid_model.nodes, index=self.thermal_grid_model.timesteps, dtype=float)
        )
        thermal_grid_pump_dlmp_node_thermal_power = (
            pd.DataFrame(columns=self.thermal_grid_model.nodes, index=self.thermal_grid_model.timesteps, dtype=float)
        )

        thermal_grid_energy_dlmp_der_thermal_power = (
            pd.DataFrame(columns=self.thermal_grid_model.ders, index=self.thermal_grid_model.timesteps, dtype=float)
        )
        thermal_grid_head_dlmp_der_thermal_power = (
            pd.DataFrame(columns=self.thermal_grid_model.ders, index=self.thermal_grid_model.timesteps, dtype=float)
        )
        thermal_grid_congestion_dlmp_der_thermal_power = (
            pd.DataFrame(columns=self.thermal_grid_model.ders, index=self.thermal_grid_model.timesteps, dtype=float)
        )
        thermal_grid_pump_dlmp_der_thermal_power = (
            pd.DataFrame(columns=self.thermal_grid_model.ders, index=self.thermal_grid_model.timesteps, dtype=float)
        )

        # Obtain DLMPs.
        for timestep in self.thermal_grid_model.timesteps:
            thermal_grid_energy_dlmp_node_thermal_power.loc[timestep, :] = (
                price_data.price_timeseries.at[timestep, ('thermal_power', 'source', 'source')]
                / self.thermal_grid_model.cooling_plant_efficiency
            )
            thermal_grid_head_dlmp_node_thermal_power.loc[timestep, :] = (
                (
                    self.sensitivity_node_head_by_node_power.transpose()
                    @ np.transpose([node_head_vector_minimum_dual.loc[timestep, :].values])
                ).ravel()
            )
            thermal_grid_congestion_dlmp_node_thermal_power.loc[timestep, :] = (
                (
                    self.sensitivity_branch_flow_by_node_power.transpose()
                    @ np.transpose([branch_flow_vector_maximum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    self.sensitivity_branch_flow_by_node_power.transpose()
                    @ np.transpose([branch_flow_vector_minimum_dual.loc[timestep, :].values])
                ).ravel()
            )
            thermal_grid_pump_dlmp_node_thermal_power.loc[timestep, :] = (
                -1.0
                * self.sensitivity_pump_power_by_node_power.ravel()
                * price_data.price_timeseries.at[timestep, ('thermal_power', 'source', 'source')]
            )

            thermal_grid_energy_dlmp_der_thermal_power.loc[timestep, :] = (
                price_data.price_timeseries.at[timestep, ('thermal_power', 'source', 'source')]
                / self.thermal_grid_model.cooling_plant_efficiency
            )
            thermal_grid_head_dlmp_der_thermal_power.loc[timestep, :] = (
                (
                    self.sensitivity_node_head_by_der_power.transpose()
                    @ np.transpose([node_head_vector_minimum_dual.loc[timestep, :].values])
                ).ravel()
            )
            thermal_grid_congestion_dlmp_der_thermal_power.loc[timestep, :] = (
                (
                    self.sensitivity_branch_flow_by_der_power.transpose()
                    @ np.transpose([branch_flow_vector_maximum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    self.sensitivity_branch_flow_by_der_power.transpose()
                    @ np.transpose([branch_flow_vector_minimum_dual.loc[timestep, :].values])
                ).ravel()
            )
            thermal_grid_pump_dlmp_der_thermal_power.loc[timestep, :] = (
                -1.0
                * self.sensitivity_pump_power_by_der_power.ravel()
                * price_data.price_timeseries.at[timestep, ('thermal_power', 'source', 'source')]
            )

        thermal_grid_total_dlmp_node_thermal_power = (
            thermal_grid_energy_dlmp_node_thermal_power
            + thermal_grid_head_dlmp_node_thermal_power
            + thermal_grid_congestion_dlmp_node_thermal_power
            + thermal_grid_pump_dlmp_node_thermal_power
        )
        thermal_grid_total_dlmp_der_thermal_power = (
            thermal_grid_energy_dlmp_der_thermal_power
            + thermal_grid_head_dlmp_der_thermal_power
            + thermal_grid_congestion_dlmp_der_thermal_power
            + thermal_grid_pump_dlmp_der_thermal_power
        )

        # Obtain total DLMPs in format similar to `fledge.data_interface.PriceData.price_timeseries`.
        thermal_grid_total_dlmp_price_timeseries = (
            pd.concat(
                [
                    price_data.price_timeseries.loc[:, ('thermal_power', 'source', 'source')].rename(
                        ('source', 'source')
                    ),
                    thermal_grid_total_dlmp_der_thermal_power
                ],
                axis='columns',
                keys=['thermal_power', 'thermal_power'],
                names=['commodity_type']
            )
        )
        # Redefine columns to avoid slicing issues.
        thermal_grid_total_dlmp_price_timeseries.columns = (
            price_data.price_timeseries.columns[
                price_data.price_timeseries.columns.isin(thermal_grid_total_dlmp_price_timeseries.columns)
            ]
        )

        return ThermalGridDLMPResults(
            thermal_grid_energy_dlmp_node_thermal_power=thermal_grid_energy_dlmp_node_thermal_power,
            thermal_grid_head_dlmp_node_thermal_power=thermal_grid_head_dlmp_node_thermal_power,
            thermal_grid_congestion_dlmp_node_thermal_power=thermal_grid_congestion_dlmp_node_thermal_power,
            thermal_grid_pump_dlmp_node_thermal_power=thermal_grid_pump_dlmp_node_thermal_power,
            thermal_grid_total_dlmp_node_thermal_power=thermal_grid_total_dlmp_node_thermal_power,
            thermal_grid_energy_dlmp_der_thermal_power=thermal_grid_energy_dlmp_der_thermal_power,
            thermal_grid_head_dlmp_der_thermal_power=thermal_grid_head_dlmp_der_thermal_power,
            thermal_grid_congestion_dlmp_der_thermal_power=thermal_grid_congestion_dlmp_der_thermal_power,
            thermal_grid_pump_dlmp_der_thermal_power=thermal_grid_pump_dlmp_der_thermal_power,
            thermal_grid_total_dlmp_der_thermal_power=thermal_grid_total_dlmp_der_thermal_power,
            thermal_grid_total_dlmp_price_timeseries=thermal_grid_total_dlmp_price_timeseries
        )

    def get_optimization_results(
            self,
            optimization_problem: fledge.utils.OptimizationProblem
    ) -> ThermalGridOperationResults:

        # Instantiate results variables.
        der_thermal_power_vector = (
            pd.DataFrame(
                (
                    optimization_problem.der_thermal_power_vector.value
                    * np.array([self.thermal_grid_model.der_thermal_power_vector_reference])
                ),
                columns=self.thermal_grid_model.ders,
                index=self.thermal_grid_model.timesteps
            )
        )
        node_head_vector = (
            pd.DataFrame(
                (
                    optimization_problem.node_head_vector.value
                    * np.array([self.thermal_grid_model.node_head_vector_reference])
                ),
                columns=self.thermal_grid_model.nodes,
                index=self.thermal_grid_model.timesteps
            )
        )
        branch_flow_vector = (
            pd.DataFrame(
                (
                    optimization_problem.branch_flow_vector.value
                    * np.array([self.thermal_grid_model.branch_flow_vector_reference])
                ),
                columns=self.thermal_grid_model.branches,
                index=self.thermal_grid_model.timesteps
            )
        )
        pump_power = (
            pd.DataFrame(
                optimization_problem.pump_power.value,
                columns=['total'],
                index=self.thermal_grid_model.timesteps
            )
        )

        # Convert in per-unit values.
        der_thermal_power_vector_per_unit = (
            der_thermal_power_vector
            / self.thermal_grid_model.der_thermal_power_vector_reference
        )
        node_head_vector_per_unit = (
            node_head_vector
            / self.thermal_grid_model.node_head_vector_reference
        )
        branch_flow_vector_per_unit = (
            branch_flow_vector
            / self.thermal_grid_model.branch_flow_vector_reference
        )

        return ThermalGridOperationResults(
            thermal_grid_model=self.thermal_grid_model,
            der_thermal_power_vector=der_thermal_power_vector,
            der_thermal_power_vector_per_unit=der_thermal_power_vector_per_unit,
            node_head_vector=node_head_vector,
            node_head_vector_per_unit=node_head_vector_per_unit,
            branch_flow_vector=branch_flow_vector,
            branch_flow_vector_per_unit=branch_flow_vector_per_unit,
            pump_power=pump_power
        )
