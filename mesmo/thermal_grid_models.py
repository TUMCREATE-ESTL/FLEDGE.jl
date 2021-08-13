"""Thermal grid models module."""

import cvxpy as cp
import itertools
from multimethod import multimethod
import numpy as np
import pandas as pd
import scipy.constants
import scipy.sparse as sp
import scipy.sparse.linalg
import typing

import mesmo.config
import mesmo.data_interface
import mesmo.der_models
import mesmo.utils

logger = mesmo.config.get_logger(__name__)


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
    branch_node_incidence_matrix: sp.spmatrix
    der_node_incidence_matrix: sp.spmatrix
    der_thermal_power_vector_reference: np.ndarray
    branch_flow_vector_reference: np.ndarray
    node_head_vector_reference: np.ndarray

    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain thermal grid data.
        thermal_grid_data = mesmo.data_interface.ThermalGridData(scenario_name)

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
            sp.dok_matrix((len(self.nodes), len(self.branches)), dtype=int)
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
            sp.dok_matrix((len(self.nodes), len(self.ders)), dtype=int)
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

        # Obtain DER model source node.
        # TODO: Use state space model for simulation / optimization.
        self.source_der_model = (
            mesmo.der_models.make_der_model(
                thermal_grid_data.thermal_grid.at['source_der_model_name'],
                thermal_grid_data.der_data,
                is_standalone=True
            )
        )
        # TODO: Remove temporary workaround: Obtain efficiency factors.
        if thermal_grid_data.thermal_grid.at['source_der_type'] == 'cooling_plant':
            self.plant_efficiency = self.source_der_model.cooling_plant_efficiency
        elif thermal_grid_data.thermal_grid.at['source_der_type'] == 'heat_pump':
            self.plant_efficiency = self.source_der_model.heat_pump_efficiency
        else:
            raise ValueError(f"Incompatible der model type: {thermal_grid_data.thermal_grid.at['source_der_type']}")


class ThermalGridDEROperationResults(mesmo.utils.ResultsBase):

    der_thermal_power_vector: pd.DataFrame
    der_thermal_power_vector_per_unit: pd.DataFrame


class ThermalGridOperationResults(ThermalGridDEROperationResults):

    thermal_grid_model: ThermalGridModel
    node_head_vector: pd.DataFrame
    node_head_vector_per_unit: pd.DataFrame
    branch_flow_vector: pd.DataFrame
    branch_flow_vector_per_unit: pd.DataFrame
    pump_power: pd.DataFrame


class ThermalGridDLMPResults(mesmo.utils.ResultsBase):

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
    source_thermal_power_cooling_plant: float

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
            / mesmo.config.water_density
            / thermal_grid_model.enthalpy_difference_distribution_water
        )
        self.source_flow = (
            -1.0 * np.sum(self.der_flow_vector)
        )

        # Obtain branch volume flow vector.
        self.branch_flow_vector = (
            scipy.sparse.linalg.spsolve(
                thermal_grid_model.branch_node_incidence_matrix[
                    mesmo.utils.get_index(thermal_grid_model.nodes, node_type='no_source'),
                    :
                ],
                thermal_grid_model.der_node_incidence_matrix[
                    mesmo.utils.get_index(thermal_grid_model.nodes, node_type='no_source'),
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
            / mesmo.config.water_kinematic_viscosity
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
                    logger.warning("Exceeding validity range of Swamee-Jain formula for calculation of friction factor.")
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
                mesmo.config.gravitational_acceleration
                * thermal_grid_model.line_diameter_vector ** 5
                * np.pi ** 2
            )
        )

        # Obtain node / source head vector.
        node_head_vector_no_source = (
            scipy.sparse.linalg.spsolve(
                np.transpose(
                    thermal_grid_model.branch_node_incidence_matrix[
                        mesmo.utils.get_index(thermal_grid_model.nodes, node_type='no_source'),
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
        self.node_head_vector[mesmo.utils.get_index(thermal_grid_model.nodes, node_type='no_source')] = (
            node_head_vector_no_source
        )

        # Obtain secondary pump / cooling plant thermal power.
        self.pump_power = (
            (
                2.0 * self.source_head
                + thermal_grid_model.energy_transfer_station_head_loss
            )
            * self.source_flow
            * mesmo.config.water_density
            * mesmo.config.gravitational_acceleration
            / thermal_grid_model.distribution_pump_efficiency
        )
        self.source_thermal_power_cooling_plant = (
            self.source_flow
            * thermal_grid_model.enthalpy_difference_distribution_water
            * mesmo.config.water_density
            / thermal_grid_model.plant_efficiency
        )


class ThermalPowerFlowSolutionSet(object):

    power_flow_solutions: typing.Dict[pd.Timestamp, ThermalPowerFlowSolution]
    thermal_grid_model: ThermalGridModel
    der_thermal_power_vector: pd.DataFrame
    timesteps: pd.Index

    @multimethod
    def __init__(
            self,
            thermal_grid_model: ThermalGridModel,
            der_operation_results: ThermalGridDEROperationResults,
            **kwargs
    ):

        der_thermal_power_vector = der_operation_results.der_thermal_power_vector

        self.__init__(
            thermal_grid_model,
            der_thermal_power_vector,
            **kwargs
        )

    @multimethod
    def __init__(
            self,
            thermal_grid_model: ThermalGridModel,
            der_thermal_power_vector: pd.DataFrame,
            power_flow_solution_method=ThermalPowerFlowSolution
    ):

        # Store attributes.
        self.thermal_grid_model = thermal_grid_model
        self.der_thermal_power_vector = der_thermal_power_vector
        self.timesteps = self.thermal_grid_model.timesteps

        # Obtain power flow solutions.
        power_flow_solutions = (
            mesmo.utils.starmap(
                power_flow_solution_method,
                zip(
                    itertools.repeat(self.thermal_grid_model),
                    der_thermal_power_vector.values
                )
            )
        )
        self.power_flow_solutions = dict(zip(self.timesteps, power_flow_solutions))

    def get_results(self) -> ThermalGridOperationResults:

        raise NotImplementedError


class LinearThermalGridModel(object):
    """Linear thermal grid model object."""

    thermal_grid_model: ThermalGridModel
    thermal_power_flow_solution: ThermalPowerFlowSolution
    sensitivity_branch_flow_by_node_power: sp.spmatrix
    sensitivity_branch_flow_by_der_power: sp.spmatrix
    sensitivity_node_head_by_node_power: sp.spmatrix
    sensitivity_node_head_by_der_power: sp.spmatrix
    sensitivity_pump_power_by_node_power: np.array
    sensitivity_pump_power_by_der_power: np.array

    @multimethod
    def __init__(
            self,
            scenario_name: str,
    ):

        # Obtain thermal grid model.
        thermal_grid_model = (
            ThermalGridModel(scenario_name)
        )

        # Obtain DER power vector.
        der_thermal_power_vector = (
            thermal_grid_model.der_thermal_power_vector_reference
        )

        # Obtain thermal power flow solution.
        thermal_power_flow_solution = (
            ThermalPowerFlowSolution(
                thermal_grid_model,
                der_thermal_power_vector
            )
        )

        self.__init__(
            thermal_grid_model,
            thermal_power_flow_solution
        )

    @multimethod
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
            mesmo.utils.get_index(self.thermal_grid_model.nodes, node_type='no_source')  # Define shorthand.
        )
        branch_node_incidence_matrix_inverse = (
            sp.dok_matrix(
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
            sp.dok_matrix(
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
            / mesmo.config.water_density
            / self.thermal_grid_model.enthalpy_difference_distribution_water
        )
        self.sensitivity_branch_flow_by_der_power = (
            self.sensitivity_branch_flow_by_node_power
            @ self.sensitivity_node_power_by_der_power
        )
        self.sensitivity_node_head_by_node_power = (
            branch_node_incidence_matrix_transpose_inverse
            @ sp.diags(
                np.abs(thermal_power_flow_solution.branch_flow_vector)
                * thermal_power_flow_solution.branch_friction_factor_vector
                * 8.0 * self.thermal_grid_model.line_length_vector
                / (
                    mesmo.config.gravitational_acceleration
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
                * mesmo.config.water_density
                * mesmo.config.gravitational_acceleration
                / self.thermal_grid_model.distribution_pump_efficiency
            )
            + (
                -1.0
                * self.thermal_grid_model.energy_transfer_station_head_loss
                * mesmo.config.gravitational_acceleration
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


# TODO: Split global / local approximation methods.
LinearThermalGridModelGlobal = LinearThermalGridModel


class LinearThermalGridModelSet(object):

    linear_thermal_grid_models: typing.Dict[pd.Timestamp, LinearThermalGridModel]
    thermal_grid_model: ThermalGridModel
    timesteps: pd.Index

    @multimethod
    def __init__(
            self,
            thermal_grid_model: ThermalGridModel,
            thermal_power_flow_solution_set: ThermalPowerFlowSolutionSet,
            linear_thermal_grid_model_method: typing.Type[LinearThermalGridModel] = LinearThermalGridModelGlobal
    ):

        self.check_linear_thermal_grid_model_method(linear_thermal_grid_model_method)

        # Obtain linear thermal grid models.
        linear_thermal_grid_models = (
            mesmo.utils.starmap(
                linear_thermal_grid_model_method,
                zip(
                    itertools.repeat(thermal_grid_model),
                    thermal_power_flow_solution_set.power_flow_solutions.values()
                )
            )
        )
        linear_thermal_grid_models = (
            dict(zip(thermal_grid_model.timesteps, linear_thermal_grid_models))
        )

        self.__init__(
            thermal_grid_model,
            linear_thermal_grid_models
        )

    @multimethod
    def __init__(
            self,
            thermal_grid_model: ThermalGridModel,
            thermal_power_flow_solution: ThermalPowerFlowSolution,
            linear_thermal_grid_model_method: typing.Type[LinearThermalGridModel] = LinearThermalGridModelGlobal
    ):

        self.check_linear_thermal_grid_model_method(linear_thermal_grid_model_method)

        # Obtain linear thermal grid models.
        linear_thermal_grid_model = LinearThermalGridModelGlobal(thermal_grid_model, thermal_power_flow_solution)
        linear_thermal_grid_models = (
            dict(zip(thermal_grid_model.timesteps, itertools.repeat(linear_thermal_grid_model)))
        )

        self.__init__(
            thermal_grid_model,
            linear_thermal_grid_models
        )

    @multimethod
    def __init__(
            self,
            thermal_grid_model: ThermalGridModel,
            linear_thermal_grid_models: typing.Dict[pd.Timestamp, LinearThermalGridModel]
    ):

        # Store attributes.
        self.thermal_grid_model = thermal_grid_model
        self.timesteps = self.thermal_grid_model.timesteps
        self.linear_thermal_grid_models = linear_thermal_grid_models

    @staticmethod
    def check_linear_thermal_grid_model_method(linear_thermal_grid_model_method):

        if not issubclass(linear_thermal_grid_model_method, LinearThermalGridModel):
            raise ValueError(f"Invalid linear thermal grid model method: {linear_thermal_grid_model_method}")

    def define_optimization_problem(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            price_data: mesmo.data_interface.PriceData,
            **kwargs
    ):

        # Defined optimization problem definitions through respective sub-methods.
        self.define_optimization_variables(optimization_problem)
        self.define_optimization_parameters(
            optimization_problem,
            price_data,
            **kwargs
        )
        self.define_optimization_constraints(optimization_problem)
        self.define_optimization_objective(optimization_problem)

    def define_optimization_variables(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem
    ):

        # Define DER power vector variables.
        # - Only if these have not yet been defined within `DERModelSet`.
        if 'der_thermal_power_vector' not in optimization_problem.variables.loc[:, 'name'].values:
            optimization_problem.define_variable(
                'der_thermal_power_vector', timestep=self.timesteps, der=self.thermal_grid_model.ders
            )

        # Define node head, branch flow and pump power variables.
        optimization_problem.define_variable(
            'node_head_vector', timestep=self.timesteps, node=self.thermal_grid_model.nodes
        )
        optimization_problem.define_variable(
            'branch_flow_vector', timestep=self.timesteps, branch=self.thermal_grid_model.branches
        )
        optimization_problem.define_variable(
            'pump_power', timestep=self.timesteps
        )

    def define_optimization_parameters(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            price_data: mesmo.data_interface.PriceData,
            node_head_vector_minimum: np.ndarray = None,
            branch_flow_vector_maximum: np.ndarray = None,
    ):

        # Obtain timestep interval in hours, for conversion of power to energy.
        timestep_interval_hours = (self.timesteps[1] - self.timesteps[0]) / pd.Timedelta('1h')

        # Define head variable term.
        optimization_problem.define_parameter(
            'head_variable',
            sp.block_diag([
                sp.diags(linear_thermal_grid_model.thermal_grid_model.node_head_vector_reference ** -1)
                @ linear_thermal_grid_model.sensitivity_node_head_by_der_power
                @ sp.diags(linear_thermal_grid_model.thermal_grid_model.der_thermal_power_vector_reference)
                for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
            ])
        )

        # Define head constant term.
        optimization_problem.define_parameter(
            'head_constant',
            np.concatenate([
                sp.diags(linear_thermal_grid_model.thermal_grid_model.node_head_vector_reference ** -1)
                @ (
                    np.transpose([linear_thermal_grid_model.thermal_power_flow_solution.node_head_vector])
                    - linear_thermal_grid_model.sensitivity_node_head_by_der_power
                    @ np.transpose([linear_thermal_grid_model.thermal_power_flow_solution.der_thermal_power_vector])
                ) for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
            ])
        )

        # Define branch flow variable term.
        optimization_problem.define_parameter(
            'branch_flow_variable',
            sp.block_diag([
                sp.diags(linear_thermal_grid_model.thermal_grid_model.branch_flow_vector_reference ** -1)
                @ linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
                @ sp.diags(linear_thermal_grid_model.thermal_grid_model.der_thermal_power_vector_reference)
                for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
            ])
        )

        # Define branch flow constant term.
        optimization_problem.define_parameter(
            'branch_flow_constant',
            np.concatenate([
                sp.diags(linear_thermal_grid_model.thermal_grid_model.branch_flow_vector_reference ** -1)
                @ (
                    np.transpose([linear_thermal_grid_model.thermal_power_flow_solution.branch_flow_vector])
                    - linear_thermal_grid_model.sensitivity_branch_flow_by_der_power
                    @ np.transpose([linear_thermal_grid_model.thermal_power_flow_solution.der_thermal_power_vector])
                ) for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
            ])
        )

        # Define pump power variable term.
        optimization_problem.define_parameter(
            'pump_power_variable',
            sp.block_diag([
                linear_thermal_grid_model.sensitivity_pump_power_by_der_power
                @ sp.diags(linear_thermal_grid_model.thermal_grid_model.der_thermal_power_vector_reference)
                for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
            ])
        )

        # Define pump power constant term.
        optimization_problem.define_parameter(
            'pump_power_constant',
            np.concatenate([
                # TODO: Fix pump power sensitivity.
                [0.0]
                # linear_thermal_grid_model.thermal_power_flow_solution.pump_power
                # - linear_thermal_grid_model.sensitivity_pump_power_by_der_power
                # @ np.transpose([linear_thermal_grid_model.thermal_power_flow_solution.der_thermal_power_vector])
                for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
            ])
        )

        # Define head limits.
        optimization_problem.define_parameter(
            'node_head_minimum',
            np.concatenate([
                node_head_vector_minimum.ravel()
                / linear_thermal_grid_model.thermal_grid_model.node_head_vector_reference
                for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
            ])
            if node_head_vector_minimum is not None
            else -np.inf * np.ones((len(self.thermal_grid_model.nodes) * len(self.timesteps), ))
        )

        # Define branch flow limits.
        optimization_problem.define_parameter(
            'branch_flow_minimum',
            np.concatenate([
                - branch_flow_vector_maximum.ravel()
                / linear_thermal_grid_model.thermal_grid_model.branch_flow_vector_reference
                for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
            ])
            if branch_flow_vector_maximum is not None
            else -np.inf * np.ones((len(self.thermal_grid_model.branches) * len(self.timesteps), ))
        )
        optimization_problem.define_parameter(
            'branch_flow_maximum',
            np.concatenate([
                branch_flow_vector_maximum.ravel()
                / linear_thermal_grid_model.thermal_grid_model.branch_flow_vector_reference
                for linear_thermal_grid_model in self.linear_thermal_grid_models.values()
            ])
            if branch_flow_vector_maximum is not None
            else +np.inf * np.ones((len(self.thermal_grid_model.branches) * len(self.timesteps), ))
        )

        # Define objective parameters.
        optimization_problem.define_parameter(
            'thermal_grid_thermal_power_cost',
            np.array([price_data.price_timeseries.loc[:, ('thermal_power', 'source', 'source')].values])
            * -1.0 * timestep_interval_hours  # In Wh.
            / self.thermal_grid_model.plant_efficiency
            @ sp.block_diag(
                [np.array([self.thermal_grid_model.der_thermal_power_vector_reference])] * len(self.timesteps)
            )
        )
        optimization_problem.define_parameter(
            'thermal_grid_thermal_power_cost_sensitivity',
            price_data.price_sensitivity_coefficient
            * timestep_interval_hours  # In Wh.
            * np.concatenate(
                [np.array([self.thermal_grid_model.der_thermal_power_vector_reference ** 2])] * len(self.timesteps),
                axis=1
            )
        )
        optimization_problem.define_parameter(
            'thermal_grid_pump_power_cost',
            price_data.price_timeseries.loc[:, ('thermal_power', 'source', 'source')].values
            * timestep_interval_hours  # In Wh.
        )
        optimization_problem.define_parameter(
            'thermal_grid_pump_power_cost_sensitivity',
            price_data.price_sensitivity_coefficient
            * timestep_interval_hours  # In Wh.
        )

    def define_optimization_constraints(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
    ):

        # Define head equation.
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='node_head_vector', timestep=self.timesteps,
                node=self.thermal_grid_model.nodes
            )),
            '==',
            ('variable', 'head_variable', dict(
                name='der_thermal_power_vector', timestep=self.timesteps,
                der=self.thermal_grid_model.ders
            )),
            ('constant', 'head_constant', dict(timestep=self.timesteps)),
        )

        # Define branch flow equation.
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='branch_flow_vector', timestep=self.timesteps,
                branch=self.thermal_grid_model.branches
            )),
            '==',
            ('variable', 'branch_flow_variable', dict(
                name='der_thermal_power_vector', timestep=self.timesteps,
                der=self.thermal_grid_model.ders
            )),
            ('constant', 'branch_flow_constant', dict(timestep=self.timesteps)),
        )

        # Define pump power equation.
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(name='pump_power', timestep=self.timesteps)),
            '==',
            ('variable', 'pump_power_variable', dict(
                name='der_thermal_power_vector', timestep=self.timesteps,
                der=self.thermal_grid_model.ders
            )),
            ('constant', 'pump_power_constant', dict(timestep=self.timesteps)),
        )

        # Define head limits.
        # Add dedicated keys to enable retrieving dual variables.
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='node_head_vector', timestep=self.timesteps,
                node=self.thermal_grid_model.nodes
            )),
            '>=',
            ('constant', 'node_head_minimum', dict(timestep=self.timesteps)),
            keys=dict(
                name='node_head_vector_minimum_constraint', timestep=self.timesteps,
                node=self.thermal_grid_model.nodes
            ),
        )

        # Define branch flow limits.
        # Add dedicated keys to enable retrieving dual variables.
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='branch_flow_vector', timestep=self.timesteps,
                branch=self.thermal_grid_model.branches
            )),
            '>=',
            ('constant', 'branch_flow_minimum', dict(timestep=self.timesteps)),
            keys=dict(
                name='branch_flow_vector_minimum_constraint', timestep=self.timesteps,
                branch=self.thermal_grid_model.branches
            ),
        )
        optimization_problem.define_constraint(
            ('variable', 1.0, dict(
                name='branch_flow_vector', timestep=self.timesteps,
                branch=self.thermal_grid_model.branches
            )),
            '<=',
            ('constant', 'branch_flow_maximum', dict(timestep=self.timesteps)),
            keys=dict(
                name='branch_flow_vector_maximum_constraint', timestep=self.timesteps,
                branch=self.thermal_grid_model.branches
            ),
        )

    def define_optimization_objective(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem
    ):

        # Set objective flag.
        optimization_problem.flags['has_thermal_grid_objective'] = True

        # Define objective for thermal loads.
        # - Defined as cost of thermal supply at thermal grid source node.
        # - Only defined here, if not yet defined as cost of thermal power supply at the DER node
        #   in `mesmo.der_models.DERModel.define_optimization_objective`.
        if not optimization_problem.flags.get('has_der_objective'):

            # Thermal power cost / revenue.
            # - Cost for load / demand, revenue for generation / supply.
            optimization_problem.define_objective(
                (
                    'variable',
                    'thermal_grid_thermal_power_cost',
                    dict(name='der_thermal_power_vector', timestep=self.timesteps, der=self.thermal_grid_model.ders)
                ), (
                    'variable',
                    'thermal_grid_thermal_power_cost_sensitivity',
                    dict(name='der_thermal_power_vector', timestep=self.timesteps, der=self.thermal_grid_model.ders),
                    dict(name='der_thermal_power_vector', timestep=self.timesteps, der=self.thermal_grid_model.ders)
                )
            )

        # Define pump power cost.
        optimization_problem.define_objective(
            (
                'variable',
                'thermal_grid_pump_power_cost',
                dict(name='pump_power', timestep=self.timesteps),
            ), (
                'variable',
                'thermal_grid_pump_power_cost_sensitivity',
                dict(name='pump_power', timestep=self.timesteps),
                dict(name='pump_power', timestep=self.timesteps)
            )
        )

    def evaluate_optimization_objective(
            self,
            results: ThermalGridOperationResults,
            price_data: mesmo.data_interface.PriceData
    ) -> float:

        # Instantiate optimization problem.
        optimization_problem = mesmo.utils.OptimizationProblem()
        self.define_optimization_parameters(optimization_problem, price_data)
        self.define_optimization_variables(optimization_problem)
        self.define_optimization_objective(optimization_problem)

        # Instantiate variable vector.
        x_vector = np.zeros((len(optimization_problem.variables), 1))

        # Set variable vector values.
        objective_variable_names = [
            'der_thermal_power_vector_per_unit',
            'pump_power'
        ]
        for variable_name in objective_variable_names:
            index = mesmo.utils.get_index(optimization_problem.variables, name=variable_name.replace('_per_unit', ''))
            x_vector[index, 0] = results[variable_name].values.ravel()

        # Obtain objective value.
        objective = optimization_problem.evaluate_objective(x_vector)

        return objective

    def get_optimization_dlmps(
            self,
            optimization_problem: mesmo.utils.OptimizationProblem,
            price_data: mesmo.data_interface.PriceData
    ) -> ThermalGridDLMPResults:

        # Obtain individual duals.
        node_head_vector_minimum_dual = (
            optimization_problem.duals['node_head_vector_minimum_constraint'].loc[
                self.thermal_grid_model.timesteps, self.thermal_grid_model.nodes
            ]
            / np.array([(self.thermal_grid_model.node_head_vector_reference)])
        )
        branch_flow_vector_minimum_dual = (
            optimization_problem.duals['branch_flow_vector_minimum_constraint'].loc[
                self.thermal_grid_model.timesteps, self.thermal_grid_model.branches
            ]
            / np.array([self.thermal_grid_model.branch_flow_vector_reference])
        )
        branch_flow_vector_maximum_dual = (
            -1.0 * optimization_problem.duals['branch_flow_vector_maximum_constraint'].loc[
                self.thermal_grid_model.timesteps, self.thermal_grid_model.branches
            ]
            / np.array([self.thermal_grid_model.branch_flow_vector_reference])
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
                / self.thermal_grid_model.plant_efficiency
            )
            thermal_grid_head_dlmp_node_thermal_power.loc[timestep, :] = (
                (
                    self.linear_thermal_grid_models[timestep].sensitivity_node_head_by_node_power.transpose()
                    @ np.transpose([node_head_vector_minimum_dual.loc[timestep, :].values])
                ).ravel()
            )
            thermal_grid_congestion_dlmp_node_thermal_power.loc[timestep, :] = (
                (
                    self.linear_thermal_grid_models[timestep].sensitivity_branch_flow_by_node_power.transpose()
                    @ np.transpose([branch_flow_vector_maximum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    self.linear_thermal_grid_models[timestep].sensitivity_branch_flow_by_node_power.transpose()
                    @ np.transpose([branch_flow_vector_minimum_dual.loc[timestep, :].values])
                ).ravel()
            )
            thermal_grid_pump_dlmp_node_thermal_power.loc[timestep, :] = (
                -1.0 * self.linear_thermal_grid_models[timestep].sensitivity_pump_power_by_node_power.ravel()
                * price_data.price_timeseries.at[timestep, ('thermal_power', 'source', 'source')]
            )

            thermal_grid_energy_dlmp_der_thermal_power.loc[timestep, :] = (
                price_data.price_timeseries.at[timestep, ('thermal_power', 'source', 'source')]
                / self.thermal_grid_model.plant_efficiency
            )
            thermal_grid_head_dlmp_der_thermal_power.loc[timestep, :] = (
                (
                    self.linear_thermal_grid_models[timestep].sensitivity_node_head_by_der_power.transpose()
                    @ np.transpose([node_head_vector_minimum_dual.loc[timestep, :].values])
                ).ravel()
            )
            thermal_grid_congestion_dlmp_der_thermal_power.loc[timestep, :] = (
                (
                    self.linear_thermal_grid_models[timestep].sensitivity_branch_flow_by_der_power.transpose()
                    @ np.transpose([branch_flow_vector_maximum_dual.loc[timestep, :].values])
                ).ravel()
                + (
                    self.linear_thermal_grid_models[timestep].sensitivity_branch_flow_by_der_power.transpose()
                    @ np.transpose([branch_flow_vector_minimum_dual.loc[timestep, :].values])
                ).ravel()
            )
            thermal_grid_pump_dlmp_der_thermal_power.loc[timestep, :] = (
                -1.0 * self.linear_thermal_grid_models[timestep].sensitivity_pump_power_by_der_power.ravel()
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

        # Obtain total DLMPs in format similar to `mesmo.data_interface.PriceData.price_timeseries`.
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
            optimization_problem: mesmo.utils.OptimizationProblem
    ) -> ThermalGridOperationResults:

        # Obtain results.
        der_thermal_power_vector_per_unit = (
            optimization_problem.results['der_thermal_power_vector'].loc[
                self.thermal_grid_model.timesteps, self.thermal_grid_model.ders
            ]
        )
        der_thermal_power_vector = (
            der_thermal_power_vector_per_unit
            * (self.thermal_grid_model.der_thermal_power_vector_reference)
        )
        node_head_vector_per_unit = (
            optimization_problem.results['node_head_vector'].loc[
                self.thermal_grid_model.timesteps, self.thermal_grid_model.nodes
            ]
        )
        node_head_vector = (
            node_head_vector_per_unit
            * (self.thermal_grid_model.node_head_vector_reference)
        )
        branch_flow_vector_per_unit = (
            optimization_problem.results['branch_flow_vector'].loc[
                self.thermal_grid_model.timesteps, self.thermal_grid_model.branches
            ]
        )
        branch_flow_vector = (
            branch_flow_vector_per_unit
            * self.thermal_grid_model.branch_flow_vector_reference
        )
        pump_power = (
            optimization_problem.results['pump_power'].loc[self.thermal_grid_model.timesteps, ['pump_power']]
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
