"""Problems module for mathematical optimization and simulation problem type definitions."""

import itertools
from multimethod import multimethod
import numpy as np
import pandas as pd
import typing

import mesmo.config
import mesmo.data_interface
import mesmo.der_models
import mesmo.electric_grid_models
import mesmo.thermal_grid_models
import mesmo.utils

logger = mesmo.config.get_logger(__name__)


class Results(
    mesmo.electric_grid_models.ElectricGridOperationResults,
    mesmo.thermal_grid_models.ThermalGridOperationResults,
    mesmo.der_models.DERModelSetOperationResults,
    mesmo.electric_grid_models.ElectricGridDLMPResults,
    mesmo.thermal_grid_models.ThermalGridDLMPResults
):

    price_data: mesmo.data_interface.PriceData


class ResultsDict(typing.Dict[str, Results]):

    pass


class Problem(mesmo.utils.ObjectBase):

    pass


class ProblemDict(typing.Dict[str, Problem]):

    pass


class NominalOperationProblem(Problem):
    """Nominal operation problem object, consisting of the corresponding electric / thermal grid models,
    reference power flow solutions and DER model set for the given scenario.

    - The nominal operation problem (alias: simulation problem, power flow problem)
      represents the simulation problem of the DERs and grids considering the nominal operation schedule for all DERs.
    - The problem formulation is able to consider combined as well as individual operation of
      thermal and electric grids.
    """

    scenario_name: str
    timesteps: pd.Index
    price_data: mesmo.data_interface.PriceData
    electric_grid_model: mesmo.electric_grid_models.ElectricGridModelDefault = None
    thermal_grid_model: mesmo.thermal_grid_models.ThermalGridModel = None
    der_model_set: mesmo.der_models.DERModelSet
    results: Results

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            electric_grid_model: mesmo.electric_grid_models.ElectricGridModelDefault = None,
            thermal_grid_model: mesmo.thermal_grid_models.ThermalGridModel = None,
            der_model_set: mesmo.der_models.DERModelSet = None
    ):

        # Obtain data.
        scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
        self.price_data = mesmo.data_interface.PriceData(scenario_name)

        # Store timesteps.
        self.timesteps = scenario_data.timesteps

        # Obtain electric grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['electric_grid_name']):
            if electric_grid_model is not None:
                self.electric_grid_model = electric_grid_model
            else:
                mesmo.utils.log_time("electric grid model instantiation")
                self.electric_grid_model = mesmo.electric_grid_models.ElectricGridModelDefault(scenario_name)
                mesmo.utils.log_time("electric grid model instantiation")

        # Obtain thermal grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['thermal_grid_name']):
            if thermal_grid_model is not None:
                self.thermal_grid_model = thermal_grid_model
            else:
                mesmo.utils.log_time("thermal grid model instantiation")
                self.thermal_grid_model = mesmo.thermal_grid_models.ThermalGridModel(scenario_name)
                mesmo.utils.log_time("thermal grid model instantiation")

        # Obtain DER model set.
        if der_model_set is not None:
            self.der_model_set = der_model_set
        else:
            mesmo.utils.log_time("DER model instantiation")
            self.der_model_set = mesmo.der_models.DERModelSet(scenario_name)
            mesmo.utils.log_time("DER model instantiation")

    def solve(self):

        # Instantiate results variables.
        if self.electric_grid_model is not None:
            der_power_vector = (
                pd.DataFrame(columns=self.electric_grid_model.ders, index=self.timesteps, dtype=complex)
            )
            node_voltage_vector = (
                pd.DataFrame(columns=self.electric_grid_model.nodes, index=self.timesteps, dtype=complex)
            )
            branch_power_vector_1 = (
                pd.DataFrame(columns=self.electric_grid_model.branches, index=self.timesteps, dtype=complex)
            )
            branch_power_vector_2 = (
                pd.DataFrame(columns=self.electric_grid_model.branches, index=self.timesteps, dtype=complex)
            )
            loss = pd.DataFrame(columns=['total'], index=self.timesteps, dtype=complex)
        if self.thermal_grid_model is not None:
            der_thermal_power_vector = (
                pd.DataFrame(columns=self.thermal_grid_model.ders, index=self.timesteps, dtype=float)
            )
            node_head_vector = (
                pd.DataFrame(columns=self.thermal_grid_model.nodes, index=self.timesteps, dtype=float)
            )
            branch_flow_vector = (
                pd.DataFrame(columns=self.thermal_grid_model.branches, index=self.timesteps, dtype=float)
            )
            pump_power = pd.DataFrame(columns=['total'], index=self.timesteps, dtype=float)

        # Obtain nominal DER power vector.
        if self.electric_grid_model is not None:
            for der in self.electric_grid_model.ders:
                # TODO: Use ders instead of der_names for der_models index.
                der_name = der[1]
                der_power_vector.loc[:, der] = (
                    self.der_model_set.der_models[der_name].active_power_nominal_timeseries
                    + (1.0j * self.der_model_set.der_models[der_name].reactive_power_nominal_timeseries)
                )
        if self.thermal_grid_model is not None:
            for der in self.electric_grid_model.ders:
                der_name = der[1]
                der_thermal_power_vector.loc[:, der] = (
                    self.der_model_set.der_models[der_name].thermal_power_nominal_timeseries
                )

        # Solve power flow.
        mesmo.utils.log_time("power flow solution")
        if self.electric_grid_model is not None:
            power_flow_solutions = (
                mesmo.utils.starmap(
                    mesmo.electric_grid_models.PowerFlowSolutionFixedPoint,
                    zip(
                        itertools.repeat(self.electric_grid_model),
                        der_power_vector.values
                    )
                )
            )
            power_flow_solutions = dict(zip(self.timesteps, power_flow_solutions))
        if self.thermal_grid_model is not None:
            thermal_power_flow_solutions = (
                mesmo.utils.starmap(
                    mesmo.thermal_grid_models.ThermalPowerFlowSolution,
                    [(self.thermal_grid_model, row) for row in der_thermal_power_vector.values]
                )
            )
            thermal_power_flow_solutions = dict(zip(self.timesteps, thermal_power_flow_solutions))
        mesmo.utils.log_time("power flow solution")

        # Obtain results.
        if self.electric_grid_model is not None:
            for timestep in self.timesteps:
                power_flow_solution = power_flow_solutions[timestep]
                # TODO: Flatten power flow solution arrays.
                node_voltage_vector.loc[timestep, :] = power_flow_solution.node_voltage_vector
                branch_power_vector_1.loc[timestep, :] = power_flow_solution.branch_power_vector_1
                branch_power_vector_2.loc[timestep, :] = power_flow_solution.branch_power_vector_2
                loss.loc[timestep, :] = power_flow_solution.loss
            der_active_power_vector = der_power_vector.apply(np.real)
            der_reactive_power_vector = der_power_vector.apply(np.imag)
            node_voltage_magnitude_vector = np.abs(node_voltage_vector)
            branch_power_magnitude_vector_1 = np.abs(branch_power_vector_1)
            branch_power_magnitude_vector_2 = np.abs(branch_power_vector_2)
            loss_active = loss.apply(np.real)
            loss_reactive = loss.apply(np.imag)
        if self.thermal_grid_model is not None:
            for timestep in self.timesteps:
                thermal_power_flow_solution = thermal_power_flow_solutions[timestep]
                node_head_vector.loc[timestep, :] = thermal_power_flow_solution.node_head_vector
                branch_flow_vector.loc[timestep, :] = thermal_power_flow_solution.branch_flow_vector
                pump_power.loc[timestep, :] = thermal_power_flow_solution.pump_power

        # Obtain per-unit values.
        if self.electric_grid_model is not None:
            der_active_power_vector_per_unit = (
                der_active_power_vector
                / np.real(self.electric_grid_model.der_power_vector_reference)
            )
            der_reactive_power_vector_per_unit = (
                der_reactive_power_vector
                / np.imag(self.electric_grid_model.der_power_vector_reference)
            )
            node_voltage_magnitude_vector_per_unit = (
                node_voltage_magnitude_vector
                / np.abs(self.electric_grid_model.node_voltage_vector_reference)
            )
            branch_power_magnitude_vector_1_per_unit = (
                branch_power_magnitude_vector_1
                / self.electric_grid_model.branch_power_vector_magnitude_reference
            )
            branch_power_magnitude_vector_2_per_unit = (
                branch_power_magnitude_vector_2
                / self.electric_grid_model.branch_power_vector_magnitude_reference
            )
        if self.thermal_grid_model is not None:
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

        # Store results.
        self.results = (
            Results(
                price_data=self.price_data,
                der_model_set=self.der_model_set
            )
        )
        if self.electric_grid_model is not None:
            self.results.update(
                Results(
                    electric_grid_model=self.electric_grid_model,
                    der_active_power_vector=der_active_power_vector,
                    der_active_power_vector_per_unit=der_active_power_vector_per_unit,
                    der_reactive_power_vector=der_reactive_power_vector,
                    der_reactive_power_vector_per_unit=der_reactive_power_vector_per_unit,
                    node_voltage_magnitude_vector=node_voltage_magnitude_vector,
                    node_voltage_magnitude_vector_per_unit=node_voltage_magnitude_vector_per_unit,
                    branch_power_magnitude_vector_1=branch_power_magnitude_vector_1,
                    branch_power_magnitude_vector_1_per_unit=branch_power_magnitude_vector_1_per_unit,
                    branch_power_magnitude_vector_2=branch_power_magnitude_vector_2,
                    branch_power_magnitude_vector_2_per_unit=branch_power_magnitude_vector_2_per_unit,
                    loss_active=loss_active,
                    loss_reactive=loss_reactive
                )
            )
        if self.thermal_grid_model is not None:
            self.results.update(
                Results(
                    thermal_grid_model=self.thermal_grid_model,
                    der_thermal_power_vector=der_thermal_power_vector,
                    der_thermal_power_vector_per_unit=der_thermal_power_vector_per_unit,
                    node_head_vector=node_head_vector,
                    node_head_vector_per_unit=node_head_vector_per_unit,
                    branch_flow_vector=branch_flow_vector,
                    branch_flow_vector_per_unit=branch_flow_vector_per_unit,
                    pump_power=pump_power
                )
            )

    def get_results(self):

        return self.results


class OptimalOperationProblem(Problem):
    """Optimal operation problem object, consisting of an optimization problem as well as the corresponding
    electric / thermal grid models, reference power flow solutions, linear grid models and DER model set
    for the given scenario.

    - The optimal operation problem (alias: optimal dispatch problem, optimal power flow problem)
      formulates the optimization problem for minimizing the objective functions of DERs and grid operators
      subject to the model constraints of all DERs and grids.
    - The problem formulation is able to consider combined as well as individual operation of
      thermal and electric grids.
    """

    scenario_name: str
    timesteps: pd.Index
    price_data: mesmo.data_interface.PriceData
    electric_grid_model: mesmo.electric_grid_models.ElectricGridModelDefault = None
    power_flow_solution_reference: mesmo.electric_grid_models.PowerFlowSolution = None
    linear_electric_grid_model_set: mesmo.electric_grid_models.LinearElectricGridModelSet = None
    thermal_grid_model: mesmo.thermal_grid_models.ThermalGridModel = None
    thermal_power_flow_solution_reference: mesmo.thermal_grid_models.ThermalPowerFlowSolution = None
    linear_thermal_grid_model_set: mesmo.thermal_grid_models.LinearThermalGridModelSet = None
    der_model_set: mesmo.der_models.DERModelSet
    optimization_problem: mesmo.utils.OptimizationProblem
    results: Results

    @multimethod
    def __init__(
            self,
            scenario_name: str,
            electric_grid_model: mesmo.electric_grid_models.ElectricGridModelDefault = None,
            thermal_grid_model: mesmo.thermal_grid_models.ThermalGridModel = None,
            der_model_set: mesmo.der_models.DERModelSet = None
    ):

        # Obtain data.
        scenario_data = mesmo.data_interface.ScenarioData(scenario_name)
        self.price_data = mesmo.data_interface.PriceData(scenario_name)

        # Store timesteps.
        self.timesteps = scenario_data.timesteps

        # Obtain electric grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['electric_grid_name']):
            mesmo.utils.log_time("electric grid model instantiation")
            if electric_grid_model is not None:
                self.electric_grid_model = electric_grid_model
            else:
                self.electric_grid_model = mesmo.electric_grid_models.ElectricGridModelDefault(scenario_name)
            self.power_flow_solution_reference = (
                mesmo.electric_grid_models.PowerFlowSolutionFixedPoint(self.electric_grid_model)
            )
            self.linear_electric_grid_model_set = (
                mesmo.electric_grid_models.LinearElectricGridModelSet(
                    self.electric_grid_model,
                    self.power_flow_solution_reference,
                    linear_electric_grid_model_method=mesmo.electric_grid_models.LinearElectricGridModelGlobal
                )
            )
            mesmo.utils.log_time("electric grid model instantiation")

        # Obtain thermal grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['thermal_grid_name']):
            mesmo.utils.log_time("thermal grid model instantiation")
            if thermal_grid_model is not None:
                self.thermal_grid_model = thermal_grid_model
            else:
                self.thermal_grid_model = mesmo.thermal_grid_models.ThermalGridModel(scenario_name)
            self.thermal_power_flow_solution_reference = (
                mesmo.thermal_grid_models.ThermalPowerFlowSolution(self.thermal_grid_model)
            )
            self.linear_thermal_grid_model_set = (
                mesmo.thermal_grid_models.LinearThermalGridModelSet(
                    self.thermal_grid_model,
                    self.thermal_power_flow_solution_reference
                )
            )
            mesmo.utils.log_time("thermal grid model instantiation")

        # Obtain DER model set.
        if der_model_set is not None:
            self.der_model_set = der_model_set
        else:
            mesmo.utils.log_time("DER model instantiation")
            self.der_model_set = mesmo.der_models.DERModelSet(scenario_name)
            mesmo.utils.log_time("DER model instantiation")

        # Instantiate optimization problem.
        self.optimization_problem = mesmo.utils.OptimizationProblem()

        # Define electric grid problem.
        if self.electric_grid_model is not None:
            self.linear_electric_grid_model_set.define_optimization_variables(self.optimization_problem)
            node_voltage_magnitude_vector_minimum = (
                scenario_data.scenario['voltage_per_unit_minimum']
                * np.abs(self.electric_grid_model.node_voltage_vector_reference)
                if pd.notnull(scenario_data.scenario['voltage_per_unit_minimum'])
                else None
            )
            node_voltage_magnitude_vector_maximum = (
                scenario_data.scenario['voltage_per_unit_maximum']
                * np.abs(self.electric_grid_model.node_voltage_vector_reference)
                if pd.notnull(scenario_data.scenario['voltage_per_unit_maximum'])
                else None
            )
            branch_power_magnitude_vector_maximum = (
                scenario_data.scenario['branch_flow_per_unit_maximum']
                * self.electric_grid_model.branch_power_vector_magnitude_reference
                if pd.notnull(scenario_data.scenario['branch_flow_per_unit_maximum'])
                else None
            )
            self.linear_electric_grid_model_set.define_optimization_parameters(
                self.optimization_problem,
                self.price_data,
                node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
                node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
                branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
            )
            self.linear_electric_grid_model_set.define_optimization_constraints(self.optimization_problem)
            self.linear_electric_grid_model_set.define_optimization_objective(self.optimization_problem)

        # Define thermal grid problem.
        if self.thermal_grid_model is not None:
            self.linear_thermal_grid_model_set.define_optimization_variables(self.optimization_problem)
            node_head_vector_minimum = (
                scenario_data.scenario['node_head_per_unit_maximum']
                * self.thermal_power_flow_solution_reference.node_head_vector
                if pd.notnull(scenario_data.scenario['voltage_per_unit_maximum'])
                else None
            )
            branch_flow_vector_maximum = (
                scenario_data.scenario['pipe_flow_per_unit_maximum']
                * self.thermal_power_flow_solution_reference.branch_flow_vector
                if pd.notnull(scenario_data.scenario['pipe_flow_per_unit_maximum'])
                else None
            )
            self.linear_thermal_grid_model_set.define_optimization_parameters(
                self.optimization_problem,
                self.price_data,
                node_head_vector_minimum=node_head_vector_minimum,
                branch_flow_vector_maximum=branch_flow_vector_maximum
            )
            self.linear_thermal_grid_model_set.define_optimization_constraints(self.optimization_problem)
            self.linear_thermal_grid_model_set.define_optimization_objective(self.optimization_problem)

        # Define DER problem.
        self.der_model_set.define_optimization_variables(self.optimization_problem)
        self.der_model_set.define_optimization_parameters(self.optimization_problem, self.price_data)
        self.der_model_set.define_optimization_constraints(self.optimization_problem)
        self.der_model_set.define_optimization_objective(self.optimization_problem)

    def solve(self):

        # Solve optimization problem.
        self.optimization_problem.solve()

    def get_results(self) -> Results:

        # Instantiate results.
        self.results = (Results(price_data=self.price_data))

        # Obtain electric grid results.
        if self.electric_grid_model is not None:
            self.results.update(self.linear_electric_grid_model_set.get_optimization_results(self.optimization_problem))

        # Obtain thermal grid results.
        if self.thermal_grid_model is not None:
            self.results.update(self.linear_thermal_grid_model_set.get_optimization_results(self.optimization_problem))

        # Obtain DER results.
        self.results.update(self.der_model_set.get_optimization_results(self.optimization_problem))

        # Obtain electric DLMPs.
        if self.electric_grid_model is not None:
            self.results.update(
                self.linear_electric_grid_model_set.get_optimization_dlmps(
                    self.optimization_problem,
                    self.price_data
                )
            )

        # Obtain thermal DLMPs.
        if self.thermal_grid_model is not None:
            self.results.update(
                self.linear_thermal_grid_model_set.get_optimization_dlmps(
                    self.optimization_problem,
                    self.price_data
                )
            )

        return self.results
