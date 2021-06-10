"""Problems module for mathematical optimization and simulation problem type definitions."""

import itertools
from multimethod import multimethod
import numpy as np
import pandas as pd

import fledge.config
import fledge.data_interface
import fledge.der_models
import fledge.electric_grid_models
import fledge.thermal_grid_models
import fledge.utils

logger = fledge.config.get_logger(__name__)


class Results(
    fledge.electric_grid_models.ElectricGridOperationResults,
    fledge.thermal_grid_models.ThermalGridOperationResults,
    fledge.der_models.DERModelSetOperationResults,
    fledge.electric_grid_models.ElectricGridDLMPResults,
    fledge.thermal_grid_models.ThermalGridDLMPResults
):

    price_data: fledge.data_interface.PriceData


class NominalOperationProblem(object):
    """Nominal operation problem object, consisting of the corresponding electric / thermal grid models,
    reference power flow solutions and DER model set for the given scenario.

    - The nominal operation problem (alias: simulation problem, power flow problem)
      represents the simulation problem of the DERs and grids considering the nominal operation schedule for all DERs.
    - The problem formulation is able to consider combined as well as individual operation of
      thermal and electric grids.
    """

    scenario_name: str
    timesteps: pd.Index
    price_data: fledge.data_interface.PriceData
    electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault = None
    power_flow_solution_reference: fledge.electric_grid_models.PowerFlowSolution = None
    thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel = None
    thermal_power_flow_solution_reference: fledge.thermal_grid_models.ThermalPowerFlowSolution = None
    der_model_set: fledge.der_models.DERModelSet
    results: Results

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain data.
        scenario_data = fledge.data_interface.ScenarioData(scenario_name)
        self.price_data = fledge.data_interface.PriceData(scenario_name)

        # Store timesteps.
        self.timesteps = scenario_data.timesteps

        # Obtain electric grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['electric_grid_name']):
            fledge.utils.log_time("electric grid model instantiation")
            self.electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
            self.power_flow_solution_reference = (
                fledge.electric_grid_models.PowerFlowSolutionFixedPoint(self.electric_grid_model)
            )
            fledge.utils.log_time("electric grid model instantiation")

        # Obtain thermal grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['thermal_grid_name']):
            fledge.utils.log_time("thermal grid model instantiation")
            self.thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
            self.thermal_power_flow_solution_reference = (
                fledge.thermal_grid_models.ThermalPowerFlowSolution(self.thermal_grid_model)
            )
            fledge.utils.log_time("thermal grid model instantiation")

        # Obtain DER model set.
        fledge.utils.log_time("DER model instantiation")
        self.der_model_set = fledge.der_models.DERModelSet(scenario_name)
        fledge.utils.log_time("DER model instantiation")

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
        fledge.utils.log_time("power flow solution")
        if self.electric_grid_model is not None:
            power_flow_solutions = (
                fledge.utils.starmap(
                    fledge.electric_grid_models.PowerFlowSolutionFixedPoint,
                    zip(
                        itertools.repeat(self.electric_grid_model),
                        der_power_vector.values
                    )
                )
            )
            power_flow_solutions = dict(zip(self.timesteps, power_flow_solutions))
        if self.thermal_grid_model is not None:
            thermal_power_flow_solutions = (
                fledge.utils.starmap(
                    fledge.thermal_grid_models.ThermalPowerFlowSolution,
                    [(self.thermal_grid_model, row) for row in der_thermal_power_vector.values]
                )
            )
            thermal_power_flow_solutions = dict(zip(self.timesteps, thermal_power_flow_solutions))
        fledge.utils.log_time("power flow solution")

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


class OptimalOperationProblem(object):
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
    price_data: fledge.data_interface.PriceData
    electric_grid_model: fledge.electric_grid_models.ElectricGridModelDefault = None
    power_flow_solution_reference: fledge.electric_grid_models.PowerFlowSolution = None
    linear_electric_grid_model: fledge.electric_grid_models.LinearElectricGridModel = None
    thermal_grid_model: fledge.thermal_grid_models.ThermalGridModel = None
    thermal_power_flow_solution_reference: fledge.thermal_grid_models.ThermalPowerFlowSolution = None
    linear_thermal_grid_model: fledge.thermal_grid_models.LinearThermalGridModel = None
    der_model_set: fledge.der_models.DERModelSet
    optimization_problem: fledge.utils.OptimizationProblem
    results: Results

    @multimethod
    def __init__(
            self,
            scenario_name: str
    ):

        # Obtain data.
        scenario_data = fledge.data_interface.ScenarioData(scenario_name)
        self.price_data = fledge.data_interface.PriceData(scenario_name)

        # Store timesteps.
        self.timesteps = scenario_data.timesteps

        # Obtain electric grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['electric_grid_name']):
            self.electric_grid_model = fledge.electric_grid_models.ElectricGridModelDefault(scenario_name)
            self.power_flow_solution_reference = (
                fledge.electric_grid_models.PowerFlowSolutionFixedPoint(self.electric_grid_model)
            )
            self.linear_electric_grid_model = (
                fledge.electric_grid_models.LinearElectricGridModelGlobal(
                    self.electric_grid_model,
                    self.power_flow_solution_reference
                )
            )

        # Obtain thermal grid model, power flow solution and linear model, if defined.
        if pd.notnull(scenario_data.scenario.at['thermal_grid_name']):
            self.thermal_grid_model = fledge.thermal_grid_models.ThermalGridModel(scenario_name)
            self.thermal_grid_model.energy_transfer_station_head_loss = 0.0  # TODO: Remove modifications.
            self.thermal_grid_model.cooling_plant_efficiency = 10.0  # TODO: Remove modifications.
            self.thermal_power_flow_solution_reference = (
                fledge.thermal_grid_models.ThermalPowerFlowSolution(self.thermal_grid_model)
            )
            self.linear_thermal_grid_model = (
                fledge.thermal_grid_models.LinearThermalGridModel(
                    self.thermal_grid_model,
                    self.thermal_power_flow_solution_reference
                )
            )

        # Obtain DER model set.
        self.der_model_set = fledge.der_models.DERModelSet(scenario_name)

        # Instantiate optimization problem.
        self.optimization_problem = fledge.utils.OptimizationProblem()

        # Define linear electric grid model variables and constraints.
        if self.electric_grid_model is not None:
            self.linear_electric_grid_model.define_optimization_variables(self.optimization_problem)
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
            self.linear_electric_grid_model.define_optimization_constraints(
                self.optimization_problem,
                node_voltage_magnitude_vector_minimum=node_voltage_magnitude_vector_minimum,
                node_voltage_magnitude_vector_maximum=node_voltage_magnitude_vector_maximum,
                branch_power_magnitude_vector_maximum=branch_power_magnitude_vector_maximum
            )

        # Define thermal grid model variables and constraints.
        if self.thermal_grid_model is not None:
            self.linear_thermal_grid_model.define_optimization_variables(self.optimization_problem)
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
            self.linear_thermal_grid_model.define_optimization_constraints(
                self.optimization_problem,
                node_head_vector_minimum=node_head_vector_minimum,
                branch_flow_vector_maximum=branch_flow_vector_maximum
            )

        # Define DER variables and constraints.
        self.der_model_set.define_optimization_variables(self.optimization_problem)
        self.der_model_set.define_optimization_constraints(self.optimization_problem)

        # Define objective.
        if self.thermal_grid_model is not None:
            self.linear_thermal_grid_model.define_optimization_objective(
                self.optimization_problem,
                self.price_data
            )
        if self.electric_grid_model is not None:
            self.linear_electric_grid_model.define_optimization_objective(
                self.optimization_problem,
                self.price_data
            )
        self.der_model_set.define_optimization_objective(
            self.optimization_problem,
            self.price_data
        )

    def solve(self):

        # Solve optimization problem.
        self.optimization_problem.solve()

    def get_results(self) -> Results:

        # Instantiate results.
        self.results = (Results(price_data=self.price_data))

        # Obtain electric grid results.
        if self.electric_grid_model is not None:
            self.results.update(self.linear_electric_grid_model.get_optimization_results(self.optimization_problem))

        # Obtain thermal grid results.
        if self.thermal_grid_model is not None:
            self.results.update(self.linear_thermal_grid_model.get_optimization_results(self.optimization_problem))

        # Obtain DER results.
        self.results.update(self.der_model_set.get_optimization_results(self.optimization_problem))

        # Obtain electric DLMPs.
        if self.electric_grid_model is not None:
            self.results.update(
                self.linear_electric_grid_model.get_optimization_dlmps(
                    self.optimization_problem,
                    self.price_data
                )
            )

        # Obtain thermal DLMPs.
        if self.thermal_grid_model is not None:
            self.results.update(
                self.linear_thermal_grid_model.get_optimization_dlmps(
                    self.optimization_problem,
                    self.price_data
                )
            )

        return self.results
