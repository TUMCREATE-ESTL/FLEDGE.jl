"""Test electric grid models."""

import unittest

import fledge

logger = fledge.config.get_logger(__name__)


class TestElectricGridModels(unittest.TestCase):

    def test_electric_grid_model_default(self):
        # Get result.
        fledge.utils.log_time("test_electric_grid_model_default", log_level='info', logger_object=logger)
        fledge.electric_grid_models.ElectricGridModelDefault(fledge.config.config['tests']['scenario_name'])
        fledge.utils.log_time("test_electric_grid_model_default", log_level='info', logger_object=logger)

    def test_electric_grid_model_opendss(self):
        # Get result.
        fledge.utils.log_time("test_electric_grid_model_opendss", log_level='info', logger_object=logger)
        fledge.electric_grid_models.ElectricGridModelOpenDSS(fledge.config.config['tests']['scenario_name'])
        fledge.utils.log_time("test_electric_grid_model_opendss", log_level='info', logger_object=logger)

    def test_power_flow_solution_fixed_point(self):
        # Get result.
        fledge.utils.log_time("test_power_flow_solution_fixed_point", log_level='info', logger_object=logger)
        fledge.electric_grid_models.PowerFlowSolutionFixedPoint(fledge.config.config['tests']['scenario_name'])
        fledge.utils.log_time("test_power_flow_solution_fixed_point", log_level='info', logger_object=logger)

    def test_power_flow_solution_z_bus(self):
        # Get result.
        fledge.utils.log_time("test_power_flow_solution_z_bus", log_level='info', logger_object=logger)
        fledge.electric_grid_models.PowerFlowSolutionZBus(fledge.config.config['tests']['scenario_name'])
        fledge.utils.log_time("test_power_flow_solution_z_bus", log_level='info', logger_object=logger)

    def test_power_flow_solution_opendss(self):
        # Get result.
        fledge.utils.log_time("test_power_flow_solution_opendss", log_level='info', logger_object=logger)
        fledge.electric_grid_models.PowerFlowSolutionOpenDSS(fledge.config.config['tests']['scenario_name'])
        fledge.utils.log_time("test_power_flow_solution_opendss", log_level='info', logger_object=logger)

    def test_linear_electric_grid_model_global(self):
        # Get result.
        fledge.utils.log_time("test_linear_electric_grid_model_global", log_level='info', logger_object=logger)
        fledge.electric_grid_models.LinearElectricGridModelGlobal(fledge.config.config['tests']['scenario_name'])
        fledge.utils.log_time("test_linear_electric_grid_model_global", log_level='info', logger_object=logger)


if __name__ == '__main__':
    unittest.main()
