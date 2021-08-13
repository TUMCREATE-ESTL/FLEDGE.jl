"""Test electric grid models."""

import unittest

import mesmo

logger = mesmo.config.get_logger(__name__)


class TestElectricGridModels(unittest.TestCase):

    def test_electric_grid_model_default(self):
        # Get result.
        mesmo.utils.log_time("test_electric_grid_model_default", log_level='info', logger_object=logger)
        mesmo.electric_grid_models.ElectricGridModelDefault(mesmo.config.config['tests']['scenario_name'])
        mesmo.utils.log_time("test_electric_grid_model_default", log_level='info', logger_object=logger)

    def test_electric_grid_model_opendss(self):
        # Get result.
        mesmo.utils.log_time("test_electric_grid_model_opendss", log_level='info', logger_object=logger)
        mesmo.electric_grid_models.ElectricGridModelOpenDSS(mesmo.config.config['tests']['scenario_name'])
        mesmo.utils.log_time("test_electric_grid_model_opendss", log_level='info', logger_object=logger)

    def test_power_flow_solution_fixed_point(self):
        # Get result.
        mesmo.utils.log_time("test_power_flow_solution_fixed_point", log_level='info', logger_object=logger)
        mesmo.electric_grid_models.PowerFlowSolutionFixedPoint(mesmo.config.config['tests']['scenario_name'])
        mesmo.utils.log_time("test_power_flow_solution_fixed_point", log_level='info', logger_object=logger)

    def test_power_flow_solution_z_bus(self):
        # Get result.
        mesmo.utils.log_time("test_power_flow_solution_z_bus", log_level='info', logger_object=logger)
        mesmo.electric_grid_models.PowerFlowSolutionZBus(mesmo.config.config['tests']['scenario_name'])
        mesmo.utils.log_time("test_power_flow_solution_z_bus", log_level='info', logger_object=logger)

    def test_power_flow_solution_opendss(self):
        # Get result.
        mesmo.utils.log_time("test_power_flow_solution_opendss", log_level='info', logger_object=logger)
        mesmo.electric_grid_models.PowerFlowSolutionOpenDSS(mesmo.config.config['tests']['scenario_name'])
        mesmo.utils.log_time("test_power_flow_solution_opendss", log_level='info', logger_object=logger)

    def test_linear_electric_grid_model_global(self):
        # Get result.
        mesmo.utils.log_time("test_linear_electric_grid_model_global", log_level='info', logger_object=logger)
        mesmo.electric_grid_models.LinearElectricGridModelGlobal(mesmo.config.config['tests']['scenario_name'])
        mesmo.utils.log_time("test_linear_electric_grid_model_global", log_level='info', logger_object=logger)


if __name__ == '__main__':
    unittest.main()
