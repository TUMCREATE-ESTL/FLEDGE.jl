"""Test script for thermal grid models module."""

import unittest

import mesmo

logger = mesmo.config.get_logger(__name__)


class TestThermalGridModels(unittest.TestCase):

    def test_thermal_grid_model(self):
        # Get result.
        mesmo.utils.log_time("test_thermal_grid_model", log_level='info', logger_object=logger)
        mesmo.thermal_grid_models.ThermalGridModel('singapore_tanjongpagar')
        mesmo.utils.log_time("test_thermal_grid_model", log_level='info', logger_object=logger)

    def test_thermal_power_flow_solution(self):
        # Get result.
        mesmo.utils.log_time("test_thermal_power_flow_solution", log_level='info', logger_object=logger)
        mesmo.thermal_grid_models.ThermalPowerFlowSolution('singapore_tanjongpagar')
        mesmo.utils.log_time("test_thermal_power_flow_solution", log_level='info', logger_object=logger)


if __name__ == '__main__':
    unittest.main()
