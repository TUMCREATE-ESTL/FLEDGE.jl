"""Test script for thermal grid models module."""

import unittest

import fledge

logger = fledge.config.get_logger(__name__)


class TestThermalGridModels(unittest.TestCase):

    def test_thermal_grid_model(self):
        # Get result.
        fledge.utils.log_time("test_thermal_grid_model", log_level='info', logger_object=logger)
        fledge.thermal_grid_models.ThermalGridModel('singapore_tanjongpagar')
        fledge.utils.log_time("test_thermal_grid_model", log_level='info', logger_object=logger)

    def test_thermal_power_flow_solution(self):
        # Get result.
        fledge.utils.log_time("test_thermal_power_flow_solution", log_level='info', logger_object=logger)
        fledge.thermal_grid_models.ThermalPowerFlowSolution('singapore_tanjongpagar')
        fledge.utils.log_time("test_thermal_power_flow_solution", log_level='info', logger_object=logger)


if __name__ == '__main__':
    unittest.main()
