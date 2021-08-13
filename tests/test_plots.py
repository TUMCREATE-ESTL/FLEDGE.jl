"""Test plots."""

import unittest

import mesmo

logger = mesmo.config.get_logger(__name__)


class TestPlots(unittest.TestCase):

    def test_electric_grid_graph(self):
        # Get result.
        mesmo.utils.log_time("test_electric_grid_graph", log_level='info', logger_object=logger)
        mesmo.plots.ElectricGridGraph(mesmo.config.config['tests']['scenario_name'])
        mesmo.utils.log_time("test_electric_grid_graph", log_level='info', logger_object=logger)

    def test_thermal_grid_graph(self):
        # Get result.
        mesmo.utils.log_time("test_thermal_grid_graph", log_level='info', logger_object=logger)
        mesmo.plots.ThermalGridGraph('singapore_tanjongpagar')
        mesmo.utils.log_time("test_thermal_grid_graph", log_level='info', logger_object=logger)
