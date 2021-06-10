"""Test plots."""

import unittest

import fledge

logger = fledge.config.get_logger(__name__)


class TestPlots(unittest.TestCase):

    def test_electric_grid_graph(self):
        # Get result.
        fledge.utils.log_time("test_electric_grid_graph", log_level='info', logger_object=logger)
        fledge.plots.ElectricGridGraph(fledge.config.config['tests']['scenario_name'])
        fledge.utils.log_time("test_electric_grid_graph", log_level='info', logger_object=logger)

    def test_thermal_grid_graph(self):
        # Get result.
        fledge.utils.log_time("test_thermal_grid_graph", log_level='info', logger_object=logger)
        fledge.plots.ThermalGridGraph('singapore_tanjongpagar')
        fledge.utils.log_time("test_thermal_grid_graph", log_level='info', logger_object=logger)
