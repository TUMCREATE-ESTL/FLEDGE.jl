"""Test plots."""

import time
import unittest

import fledge

logger = fledge.config.get_logger(__name__)


class TestPlots(unittest.TestCase):

    def test_electric_grid_graph(self):

        time_start = time.time()
        fledge.plots.ElectricGridGraph(fledge.config.config['tests']['scenario_name'])
        time_duration = time.time() - time_start
        logger.info(f"Test ElectricGridGraph: Completed in {time_duration:.6f} seconds.")

    def test_thermal_grid_graph(self):

        time_start = time.time()
        fledge.plots.ThermalGridGraph('singapore_tanjongpagar')
        time_duration = time.time() - time_start
        logger.info(f"Test ThermalGridGraph: Completed in {time_duration:.6f} seconds.")
