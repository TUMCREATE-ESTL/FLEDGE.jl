"""Test database interface."""

import sqlite3
import unittest

import fledge

logger = fledge.config.get_logger(__name__)


class TestDatabaseInterface(unittest.TestCase):

    def test_recreate_database(self):
        # Get result.
        fledge.utils.log_time("test_recreate_database", log_level='info', logger_object=logger)
        fledge.data_interface.recreate_database()
        fledge.utils.log_time("test_recreate_database", log_level='info', logger_object=logger)

    def test_connect_database(self):
        # Define expected result.
        expected = sqlite3.dbapi2.Connection

        # Get actual result.
        fledge.utils.log_time("test_connect_database", log_level='info', logger_object=logger)
        actual = type(fledge.data_interface.connect_database())
        fledge.utils.log_time("test_connect_database", log_level='info', logger_object=logger)

        # Compare expected and actual.
        self.assertEqual(actual, expected)

    def test_scenario_data(self):
        # Get result.
        fledge.utils.log_time("test_scenario_data", log_level='info', logger_object=logger)
        fledge.data_interface.ScenarioData(fledge.config.config['tests']['scenario_name'])
        fledge.utils.log_time("test_scenario_data", log_level='info', logger_object=logger)

    def test_electric_grid_data(self):
        # Get result.
        fledge.utils.log_time("test_electric_grid_data", log_level='info', logger_object=logger)
        fledge.data_interface.ElectricGridData(fledge.config.config['tests']['scenario_name'])
        fledge.utils.log_time("test_electric_grid_data", log_level='info', logger_object=logger)

    def test_thermal_grid_data(self):
        # Get result.
        fledge.utils.log_time("test_thermal_grid_data", log_level='info', logger_object=logger)
        fledge.data_interface.ThermalGridData('singapore_tanjongpagar')
        fledge.utils.log_time("test_thermal_grid_data", log_level='info', logger_object=logger)

    def test_electric_grid_der_data(self):
        # Get result.
        fledge.utils.log_time("test_electric_grid_der_data", log_level='info', logger_object=logger)
        fledge.data_interface.DERData(fledge.config.config['tests']['scenario_name'])
        fledge.utils.log_time("test_electric_grid_der_data", log_level='info', logger_object=logger)

    def test_price_data(self):
        # Get result.
        fledge.utils.log_time("test_price_data", log_level='info', logger_object=logger)
        fledge.data_interface.PriceData(fledge.config.config['tests']['scenario_name'])
        fledge.utils.log_time("test_price_data", log_level='info', logger_object=logger)


if __name__ == '__main__':
    unittest.main()
