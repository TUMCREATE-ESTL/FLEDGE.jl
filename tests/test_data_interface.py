"""Test database interface."""

import sqlite3
import unittest

import mesmo

logger = mesmo.config.get_logger(__name__)


class TestDatabaseInterface(unittest.TestCase):

    def test_recreate_database(self):
        # Get result.
        mesmo.utils.log_time("test_recreate_database", log_level='info', logger_object=logger)
        mesmo.data_interface.recreate_database()
        mesmo.utils.log_time("test_recreate_database", log_level='info', logger_object=logger)

    def test_connect_database(self):
        # Define expected result.
        expected = sqlite3.dbapi2.Connection

        # Get actual result.
        mesmo.utils.log_time("test_connect_database", log_level='info', logger_object=logger)
        actual = type(mesmo.data_interface.connect_database())
        mesmo.utils.log_time("test_connect_database", log_level='info', logger_object=logger)

        # Compare expected and actual.
        self.assertEqual(actual, expected)

    def test_scenario_data(self):
        # Get result.
        mesmo.utils.log_time("test_scenario_data", log_level='info', logger_object=logger)
        mesmo.data_interface.ScenarioData(mesmo.config.config['tests']['scenario_name'])
        mesmo.utils.log_time("test_scenario_data", log_level='info', logger_object=logger)

    def test_electric_grid_data(self):
        # Get result.
        mesmo.utils.log_time("test_electric_grid_data", log_level='info', logger_object=logger)
        mesmo.data_interface.ElectricGridData(mesmo.config.config['tests']['scenario_name'])
        mesmo.utils.log_time("test_electric_grid_data", log_level='info', logger_object=logger)

    def test_thermal_grid_data(self):
        # Get result.
        mesmo.utils.log_time("test_thermal_grid_data", log_level='info', logger_object=logger)
        mesmo.data_interface.ThermalGridData('singapore_tanjongpagar')
        mesmo.utils.log_time("test_thermal_grid_data", log_level='info', logger_object=logger)

    def test_electric_grid_der_data(self):
        # Get result.
        mesmo.utils.log_time("test_electric_grid_der_data", log_level='info', logger_object=logger)
        mesmo.data_interface.DERData(mesmo.config.config['tests']['scenario_name'])
        mesmo.utils.log_time("test_electric_grid_der_data", log_level='info', logger_object=logger)

    def test_price_data(self):
        # Get result.
        mesmo.utils.log_time("test_price_data", log_level='info', logger_object=logger)
        mesmo.data_interface.PriceData(mesmo.config.config['tests']['scenario_name'])
        mesmo.utils.log_time("test_price_data", log_level='info', logger_object=logger)


if __name__ == '__main__':
    unittest.main()
