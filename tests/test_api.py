"""Test API."""

import gurobipy as gp
import unittest

import mesmo

logger = mesmo.config.get_logger(__name__)


class TestAPI(unittest.TestCase):

    def test_run_nominal_operation_problem(self):
        # Get result.
        mesmo.utils.log_time("test_run_nominal_operation_problem", log_level='info', logger_object=logger)
        mesmo.api.run_nominal_operation_problem('singapore_tanjongpagar')
        mesmo.utils.log_time("test_run_nominal_operation_problem", log_level='info', logger_object=logger)

    def test_run_optimal_operation_problem(self):
        # Get result.
        mesmo.utils.log_time("test_run_optimal_operation_problem", log_level='info', logger_object=logger)
        try:
            mesmo.api.run_optimal_operation_problem('singapore_tanjongpagar')
        except gp.GurobiError:
            # Soft fail: Only raise warning on selected errors, since it may be due to solver not installed.
            logger.warning(f"Test run_optimal_operation_problem failed due to solver error.", exc_info=True)
        mesmo.utils.log_time("test_run_optimal_operation_problem", log_level='info', logger_object=logger)
