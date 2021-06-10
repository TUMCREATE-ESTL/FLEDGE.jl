"""Test API."""

import cvxpy as cp
import unittest

import fledge

logger = fledge.config.get_logger(__name__)


class TestAPI(unittest.TestCase):

    def test_run_nominal_operation_problem(self):
        # Get result.
        fledge.utils.log_time("test_run_nominal_operation_problem", log_level='info', logger_object=logger)
        fledge.api.run_nominal_operation_problem('singapore_tanjongpagar')
        fledge.utils.log_time("test_run_nominal_operation_problem", log_level='info', logger_object=logger)

    def test_run_optimal_operation_problem(self):
        # Get result.
        fledge.utils.log_time("test_run_optimal_operation_problem", log_level='info', logger_object=logger)
        try:
            fledge.api.run_optimal_operation_problem('singapore_tanjongpagar')
        except (cp.SolverError, AttributeError):
            # Soft fail: Only raise warning on selected errors, since it may be due to solver not installed.
            logger.warning(f"Test run_optimal_operation_problem failed due to solver error.", exc_info=True)
        fledge.utils.log_time("test_run_optimal_operation_problem", log_level='info', logger_object=logger)
