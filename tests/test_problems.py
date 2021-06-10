"""Test problems."""

import unittest

import fledge

logger = fledge.config.get_logger(__name__)


class TestProblems(unittest.TestCase):

    def test_nominal_operation_problem(self):
        # Get result.
        fledge.utils.log_time("test_nominal_operation_problem", log_level='info', logger_object=logger)
        fledge.problems.NominalOperationProblem('singapore_tanjongpagar')
        fledge.utils.log_time("test_nominal_operation_problem", log_level='info', logger_object=logger)

    def test_optimal_operation_problem(self):
        # Get result.
        fledge.utils.log_time("test_optimal_operation_problem", log_level='info', logger_object=logger)
        fledge.problems.OptimalOperationProblem('singapore_tanjongpagar')
        fledge.utils.log_time("test_optimal_operation_problem", log_level='info', logger_object=logger)
