"""Test template."""

import unittest

import fledge

logger = fledge.config.get_logger(__name__)


class TestTemplate(unittest.TestCase):

    def test_equal(self):
        # Define expected result.
        expected = 4

        # Get actual result.
        fledge.utils.log_time("test_equal", log_level='info', logger_object=logger)
        actual = 2 + 2
        fledge.utils.log_time("test_equal", log_level='info', logger_object=logger)

        # Compare expected and actual.
        self.assertEqual(actual, expected)

    def test_not_equal(self):
        # Define expected result.
        expected = 4

        # Get actual result.
        fledge.utils.log_time("test_not_equal", log_level='info', logger_object=logger)
        actual = 2 + 1
        fledge.utils.log_time("test_not_equal", log_level='info', logger_object=logger)

        # Compare expected and actual.
        self.assertNotEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
