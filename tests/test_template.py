"""Test template."""

import unittest

import mesmo

logger = mesmo.config.get_logger(__name__)


class TestTemplate(unittest.TestCase):

    def test_equal(self):
        # Define expected result.
        expected = 4

        # Get actual result.
        mesmo.utils.log_time("test_equal", log_level='info', logger_object=logger)
        actual = 2 + 2
        mesmo.utils.log_time("test_equal", log_level='info', logger_object=logger)

        # Compare expected and actual.
        self.assertEqual(actual, expected)

    def test_not_equal(self):
        # Define expected result.
        expected = 4

        # Get actual result.
        mesmo.utils.log_time("test_not_equal", log_level='info', logger_object=logger)
        actual = 2 + 1
        mesmo.utils.log_time("test_not_equal", log_level='info', logger_object=logger)

        # Compare expected and actual.
        self.assertNotEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
