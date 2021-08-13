"""Run script which executes all example scripts, for testing."""

import glob
import os
import subprocess
import sys

import mesmo


def main():

    # Obtain example scripts.
    examples_path = os.path.join(mesmo.config.base_path, 'examples')
    example_scripts = glob.glob(os.path.join(examples_path, '*.py'))

    # Run all example scripts.
    for example_script in example_scripts:
        mesmo.utils.log_time(example_script, log_level='info')
        subprocess.check_call([sys.executable, os.path.join(examples_path, example_script)])
        mesmo.utils.log_time(example_script, log_level='info')


if __name__ == '__main__':
    main()
