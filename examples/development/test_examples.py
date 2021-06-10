"""Run script which executes all example scripts, for testing."""

import glob
import os
import subprocess
import sys

import fledge


def main():

    # Obtain example scripts.
    examples_path = os.path.join(fledge.config.base_path, 'examples')
    example_scripts = glob.glob(os.path.join(examples_path, '*.py'))

    # Run all example scripts.
    for example_script in example_scripts:
        fledge.utils.log_time(example_script, log_level='info')
        subprocess.check_call([sys.executable, os.path.join(examples_path, example_script)])
        fledge.utils.log_time(example_script, log_level='info')


if __name__ == '__main__':
    main()
