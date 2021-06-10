"""Setup script."""

import os
import setuptools
import subprocess
import sys

submodules = [
    'cobmo'
]

# Check if submodules are loaded.
for submodule in submodules:
    if not os.path.exists(os.path.join(submodule, 'setup.py')):
        raise FileNotFoundError(
            f"No setup file found for submodule `{submodule}`. "
            "Please check if the submodule is loaded correctly."
        )

# Install submodules. Use `pip -v` to see subprocess outputs.
for submodule in submodules:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', submodule])

# Install Gurobi interface. Use `pip -v` to see subprocess outputs.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-i', 'https://pypi.gurobi.com', 'gurobipy'])

setuptools.setup(
    name='fledge',
    version='0.4.1',
    py_modules=setuptools.find_packages(),
    install_requires=[
        # Please note: Dependencies must also be added in `docs/conf.py` to `autodoc_mock_imports`.
        'cvxpy',
        'dill',
        'diskcache',
        'kaleido',  # For static plot output with plotly.
        'matplotlib',
        'multimethod',
        'networkx',
        'natsort',
        'numpy',
        'opencv-python',
        'OpenDSSDirect.py',
        'pandas',
        'parameterized',  # For tests.
        'plotly',
        'pyyaml',
        'ray',
        'scipy',
    ]
)
