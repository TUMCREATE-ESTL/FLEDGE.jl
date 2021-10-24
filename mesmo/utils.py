"""Utility functions module."""

import collections
import copy
import cvxpy as cp
import gurobipy as gp
import datetime
import functools
import glob
import itertools
import logging
import numpy as np
import os
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.io as pio
import ray
import re
import time
import tqdm
import typing
import scipy.sparse as sp
import subprocess
import sys

import mesmo.config

logger = mesmo.config.get_logger(__name__)

# Instantiate dictionary for execution time logging.
log_times = dict()


class ObjectBase(object):
    """MESMO object base class, which extends the Python object base class.

    - Requires all attributes, i.e. parameters or object variables, to be defined with type declaration at the
      beginning of the class definition. Setting a value to an attribute which has not been defined will raise
      a warning. This is to ensure consistent definition structure of MESMO classes.
    - String representation of the object is the concatenation of the string representation of all its attributes.
      Thus, printing the object will print all its attributes.

    Example:

        Attributes should be defined in the beginning of the class definition as follows::

            class ExampleClass(ObjectBase):

                example_attribute1: str
                example_attribute2: pd.DataFrame

        In this case, ``example_attribute1`` and ``example_attribute2`` are valid attributes of the class.
    """

    def __setattr__(
            self,
            attribute_name,
            value
    ):

        # Assert that attribute name is valid.
        # - Valid attributes are those which are defined as results class attributes with type declaration.
        if not (attribute_name in typing.get_type_hints(type(self))):
            logger.warning(
                f"Setting undefined attribute '{attribute_name}'. "
                f"Please ensure that the attribute has been defined by a type declaration in the class definition."
            )

        # Set attribute value.
        super().__setattr__(attribute_name, value)

    def __repr__(self) -> str:
        """Obtain string representation."""

        # Obtain attributes.
        attributes = vars(self)

        # Obtain representation string.
        repr_string = ""
        for attribute_name in attributes:
            repr_string += f"{attribute_name} = \n{attributes[attribute_name]}\n"

        return repr_string

    def copy(self):
        """Return a copy of this object. A new object will be created with a copy of the calling object’s attributes.
        Modifications to the attributes of the copy will not be reflected in the original object.
        """

        return copy.deepcopy(self)


class ResultsBase(ObjectBase):
    """Results object base class."""

    def __init__(
            self,
            **kwargs
    ):

        # Set all keyword arguments as attributes.
        for attribute_name in kwargs:
            self.__setattr__(attribute_name, kwargs[attribute_name])

    def __getitem__(self, key):
        # Enable dict-like attribute getting.
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        # Enable dict-like attribute setting.
        self.__setattr__(key, value)

    def update(
            self,
            other_results
    ):

        # Obtain attributes of other results object.
        attributes = vars(other_results)

        # Update attributes.
        # - Existing attributes are overwritten with values from the other results object.
        for attribute_name in attributes:
            if attributes[attribute_name] is not None:
                self.__setattr__(attribute_name, attributes[attribute_name])

    def save(
            self,
            results_path: str
    ):
        """Store results to files at given results path.

        - Each results variable / attribute will be stored as separate file with the attribute name as file name.
        - Pandas Series / DataFrame are stored to CSV.
        - Other objects are stored to pickle binary file (PKL).
        """

        # Obtain results attributes.
        attributes = vars(self)

        # Store each attribute to a separate file.
        for attribute_name in attributes:
            if type(attributes[attribute_name]) in (pd.Series, pd.DataFrame):
                # Pandas Series / DataFrame are stored to CSV.
                attributes[attribute_name].to_csv(os.path.join(results_path, f'{attribute_name}.csv'))
            else:
                # Other objects are stored to pickle binary file (PKL).
                with open(os.path.join(results_path, f'{attribute_name}.pkl'), 'wb') as output_file:
                    pickle.dump(attributes[attribute_name], output_file, pickle.HIGHEST_PROTOCOL)

    def load(
            self,
            results_path: str
    ):
        """Load results from given path."""

        # Obtain all CSV and PKL files at results path.
        files = glob.glob(os.path.join(results_path, '*.csv')) + glob.glob(os.path.join(results_path, '*.pkl'))

        # Load all files which correspond to valid attributes.
        for file in files:

            # Obtain file extension / attribute name.
            file_extension = os.path.splitext(file)[1]
            attribute_name = os.path.basename(os.path.splitext(file)[0])

            # Load file and set attribute value.
            if attribute_name in typing.get_type_hints(type(self)):
                if file_extension.lower() == '.csv':
                    value = pd.read_csv(file)
                else:
                    with open(file, 'rb') as input_file:
                        value = pickle.load(input_file)
                self.__setattr__(attribute_name, value)
            else:
                # Files which do not match any valid results attribute are not loaded.
                logger.debug(f"Skipping results file which does match any valid results attribute: {file}")

        return self


class OptimizationProblem(ObjectBase):
    r"""Optimization problem object class, which allows the definition and solution of convex optimization problems.
    The optimization problem object serves as a container for the parameters, variables, constraints and objective
    terms. The object provides methods for defining variables, parameters, constraints and objectives as well as
    methods for solving the numerical optimization problem and obtaining results for variables, objective value and
    dual values.

    - This documentation assumes a fundamental understanding of convex optimization. As a general reference
      on this topic, refer to: *S. P. Boyd and L. Vandenberghe, Convex optimization. Cambridge University Press, 2004.*
      Available at: https://web.stanford.edu/~boyd/cvxbook/
    - The optimization problem object currently supports convex optimization problems in the form of
      1) linear program (LP) or 2) quadratic program (QP) with only linear constraints.
    - The solve method currently implements interfaces to 1) Gurobi and 2) CVXPY, where the latter is a high-level
      convex optimization interface, which in turn allows interfacing further third-party solvers. The intention is
      to implement more direct solver interfaces on a as-need basis (please raise an issue!), as these interfaces are
      assumed to allow higher performance than CVXPY for large-scale problems. However, CVXPY is kept as a fallback to
      allow a high degree of compatibility with various solvers.

    The optimization problem object internally translates optimizations into LP / QP standard form. Where the following
    formulation is assumed for the standard form:

    .. math::
        \begin{align}
            \min_{\boldsymbol{x}} \quad
            & \boldsymbol{c}^{\intercal} \boldsymbol{x}
            + \frac{1}{2} \boldsymbol{x}^{\intercal} \boldsymbol{Q} \boldsymbol{x} + d \\
            \text{s.t.} \quad
            & \boldsymbol{A} \boldsymbol{x} \leq \boldsymbol{b} \quad : \ \boldsymbol{\mu}
        \end{align}

    The vectors :math:`\boldsymbol{x}` and :math:`\boldsymbol{\mu}` are the variable vector and
    associated constraint dual variable vector. The matrix :math:`\boldsymbol{A}` defines the linear
    constraint coefficients, whereas the matrix :math:`\boldsymbol{Q}` defines quadradtic objective coefficients.
    The vectors :math:`\boldsymbol{b}` and :math:`\boldsymbol{c}` define constant constraint terms
    and linear objective coefficients. Lastly, the scalar :math:`d` defines the constant objective term.
    Note that the scalar :math:`d` represents a slight abuse of the standard form to include constant objective term,
    which may prove useful for comparing objective values across different problem definitions.

    Example:

        Consider the following optimization problem:

        .. math::
            \begin{align}
                \min_{\boldsymbol{a},\boldsymbol{b}} \quad
                & \sum_{i=1}^{n=1000} b_i \\
                \text{s.t.} \quad
                & \boldsymbol{b} = \boldsymbol{a} \cdot \boldsymbol{P} \\
                & -10 \leq \boldsymbol{a} \leq +10
            \end{align}

        The matrix :math:`\boldsymbol{P} \in \mathbb{R}^{n \times n}` is an abitrary parameter matrix. The vectors
        :math:`\boldsymbol{a}, \boldsymbol{b} \in \mathbb{R}^{n \times 1}` are decision variable vectors. The symbol
        :math:`n` defines the problem dimension.
        This problem can be defined and solved with the optimization problem interface as follows::

            # Instantiate optimization problem.
            optimization_problem = mesmo.utils.OptimizationProblem()

            # Define optimization parameters.
            optimization_problem.define_parameter('parameter_matrix', parameter_matrix)

            # Define optimization variables.
            optimization_problem.define_variable('a_vector', a_index=range(dimension))
            optimization_problem.define_variable('b_vector', b_index=range(dimension))

            # Define optimization constraints.
            optimization_problem.define_constraint(
                ('variable', 1.0, dict(name='b_vector')),
                '==',
                ('variable', 'parameter_matrix', dict(name='a_vector')),
            )
            optimization_problem.define_constraint(
                ('constant', -10.0),
                '<=',
                ('variable', 1.0, dict(name='a_vector')),
            )
            optimization_problem.define_constraint(
                ('constant', +10.0),
                '>=',
                ('variable', 1.0, dict(name='a_vector')),
            )

            # Define optimization objective.
            optimization_problem.define_objective(('variable', 1.0, dict(name='b_vector')))

            # Solve optimization problem.
            optimization_problem.solve()

            # Obtain results.
            results = optimization_problem.get_results()
            a_vector = results['a_vector']
            b_vector = results['b_vector']

        This example is also available as standalone script at: ``examples/run_general_optimization_problem.py``
    """

    variables: pd.DataFrame
    constraints: pd.DataFrame
    constraints_len: int
    parameters: dict
    flags: dict
    a_dict: dict
    b_dict: dict
    c_dict: dict
    q_dict: dict
    d_dict: dict
    x_vector: np.ndarray
    mu_vector: np.ndarray
    results: dict
    duals: dict
    objective: float

    def __init__(self):

        # Instantiate index sets.
        # - Variables are instantiated with 'name' and 'timestep' keys, but more may be added in ``define_variable()``.
        # - Constraints are instantiated with 'name', 'timestep' and 'constraint_type' keys,
        #   but more may be added in ``define_constraint()``.
        self.variables = pd.DataFrame(columns=['name', 'timestep'])
        self.constraints = pd.DataFrame(columns=['name', 'timestep', 'constraint_type'])
        self.constraints_len = 0

        # Instantiate parameters / flags dictionary.
        self.parameters = dict()
        self.flags = dict()

        # Instantiate A matrix / b vector / c vector / Q matrix / d constant dictionaries.
        # - Final matrix / vector are only created in ``get_a_matrix()``, ``get_b_vector()``, ``get_c_vector()``,
        #   ``get_q_matrix()`` and ``get_d_constant()``.
        # - Uses `defaultdict(list)` to enable more convenient collecting of elements into lists. This avoids
        #   accidental overwriting of dictionary entries.
        self.a_dict = collections.defaultdict(list)
        self.b_dict = collections.defaultdict(list)
        self.c_dict = collections.defaultdict(list)
        self.q_dict = collections.defaultdict(list)
        self.d_dict = collections.defaultdict(list)

    def define_variable(
            self,
            name: str,
            **keys
    ):
        """Define decision variable with given name and key set.

        - Variables are defined by passing a name string and index key sets. The variable dimension is determined by
          the dimension of the index key sets. Accepted key set values are 1) lists, 2) tuples, 3) numpy arrays,
          4) pandas index objects and 5) range objects.
        - If multiple index key sets are passed, the variable dimension is determined as the cartesian product of
          the key sets. However, note that variables always take the shape of column vectors in constraint and
          objective definitions. That means, multiple key sets are not interpreted as array dimensions.
        """

        # Obtain new variables based on ``keys``.
        # - Variable dimensions are constructed based by taking the product of the given key sets.
        new_variables = (
            pd.DataFrame(itertools.product([name], *[
                list(value)
                if type(value) in [pd.MultiIndex, pd.Index, pd.DatetimeIndex, np.ndarray, list, tuple, range]
                else [value]
                for value in keys.values()
            ]), columns=['name', *keys.keys()])
        )
        # Add new variables to index.
        # - Duplicate definitions are automatically removed.
        self.variables = (
            pd.concat([self.variables, new_variables], ignore_index=True).drop_duplicates(ignore_index=True)
        )

    def define_parameter(
            self,
            name: str,
            value: typing.Union[float, np.ndarray, sp.spmatrix]
    ):
        """Define constant parameters with given name and numerical value.

        - Numerical values can be numerical value can be real-valued 1) float, 2) numpy array and
          3) scipy sparse matrix.
        - Defining parameters is optional. – Numerical values can also be directly passed in the constraints /
          objective definitions. However, using parameters allows updating the numerical values of the problem
          without re-defining the complete problem.
        """

        # Validate dimensions, if parameter already defined.
        if name in self.parameters.keys():
            if np.shape(value) != np.shape(self.parameters[name]):
                ValueError(f"Mismatch of redefined parameter: {name}")

        # Set parameter value.
        self.parameters[name] = value

    def define_constraint(
            self,
            *elements: typing.Union[
                str,
                typing.Tuple[str, typing.Union[str, float, np.ndarray, sp.spmatrix]],
                typing.Tuple[str, typing.Union[str, float, np.ndarray, sp.spmatrix], dict]
            ],
            **kwargs
    ):
        """Define linear constraint for given list of constraint elements.

        - Constraints are defined as list of tuples and strings, where tuples are either 1) variable terms or
          2) constant terms and strings represent operators (==, <= or >=). If multiple variable and constant
          terms are on either side of the operator, these are interpreted as summation of the variables / constants.
        - Constant terms are tuples in the form (‘constant’, numerical value), where the numerical value can be
          real-valued 1) float, 2) numpy array, 3) scipy sparse matrix or 4) a parameter name string.
          The numerical value is expected to represent a column vector with appropriate size matching the
          constraint dimension. If a float value is given as numerical value, the value is multiplied with
          a column vector of ones of appropriate size.
        - Variable terms are tuples in the form (‘variable’, numerical factor, dict(name=variable name, keys…)),
          where the numerical factor can be real-valued 1) float, 2) numpy array, 3) scipy sparse matrix or
          4) a parameter name string. The numerical factor is multiplied with the variable vector and is expected
          to represent a matrix of appropriate size for the multiplication. If a float value is given as
          numerical factor, the value is multiplied with a identity matrix of appropriate size. Keys can
          be optionally given to select / slice a portion of the variable vector.
          Note that variables always take the shape of column vectors.
        """

        # Instantiate constraint element aggregation variables.
        variables = list()
        constants = list()
        operator = None

        # Instantiate left-hand / right-hand side indicator. Starting from left-hand side.
        side = 'left'

        # Aggregate constraint elements.
        for element in elements:

            # Tuples are variables / constants.
            if isinstance(element, tuple):

                # Obtain element attributes.
                element_type = element[0]
                element_value = element[1]
                element_keys = element[2] if len(element) > 2 else None

                # Identify variables.
                if element_type in ('variable', 'var', 'v'):

                    # Move right-hand variables to left-hand side.
                    if side == 'right':
                        factor = -1.0
                    else:
                        factor = 1.0

                    # Raise error if no keys defined.
                    if element_keys is None:
                        raise ValueError(f"Missing keys for variable: \n{element}")

                    # Append element to variables.
                    variables.append((factor, element_value, element_keys))

                # Identify constants.
                elif element_type in ('constant', 'con', 'c'):

                    # Move left-hand constants to right-hand side.
                    if side == 'left':
                        factor = -1.0
                    else:
                        factor = 1.0

                    # Append element to constants.
                    constants.append((factor, element_value, element_keys))

                # Raise error if element type cannot be identified.
                else:
                    raise ValueError(f"Invalid constraint element type: {element_type}")

            # Strings are operators.
            elif element in ['==', '<=', '>=']:

                # Raise error if operator is first element.
                if element == elements[0]:
                    ValueError(f"Operator is first element of a constraint.")

                # Raise error if operator is last element.
                if element == elements[-1]:
                    ValueError(f"Operator is last element of a constraint.")

                # Raise error if operator is already defined.
                if operator is not None:
                    ValueError(f"Multiple operators defined in one constraint.")

                # Set operator.
                operator = element

                # Update left-hand / right-hand side indicator. Moving to right-hand side.
                side = 'right'

            # Raise error if element type cannot be identified.
            else:
                raise ValueError(f"Invalid constraint element: \n{element}")

        # Raise error if operator missing.
        if operator is None:
            raise ValueError("Cannot define constraint without operator (==, <= or >=).")

        self.define_constraint_low_level(
            variables,
            operator,
            constants,
            **kwargs
        )

    def define_constraint_low_level(
            self,
            variables: typing.List[
                typing.Tuple[float, typing.Union[str, float, np.ndarray, sp.spmatrix], dict]
            ],
            operator: str,
            constants: typing.List[
                typing.Tuple[float, typing.Union[str, float, np.ndarray, sp.spmatrix], dict]
            ],
            keys: dict = None,
            broadcast: typing.Union[str, list, tuple] = None
    ):

        # Raise error if no variables in constraint.
        if len(variables) == 0:
            raise ValueError(f"Cannot define constraint without variables.")

        # Run checks for constraint index keys.
        if keys is not None:

            # Raise error if ``keys`` is not a dictionary.
            if type(keys) is not dict:
                raise TypeError(f"Constraint `keys` parameter must be a dictionary, but instead is: {type(keys)}")

            # Raise error if no 'name' key was defined.
            if 'name' not in keys.keys():
                raise ValueError(f"'name' key is required in constraint `keys` dictionary. Only found: {keys.keys()}")

            # TODO: Raise error if using reserved 'constraint_type' key.

        # Run type checks for broadcast argument.
        if broadcast is not None:
            if type(broadcast) is str:
                broadcast = [broadcast]
            elif type(broadcast) not in [list, tuple]:
                raise ValueError(f"Invalid type of broadcast argument: {type(broadcast)}")

        # For equality constraint, define separate upper / lower inequality.
        if operator in ['==']:

            # Define upper inequality.
            self.define_constraint_low_level(
                variables,
                '>=',
                constants,
                keys=dict(keys, constraint_type='==>=') if keys is not None else None,
                broadcast=broadcast
            )

            # Define lower inequality.
            self.define_constraint_low_level(
                variables,
                '<=',
                constants,
                keys=dict(keys, constraint_type='==<=') if keys is not None else None,
                broadcast=broadcast
            )

        # For inequality constraint, add into A matrix / b vector dictionaries.
        elif operator in ['<=', '>=']:

            # If greater-than-equal, invert signs.
            if operator == '>=':
                operator_factor = -1.0
            else:
                operator_factor = 1.0

            # Instantiate constraint index.
            constraint_index = None

            # Process variables.
            for variable_factor, variable_value, variable_keys in variables:

                # If any variable key values are empty, ignore variable & do not add any A matrix entry.
                for key_value in variable_keys.values():
                    if isinstance(key_value, (list, tuple, pd.MultiIndex, pd.Index, np.ndarray)):
                        if len(key_value) == 0:
                            continue  # Skip variable & go to next iteration.

                # Obtain variable integer index & raise error if variable or key does not exist.
                variable_index = (
                    tuple(mesmo.utils.get_index(self.variables, **variable_keys, raise_empty_index_error=True))
                )

                # Obtain broadcast dimension length for variable.
                if broadcast is not None:
                    broadcast_len = 1
                    for broadcast_key in broadcast:
                        if broadcast_key not in variable_keys.keys():
                            raise ValueError(f"Invalid broadcast dimension: {broadcast_key}")
                        else:
                            broadcast_len *= len(variable_keys[broadcast_key])
                else:
                    broadcast_len = 1

                # String values are interpreted as parameter name.
                if type(variable_value) is str:
                    parameter_name = variable_value
                    variable_value = self.parameters[parameter_name]
                else:
                    parameter_name = None
                # Flat arrays are interpreted as row vectors (1, n).
                if len(np.shape(variable_value)) == 1:
                    variable_value = np.array([variable_value])
                # Scalar values are multiplied with identity matrix of appropriate size.
                if len(np.shape(variable_value)) == 0:
                    variable_value = variable_value * sp.eye(len(variable_index))
                # If broadcasting, value is repeated in block-diagonal matrix.
                elif broadcast_len > 1:
                    if type(variable_value) is np.matrix:
                        variable_value = np.array(variable_value)
                    variable_value = sp.block_diag([variable_value] * broadcast_len)

                # If not yet defined, obtain constraint index based on dimension of first variable.
                if constraint_index is None:
                    constraint_index = (
                        tuple(range(self.constraints_len, self.constraints_len + np.shape(variable_value)[0]))
                    )

                # Raise error if variable dimensions are inconsistent.
                if np.shape(variable_value) != (len(constraint_index), len(variable_index)):
                    raise ValueError(f"Dimension mismatch at variable: \n{variable_keys}")

                # Append A matrix entry.
                # - If parameter, pass tuple of factor, parameter name and broadcasting dimension length.
                if parameter_name is None:
                    self.a_dict[constraint_index, variable_index].append(
                        operator_factor * variable_factor * variable_value
                    )
                else:
                    self.a_dict[constraint_index, variable_index].append(
                        (operator_factor * variable_factor, parameter_name, broadcast_len)
                    )

            # Process constants.
            for constant_factor, constant_value, constant_keys in constants:

                # If constant value is string, it is interpreted as parameter.
                if type(constant_value) is str:
                    parameter_name = constant_value
                    constant_value = self.parameters[parameter_name]
                else:
                    parameter_name = None

                # Obtain broadcast dimension length for constant.
                if (broadcast is not None) and (constant_keys is not None):
                    broadcast_len = 1
                    for broadcast_key in broadcast:
                        if broadcast_key in constant_keys.keys():
                            broadcast_len *= len(constant_keys[broadcast_key])
                else:
                    broadcast_len = 1

                # If constant is sparse, convert to dense array.
                if isinstance(constant_value, sp.spmatrix):
                    constant_value = constant_value.toarray()

                # If constant is scalar, cast into vector of appropriate size.
                if len(np.shape(constant_value)) == 0:
                    constant_value = constant_value * np.ones(len(constraint_index))
                # If broadcasting, values are repeated along broadcast dimension.
                elif broadcast_len > 1:
                    constant_value = np.concatenate([constant_value] * broadcast_len, axis=0)

                # Raise error if constant is not a scalar, column vector (n, 1) or flat array (n, ).
                if len(np.shape(constant_value)) > 1:
                    if np.shape(constant_value)[1] > 1:
                        raise ValueError(f"Constant must be column vector (n, 1), not row vector (1, n).")

                # If not yet defined, obtain constraint index based on dimension of first constant.
                if constraint_index is None:
                    constraint_index = tuple(range(self.constraints_len, self.constraints_len + len(constant_value)))

                # Raise error if constant dimensions are inconsistent.
                if len(constant_value) != len(constraint_index):
                    raise ValueError(f"Dimension mismatch at constant: \n{constant_keys}")

                # Append b vector entry.
                if parameter_name is None:
                    self.b_dict[constraint_index].append(
                        operator_factor * constant_factor * constant_value
                    )
                else:
                    self.b_dict[constraint_index].append(
                        (operator_factor * constant_factor, parameter_name, broadcast_len)
                    )

            # Append constraints index entries.
            if keys is not None:
                # Set constraint type:
                if 'constraint_type' in keys.keys():
                    if keys['constraint_type'] not in ('==>=', '==<='):
                        keys['constraint_type'] = operator
                else:
                    keys['constraint_type'] = operator
                # Obtain new constraints based on ``keys``.
                # - Constraint dimensions are constructed based by taking the product of the given key sets.
                new_constraints = (
                    pd.DataFrame(itertools.product(*[
                        list(value)
                        if type(value) in [pd.MultiIndex, pd.Index, pd.DatetimeIndex, np.ndarray, list, tuple]
                        else [value]
                        for value in keys.values()
                    ]), columns=keys.keys())
                )
                # Raise error if key set dimension does not align with constant dimension.
                if len(new_constraints) != len(constraint_index):
                    raise ValueError(
                        f"Constraint key set dimension ({len(new_constraints)})"
                        f" does not align with constraint value dimension ({len(constraint_index)})."
                    )
                # Add new constraints to index.
                new_constraints.index = constraint_index
                self.constraints = self.constraints.append(new_constraints)
                self.constraints_len += len(constraint_index)
            else:
                # Only change constraints size, if no ``keys`` defined.
                # - This is for speedup, as updating the constraints index set with above operation is slow.
                self.constraints_len += len(constraint_index)

        # Raise error for invalid operator.
        else:
            ValueError(f"Invalid constraint operator: {operator}")

    def define_objective(
            self,
            *elements: typing.Union[
                str,
                typing.Tuple[str, typing.Union[str, float, np.ndarray, sp.spmatrix]],
                typing.Tuple[str, typing.Union[str, float, np.ndarray, sp.spmatrix], dict],
                typing.Tuple[str, typing.Union[str, float, np.ndarray, sp.spmatrix], dict, dict]
            ],
            **kwargs
    ):
        """Define objective terms for the given list of objective elements.

        - Objective terms are defined as list of tuples, where tuples are either 1) variable terms or
          2) constant terms. Each term is expected to evaluate to a scalar value. If multiple variable and
          constant terms are defined, these are interpreted as summation of the variables / constants.
        - Constant terms are tuples in the form (‘constant’, numerical value), where the numerical value can be
          1) float value or 2) a parameter name string.
        - Variable terms are tuples in the form (‘variable’, numerical factor, dict(name=variable name, keys…)),
          where the numerical factor can be 1) float value, 2) numpy array, 3) scipy sparse matrix or
          4) a parameter name string. The numerical factor is multiplied with the variable vector and is expected
          to represent a matrix of appropriate size for the multiplication, such that the multiplication evaluates
          to a scalar. If a float value is given as numerical factor, the value is multiplied with a row vector of
          ones of appropriate size. Keys can be optionally given to select / slice a portion of the variable vector.
          Note that variables always take the shape of column vectors.
        """

        # Instantiate objective element aggregation variables.
        variables = list()
        variables_quadratic = list()
        constants = list()

        # Aggregate objective elements.
        for element in elements:

            # Tuples are variables / constants.
            if isinstance(element, tuple):

                # Obtain element attributes.
                element_type = element[0]
                element_value = element[1]
                element_keys_1 = element[2] if len(element) > 2 else None
                element_keys_2 = element[3] if len(element) > 3 else None

                # Identify variables.
                if element_type in ('variable', 'var', 'v'):

                    # Append element to variables / quadratic variables.
                    if element_keys_2 is None:
                        variables.append((element_value, element_keys_1))
                    else:
                        variables_quadratic.append((element_value, element_keys_1, element_keys_2))

                # Identify constants.
                elif element_type in ('constant', 'con', 'c'):

                    # Add element to constant.
                    constants.append((element_value, element_keys_1))

                # Raise error if element type cannot be identified.
                else:
                    raise ValueError(f"Invalid objective element type: {element[0]}")

            # Raise error if element type cannot be identified.
            else:
                raise ValueError(f"Invalid objective element: \n{element}")

        self.define_objective_low_level(
            variables,
            variables_quadratic,
            constants,
            **kwargs
        )

    def define_objective_low_level(
            self,
            variables: typing.List[
                typing.Tuple[typing.Union[str, float, np.ndarray, sp.spmatrix], dict]
            ],
            variables_quadratic: typing.List[
                typing.Tuple[typing.Union[str, float, np.ndarray, sp.spmatrix], dict, dict]
            ],
            constants: typing.List[
                typing.Tuple[typing.Union[str, float, np.ndarray, sp.spmatrix], dict]
            ],
            broadcast: typing.Union[str, list, tuple] = None
    ):

        # Run type checks for broadcast argument.
        if broadcast is not None:
            if type(broadcast) is str:
                broadcast = [broadcast]
            elif type(broadcast) not in [list, tuple]:
                raise ValueError(f"Invalid type of broadcast argument: {type(broadcast)}")

        # Process variables.
        for variable_value, variable_keys in variables:

            # If any variable key values are empty, ignore variable & do not add any c vector entry.
            for key_value in variable_keys.values():
                if isinstance(key_value, (list, tuple, pd.MultiIndex, pd.Index, np.ndarray)):
                    if len(key_value) == 0:
                        continue  # Skip variable & go to next iteration.

            # Obtain variable index & raise error if variable or key does not exist.
            variable_index = (
                tuple(mesmo.utils.get_index(self.variables, **variable_keys, raise_empty_index_error=True))
            )

            # Obtain broadcast dimension length for variable.
            if broadcast is not None:
                broadcast_len = 1
                for broadcast_key in broadcast:
                    if broadcast_key not in variable_keys.keys():
                        raise ValueError(f"Invalid broadcast dimension: {broadcast_key}")
                    else:
                        broadcast_len *= len(variable_keys[broadcast_key])
            else:
                broadcast_len = 1

            # String values are interpreted as parameter name.
            if type(variable_value) is str:
                parameter_name = variable_value
                variable_value = self.parameters[parameter_name]
            else:
                parameter_name = None
            # Scalar values are multiplied with row vector of ones of appropriate size.
            if len(np.shape(variable_value)) == 0:
                variable_value = variable_value * np.ones((1, len(variable_index)))
            # If broadcasting, values are repeated along broadcast dimension.
            else:
                if broadcast_len > 1:
                    if len(np.shape(variable_value)) > 1:
                        variable_value = np.concatenate([variable_value] * broadcast_len, axis=1)
                    else:
                        variable_value = np.concatenate([[variable_value]] * broadcast_len, axis=1)

            # Raise error if vector is not a row vector (1, n) or flat array (n, ).
            if len(np.shape(variable_value)) > 1:
                if np.shape(variable_value)[0] > 1:
                    raise ValueError(
                        f"Objective factor must be row vector (1, n) or flat array (n, ),"
                        f" not column vector (n, 1) nor matrix (m, n)."
                    )

            # Raise error if variable dimensions are inconsistent.
            if (
                    np.shape(variable_value)[1] != len(variable_index)
                    if len(np.shape(variable_value)) > 1
                    else np.shape(variable_value)[0] != len(variable_index)
            ):
                raise ValueError(f"Objective factor dimension mismatch at variable: \n{variable_keys}")

            # Add c vector entry.
            # - If parameter, pass tuple of parameter name and broadcasting dimension length.
            if parameter_name is None:
                self.c_dict[variable_index].append(variable_value)
            else:
                self.c_dict[variable_index].append((parameter_name, broadcast_len))

        # Process quadratic variables.
        for variable_value, variable_keys_1, variable_keys_2 in variables_quadratic:

            # If any variable key values are empty, ignore variable & do not add any c vector entry.
            for key_value in list(variable_keys_1.values()) + list(variable_keys_2.values()):
                if isinstance(key_value, (list, tuple, pd.MultiIndex, pd.Index, np.ndarray)):
                    if len(key_value) == 0:
                        continue  # Skip variable & go to next iteration.

            # Obtain variable index & raise error if variable or key does not exist.
            variable_1_index = (
                tuple(mesmo.utils.get_index(self.variables, **variable_keys_1, raise_empty_index_error=True))
            )
            variable_2_index = (
                tuple(mesmo.utils.get_index(self.variables, **variable_keys_2, raise_empty_index_error=True))
            )

            # Obtain broadcast dimension length for variable.
            if broadcast is not None:
                broadcast_len = 1
                for broadcast_key in broadcast:
                    if broadcast_key not in variable_keys_1.keys():
                        raise ValueError(f"Invalid broadcast dimension: {broadcast_key}")
                    else:
                        broadcast_len *= len(variable_keys_1[broadcast_key])
            else:
                broadcast_len = 1

            # String values are interpreted as parameter name.
            if type(variable_value) is str:
                parameter_name = variable_value
                variable_value = self.parameters[parameter_name]
            else:
                parameter_name = None
            # Flat arrays are interpreted as diagonal matrix.
            if len(np.shape(variable_value)) == 1:
                # TODO: Raise error for flat arrays instead?
                variable_value = sp.diags(variable_value)
            # Scalar values are multiplied with diagonal matrix of ones of appropriate size.
            if len(np.shape(variable_value)) == 0:
                variable_value = variable_value * sp.eye(len(variable_1_index))
            # If broadcasting, values are repeated along broadcast dimension.
            else:
                if type(variable_value) is np.matrix:
                    variable_value = np.array(variable_value)
                variable_value = sp.block_diag([variable_value] * broadcast_len)

            # Raise error if variable dimensions are inconsistent.
            if np.shape(variable_value)[0] != len(variable_1_index):
                raise ValueError(
                    f"Quadratic objective factor dimension mismatch at variable 1: \n{variable_keys_1}"
                    f"\nThe shape of quadratic objective factor matrix must be "
                    f"{(len(variable_1_index), len(variable_2_index))}, based on the variable dimensions."
                )
            if np.shape(variable_value)[1] != len(variable_2_index):
                raise ValueError(
                    f"Quadratic objective factor dimension mismatch at variable 2: \n{variable_keys_2}"
                    f"\nThe shape of quadratic objective factor matrix must be "
                    f"{(len(variable_1_index), len(variable_2_index))}, based on the variable dimensions."
                )

            # Add Q matrix entry.
            # - If parameter, pass tuple of parameter name and broadcasting dimension length.
            if parameter_name is None:
                self.q_dict[variable_1_index, variable_2_index].append(variable_value)
            else:
                self.q_dict[variable_1_index, variable_2_index].append((parameter_name, broadcast_len))

        # Process constants.
        for constant_value, constant_keys in constants:

            # If constant value is string, it is interpreted as parameter.
            if type(constant_value) is str:
                parameter_name = constant_value
                constant_value = self.parameters[parameter_name]
            else:
                parameter_name = None

            # Obtain broadcast dimension length for constant.
            if (broadcast is not None) and (constant_keys is not None):
                broadcast_len = 1
                for broadcast_key in broadcast:
                    if broadcast_key in constant_keys.keys():
                        broadcast_len *= len(constant_keys[broadcast_key])
            else:
                broadcast_len = 1

            # Raise error if constant is not a scalar (1, ) or (1, 1) or float.
            if type(constant_value) is not float:
                if np.shape(constant_value) not in [(1, ), (1, 1)]:
                    raise ValueError(f"Objective constant must be scalar or (1, ) or (1, 1).")

            # If broadcasting, value is repeated along broadcast dimension.
            if broadcast_len > 1:
                constant_value = constant_value * broadcast_len

            # Append d constant entry.
            if parameter_name is None:
                self.d_dict[0].append(constant_value)
            else:
                self.d_dict[0].append((parameter_name, broadcast_len))

    def get_a_matrix(self) -> sp.csr_matrix:
        r"""Obtain :math:`\boldsymbol{A}` matrix for the standard-form problem (see :class:`OptimizationProblem`)."""

        # Log time.
        log_time('get optimization problem A matrix')

        # Instantiate collections.
        values_list = list()
        rows_list = list()
        columns_list = list()

        # Collect matrix entries.
        for constraint_index, variable_index in self.a_dict:
            for values in self.a_dict[constraint_index, variable_index]:
                # If value is tuple, treat as parameter.
                if type(values) is tuple:
                    factor, parameter_name, broadcast_len = values
                    values = self.parameters[parameter_name]
                    if len(np.shape(values)) == 1:
                        values = np.array([values])
                    if len(np.shape(values)) == 0:
                        values = values * sp.eye(len(variable_index))
                    elif broadcast_len > 1:
                        if type(values) is np.matrix:
                            values = np.array(values)
                        values = sp.block_diag([values] * broadcast_len)
                    values = values * factor
                # Obtain row index, column index and values for entry in A matrix.
                rows, columns, values = sp.find(values)
                rows = np.array(constraint_index)[rows]
                columns = np.array(variable_index)[columns]
                # Insert entry in collections.
                values_list.append(values)
                rows_list.append(rows)
                columns_list.append(columns)

        # Instantiate A matrix.
        a_matrix = (
            sp.coo_matrix(
                (np.concatenate(values_list), (np.concatenate(rows_list), np.concatenate(columns_list))),
                shape=(self.constraints_len, len(self.variables))
            ).tocsr()
        )

        # Log time.
        log_time('get optimization problem A matrix')

        return a_matrix

    def get_b_vector(self) -> np.ndarray:
        r"""Obtain :math:`\boldsymbol{b}` vector for the standard-form problem (see :class:`OptimizationProblem`)."""

        # Log time.
        log_time('get optimization problem b vector')

        # Instantiate array.
        b_vector = np.zeros((self.constraints_len, 1))

        # Fill vector entries.
        for constraint_index in self.b_dict:
            for values in self.b_dict[constraint_index]:
                # If value is tuple, treat as parameter.
                if type(values) is tuple:
                    factor, parameter_name, broadcast_len = values
                    values = self.parameters[parameter_name]
                    if len(np.shape(values)) == 0:
                        values = values * np.ones(len(constraint_index))
                    elif broadcast_len > 1:
                        values = np.concatenate([values] * broadcast_len, axis=0)
                    values = values * factor
                # Insert entry in b vector.
                b_vector[constraint_index, 0] += values.ravel()

        # Log time.
        log_time('get optimization problem b vector')

        return b_vector

    def get_c_vector(self) -> np.ndarray:
        r"""Obtain :math:`\boldsymbol{c}` vector for the standard-form problem (see :class:`OptimizationProblem`)."""

        # Log time.
        log_time('get optimization problem c vector')

        # Instantiate array.
        c_vector = np.zeros((1, len(self.variables)))

        # Fill vector entries.
        for variable_index in self.c_dict:
            for values in self.c_dict[variable_index]:
                # If value is tuple, treat as parameter.
                if type(values) is tuple:
                    parameter_name, broadcast_len = values
                    values = self.parameters[parameter_name]
                    if len(np.shape(values)) == 0:
                        values = values * np.ones(len(variable_index))
                    elif broadcast_len > 1:
                        if len(np.shape(values)) > 1:
                            values = np.concatenate([values] * broadcast_len, axis=1)
                        else:
                            values = np.concatenate([[values]] * broadcast_len, axis=1)
                # Insert entry in c vector.
                c_vector[0, variable_index] += values.ravel()

        # Log time.
        log_time('get optimization problem c vector')

        return c_vector

    def get_q_matrix(self) -> sp.spmatrix:
        r"""Obtain :math:`\boldsymbol{Q}` matrix for the standard-form problem (see :class:`OptimizationProblem`)."""

        # Log time.
        log_time('get optimization problem Q matrix')

        # Instantiate collections.
        values_list = list()
        rows_list = list()
        columns_list = list()

        # Collect matrix entries.
        for variable_1_index, variable_2_index in self.q_dict:
            for values in self.q_dict[variable_1_index, variable_2_index]:
                # If value is tuple, treat as parameter.
                if type(values) is tuple:
                    parameter_name, broadcast_len = values
                    values = self.parameters[parameter_name]
                    if len(np.shape(values)) == 1:
                        values = sp.diags(values)
                    if len(np.shape(values)) == 0:
                        values = values * sp.eye(len(variable_1_index))
                    elif broadcast_len > 1:
                        if type(values) is np.matrix:
                            values = np.array(values)
                        values = sp.block_diag([values] * broadcast_len)
                # Obtain row index, column index and values for entry in Q matrix.
                rows, columns, values = sp.find(values)
                rows = np.array(variable_1_index)[rows]
                columns = np.array(variable_2_index)[columns]
                # Insert entry in collections.
                values_list.append(values)
                rows_list.append(rows)
                columns_list.append(columns)

        # Instantiate Q matrix.
        q_matrix = (
            sp.coo_matrix(
                (np.concatenate(values_list), (np.concatenate(rows_list), np.concatenate(columns_list))),
                shape=(len(self.variables), len(self.variables))
            ).tocsr()
            if len(self.q_dict) > 0 else sp.csr_matrix((len(self.variables), len(self.variables)))
        )

        # Log time.
        log_time('get optimization problem Q matrix')

        return q_matrix

    def get_d_constant(self) -> float:
        r"""Obtain :math:`d` value for the standard-form problem (see :class:`OptimizationProblem`)."""

        # Log time.
        log_time('get optimization problem d constant')

        # Instantiate array.
        d_constant = 0.0

        # Fill vector entries.
        for values in self.d_dict[0]:
            # If value is tuple, treat as parameter.
            if type(values) is tuple:
                parameter_name, broadcast_len = values
                values = self.parameters[parameter_name]
                if broadcast_len > 1:
                    values = values * broadcast_len
            # Insert entry to d constant.
            d_constant += float(values)

        # Log time.
        log_time('get optimization problem d constant')

        return d_constant

    def solve(self):
        r"""Solve the optimization problem.

        - The solve method compiles the standard form of the optimization problem
          (see :class:`OptimizationProblem`) and passes the standard-form problem to the optimization
          solver interface.
        - The solve method currently implements interfaces to 1) Gurobi and 2) CVXPY, where the latter is a high-level
          convex optimization interface, which in turn allows interfacing further third-party solvers. The intention is
          to implement more direct solver interfaces on a as-need basis (please raise an issue!), as these interfaces
          are assumed to allow higher performance than CVXPY for large-scale problems. However, CVXPY is kept as
          a fallback to allow a high degree of compatibility with various solvers.
        - The choice of solver and solver interface can be controlled through the config parameters
          ``optimization > solver_name`` and ``optimization > solver_interface`` (see ``mesmo/config_default.yml``).

        The default workflow of the solve method is as follows:

        1. Obtain problem definition through selected solver interface via :meth:`get_cvxpy_problem()` or
           :meth:`get_gurobi_problem()`.
        2. Solve optimization problem and obtain standard-form results via :meth:`solve_cvxpy()` or
           :meth:`solve_gurobi()`. The standard-form results include the 1) :math:`\boldsymbol{x}` variable vector
           value, 2) :math:`\boldsymbol{\mu}` dual vector value and 3) objective value, which are stored into the
           object attributes :attr:`x_vector`, :attr:`mu_vector` and :attr:`objective`.
        3. Obtain results with respect to the original problem formulation via :meth:`get_results()` and
           :meth:`get_duals()`. These results are 1) decision variable values and
           2) constraint dual values, which are stored into the object attributes :attr:`results` and :attr:`duals`.

        Low-level customizations of the problem definition are possible, e.g. definition of quadratic constraints or
        second-order conic (SOC) constraints via the solver interfaces, with the following workflow.

        1. Obtain problem definition through selected solver interface via :meth:`get_cvxpy_problem()` or
           :meth:`get_gurobi_problem()`.
        2. Customize problem definitions, e.g. add custom constraints directly with the Gurobi or CVXPY interfaces.
        3. Solve optimization problem and obtain standard-form results via :meth:`solve_cvxpy()` or
           :meth:`solve_gurobi()`.
        4. Obtain results with respect to the original problem formulation via :meth:`get_results()` and
           :meth:`get_duals()`.
        """

        # TODO: Add example for low-level customization solve workflow.

        # Log time.
        log_time(f'solve optimization problem problem')
        logger.debug(
            f"Solver name: {mesmo.config.config['optimization']['solver_name']};"
            f" Solver interface: {mesmo.config.config['optimization']['solver_interface']};"
            f" Problem statistics: {len(self.variables)} variables, {self.constraints_len} constraints"
        )

        # Use CVXPY solver interface, if selected.
        if mesmo.config.config['optimization']['solver_interface'] == 'cvxpy':
            self.solve_cvxpy(*self.get_cvxpy_problem())
        # Use direct solver interfaces, if selected.
        elif mesmo.config.config['optimization']['solver_interface'] == 'direct':
            if mesmo.config.config['optimization']['solver_name'] == 'gurobi':
                self.solve_gurobi(*self.get_gurobi_problem())
            # If no direct solver interface found, fall back to CVXPY interface.
            else:
                logger.debug(
                    f"No direct solver interface implemented for"
                    f" '{mesmo.config.config['optimization']['solver_name']}'. Falling back to CVXPY."
                )
                self.solve_cvxpy(*self.get_cvxpy_problem())
        # Raise error, if invalid solver interface selected.
        else:
            raise ValueError(f"Invalid solver interface: '{mesmo.config.config['optimization']['solver_interface']}'")

        # Get results / duals.
        self.results = self.get_results()
        self.duals = self.get_duals()

        # Log time.
        log_time(f'solve optimization problem problem')

    def get_gurobi_problem(self) -> (gp.Model, gp.MVar, gp.MConstr, gp.MQuadExpr):
        """Obtain standard-form problem via Gurobi direct interface."""

        # Instantiate Gurobi model.
        # - A Gurobi model holds a single optimization problem. It consists of a set of variables, a set of constraints,
        #   and the associated attributes.
        gurobipy_problem = gp.Model()
        # Set solver parameters.
        gurobipy_problem.setParam('OutputFlag', int(mesmo.config.config['optimization']['show_solver_output']))
        for key, value in mesmo.config.solver_parameters.items():
            gurobipy_problem.setParam(key, value)

        # Define variables.
        # - Need to express vectors as 1-D arrays to enable matrix multiplication in constraints (gurobipy limitation).
        # - Lower bound defaults to 0 and needs to be explicitly overwritten.
        x_vector = (
            gurobipy_problem.addMVar(
                shape=(len(self.variables), ),
                lb=-np.inf,
                ub=np.inf,
                vtype=gp.GRB.CONTINUOUS,
                name='x_vector'
            )
        )

        # Define constraints.
        # - 1-D arrays are interpreted as column vectors (n, 1) (based on gurobipy convention).
        constraints = self.get_a_matrix() @ x_vector <= self.get_b_vector().ravel()
        constraints = gurobipy_problem.addConstr(constraints, name='constraints')

        # Define objective.
        # - 1-D arrays are interpreted as column vectors (n, 1) (based on gurobipy convention).
        objective = (
            self.get_c_vector().ravel() @ x_vector
            + x_vector @ (0.5 * self.get_q_matrix()) @ x_vector
            + self.get_d_constant()
        )
        gurobipy_problem.setObjective(objective, gp.GRB.MINIMIZE)

        return (
            gurobipy_problem,
            x_vector,
            constraints,
            objective
        )

    def solve_gurobi(
            self,
            gurobipy_problem: gp.Model,
            x_vector: gp.MVar,
            constraints: gp.MConstr,
            objective: gp.MQuadExpr
    ) -> gp.Model:
        """Solve optimization problem via Gurobi direct interface."""

        # Solve optimization problem.
        gurobipy_problem.optimize()

        # Raise error if no optimal solution.
        status_labels = {
            gp.GRB.INFEASIBLE: "Infeasible",
            gp.GRB.INF_OR_UNBD: "Infeasible or Unbounded",
            gp.GRB.UNBOUNDED: "Unbounded",
            gp.GRB.SUBOPTIMAL: "Suboptimal"
        }
        status = gurobipy_problem.getAttr('Status')
        if status not in [gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL]:
            status = status_labels[status] if status in status_labels.keys() else f"{status} (See Gurobi documentation)"
            raise RuntimeError(f"Gurobi exited with non-optimal solution status: {status}")
        elif status == gp.GRB.SUBOPTIMAL:
            status = status_labels[status] if status in status_labels.keys() else f"{status} (See Gurobi documentation)"
            logger.warning(f"Gurobi exited with non-optimal solution status: {status}")

        # Store results.
        self.x_vector = np.transpose([x_vector.getAttr('x')])
        if gurobipy_problem.getAttr('NumQCNZs') == 0:
            self.mu_vector = np.transpose([constraints.getAttr('Pi')])
        else:
            # Duals are not retrieved if quadratic or SOC constraints have been added to the model.
            logger.warning(
                f"Duals of the optimization problem's constraints are not retrieved,"
                f" because quadratic or SOC constraints have been added to the problem."
                f"\nPlease retrieve the duals manually."
            )
            self.mu_vector = np.nan * np.zeros(constraints.shape)
        self.objective = float(objective.getValue())

        return gurobipy_problem

    def get_cvxpy_problem(self) -> (cp.Variable, typing.List[typing.Union[cp.NonPos, cp.Zero, cp.SOC, cp.PSD]], cp.Expression):
        """Obtain standard-form problem via CVXPY interface."""

        # Define variables.
        x_vector = cp.Variable(shape=(len(self.variables), 1), name='x_vector')

        # Define constraints.
        constraints = [self.get_a_matrix() @ x_vector <= self.get_b_vector()]

        # Define objective.
        objective = (
            self.get_c_vector() @ x_vector
            + cp.quad_form(x_vector, 0.5 * self.get_q_matrix())
            + self.get_d_constant()
        )

        return (
            x_vector,
            constraints,
            objective
        )

    def solve_cvxpy(
            self,
            x_vector: cp.Variable,
            constraints: typing.List[typing.Union[cp.NonPos, cp.Zero, cp.SOC, cp.PSD]],
            objective: cp.Expression
    ) -> cp.Problem:
        """Solve optimization problem via CVXPY interface."""

        # Instantiate CVXPY problem.
        cvxpy_problem = cp.Problem(cp.Minimize(objective), constraints)

        # Solve optimization problem.
        cvxpy_problem.solve(
            solver=(
                mesmo.config.config['optimization']['solver_name'].upper()
                if mesmo.config.config['optimization']['solver_name'] is not None
                else None
            ),
            verbose=mesmo.config.config['optimization']['show_solver_output'],
            **mesmo.config.solver_parameters
        )

        # Assert that solver exited with an optimal solution. If not, raise an error.
        if not (cvxpy_problem.status == cp.OPTIMAL):
            raise RuntimeError(f"CVXPY exited with non-optimal solution status: {cvxpy_problem.status}")

        # Store results.
        self.x_vector = x_vector.value
        self.mu_vector = constraints[0].dual_value
        self.objective = float(cvxpy_problem.objective.value)

        return cvxpy_problem

    def get_results(
        self,
        x_vector: typing.Union[cp.Variable, np.ndarray] = None
    ) -> dict:
        """Obtain results for decisions variables.

        - Results are returned as dictionary with keys corresponding to the variable names that have been defined.
        """

        # Log time.
        log_time('get optimization problem results')

        # Obtain x vector.
        if x_vector is None:
            x_vector = self.x_vector
        elif type(x_vector) is cp.Variable:
            x_vector = x_vector.value

        # Instantiate results object.
        results = dict.fromkeys(self.variables.loc[:, 'name'].unique())

        # Obtain results for each variable.
        for name in results:

            # Get variable dimensions.
            variable_dimensions = (
                self.variables.iloc[mesmo.utils.get_index(self.variables, name=name), :]
                .drop(['name'], axis=1).drop_duplicates().dropna(axis=1)
            )

            if len(variable_dimensions.columns) > 0:

                # Get results from x vector as pandas series.
                results[name] = (
                    pd.Series(
                        x_vector[mesmo.utils.get_index(self.variables, name=name), 0],
                        index=pd.MultiIndex.from_frame(variable_dimensions)
                    )
                )

                # Reshape to dataframe with timesteps as index and other variable dimensions as columns.
                if 'timestep' in variable_dimensions.columns:
                    results[name] = (
                        results[name].unstack(level=[key for key in variable_dimensions.columns if key != 'timestep'])
                    )

                # If results are obtained as series, convert to dataframe with variable name as column.
                if type(results[name]) is pd.Series:
                    results[name] = pd.DataFrame(results[name], columns=[name])

            else:

                # Scalar values are obtained as float.
                results[name] = float(x_vector[mesmo.utils.get_index(self.variables, name=name), 0])

        # Log time.
        log_time('get optimization problem results')

        return results

    def get_duals(self) -> dict:
        """Obtain results for constraint dual variables.

        - Duals are returned as dictionary with keys corresponding to the constraint names that have been defined.
        """

        # Log time.
        log_time('get optimization problem duals')

        # Instantiate results object.
        results = dict.fromkeys(self.constraints.loc[:, 'name'].unique())

        # Obtain results for each constraint.
        for name in results:

            # Get constraint dimensions & constraint type.
            # TODO: Check if this works for scalar constraints without timesteps.
            constraint_dimensions = (
                pd.MultiIndex.from_frame(
                    self.constraints.iloc[mesmo.utils.get_index(self.constraints, name=name), :]
                    .drop(['name', 'constraint_type'], axis=1).drop_duplicates().dropna(axis=1)
                )
            )
            constraint_type = (
                pd.Series(self.constraints.loc[self.constraints.loc[:, 'name'] == name, 'constraint_type'].unique())
            )

            # Get results from x vector as pandas series.
            if constraint_type.str.contains('==').any():
                results[name] = (
                    pd.Series(
                        0.0
                        - self.mu_vector[self.constraints.index[
                            mesmo.utils.get_index(self.constraints, name=name, constraint_type='==>=')
                        ], 0]
                        - self.mu_vector[self.constraints.index[
                            mesmo.utils.get_index(self.constraints, name=name, constraint_type='==<=')
                        ], 0],
                        index=constraint_dimensions
                    )
                )
            elif constraint_type.str.contains('>=').any():
                results[name] = (
                    pd.Series(
                        0.0
                        - self.mu_vector[self.constraints.index[
                            mesmo.utils.get_index(self.constraints, name=name, constraint_type='>=')
                        ], 0],
                        index=constraint_dimensions
                    )
                )
            elif constraint_type.str.contains('<=').any():
                results[name] = (
                    pd.Series(
                        0.0
                        - self.mu_vector[self.constraints.index[
                            mesmo.utils.get_index(self.constraints, name=name, constraint_type='<=')
                        ], 0],
                        index=constraint_dimensions
                    )
                )

            # Reshape to dataframe with timesteps as index and other constraint dimensions as columns.
            results[name] = (
                results[name].unstack(level=[key for key in constraint_dimensions.names if key != 'timestep'])
            )
            # If no other dimensions, e.g. for scalar constraints, convert to dataframe with constraint name as column.
            if type(results[name]) is pd.Series:
                results[name] = pd.DataFrame(results[name], columns=[name])

        # Log time.
        log_time('get optimization problem duals')

        return results

    def evaluate_objective(
            self,
            x_vector: np.ndarray
    ) -> float:
        r"""Utility function for evaluating the objective value for a given :math:`x` vector value."""

        objective = float(
            self.get_c_vector() @ x_vector
            + x_vector.T @ (0.5 * self.get_q_matrix()) @ x_vector
            + self.get_d_constant()
        )

        return objective


def starmap(
        function: typing.Callable,
        argument_sequence: typing.Iterable[tuple],
        keyword_arguments: dict = None
) -> list:
    """Utility function to execute a function for a sequence of arguments, effectively replacing a for-loop.
    Allows running repeated function calls in-parallel, based on Python's `multiprocessing` module.

    - If configuration parameter `run_parallel` is set to True, execution is passed to `starmap`
      of multiprocessing pool, hence running the function calls in parallel.
    - Otherwise, execution is passed to `itertools.starmap`, which is the non-parallel equivalent.
    """

    # Apply keyword arguments.
    if keyword_arguments is not None:
        function_partial = functools.partial(function, **keyword_arguments)
    else:
        function_partial = function

    # Ensure that argument sequence is list.
    argument_sequence = list(argument_sequence)

    if mesmo.config.config['multiprocessing']['run_parallel']:
        # TODO: Remove old parallel pool traces.
        # # If `run_parallel`, use starmap from multiprocessing pool for parallel execution.
        # if mesmo.config.parallel_pool is None:
        #     # Setup parallel pool on first execution.
        #     log_time('parallel pool setup')
        #     mesmo.config.parallel_pool = mesmo.config.get_parallel_pool()
        #     log_time('parallel pool setup')
        # results = mesmo.config.parallel_pool.starmap(function_partial, list(argument_sequence))

        # If `run_parallel`, use `ray_starmap` for parallel execution.
        if mesmo.config.parallel_pool is None:
            log_time('parallel pool setup')
            ray.init()
            mesmo.config.parallel_pool = True
            log_time('parallel pool setup')
        results = ray_starmap(function_partial, argument_sequence)
    else:
        # If not `run_parallel`, use for loop for sequential execution.
        results = [
            function_partial(*arguments) for arguments in tqdm.tqdm(
                argument_sequence,
                total=len(argument_sequence),
                disable=(mesmo.config.config['logs']['level'] != 'debug')  # Progress bar only shown in debug mode.
            )
        ]

    return results


def chunk_dict(
        dict_in: dict,
        chunk_count: int = os.cpu_count()
):
    """Divide dictionary into equally sized chunks."""

    chunk_size = int(np.ceil(len(dict_in) / chunk_count))
    dict_iter = iter(dict_in)

    return [
        {j: dict_in[j] for j in itertools.islice(dict_iter, chunk_size)}
        for i in range(0, len(dict_in), chunk_size)
    ]


def chunk_list(
        list_in: typing.Union[typing.Iterable, typing.Sized],
        chunk_count: int = os.cpu_count()
):
    """Divide list into equally sized chunks."""

    chunk_size = int(np.ceil(len(list_in) / chunk_count))
    list_iter = iter(list_in)

    return [
        [j for j in itertools.islice(list_iter, chunk_size)]
        for i in range(0, len(list_in), chunk_size)
    ]


def ray_iterator(objects: list):
    """Utility iterator for a list of parallelized ``ray`` objects.

    - This iterator enables progress reporting with ``tqdm`` in :func:`ray_get`.
    """

    while objects:
        done, objects = ray.wait(objects)
        yield ray.get(done[0])


def ray_get(objects: list):
    """Utility function for parallelized execution of a list of ``ray`` objects.

    - This function enables the parallelized execution with built-in progress reporting.
    """

    try:
        for _ in tqdm.tqdm(
                ray_iterator(objects),
                total=len(objects),
                disable=(mesmo.config.config['logs']['level'] != 'debug')  # Progress bar only shown in debug mode.
        ):
            pass
    except TypeError:
        pass
    return ray.get(objects)


def ray_starmap(
        function_handle: typing.Callable,
        argument_sequence: list
):
    """Utility function to provide an interface similar to ``itertools.starmap`` for ``ray``.

    - This replicates the ``starmap`` interface of the ``multiprocessing`` API, which ray also supports,
      but allows for additional modifications, e.g. progress reporting via :func:`ray_get`.
    """

    return ray_get([
        ray.remote(lambda *args: function_handle(*args)).remote(*arguments)
        for arguments in argument_sequence
    ])


def log_time(
        label: str,
        log_level: str = 'debug',
        logger_object: logging.Logger = logger
):
    """Log start / end message and time duration for given label.

    - When called with given label for the first time, will log start message.
    - When called subsequently with the same / previously used label, will log end message and time duration since
      logging the start message.
    - The log level for start / end messages can be given as keyword argument, By default, messages are logged as
      debug messages.
    - The logger object can be given as keyword argument. By default, uses ``utils.logger`` as logger.
    - Start message: "Starting ``label``."
    - End message: "Completed ``label`` in ``duration`` seconds."

    Arguments:
        label (str): Label for the start / end message.

    Keyword Arguments:
        log_level (str): Log level to which the start / end messages are output. Choices: 'debug', 'info'.
            Default: 'debug'.
        logger_object (logging.logger.Logger): Logger object to which the start / end messages are output. Default:
            ``utils.logger``.
    """

    time_now = time.time()

    if log_level == 'debug':
        logger_handle = lambda message: logger_object.debug(message)
    elif log_level == 'info':
        logger_handle = lambda message: logger_object.info(message)
    else:
        raise ValueError(f"Invalid log level: '{log_level}'")

    if label in log_times.keys():
        logger_handle(f"Completed {label} in {(time_now - log_times.pop(label)):.6f} seconds.")
    else:
        log_times[label] = time_now
        logger_handle(f"Starting {label}.")


def get_index(
        index_set: typing.Union[pd.Index, pd.DataFrame],
        raise_empty_index_error: bool = True,
        **levels_values
):
    """Utility function for obtaining the integer index array for given index set / level / value list combination.

    :syntax:
        - ``get_index(electric_grid_model.nodes, node_type='source', phase=1)``: Get index array for entries in
          index set `electric_grid_model.nodes` with given `node_type` and `phase`.

    Arguments:
        index_set (pd.Index): Index set, e.g., `electric_grid_model.nodes`.

    Keyword Arguments:
        raise_empty_index_error (bool): If true, raise an exception if obtained index array is empty. This is
            the default behavior, because it is usually caused by an invalid level / value combination.
        level (value): All other keyword arguments are interpreted as level / value combinations, where `level`
            must correspond to a level name of the index set.
    """

    # Define handle for get_level_values() depending on index set type.
    if isinstance(index_set, pd.Index):
        get_level_values = lambda level: index_set.get_level_values(level)
    elif isinstance(index_set, pd.DataFrame):
        # get_level_values = lambda level: index_set.get(level, pd.Series(index=index_set.index, name=level))
        get_level_values = lambda level: index_set.loc[:, level]
    else:
        raise TypeError(f"Invalid index set type: {type(index_set)}")

    # Obtain mask for each level / values combination keyword arguments.
    mask = np.ones(len(index_set), dtype=bool)
    for level, values in levels_values.items():

        # Ensure that values are passed as list.
        if isinstance(values, list):
            pass
        elif isinstance(values, tuple):
            # If values are passed as tuple, wrap in list, but only if index
            # level values are tuples. Otherwise, convert to list.
            if isinstance(get_level_values(level).dropna().values[0], tuple):
                values = [values]
            else:
                values = list(values)
        elif isinstance(values, range):
            # Convert range to list.
            values = list(values)
        elif isinstance(values, np.ndarray):
            # Convert numpy arrays to list.
            values = values.tolist()
            values = [values] if not isinstance(values, list) else values
        elif isinstance(values, pd.Index):
            # Convert pandas index to list.
            values = values.to_list()
        else:
            # Convert single values into list with one item.
            values = [values]

        # Obtain mask.
        mask &= get_level_values(level).isin(values)

    # Obtain integer index array.
    index = np.flatnonzero(mask)

    # Assert that index is not empty.
    if raise_empty_index_error:
        if not (len(index) > 0):
            raise ValueError(f"Empty index returned for: {levels_values}")

    return index


def get_element_phases_array(element: pd.Series):
    """Utility function for obtaining the list of connected phases for given element data."""

    # Obtain list of connected phases.
    phases_array = (
        np.flatnonzero([
            False,  # Ground / '0' phase connection is not considered.
            element.at['is_phase_1_connected'] == 1,
            element.at['is_phase_2_connected'] == 1,
            element.at['is_phase_3_connected'] == 1
        ])
    )

    return phases_array


def get_element_phases_string(element: pd.Series):
    """Utility function for obtaining the OpenDSS phases string for given element data."""

    # Obtain string of connected phases.
    phases_string = ""
    if element.at['is_phase_1_connected'] == 1:
        phases_string += ".1"
    if element.at['is_phase_2_connected'] == 1:
        phases_string += ".2"
    if element.at['is_phase_3_connected'] == 1:
        phases_string += ".3"

    return phases_string


def get_inverse_with_zeros(array: np.ndarray) -> np.ndarray:
    """Obtain the inverse of an array, but do not take the inverse of zero values to avoid numerical errors.

    - Takes the inverse of an array and replaces the inverse of any zero values with zero,
      thus avoiding `inf` or `nan` values in the inverse.
    """

    # Take inverse.
    array_inverse = array ** -1

    # Replace inverse of zero values.
    array_inverse[array == 0.0] = array[array == 0.0]

    return array_inverse


def get_timestamp(
        time: datetime.datetime = None
) -> str:
    """Generate formatted timestamp string, e.g., for saving results with timestamp."""

    if time is None:
        time = datetime.datetime.now()

    return time.strftime('%Y-%m-%d_%H-%M-%S')


def get_results_path(
        base_name: str,
        scenario_name: str = None
) -> str:
    """Generate results path, which is a new subfolder in the results directory. The subfolder name is
    assembled of the given base name, scenario name and current timestamp. The new subfolder is
    created on disk along with this.

    - Non-alphanumeric characters are removed from `base_name` and `scenario_name`.
    - If is a script file path or `__file__` is passed as `base_name`, the base file name without extension
      will be taken as base name.
    """

    # Preprocess results path name components, including removing non-alphanumeric characters.
    base_name = re.sub(r'\W-+', '', os.path.basename(os.path.splitext(base_name)[0])) + '_'
    scenario_name = '' if scenario_name is None else re.sub(r'\W-+', '', scenario_name) + '_'
    timestamp = mesmo.utils.get_timestamp()

    # Obtain results path.
    results_path = os.path.join(mesmo.config.config['paths']['results'], f'{base_name}{scenario_name}{timestamp}')

    # Instantiate results directory.
    # TODO: Catch error if dir exists.
    os.mkdir(results_path)

    return results_path


def get_alphanumeric_string(
        string: str
):
    """Create lowercase alphanumeric string from given string, replacing non-alphanumeric characters with underscore."""

    return re.sub(r'[^0-9a-zA-Z_]+', '_', string).strip('_').lower()


def launch(path):
    """Launch the file at given path with its associated application. If path is a directory, open in file explorer."""

    if not os.path.exists(path):
        raise FileNotFoundError(f'Cannot launch file or directory that does not exist: {path}')

    if sys.platform == 'win32':
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path], cwd="/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        subprocess.Popen(['xdg-open', path], cwd="/", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def write_figure_plotly(
        figure: go.Figure,
        results_path: str,
        file_format=mesmo.config.config['plots']['file_format']
):
    """Utility function for writing / storing plotly figure to output file. File format can be given with
    `file_format` keyword argument, otherwise the default is obtained from config parameter `plots/file_format`.

    - `results_path` should be given as file name without file extension, because the file extension is appended
      automatically based on given `file_format`.
    - Valid file formats: 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json'
    """

    if file_format in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf']:
        pio.write_image(
            figure,
            f"{results_path}.{file_format}",
            width=mesmo.config.config['plots']['plotly_figure_width'],
            height=mesmo.config.config['plots']['plotly_figure_height']
        )
    elif file_format in ['html']:
        pio.write_html(figure, f"{results_path}.{file_format}")
    elif file_format in ['json']:
        pio.write_json(figure, f"{results_path}.{file_format}")
    else:
        raise ValueError(
            f"Invalid `file_format` for `write_figure_plotly`: {file_format}"
            f" - Valid file formats: 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'html', 'json'"
        )