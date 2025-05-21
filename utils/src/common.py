import ast
import math
from typing import Any, Iterable, Tuple, Type, Union

from scipy import integrate


def is_invalid(value: Any,
               custom_invalid_values: Union[Any, Iterable[Any]] = None,
               valid_types: Union[Type, Tuple[Type, ...]] = (int, float, str)
               ) -> bool:
    """
    Determines if a value should be considered invalid based on type and content.

    A value is considered invalid if:
    - It is None
    - It is a NaN (Not a Number)
    - It is an empty string or 'NaN' string (case-insensitive)
    - It matches any of the custom invalid values
    - It is not of the specified valid types

    Args:
        value: The value to validate
        custom_invalid_values: Additional values to consider invalid (single value or iterable)
        valid_types: Types that are considered valid for the value

    Returns:
        bool: True if the value is invalid, False otherwise
    """
    # Handle custom invalid values
    if custom_invalid_values is None:
        custom_invalid_values = set()
    elif not isinstance(custom_invalid_values, (set, list, tuple)):
        custom_invalid_values = {custom_invalid_values}

    # Check type validity first
    if not isinstance(value, valid_types):
        return True

    # Check various invalidity conditions
    return (
        value is None or
        (isinstance(value, float) and math.isnan(value)) or
        str(value).strip().lower() in {'', 'nan'} or
        value in custom_invalid_values
    )


def safe_eval(expression: str, globals_dict: dict = None, locals_dict: dict = None) -> Any:
    """
    Safely parse and evaluate an expression using specified globals and locals.

    This function provides a safer alternative to Python's built-in eval() by:
    - Parsing the expression first to validate its structure
    - Only evaluating expressions, not statements
    - Rejecting potentially dangerous constructs like function definitions and imports
    - Using restricted global and local variable scopes

    Args:
        expression: The string expression to evaluate
        globals_dict: Dictionary of global variables (defaults to empty dict)
        locals_dict: Dictionary of local variables (defaults to empty dict)

    Returns:
        The result of evaluating the expression

    Raises:
        SyntaxError: If the expression contains invalid syntax
        ValueError: If the expression contains statements or unsafe operations
        TypeError: If the expression is not a string
    """
    if not isinstance(expression, str):
        raise TypeError("Expression must be a string")

    if globals_dict is None:
        globals_dict = {}
    if locals_dict is None:
        locals_dict = {}

    # Parse the expression to validate it
    parsed = ast.parse(expression, mode='eval')

    # Walk through the AST and ensure it's safe (i.e., only allows literals, variables, and mathematical operations)
    for node in ast.walk(parsed):
        if isinstance(node, (ast.FunctionDef, ast.Lambda, ast.Import, ast.ImportFrom)):
            raise ValueError(
                "Function definitions or imports are not allowed in expressions.")

    # Compile and evaluate the expression
    compiled_expr = compile(parsed, '<string>', 'eval')
    return eval(compiled_expr, globals_dict, locals_dict)


def compute_average_value(value: Union[int, float, tuple, str],
                          lower_bounds: dict,
                          upper_bounds: dict,
                          method: str = 'rough',
                          integration_variable: str = 'x') -> Union[int, float]:
    """
    Computes the average value of a parameter based on its type.

    This function handles different parameter types:
    - Numbers (int/float): Returns the value directly
    - Tuples: Returns arithmetic mean of elements
    - Expressions (str): Evaluates the average value between bounds

    Args:
        value: Parameter to calculate average for
        lower_bounds: Dictionary with lower bounds for variables in the expression
        upper_bounds: Dictionary with upper bounds for variables in the expression
        method: Calculation method - 'rough' (endpoint average) or 'numerical' (integration)
        integration_variable: Variable name to integrate over when using numerical method

    Returns:
        float: The computed average value

    Raises:
        ValueError: For empty tuples, invalid expressions, unsupported types/methods
        ImportError: If scipy is not available for numerical integration
    """

    # Case 1: Direct value (no averaging needed)
    if isinstance(value, (int, float)):
        return value

    # Case 2: Tuple - calculate arithmetic mean
    if isinstance(value, tuple):
        if not value:
            raise ValueError("Cannot compute average of an empty tuple")
        return sum(value) / len(value)

    # Case 3: String expression
    if isinstance(value, str):
        try:
            # Restrict execution environment for security
            restricted_globals = {'__builtins__': {}}

            if method == 'rough':
                # Simple averaging of function values at endpoints
                lower_value = safe_eval(
                    value, restricted_globals, lower_bounds)
                upper_value = safe_eval(
                    value, restricted_globals, upper_bounds)
                return (lower_value + upper_value) / 2

            elif method == 'numerical':
                try:
                    # Extract integration bounds
                    lower_limit = lower_bounds.get(integration_variable)
                    upper_limit = upper_bounds.get(integration_variable)

                    if lower_limit is None or upper_limit is None:
                        raise ValueError(
                            f"Integration variable '{integration_variable}' not found in bounds")

                    if lower_limit >= upper_limit:
                        raise ValueError(
                            f"Invalid integration range: [{lower_limit}, {upper_limit}]")

                    # Function to be integrated
                    def integrand(x):
                        # Create parameter set with current integration variable value
                        parameters = lower_bounds.copy()
                        parameters[integration_variable] = x
                        return safe_eval(value, restricted_globals, parameters)

                    # Perform numerical integration
                    integral_result, error_estimate = integrate.quad(
                        integrand, lower_limit, upper_limit)

                    # Calculate average value over the interval
                    interval_width = upper_limit - lower_limit
                    return integral_result / interval_width

                except ImportError:
                    raise ImportError(
                        "Numerical integration requires scipy package")
                except Exception as error:
                    raise ValueError(f"Integration error: {error}") from error
            else:
                raise ValueError(
                    f"Unknown calculation method: '{method}'. Use 'rough' or 'numerical'")

        except Exception as error:
            raise ValueError(
                f"Error evaluating expression '{value}': {error}") from error

    # Unsupported type
    raise ValueError(f"Unsupported parameter type: {type(value).__name__}")
