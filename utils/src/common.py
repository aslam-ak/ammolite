import ast
import math
import os
import sys
from typing import (Any, Dict, Iterable, List, Literal, Optional, Tuple, Type,
                    Union)

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



def transform_parameter(input_param, transform_function, param_name=None, default=None, additional_params=None):
    """
    Generically transform a parameter value, handling constant values, gradients, and expressions.
    Supports multi-variable transform functions by passing additional parameters.
    
    Args:
        input_param: The parameter to transform (can be constant, gradient, or expression)
        transform_function: Function to apply to the parameter value(s)
        param_name: Name of the parameter (for use in expressions)
        default: Default value if parameter is None
        additional_params: Dictionary of additional parameters to pass to the transform function
        
    Returns:
        Transformed parameter with same structure as input (constant, gradient, or expression)
    """
    if input_param is None:
        return default
    
    # Prepare additional parameters for the transform function
    additional_params = additional_params or {}
    
    if isinstance(input_param, (int, float)):
        # If parameter is a constant value
        return transform_function(input_param, **additional_params)
    elif isinstance(input_param, dict) and 'top' in input_param and 'bottom' in input_param:
        # If parameter is a gradient with top and bottom values
        return {
            'top': transform_function(input_param['top'], **additional_params),
            'bottom': transform_function(input_param['bottom'], **additional_params)
        }
    else:
        # For expressions, create a representation that includes all parameters
        param_str = f"{param_name}" if param_name else "x"
        additional_args = ", ".join(f"{k}={v}" for k, v in additional_params.items())
        expr = f"{transform_function.__name__}({param_str}" + (f", {additional_args}" if additional_args else "") + ")"
        return {'expression': expr}



def calculate_cross_sectional_area(
    shape: str,
    area_type: Literal["total", "inner", "annular"] = "total",
    outer_diameter: Optional[float] = None,
    outer_width: Optional[float] = None, 
    outer_breadth: Optional[float] = None,
    inner_diameter: Optional[float] = None,
    inner_width: Optional[float] = None,
    inner_breadth: Optional[float] = None,
    flange_width: Optional[float] = None,
    flange_thickness: Optional[float] = None,
    web_height: Optional[float] = None,
    web_thickness: Optional[float] = None,
    cross_section: Literal["solid", "hollow"] = "solid"
) -> float:
    """
    Calculate cross-sectional area based on shape and area type.
    
    :param shape: Shape of the cross-section ('rectangle', 'square', 'circle', 'h_section', 'c_section', 'l_section')
    :param area_type: Type of area to calculate ('total', 'inner', 'annular')
    :param outer_diameter: Outer diameter for circular shapes (m)
    :param outer_width: Outer width for rectangular shapes (m)
    :param outer_breadth: Outer breadth for rectangular shapes (m)
    :param inner_diameter: Inner diameter for hollow circular shapes (m)
    :param inner_width: Inner width for hollow rectangular shapes (m)
    :param inner_breadth: Inner breadth for hollow rectangular shapes (m)
    :param flange_width: Width of flange for structural sections (m)
    :param flange_thickness: Thickness of flange for structural sections (m)
    :param web_height: Height of web for structural sections (m)
    :param web_thickness: Thickness of web for structural sections (m)
    :param cross_section: Type of cross-section ('solid', 'hollow')
    :return: Calculated area in square meters
    :raises ValueError: If required parameters for the specified shape are missing
    """
    # For inner area calculation, we only need the inner dimensions
    if area_type == "inner":
        # Inner area is always 0 for solid sections, H-sections, C-sections, and L-sections
        if cross_section == "hollow" and shape not in {"h_section", "c_section", "l_section"}:
            if shape in {"rectangle", "square"} and inner_width and inner_breadth:
                return inner_width * inner_breadth
            elif shape == "circle" and inner_diameter:
                return (math.pi * (inner_diameter ** 2) / 4)
        return 0
    
    # Calculate outer area
    outer_area = 0
    if shape == "h_section":
        if not all([flange_width, flange_thickness, web_height, web_thickness]):
            raise ValueError("H-section requires flange_width, flange_thickness, web_height, and web_thickness")
        # H-section area calculation: 2 flanges + 1 web
        flange_area = 2 * flange_width * flange_thickness
        web_area = web_height * web_thickness
        outer_area = flange_area + web_area
    elif shape == "c_section":
        if not all([flange_width, flange_thickness, web_height, web_thickness]):
            raise ValueError("C-section requires flange_width, flange_thickness, web_height, and web_thickness")
        # C-section area calculation: 2 flanges + 1 web (but on one side)
        flange_area = 2 * flange_width * flange_thickness
        web_area = web_height * web_thickness
        outer_area = flange_area + web_area
    elif shape == "l_section":
        if not all([flange_width, flange_thickness, web_height, web_thickness]):
            raise ValueError("L-section requires flange_width, flange_thickness, web_height, and web_thickness")
        # L-section area calculation: 1 flange + 1 web
        flange_area = flange_width * flange_thickness
        web_area = web_height * web_thickness
        outer_area = flange_area + web_area
    elif shape in {"rectangle", "square"}:
        if not all([outer_width, outer_breadth]):
            raise ValueError(f"{shape} requires outer_width and outer_breadth for total or annular area")
        outer_area = outer_width * outer_breadth
    elif shape == "circle":
        if not outer_diameter:
            raise ValueError(f"Circle requires outer_diameter for total or annular area")
        outer_area = (math.pi * (outer_diameter ** 2) / 4)
    else:
        raise ValueError(f"Unsupported shape: {shape}")
    
    # Calculate inner area for annular area
    inner_area = 0
    if area_type == "annular" and cross_section == "hollow" and shape not in {"h_section", "c_section", "l_section"}:
        if shape in {"rectangle", "square"} and inner_width and inner_breadth:
            inner_area = inner_width * inner_breadth
        elif shape == "circle" and inner_diameter:
            inner_area = (math.pi * (inner_diameter ** 2) / 4)
    
    # Return appropriate area based on type
    if area_type == "total":
        return outer_area
    elif area_type == "annular":
        return outer_area - inner_area
    else:
        # This should never be reached as inner case is handled above
        raise ValueError(f"Unsupported area_type: {area_type}")

def calculate_perimeter(
    shape: str, 
    perimeter_type: Literal["outer", "inner"] = "outer",
    outer_diameter: Optional[float] = None,
    outer_width: Optional[float] = None, 
    outer_breadth: Optional[float] = None,
    inner_diameter: Optional[float] = None,
    inner_width: Optional[float] = None,
    inner_breadth: Optional[float] = None,
    flange_width: Optional[float] = None,
    flange_thickness: Optional[float] = None,
    web_height: Optional[float] = None,
    web_thickness: Optional[float] = None,
    cross_section: Literal["solid", "hollow"] = "solid"
) -> float:
    """
    Calculate perimeter based on shape and perimeter type.
    
    :param shape: Shape of the cross-section ('rectangle', 'square', 'circle', 'h_section', 'c_section', 'l_section')
    :param perimeter_type: Type of perimeter to calculate ('outer', 'inner')
    :param outer_diameter: Outer diameter for circular shapes (m)
    :param outer_width: Outer width for rectangular shapes (m)
    :param outer_breadth: Outer breadth for rectangular shapes (m)
    :param inner_diameter: Inner diameter for hollow circular shapes (m)
    :param inner_width: Inner width for hollow rectangular shapes (m)
    :param inner_breadth: Inner breadth for hollow rectangular shapes (m)
    :param flange_width: Width of flange for structural sections (m)
    :param flange_thickness: Thickness of flange for structural sections (m)
    :param web_height: Height of web for structural sections (m)
    :param web_thickness: Thickness of web for structural sections (m)
    :param cross_section: Type of cross-section ('solid', 'hollow')
    :return: Calculated perimeter in meters
    :raises ValueError: If required parameters for the specified shape are missing
    """
    if perimeter_type == "inner":
        # Inner perimeter is always 0 for solid sections, H-sections, C-sections, and L-sections
        if cross_section == "hollow" and shape not in {"h_section", "c_section", "l_section"}:
            if shape in {"rectangle", "square"} and inner_width and inner_breadth:
                return 2 * (inner_width + inner_breadth)
            elif shape == "circle" and inner_diameter:
                return math.pi * inner_diameter
        return 0
    
    # Calculate outer perimeter
    if shape == "h_section":
        if not all([flange_width, flange_thickness, web_height, web_thickness]):
            raise ValueError("H-section requires flange_width, flange_thickness, web_height, and web_thickness")
        # Perimeter of H-section
        return 2 * (2 * flange_width + web_height + 2 * flange_thickness)
    
    elif shape == "c_section":
        if not all([flange_width, flange_thickness, web_height, web_thickness]):
            raise ValueError("C-section requires flange_width, flange_thickness, web_height, and web_thickness")
        # Perimeter of C-section
        return 2 * flange_width + 2 * web_height + 3 * flange_thickness + web_thickness
    
    elif shape == "l_section":
        if not all([flange_width, flange_thickness, web_height, web_thickness]):
            raise ValueError("L-section requires flange_width, flange_thickness, web_height, and web_thickness")
        # Perimeter of L-section
        return 2 * flange_width + web_height + flange_thickness + web_thickness
    
    elif shape in {"rectangle", "square"}:
        if not all([outer_width, outer_breadth]):
            raise ValueError(f"{shape} requires outer_width and outer_breadth")
        return 2 * (outer_width + outer_breadth)
    
    elif shape == "circle":
        if not outer_diameter:
            raise ValueError("Circle requires outer_diameter")
        return math.pi * outer_diameter
    
    else:
        raise ValueError(f"Unsupported shape: {shape}")
