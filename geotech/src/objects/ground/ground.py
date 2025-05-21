import ast
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Set up the root path and ensure it's in the system path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), R"..\.."))
if root_path not in sys.path:
    sys.path.append(root_path)
    
from utils.src.common import is_invalid


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



class GroundSection(BaseModel):
    """
    Represents a section of the ground with specific parameters and properties.

    Attributes:
        section_id (str): Unique identifier for the section
        depth_from (float): Starting depth of the section (in meters)
        depth_to (float): Ending depth of the section (in meters)
        behaviour (str): Soil behavior type ('cohesive', 'frictional', 'c_phi', 'rock', 'calcareous')
        main_ground_unit (str): Primary ground material type ('sand', 'clay', 'silt', 'rock')
        name (str, optional): Descriptive name for the section
        slope (float): Slope angle in degrees (0-89)
        params (Dict[str, Any]): Dictionary of user-defined soil parameters:
            - unit_weight_sat (float): Saturated unit weight
            - unit_weight_bulk (float): Bulk unit weight
            - cohesion (float): Soil cohesion
            - angle_friction (float): Internal friction angle
            - ratio_poisson (float): Poisson's ratio
            - modulus_elastic (float): Elastic modulus
            - additional custom parameters as needed
            Each parameter can be:
            - A single value (constant throughout section)
            - A gradient definition with 'top' and 'bottom' values
            - An expression as a string that will be evaluated based on depth
        props (Dict[str, Union[Any, Dict[str, Any]]]): Dictionary of computed soil properties:
            Each property can be:
            - A single value (constant throughout section)
            - A gradient definition with 'top' and 'bottom' values
            - An expression as a string that will be evaluated based on depth
    """
    section_id: str
    depth_from: float = Field(..., ge=0, metadata={'unit': 'm', 'desc': 'Depth below ground surface'})
    depth_to: float = Field(..., gt=0, metadata={'unit': 'm', 'desc': 'Depth below ground surface'})
    behaviour: str
    main_ground_unit: str
    name: Optional[str] = Field(default='unnamed ground section')
    slope: float = Field(default=0.0, ge=0., le=89., metadata={'unit': 'deg','desc': 'Slope angle from the horizontal'})
    params: Dict[str, Union[float, Dict[str, float], str]] = Field(
        default_factory=dict, 
        description="Soil parameters as constant values, gradients (top/bottom), or expressions"
    )
    props: Dict[str, Union[float, Dict[str, float], str]] = Field(
        default_factory=dict,
        description="Computed properties as constant values, gradients (top/bottom), or expressions"
    )

    @field_validator('section_id', mode='before')
    def validate_section_id(cls, value):
        """Ensure section_id is a valid string."""
        if not value:
            raise ValueError("section_id cannot be empty")
        return str(value)
    
    @field_validator('name', mode='before')
    def validate_name(cls, value):
        """Ensure name is a valid string."""
        return str(value) if value is not None else 'unnamed ground section'

    @field_validator('behaviour', mode='before')
    def validate_behaviour(cls, value):
        """Ensure behaviour is one of the predefined valid types."""
        valid_behaviours = {'cohesive', 'frictional',
                            'c_phi', 'rock', 'calcareous'}
        if not isinstance(value, str) or value.lower() not in valid_behaviours:
            raise ValueError(
                f"Invalid soil behaviour '{value}', must be one of {valid_behaviours}")
        return value.lower()

    @field_validator('main_ground_unit', mode='before')
    def validate_main_ground_unit(cls, value):
        """Ensure main_ground_unit is one of the predefined valid types."""
        valid_main_ground_units = {'sand', 'clay', 'silt', 'rock'}
        if not isinstance(value, str) or value.lower() not in valid_main_ground_units:
            raise ValueError(
                f"Invalid main ground unit '{value}', must be one of {valid_main_ground_units}")
        return value.lower()

    @field_validator('params', mode='before')
    def validate_params(cls, value):
        """Ensure params is a dictionary."""
        if not isinstance(value, dict):
            raise ValueError("params must be a dictionary.")
        return value
        
    @field_validator('props', mode='before')
    def validate_props(cls, value):
        """Ensure props is a dictionary."""
        if not isinstance(value, dict):
            raise ValueError("props must be a dictionary.")
        return value

    @model_validator(mode='after')
    def validate_depth_order(self):
        """Ensure depth_from is less than depth_to."""
        if self.depth_from >= self.depth_to:
            raise ValueError(
                f"depth_from ({self.depth_from}) must be less than depth_to ({self.depth_to}).")
        return self
    
    @model_validator(mode='after')
    def validate_soil_parameters(self):
        """Ensure required soil parameters are valid based on soil behaviour."""
        unit_weight_bulk = self.params.get('unit_weight_bulk')
        unit_weight_sat = self.params.get('unit_weight_sat')
        cohesion = self.params.get('cohesion')
        angle_friction = self.params.get('angle_friction')
        additional_invalid_values = [0]

        # Define valid parameter ranges
        param_ranges = {
            'unit_weight_bulk': (10, 30),  # kN/m³
            'unit_weight_sat': (10, 30),   # kN/m³
            'cohesion': (0, 10000),        # kPa
            'angle_friction': (0, 50),     # degrees
            'ratio_poisson': (0.1, 0.5),   # dimensionless
            'modulus_elastic': (100, 1e6)  # kPa
        }

        # Check existence for required parameters
        if is_invalid(unit_weight_bulk, additional_invalid_values=additional_invalid_values):
            raise ValueError(
                f"Ground section requires a nonzero unit_weight_bulk")
        if is_invalid(unit_weight_sat, additional_invalid_values=additional_invalid_values):
            raise ValueError(
                f"Ground section requires a nonzero unit_weight_sat")
        
        behaviour_validations = {
            'cohesive': lambda: is_invalid(cohesion, additional_invalid_values=additional_invalid_values),
            'frictional': lambda: is_invalid(angle_friction, additional_invalid_values=additional_invalid_values),
            'c_phi': lambda: (is_invalid(cohesion, additional_invalid_values=additional_invalid_values) or 
                                is_invalid(angle_friction, additional_invalid_values=additional_invalid_values))
        }
        
        if self.behaviour in behaviour_validations and behaviour_validations[self.behaviour]():
            if self.behaviour == 'c_phi':
                raise ValueError(
                    f"{self.behaviour} soil requires finite nonzero cohesion and angle_friction values.")
            else:
                param_name = 'cohesion' if self.behaviour == 'cohesive' else 'angle_friction'
                raise ValueError(
                    f"{self.behaviour} soil requires a finite nonzero {param_name} value.")

        # Ensure defaults for specific soil behaviors
        if self.behaviour == 'cohesive':
            self.params['angle_friction'] = 0
        if self.behaviour == 'frictional':
            self.params['cohesion'] = 0

        # Validate parameter ranges
        for param_name, param_value in self.params.items():
            if param_name in param_ranges:
                min_val, max_val = param_ranges[param_name]
                
                # Handle different parameter types
                if isinstance(param_value, (int, float)):
                    if not (min_val <= param_value <= max_val):
                        raise ValueError(f"Parameter '{param_name}' value {param_value} is outside valid range [{min_val}, {max_val}]")
                
                # Handle gradient parameters
                elif isinstance(param_value, dict) and 'top' in param_value and 'bottom' in param_value:
                    top_val = param_value['top']
                    bottom_val = param_value['bottom']
                    if not (min_val <= top_val <= max_val):
                        raise ValueError(f"Parameter '{param_name}' top value {top_val} is outside valid range [{min_val}, {max_val}]")
                    if not (min_val <= bottom_val <= max_val):
                        raise ValueError(f"Parameter '{param_name}' bottom value {bottom_val} is outside valid range [{min_val}, {max_val}]")

        return self

    @property
    def thickness(self) -> float:
        """Calculate the thickness of the section.
        
        Returns:
            float: Thickness of section in meters
        """
        return self.depth_to - self.depth_from
    
    @property
    def midpoint_depth(self) -> float:
        """Calculate the midpoint depth of the section.
        
        Returns:
            float: Midpoint depth of the section in meters
        """
        return (self.depth_from + self.depth_to) / 2
    
    def get_property(self, property_name: str, default: Any = None) -> Any:
        """Get a soil property value at a specific depth.
        
        Args:
            property_name: Name of the soil property
            default: Default value to return if property doesn't exist
            
        Returns:
            The value of the property at the specified depth or the default value
        """
        # First check if it's a user-defined parameter
        if property_name in self.params:
            return self.params.get(property_name, default)
        
        # Then check if it's a computed property
        if property_name in self.props:
            return self.props.get(property_name, default)

        # If property not found, return default
        return default
    
    def set_property(self, property_name: str, value: Any) -> None:
        """Set a computed soil property.
        
        Args:
            property_name: Name of the soil property
            value: Value to set (can be constant, gradient dict, or expression string)
        """
        self.props[property_name] = value
        
    def set_param(self, param_name: str, value: Any) -> None:
        """Set a user-defined soil parameter.
        
        Args:
            param_name: Name of the soil parameter
            value: Value to set
        """
        self.params[param_name] = value
    def compute_default_props(self) -> None:
        """Calculate default computed props based on basic soil parameters."""
        
        # Calculate earth pressure coefficient at rest (K0) using Jaky's formula
        if 'angle_friction' in self.params:
            phi = self.params['angle_friction']
            # Define k0_transform to accept additional parameters (even if not used)
            k0_transform = lambda angle, **kwargs: 1 - math.sin(math.radians(angle))
            self.props['k0'] = transform_parameter(phi, k0_transform, 'angle_friction')


class Ground(BaseModel):
    """
    Represents a ground model consisting of multiple vertical soil sections.

    Attributes:
        gwt (float): Groundwater table depth in meters (≥ 0, default is ∞)
        sections (List[GroundSection]): List of ground sections from surface downward
        name (str, optional): Name of the ground model
    
    Properties:
        ground_depth (float): Maximum depth of the ground model
        number_of_sections (int): Total number of ground sections
    """
    gwt: float = Field(default=float('inf'), ge=0, metadata={'unit': 'm'})
    sections: List[GroundSection] = Field()
    name: Optional[str] = Field(default='unnamed ground model')

    @property
    def ground_depth(self) -> float:
        """Returns the maximum ground depth from the last section.

        Returns:
            float: Maximum depth of the ground model in meters
        """
        if not self.sections:
            return 0.0
        return self.sections[-1].depth_to

    @property
    def number_of_sections(self) -> int:
        """Returns the total number of ground sections.

        Returns:
            int: Number of sections in the ground model
        """
        return len(self.sections) if self.sections else 0

    @model_validator(mode="before")
    def validate_and_reorder_sections(cls, values):
        """Validates and orders ground sections by depth.

        - Ensures at least one section exists.
        - Sorts sections in ascending order of `depth_from`.
        - Checks for continuity (no gaps between `depth_to` and `depth_from`).
        - Ensures unique section IDs.
        - Confirms ground starts at `depth_from = 0`.

        Raises:
            ValueError: If sections are missing, unordered, discontinuous, or have duplicate IDs.
        
        Returns:
            dict: Validated and sorted sections.
        """
        sections = values.get("sections", [])

        # Ensure at least one foundation section is present
        if not sections:
            raise ValueError(
                "At least one foundation section must be present.")

        # Reorder sections in ascending order of depth_from
        sorted_sections = sorted(sections, key=lambda sec: sec.depth_from)

        # Ensure sections are continuous (no gaps in depth)
        for i in range(len(sorted_sections) - 1):
            if sorted_sections[i].depth_to != sorted_sections[i + 1].depth_from:
                raise ValueError(
                    f"Depths must be continuous, but found a gap/overlap between "
                    f"Section {sorted_sections[i].section_id} with depth_to {sorted_sections[i].depth_to}m "
                    f"and Section {sorted_sections[i + 1].section_id} with depth_from {sorted_sections[i + 1].depth_from}m."
                )

        # Check for unique section_id values
        section_ids = [sec.section_id for sec in sorted_sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("Each section must have a unique section_id.")

        # First section's depth_from should be 0
        if sorted_sections[0].depth_from != 0:
            raise ValueError("Ground must start at depth_from = 0.0")

        values["sections"] = sorted_sections
        return values

    def refine_ground_sections(self, depth_increment: float = 0.1) -> 'Ground':
        """
        Divides each ground section into multiple finer sections with the specified depth increments.

        Args:
            depth_increment (float): Depth increment for dividing sections, must be positive.
            
        Returns:
            Ground: A new Ground object with refined sections.
            
        Raises:
            ValueError: If depth_increment is not positive.
        """
        if depth_increment <= 0:
            raise ValueError("Depth increment must be a positive number.")
        
        refined_sections = []
        section_counter = 1
        
        for section in self.sections:
            current_depth = section.depth_from
            while current_depth < section.depth_to:
                next_depth = min(current_depth + depth_increment, section.depth_to)
                
                new_section = section.model_copy(deep=True)
                new_section.depth_from = current_depth
                new_section.depth_to = next_depth
                new_section.section_id = f"{section.section_id}_{section_counter}"
                refined_sections.append(new_section)
                
                current_depth = next_depth
                section_counter += 1
        
        refined_ground = self.model_copy(deep=True)
        refined_ground.sections = refined_sections
        return refined_ground

    def get_section_at_depth(self, depth: float) -> Optional[GroundSection]:
        """
        Retrieves the ground section at a specific depth.
        
        Args:
            depth (float): The depth to query in meters
            
        Returns:
            Optional[GroundSection]: The section at the specified depth or None if no section exists
        """
        if depth < 0 or depth > self.ground_depth:
            return None
            
        for section in self.sections:
            if section.depth_from <= depth < section.depth_to:
                return section
            
        # Handle the case where depth equals the maximum depth
        if depth == self.ground_depth and self.sections:
            return self.sections[-1]
            
        return None
    
    def is_below_water_table(self, depth: float) -> bool:
        """
        Determines if a depth is below the groundwater table.
        
        Args:
            depth (float): The depth to check in meters
            
        Returns:
            bool: True if depth is below water table, False otherwise
        """
        return depth >= self.gwt
    
    def get_property_profile(self, property_name: str) -> List[Tuple[float, Any]]:
        """
        Creates a depth profile of a specific soil property.
        
        Args:
            property_name (str): Name of the soil property to profile
            
        Returns:
            List[Tuple[float, Any]]: List of (depth, value) pairs for the property
        """
        profile = []
        for section in self.sections:
            value = section.get_property(property_name)
            profile.append((section.depth_from, value))
            
        # Add the final depth point
        if self.sections:
            profile.append((self.sections[-1].depth_to, 
                           self.sections[-1].get_property(property_name)))
            
        return profile


if __name__ == "__main__":
    # Sample usage example
    try:
        # Create a simple two-layer ground profile
        clay_section = GroundSection(
            section_id="clay1",
            depth_from=0.0,
            depth_to=5.0,
            behaviour="cohesive",
            main_ground_unit="clay",
            name="Soft Clay Layer",
            params={
                "unit_weight_bulk": 18.0,
                "unit_weight_sat": 19.0,
                "cohesion": 25.0,
                "modulus_elastic": 15000.0
            }
        )
        
        sand_section = GroundSection(
            section_id="sand1",
            depth_from=5.0,
            depth_to=10.0,
            behaviour="frictional",
            main_ground_unit="sand",
            name="Dense Sand Layer",
            params={
                "unit_weight_bulk": 19.0,
                "unit_weight_sat": 20.0,
                "angle_friction": 35.0,
                "modulus_elastic": 40000.0
            }
        )
        
        ground_model = Ground(
            gwt=3.0,
            sections=[clay_section, sand_section],
            name="Test Site Ground Model"
        )
        
        print(f"Ground model: {ground_model.name}")
        print(f"Ground depth: {ground_model.ground_depth}m")
        print(f"Water table depth: {ground_model.gwt}m")
        print(f"Number of sections: {ground_model.number_of_sections}")
        
        # Refine ground model
        refined_model = ground_model.refine_ground_sections(depth_increment=1.0)
        print(f"Refined model has {refined_model.number_of_sections} sections")
        
    except Exception as e:
        print(f"Error: {str(e)}")
