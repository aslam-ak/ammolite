import ast
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Set up the root path and ensure it's in the system path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), R"..\..\..\.."))
if root_path not in sys.path:
    sys.path.append(root_path)
    
import numpy as np

from utils.src.common import is_invalid, transform_parameter


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
            - unit_weight_sat (float): Saturated unit weight (kN/m³)
            - unit_weight_bulk (float): Bulk unit weight (kN/m³)
            - cohesion (float): Soil cohesion (kPa)
            - angle_friction (float): Internal friction angle (degrees)
            - ratio_poisson (float): Poisson's ratio (dimensionless)
            - modulus_elastic (float): Elastic modulus (kPa)
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
        """
        Ensure section_id is a valid string.
        
        Args:
            value: The section_id to validate
        
        Returns:
            str: The validated section_id
            
        Raises:
            ValueError: If section_id is empty
        """
        if not value:
            raise ValueError("section_id cannot be empty")
        return str(value)
    
    @field_validator('name', mode='before')
    def validate_name(cls, value):
        """
        Ensure name is a valid string.
        
        Args:
            value: The name to validate
            
        Returns:
            str: The validated name or default name if None
        """
        return str(value) if value is not None else 'unnamed ground section'

    @field_validator('behaviour', mode='before')
    def validate_behaviour(cls, value):
        """
        Ensure behaviour is one of the predefined valid types.
        
        Args:
            value: The behaviour to validate
            
        Returns:
            str: The validated behaviour in lowercase
            
        Raises:
            ValueError: If behaviour is not one of the valid types
        """
        valid_behaviours = {'cohesive', 'frictional',
                            'c_phi', 'rock', 'calcareous'}
        if not isinstance(value, str) or value.lower() not in valid_behaviours:
            raise ValueError(
                f"Invalid soil behaviour '{value}', must be one of {valid_behaviours}")
        return value.lower()

    @field_validator('main_ground_unit', mode='before')
    def validate_main_ground_unit(cls, value):
        """
        Ensure main_ground_unit is one of the predefined valid types.
        
        Args:
            value: The main_ground_unit to validate
            
        Returns:
            str: The validated main_ground_unit in lowercase
            
        Raises:
            ValueError: If main_ground_unit is not one of the valid types
        """
        valid_main_ground_units = {'sand', 'clay', 'silt', 'rock'}
        if not isinstance(value, str) or value.lower() not in valid_main_ground_units:
            raise ValueError(
                f"Invalid main ground unit '{value}', must be one of {valid_main_ground_units}")
        return value.lower()

    @field_validator('params', mode='before')
    def validate_params(cls, value):
        """
        Ensure params is a dictionary.
        
        Args:
            value: The params to validate
            
        Returns:
            dict: The validated params
            
        Raises:
            ValueError: If params is not a dictionary
        """
        if not isinstance(value, dict):
            raise ValueError("params must be a dictionary.")
        return value
        
    @field_validator('props', mode='before')
    def validate_props(cls, value):
        """
        Ensure props is a dictionary.
        
        Args:
            value: The props to validate
            
        Returns:
            dict: The validated props
            
        Raises:
            ValueError: If props is not a dictionary
        """
        if not isinstance(value, dict):
            raise ValueError("props must be a dictionary.")
        return value

    @model_validator(mode='after')
    def validate_depth_order(self):
        """
        Ensure depth_from is less than depth_to.
        
        Returns:
            GroundSection: The validated section
            
        Raises:
            ValueError: If depth_from is not less than depth_to
        """
        if self.depth_from >= self.depth_to:
            raise ValueError(
                f"depth_from ({self.depth_from}) must be less than depth_to ({self.depth_to}).")
        return self
    
    @model_validator(mode='after')
    def validate_soil_parameters(self):
        """
        Ensure required soil parameters are valid based on soil behaviour.
        
        Checks for:
        - Required unit weights (bulk and saturated)
        - Behavior-specific parameters (cohesion for cohesive, angle_friction for frictional)
        - Parameter values within valid ranges
        
        Returns:
            GroundSection: The validated section with default values set
            
        Raises:
            ValueError: If any required parameter is missing or invalid
        """
        unit_weight_bulk = self.params.get('unit_weight_bulk')
        unit_weight_sat = self.params.get('unit_weight_sat')
        cohesion = self.params.get('cohesion')
        angle_friction = self.params.get('angle_friction')
        custom_invalid_values = [0]

        # Define valid parameter ranges
        param_ranges = {
            'unit_weight_bulk': (10, 30),       # kN/m³
            'unit_weight_sat': (10, 30),        # kN/m³
            'cohesion': (0, 10000),             # kPa
            'angle_friction': (0, 50),          # degrees
            'angle_friction_peak': (0, 50),     # degrees
            'angle_friction_residual': (0, 50), # degrees
            'ratio_poisson': (0.1, 0.5),        # dimensionless
            'modulus_elastic': (100, 1e6)       # kPa
        }

        # Check existence for required parameters
        # For unit weight parameters, check if they are valid in various forms
        for param_name in ['unit_weight_bulk', 'unit_weight_sat']:
            param_value = self.params.get(param_name)
            
            # Skip if parameter doesn't exist
            if param_value is None:
                raise ValueError(f"Ground section requires {param_name}")
            
            # Handle different parameter formats
            if isinstance(param_value, (int, float)):
            # Simple value
                if is_invalid(param_value, custom_invalid_values=custom_invalid_values):
                    raise ValueError(f"Ground section requires a nonzero {param_name}")
                
            elif isinstance(param_value, dict) and 'top' in param_value and 'bottom' in param_value:
            # Gradient definition with top/bottom values
                top_value = param_value['top']
                bottom_value = param_value['bottom']
                if is_invalid(top_value, custom_invalid_values=custom_invalid_values) or \
                is_invalid(bottom_value, custom_invalid_values=custom_invalid_values):
                    raise ValueError(f"Ground section requires nonzero values for {param_name} gradient")
                
            elif isinstance(param_value, str) and param_value.strip():
            # Expression string - we assume it's valid if non-empty
            # (will be evaluated elsewhere)
                pass
            
            else:
            # Invalid format
                raise ValueError(f"Invalid format for {param_name}: must be a number, top/bottom dict, or expression")
        
        behaviour_validations = {
            'cohesive': lambda: is_invalid(cohesion, custom_invalid_values=custom_invalid_values),
            'frictional': lambda: is_invalid(angle_friction, custom_invalid_values=custom_invalid_values),
            'c_phi': lambda: (is_invalid(cohesion, custom_invalid_values=custom_invalid_values) or 
                                is_invalid(angle_friction, custom_invalid_values=custom_invalid_values))
        }
        
        if self.behaviour in behaviour_validations and behaviour_validations[self.behaviour]():
            if self.behaviour == 'c_phi':
                raise ValueError(
                    f"{self.behaviour} soil requires finite nonzero cohesion and angle_friction values.")
            else:
                param_name = 'cohesion' if self.behaviour == 'cohesive' else 'angle_friction'
                param_value = self.params.get(param_name)
                
                # Check if the parameter is a valid dictionary with top/bottom values
                if isinstance(param_value, dict) and 'top' in param_value and 'bottom' in param_value:
                    top_value = param_value['top']
                    bottom_value = param_value['bottom']
                    if is_invalid(top_value, custom_invalid_values=custom_invalid_values) and \
                       is_invalid(bottom_value, custom_invalid_values=custom_invalid_values):
                        raise ValueError(
                            f"{self.behaviour} soil requires finite nonzero values for {param_name}.")
                
                # Check if the parameter is a string expression (will be evaluated later)
                elif isinstance(param_value, str) and param_value.strip():
                    # String expressions are assumed to be valid if they're non-empty
                    pass
                
                # If it's a simple value and invalid
                else:
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
        """
        Calculate the thickness of the section.
        
        Returns:
            float: Thickness of section in meters
        """
        return self.depth_to - self.depth_from
    
    @property
    def midpoint_depth(self) -> float:
        """
        Calculate the midpoint depth of the section.
        
        Returns:
            float: Midpoint depth of the section in meters
        """
        return (self.depth_from + self.depth_to) / 2
    
    def get_property(self, property_name: str, default: Any = None) -> Any:
        """
        Get a soil property value.
        
        Retrieves property from params or props dictionaries.
        
        Args:
            property_name: Name of the soil property
            default: Default value to return if property doesn't exist
            
        Returns:
            The value of the property or the default value if property not found
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
        """
        Set a computed soil property.
        
        Args:
            property_name: Name of the soil property
            value: Value to set (can be constant, gradient dict, or expression string)
        """
        self.props[property_name] = value
        
    def set_param(self, param_name: str, value: Any) -> None:
        """
        Set a user-defined soil parameter.
        
        Args:
            param_name: Name of the soil parameter
            value: Value to set (can be constant, gradient dict, or expression string)
        """
        self.params[param_name] = value
        
    def compute_default_props(self) -> None:
        """
        Calculate default computed props based on basic soil parameters.
        
        Currently calculates:
        - Earth pressure coefficient at rest (K0) using Jaky's formula
        """
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
        name (str): Name of the ground model
        ground_depth (float): Maximum depth of the ground model in meters
        number_of_sections (int): Total number of ground sections
        gwt (float): Groundwater table depth in meters (≥ 0, default is ∞)
        unit_weight_water (float): Unit weight of water in kN/m³
        sections (List[GroundSection]): List of ground sections from surface downward
    """
    name: Optional[str] = Field(default='unnamed ground model')
    ground_depth: float = Field(default=0.0, ge=0.0, metadata={'unit': 'm', 'desc': 'Maximum depth of the ground model'})
    number_of_sections: int = Field(default=0, ge=0, metadata={'desc': 'Total number of ground sections'})
    gwt: float = Field(default=float('inf'), ge=0, metadata={'unit': 'm', 'desc': 'Groundwater table depth below ground surface'})
    unit_weight_water: float = Field(default=9.81, ge=5, le=15, metadata={'unit': 'kN/m³', 'desc': 'Unit weight of water'})
    sections: List[GroundSection] = Field(default_factory=list, description="List of ground sections from top to bottom")

    def __init__(self, **data: Any) -> None:
        """
        Initialize a Ground model with sections and calculate derived properties.
        
        Args:
            **data: Keyword arguments matching the Ground model attributes
        """
        super().__init__(**data)
        self.ground_depth = self._calculate_ground_depth()
        self.number_of_sections = self._calculate_number_of_sections()

    def _calculate_ground_depth(self) -> float:
        """
        Returns the maximum ground depth from the last section.

        Returns:
            float: Maximum depth of the ground model in meters
        """
        if not self.sections:
            return 0.0
        return self.sections[-1].depth_to

    def _calculate_number_of_sections(self) -> int:
        """
        Returns the total number of ground sections.

        Returns:
            int: Number of sections in the ground model
        """
        return len(self.sections)

    @model_validator(mode="before")
    def validate_and_reorder_sections(cls, values):
        """
        Validates and orders ground sections by depth.

        - Ensures at least one section exists.
        - Sorts sections in ascending order of `depth_from`.
        - Checks for continuity (no gaps between `depth_to` and `depth_from`).
        - Ensures unique section IDs.
        - Confirms ground starts at `depth_from = 0`.

        Args:
            values: Dictionary of model field values
            
        Raises:
            ValueError: If sections are missing, unordered, discontinuous, or have duplicate IDs.
        
        Returns:
            dict: Validated and sorted values dictionary
        """
        sections = values.get("sections", [])

        # Ensure at least one section is present
        if not sections:
            raise ValueError(
                "At least one ground section must be present.")

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
            # Find duplicate IDs for better error message
            duplicate_ids = [id for id in section_ids if section_ids.count(id) > 1]
            raise ValueError(f"Each section must have a unique section_id. Duplicate IDs found: {set(duplicate_ids)}")

        # First section's depth_from should be 0
        if sorted_sections[0].depth_from != 0:
            raise ValueError("Ground must start at depth_from = 0.0")

        values["sections"] = sorted_sections
        return values
    
    def add_ground_section(self, new_section: GroundSection) -> None:
        """
        Adds a new ground section to the model, ensuring proper depth continuity.
        
        Args:
            new_section (GroundSection): The new ground section to add
            
        Raises:
            ValueError: If the section's depth doesn't align with existing sections
                       or if its section_id is not unique
        """
        # Check if this is the first section (special case)
        if not self.sections:
            if new_section.depth_from != 0:
                raise ValueError("First ground section must start at depth_from = 0.0")
            self.sections = [new_section]
            self.ground_depth = new_section.depth_to
            self.number_of_sections = 1
            return
            
        # Check for unique section_id
        section_ids = [sec.section_id for sec in self.sections]
        if new_section.section_id in section_ids:
            raise ValueError(f"Section ID '{new_section.section_id}' already exists. Section IDs must be unique.")
            
        # Determine where to insert the new section
        if new_section.depth_from == 0:
            # New section starts at the top
            if new_section.depth_to != self.sections[0].depth_from:
                raise ValueError(f"New section ending at {new_section.depth_to}m must connect with existing section starting at {self.sections[0].depth_from}m")
            self.sections.insert(0, new_section)
        elif new_section.depth_from == self.ground_depth:
            # New section extends the ground model deeper
            if self.sections[-1].depth_to != new_section.depth_from:
                raise ValueError(f"New section must start at current maximum depth {self.ground_depth}m")
            self.sections.append(new_section)
            self.ground_depth = new_section.depth_to
        else:
            # New section goes somewhere in the middle, find the right position
            insertion_index = None
            for i, section in enumerate(self.sections):
                if section.depth_from == new_section.depth_to:
                    # Insert before this section
                    insertion_index = i
                    # Check if new section connects to previous section
                    if i > 0 and self.sections[i-1].depth_to != new_section.depth_from:
                        raise ValueError(f"New section must connect with existing sections")
                    break
                elif section.depth_to == new_section.depth_from:
                    # Insert after this section
                    insertion_index = i + 1
                    # Check if new section connects to next section
                    if i < len(self.sections) - 1 and self.sections[i+1].depth_from != new_section.depth_to:
                        raise ValueError(f"New section must connect with existing sections")
                    break
                    
            if insertion_index is None:
                raise ValueError(f"New section with depth_from={new_section.depth_from}m and depth_to={new_section.depth_to}m doesn't align with existing sections")
                
            self.sections.insert(insertion_index, new_section)
            
        # Re-sort the sections to ensure proper order
        self.sections.sort(key=lambda sec: sec.depth_from)
        
        # Update ground_depth and number_of_sections
        self.number_of_sections = len(self.sections)
        if self.sections:
            self.ground_depth = self.sections[-1].depth_to
            
    def calculate_vertical_stresses(self) -> None:
        """
        Calculates vertical total and effective stresses at midpoint and bottom of each section.
        
        The stresses are calculated cumulatively:
        - For each section, stress increments are computed:
            1. From section top to midpoint
            2. From section top to bottom
        - Stresses are cumulative, including stress from all previous sections
        - Both total stress (using appropriate unit weights) and effective stress 
            (accounting for water pressure below the water table) are calculated.
        
        Updates each section's props with:
        - 'stress_vertical_total_mid': Total stress at midpoint
        - 'stress_vertical_effective_mid': Effective stress at midpoint
        - 'stress_vertical_total_bottom': Total stress at bottom
        - 'stress_vertical_effective_bottom': Effective stress at bottom
        """
        cumulative_total_stress = 0.0
        cumulative_effective_stress = 0.0
        
        for i, section in enumerate(self.sections):
            # Get unit weights for current section
            unit_weight_bulk = section.get_property('unit_weight_bulk')
            unit_weight_sat = section.get_property('unit_weight_sat')
            
            # Check for valid unit weights
            if isinstance(unit_weight_bulk, (dict, str)) or isinstance(unit_weight_sat, (dict, str)):
                raise ValueError(f"Section {section.section_id}: Stress calculation requires constant unit weight values, not gradients or expressions")
            
            # Calculate stress increment from top to midpoint of this section
            depth_from = section.depth_from
            mid_depth = section.midpoint_depth
            depth_to = section.depth_to
            
            distance_to_mid = mid_depth - depth_from
            distance_to_bottom = depth_to - depth_from
            
            # Initialize stress increments
            total_stress_increment_to_mid = 0.0
            effective_stress_increment_to_mid = 0.0
            total_stress_increment_to_bottom = 0.0
            effective_stress_increment_to_bottom = 0.0
            
            # --- Calculate stress increment to midpoint ---
            
            # If the water table is within this section, handle differently
            if self.gwt > mid_depth:
                # Midpoint is above water table - use bulk unit weight
                total_stress_increment_to_mid = unit_weight_bulk * distance_to_mid
                effective_stress_increment_to_mid = unit_weight_bulk * distance_to_mid
            elif self.gwt <= depth_from:
                # Section entirely below water table - use saturated unit weight
                total_stress_increment_to_mid = unit_weight_sat * distance_to_mid
                effective_stress_increment_to_mid = (unit_weight_sat - self.unit_weight_water) * distance_to_mid
            else:
                # Water table within this segment (between top and midpoint)
                above_water_distance = self.gwt - depth_from
                below_water_distance = mid_depth - self.gwt
                
                # Above water contribution
                total_stress_increment_to_mid = unit_weight_bulk * above_water_distance
                effective_stress_increment_to_mid = unit_weight_bulk * above_water_distance
                
                # Below water contribution
                total_stress_increment_to_mid += unit_weight_sat * below_water_distance
                effective_stress_increment_to_mid += (unit_weight_sat - self.unit_weight_water) * below_water_distance
            
            # --- Calculate stress increment to bottom ---
            
            # If the water table is within this section, handle differently
            if self.gwt > depth_to:
                # Bottom is above water table - use bulk unit weight
                total_stress_increment_to_bottom = unit_weight_bulk * distance_to_bottom
                effective_stress_increment_to_bottom = unit_weight_bulk * distance_to_bottom
            elif self.gwt <= depth_from:
                # Section entirely below water table - use saturated unit weight
                total_stress_increment_to_bottom = unit_weight_sat * distance_to_bottom
                effective_stress_increment_to_bottom = (unit_weight_sat - self.unit_weight_water) * distance_to_bottom
            else:
                # Water table within this section
                above_water_distance = self.gwt - depth_from
                below_water_distance = depth_to - self.gwt
                
                # Above water contribution
                total_stress_increment_to_bottom = unit_weight_bulk * above_water_distance
                effective_stress_increment_to_bottom = unit_weight_bulk * above_water_distance
                
                # Below water contribution
                total_stress_increment_to_bottom += unit_weight_sat * below_water_distance
                effective_stress_increment_to_bottom += (unit_weight_sat - self.unit_weight_water) * below_water_distance
            
            # Update cumulative stresses at midpoint
            total_stress_at_mid = cumulative_total_stress + total_stress_increment_to_mid
            effective_stress_at_mid = cumulative_effective_stress + effective_stress_increment_to_mid
            
            # Update cumulative stresses at bottom (for next layer)
            cumulative_total_stress += total_stress_increment_to_bottom
            cumulative_effective_stress += effective_stress_increment_to_bottom
            
            # Store the stresses in the section's properties
            section.set_property('stress_vertical_total_mid', total_stress_at_mid)
            section.set_property('stress_vertical_effective_mid', effective_stress_at_mid)
            section.set_property('stress_vertical_total_bottom', cumulative_total_stress)
            section.set_property('stress_vertical_effective_bottom', cumulative_effective_stress)
            
            
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
        refined_ground.name = f"{self.name}_refined@{depth_increment}m"
        refined_ground.number_of_sections = len(refined_sections)
        
        # Update parameter values for gradients in the refined sections
        for refined_section in refined_sections:
            # Find the original section that this refined section belongs to
            original_section = None
            for orig_section in self.sections:
                if refined_section.section_id.startswith(orig_section.section_id + "_"):
                    original_section = orig_section
                    break
                    
            if not original_section:
                continue
            
            # Calculate the relative position within the original section
            orig_thickness = original_section.depth_to - original_section.depth_from
            
            # Update params that have top/bottom gradients
            for param_name, param_value in refined_section.params.items():
                if isinstance(param_value, dict) and 'top' in param_value and 'bottom' in param_value:
                    orig_top = param_value['top']
                    orig_bottom = param_value['bottom']
                    
                    # Calculate relative position in the original section
                    rel_top = (refined_section.depth_from - original_section.depth_from) / orig_thickness
                    rel_bottom = (refined_section.depth_to - original_section.depth_from) / orig_thickness
                    
                    # Linear interpolation
                    new_top = orig_top + rel_top * (orig_bottom - orig_top)
                    new_bottom = orig_top + rel_bottom * (orig_bottom - orig_top)
                    
                    # Update the parameter
                    refined_section.params[param_name] = {'top': new_top, 'bottom': new_bottom}
            
            # Update props that have top/bottom gradients
            for prop_name, prop_value in refined_section.props.items():
                if isinstance(prop_value, dict) and 'top' in prop_value and 'bottom' in prop_value:
                    orig_top = prop_value['top']
                    orig_bottom = prop_value['bottom']
                    
                    # Calculate relative position in the original section
                    rel_top = (refined_section.depth_from - original_section.depth_from) / orig_thickness
                    rel_bottom = (refined_section.depth_to - original_section.depth_from) / orig_thickness
                    
                    # Linear interpolation
                    new_top = orig_top + rel_top * (orig_bottom - orig_top)
                    new_bottom = orig_top + rel_bottom * (orig_bottom - orig_top)
                    
                    # Update the property
                    refined_section.props[prop_name] = {'top': new_top, 'bottom': new_bottom}
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
                "unit_weight_bulk": 18.0,  # kN/m³
                "unit_weight_sat": 19.0,   # kN/m³
                "cohesion": {'top': 20.0, 'bottom': 30.0},  # Gradient from top to bottom (kPa)
                "modulus_elastic": 15000.0  # kPa
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
                "unit_weight_bulk": 19.0,    # kN/m³
                "unit_weight_sat": 20.0,     # kN/m³
                "angle_friction": 35.0,      # degrees
                "modulus_elastic": 40000.0   # kPa
            }
        )
        
        # Create the ground model with both sections
        ground_model = Ground(
            gwt=3.0,  # Groundwater table at 3m depth
            sections=[clay_section, sand_section],
            name="Test Site Ground Model"
        )
        ground_model = refined_model
        # Display basic ground model information
        print(f"Ground model: {ground_model.name}")
        print(f"Ground depth: {ground_model.ground_depth}m")
        print(f"Water table depth: {ground_model.gwt}m")
        print(f"Number of sections: {ground_model.number_of_sections}")
        print("\nGround sections:")
        for i, section in enumerate(ground_model.sections):
            print(f"  Section {i+1}: {section.name} ({section.depth_from}m to {section.depth_to}m)")
            print(f"    - Behaviour: {section.behaviour}")
            print(f"    - Main ground unit: {section.main_ground_unit}")
            print(f"    - Unit weight bulk: {section.get_property('unit_weight_bulk')} kN/m³")
            print(f"    - Unit weight saturated: {section.get_property('unit_weight_sat')} kN/m³")
        
        # Calculate vertical stresses
        ground_model.calculate_vertical_stresses()
        print("\nVertical stresses calculated:")
        for i, section in enumerate(ground_model.sections):
            print(f"  Section {i+1}: {section.name}")
            print(f"    - Total stress at midpoint: {section.props.get('stress_vertical_total_mid'):.2f} kPa")
            print(f"    - Effective stress at midpoint: {section.props.get('stress_vertical_effective_mid'):.2f} kPa")
        
        # Refine ground model
        refined_model = ground_model.refine_ground_sections(depth_increment=1.0)
        print(f"\nRefined model has {refined_model.number_of_sections} sections")
        
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")



    # # Calculate vertical stresses
    # ground_model.calculate_vertical_stresses()
    # # Create data for plotting
    # depths = []
    # total_stress = []
    # effective_stress = []
    # gwt_line = []

    # # Extract stress values for plotting
    # for section in ground_model.sections:
    #     depths.append(section.midpoint_depth)
    #     total_stress.append(section.props.get('stress_vertical_total_mid', 0))
    #     effective_stress.append(section.props.get('stress_vertical_effective_mid', 0))

    # # Plot the stress profiles
    # import matplotlib.pyplot as plt

    # # Create two subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # # Create continuous stress profile
    # depth_points = np.linspace(0, ground_model.ground_depth, 100)
    # total_stress_curve = []
    # effective_stress_curve = []
    # unit_weight_curve = []
    # effective_unit_weight_curve = []

    # for depth in depth_points:
    #     section = ground_model.get_section_at_depth(depth)
    #     if section:
    #         # Linear interpolation between top and bottom stresses within each section
    #         if depth == section.depth_from:
    #             if section == ground_model.sections[0]:
    #                 total_stress_curve.append(0)
    #                 effective_stress_curve.append(0)
    #             else:
    #                 prev_section = ground_model.sections[ground_model.sections.index(section) - 1]
    #                 total_stress_curve.append(prev_section.props.get('stress_vertical_total_bottom', 0))
    #                 effective_stress_curve.append(prev_section.props.get('stress_vertical_effective_bottom', 0))
    #         else:
    #             # Interpolate based on position within section
    #             fraction = (depth - section.depth_from) / section.thickness
    #             sigma_top = section.props.get('stress_vertical_total_mid', 0) * 2 - section.props.get('stress_vertical_total_bottom', 0)
    #             sigma_bottom = section.props.get('stress_vertical_total_bottom', 0)
    #             total_stress_curve.append(sigma_top + fraction * (sigma_bottom - sigma_top))
                
    #             sigma_eff_top = section.props.get('stress_vertical_effective_mid', 0) * 2 - section.props.get('stress_vertical_effective_bottom', 0)
    #             sigma_eff_bottom = section.props.get('stress_vertical_effective_bottom', 0)
    #             effective_stress_curve.append(sigma_eff_top + fraction * (sigma_eff_bottom - sigma_eff_top))
        
    #         # Get unit weights
    #         if ground_model.is_below_water_table(depth):
    #             unit_weight_curve.append(section.get_property('unit_weight_sat'))
    #             effective_unit_weight_curve.append(section.get_property('unit_weight_sat') - ground_model.unit_weight_water)
    #         else:
    #             unit_weight_curve.append(section.get_property('unit_weight_bulk'))
    #             effective_unit_weight_curve.append(section.get_property('unit_weight_bulk'))

    # # Plot continuous stress curves on first subplot
    # ax1.plot(total_stress_curve, depth_points, 'r-', linewidth=2, label='Total Stress')
    # ax1.plot(effective_stress_curve, depth_points, 'b-', linewidth=2, label='Effective Stress')

    # # Add groundwater table line to both plots
    # if ground_model.gwt <= ground_model.ground_depth:
    #     ax1.axhline(y=ground_model.gwt, color='c', linestyle='--', alpha=0.7, label='Water Table')
    #     ax2.axhline(y=ground_model.gwt, color='c', linestyle='--', alpha=0.7, label='Water Table')

    # # Add stress points at section midpoints
    # ax1.scatter(total_stress, depths, color='red', s=40, zorder=5)
    # ax1.scatter(effective_stress, depths, color='blue', s=40, zorder=5)

    # # Plot unit weight curves on second subplot
    # ax2.plot(unit_weight_curve, depth_points, 'g-', linewidth=2, label='Total Unit Weight')
    # ax2.plot(effective_unit_weight_curve, depth_points, 'm-', linewidth=2, label='Effective Unit Weight')

    # # Extract unit weight values for scatter points
    # unit_weights = []
    # effective_unit_weights = []
    # for section in ground_model.sections:
    #     depth = section.midpoint_depth
    #     if ground_model.is_below_water_table(depth):
    #         unit_weights.append(section.get_property('unit_weight_sat'))
    #         effective_unit_weights.append(section.get_property('unit_weight_sat') - ground_model.unit_weight_water)
    #     else:
    #         unit_weights.append(section.get_property('unit_weight_bulk'))
    #         effective_unit_weights.append(section.get_property('unit_weight_bulk'))
        
    # # Add unit weight points at section midpoints
    # ax2.scatter(unit_weights, depths, color='green', s=40, zorder=5)
    # ax2.scatter(effective_unit_weights, depths, color='magenta', s=40, zorder=5)

    # # Plot section boundaries on both subplots
    # for section in ground_model.sections:
    #     ax1.axhline(y=section.depth_from, color='gray', linestyle='-', alpha=0.3)
    #     ax1.text(5, section.midpoint_depth, section.name, fontsize=8, 
    #         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    #     ax2.axhline(y=section.depth_from, color='gray', linestyle='-', alpha=0.3)

    # # Configure first plot - Stress Profiles
    # ax1.set_ylim(ground_model.ground_depth, 0)  # Reverse y-axis to show depth increasing downward
    # ax1.set_xlabel('Stress (kPa)')
    # ax1.set_ylabel('Depth (m)')
    # ax1.set_title('Vertical Stress Profiles')
    # ax1.grid(True, linestyle='--', alpha=0.6)
    # ax1.legend(loc='upper right')

    # # Configure second plot - Unit Weight Profiles
    # ax2.set_ylim(ground_model.ground_depth, 0)  # Reverse y-axis to show depth increasing downward
    # ax2.set_xlabel('Unit Weight (kN/m³)')
    # ax2.set_title('Unit Weight Profiles')
    # ax2.grid(True, linestyle='--', alpha=0.6)
    # ax2.legend(loc='upper right')

    # plt.tight_layout()
    # plt.show()
    