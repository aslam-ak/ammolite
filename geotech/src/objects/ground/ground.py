import math
import os
import sys
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Set up the root path and ensure it's in the system path
root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), R"..\..\..\.."))
if root_path not in sys.path:
    sys.path.append(root_path)

from utils.src.common import is_invalid, transform_parameter


class GroundSection(BaseModel):
    """
    Represents a section of the ground with specific parameters and properties.

    :param section_id: Unique identifier for the section
    :param depth_from: Starting depth of the section (in meters)
    :param depth_to: Ending depth of the section (in meters)
    :param behaviour: Soil behavior type ('cohesive', 'frictional', 'c_phi', 'rock', 'calcareous')
    :param main_ground_unit: Primary ground material type ('sand', 'clay', 'silt', 'rock')
    :param name: Descriptive name for the section
    :param slope: Slope angle in degrees (0-89)
    :param params: Dictionary of user-defined soil parameters:
        - unit_weight_sat (float): Saturated unit weight (kN/m³)
        - unit_weight_bulk (float): Bulk unit weight (kN/m³)
        - cohesion (float): Soil cohesion (kPa)
        - angle_friction (float): Internal friction angle (degrees)
        - ratio_poisson (float): Poisson's ratio (dimensionless)
        - modulus_elastic (float): Elastic modulus (kPa)
        Each parameter can be:
        - A single value (constant throughout section)
        - A gradient definition with 'top' and 'bottom' values
        - An expression as a string that will be evaluated based on depth
    :param props: Dictionary of computed soil properties
    """
    section_id: str
    depth_from: float = Field(..., ge=0, metadata={
                              'unit': 'm', 'desc': 'Depth below ground surface'})
    depth_to: float = Field(..., gt=0, metadata={
                            'unit': 'm', 'desc': 'Depth below ground surface'})
    behaviour: str
    main_ground_unit: str
    name: Optional[str] = Field(default='unnamed ground section')
    slope: float = Field(default=0.0, ge=0., le=89., metadata={
                         'unit': 'deg', 'desc': 'Slope angle from the horizontal'})
    params: Dict[str, Union[float, Dict[str, float], str]] = Field(
        default_factory=dict,
        description="Soil parameters as constant values, gradients (top/bottom), or expressions"
    )
    props: Dict[str, Union[float, Dict[str, float], str]] = Field(
        default_factory=dict,
        description="Computed properties as constant values, gradients (top/bottom), or expressions"
    )

    @field_validator('section_id', mode='before')
    def validate_section_id(cls, value) -> str:
        """
        Ensure section_id is a valid string.

        :param value: The section_id to validate
        :return: The validated section_id
        :raises ValueError: If section_id is empty
        """
        if not value:
            raise ValueError("section_id cannot be empty")
        return str(value)

    @field_validator('name', mode='before')
    def validate_name(cls, value) -> str:
        """
        Ensure name is a valid string.

        :param value: The name to validate
        :return: The validated name or default name if None
        """
        return str(value) if value is not None else 'unnamed ground section'

    @field_validator('behaviour', mode='before')
    def validate_behaviour(cls, value) -> str:
        """
        Ensure behaviour is one of the predefined valid types.

        :param value: The behaviour to validate
        :return: The validated behaviour in lowercase
        :raises ValueError: If behaviour is not one of the valid types
        """
        valid_behaviours = {'cohesive', 'frictional',
                            'c_phi', 'rock', 'calcareous'}
        if not isinstance(value, str) or value.lower() not in valid_behaviours:
            raise ValueError(
                f"Invalid soil behaviour '{value}', must be one of {valid_behaviours}")
        return value.lower()

    @field_validator('main_ground_unit', mode='before')
    def validate_main_ground_unit(cls, value) -> str:
        """
        Ensure main_ground_unit is one of the predefined valid types.

        :param value: The main_ground_unit to validate
        :return: The validated main_ground_unit in lowercase
        :raises ValueError: If main_ground_unit is not one of the valid types
        """
        valid_main_ground_units = {'sand', 'clay', 'silt', 'rock'}
        if not isinstance(value, str) or value.lower() not in valid_main_ground_units:
            raise ValueError(
                f"Invalid main ground unit '{value}', must be one of {valid_main_ground_units}")
        return value.lower()

    @field_validator('params', mode='before')
    def validate_params(cls, value) -> Dict[str, Any]:
        """
        Ensure params is a dictionary.

        :param value: The params to validate
        :return: The validated params
        :raises ValueError: If params is not a dictionary
        """
        if not isinstance(value, dict):
            raise ValueError("params must be a dictionary.")
        return value

    @field_validator('props', mode='before')
    def validate_props(cls, value) -> Dict[str, Any]:
        """
        Ensure props is a dictionary.

        :param value: The props to validate
        :return: The validated props
        :raises ValueError: If props is not a dictionary
        """
        if not isinstance(value, dict):
            raise ValueError("props must be a dictionary.")
        return value

    @model_validator(mode='after')
    def validate_depth_order(self) -> 'GroundSection':
        """
        Ensure depth_from is less than depth_to.

        :return: The validated section
        :raises ValueError: If depth_from is not less than depth_to
        """
        if self.depth_from >= self.depth_to:
            raise ValueError(
                f"depth_from ({self.depth_from}) must be less than depth_to ({self.depth_to}).")
        return self

    @model_validator(mode='after')
    def validate_soil_parameters(self) -> 'GroundSection':
        """
        Ensure required soil parameters are valid based on soil behaviour.

        Checks for:
        - Required unit weights (bulk and saturated)
        - Behavior-specific parameters (cohesion for cohesive, angle_friction for frictional)
        - Parameter values within valid ranges

        :return: The validated section with default values set
        :raises ValueError: If any required parameter is missing or invalid
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
            'angle_friction_residual': (0, 50),  # degrees
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
                    raise ValueError(
                        f"Ground section requires a nonzero {param_name}")

            elif isinstance(param_value, dict) and 'top' in param_value and 'bottom' in param_value:
                # Gradient definition with top/bottom values
                top_value = param_value['top']
                bottom_value = param_value['bottom']
                if is_invalid(top_value, custom_invalid_values=custom_invalid_values) or \
                        is_invalid(bottom_value, custom_invalid_values=custom_invalid_values):
                    raise ValueError(
                        f"Ground section requires nonzero values for {param_name} gradient")

            elif isinstance(param_value, str) and param_value.strip():
                # Expression string - we assume it's valid if non-empty
                # (will be evaluated elsewhere)
                pass

            else:
                # Invalid format
                raise ValueError(
                    f"Invalid format for {param_name}: must be a number, top/bottom dict, or expression")

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
                        raise ValueError(
                            f"Parameter '{param_name}' value {param_value} is outside valid range [{min_val}, {max_val}]")

                # Handle gradient parameters
                elif isinstance(param_value, dict) and 'top' in param_value and 'bottom' in param_value:
                    top_val = param_value['top']
                    bottom_val = param_value['bottom']
                    if not (min_val <= top_val <= max_val):
                        raise ValueError(
                            f"Parameter '{param_name}' top value {top_val} is outside valid range [{min_val}, {max_val}]")
                    if not (min_val <= bottom_val <= max_val):
                        raise ValueError(
                            f"Parameter '{param_name}' bottom value {bottom_val} is outside valid range [{min_val}, {max_val}]")

        return self

    @property
    def thickness(self) -> float:
        """
        Calculate the thickness of the section.

        :return: Thickness of section in meters
        """
        return self.depth_to - self.depth_from

    @property
    def midpoint_depth(self) -> float:
        """
        Calculate the midpoint depth of the section.

        :return: Midpoint depth of the section in meters
        """
        return (self.depth_from + self.depth_to) / 2

    def get_property(self, property_name: str, default: Any = None) -> Any:
        """
        Get a soil property value.

        Retrieves property from params or props dictionaries.

        :param property_name: Name of the soil property
        :param default: Default value to return if property doesn't exist
        :return: The value of the property or the default value if property not found
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

        :param property_name: Name of the soil property
        :param value: Value to set (can be constant, gradient dict, or expression string)
        """
        self.props[property_name] = value

    def set_param(self, param_name: str, value: Any) -> None:
        """
        Set a user-defined soil parameter.

        :param param_name: Name of the soil parameter
        :param value: Value to set (can be constant, gradient dict, or expression string)
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
            k0_transform = lambda angle, **kwargs: 1 - \
                math.sin(math.radians(angle))
            self.props['k0'] = transform_parameter(
                phi, k0_transform, 'angle_friction')


class Ground(BaseModel):
    """
    Represents a ground model consisting of multiple vertical soil sections.

    :param name: Name of the ground model
    :param ground_depth: Maximum depth of the ground model in meters
    :param number_of_sections: Total number of ground sections
    :param gwt: Groundwater table depth in meters (≥ 0, default is ∞)
    :param unit_weight_water: Unit weight of water in kN/m³
    :param sections: List of ground sections from surface downward
    """
    name: Optional[str] = Field(default='unnamed ground model')
    ground_depth: float = Field(default=0.0, ge=0.0, metadata={
                                'unit': 'm', 'desc': 'Maximum depth of the ground model'})
    number_of_sections: int = Field(default=0, ge=0, metadata={
                                    'desc': 'Total number of ground sections'})
    gwt: float = Field(default=float('inf'), ge=0, metadata={
                       'unit': 'm', 'desc': 'Groundwater table depth below ground surface'})
    unit_weight_water: float = Field(default=9.81, ge=5, le=15, metadata={
                                     'unit': 'kN/m³', 'desc': 'Unit weight of water'})
    sections: List[GroundSection] = Field(
        default_factory=list, description="List of ground sections from top to bottom")

    def __init__(self, **data: Any) -> None:
        """
        Initialize a Ground model with sections and calculate derived properties.

        :param data: Keyword arguments matching the Ground model attributes
        """
        super().__init__(**data)
        self.ground_depth = self._calculate_ground_depth()
        self.number_of_sections = self._calculate_number_of_sections()

    def _calculate_ground_depth(self) -> float:
        """
        Returns the maximum ground depth from the last section.

        :return: Maximum depth of the ground model in meters
        """
        if not self.sections:
            return 0.0
        return self.sections[-1].depth_to

    def _calculate_number_of_sections(self) -> int:
        """
        Returns the total number of ground sections.

        :return: Number of sections in the ground model
        """
        return len(self.sections)

    @model_validator(mode="before")
    def validate_and_reorder_sections(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates and orders ground sections by depth.

        - Ensures at least one section exists.
        - Sorts sections in ascending order of `depth_from`.
        - Checks for continuity (no gaps between `depth_to` and `depth_from`).
        - Ensures unique section IDs.
        - Confirms ground starts at `depth_from = 0`.

        :param values: Dictionary of model field values
        :raises ValueError: If sections are missing, unordered, discontinuous, or have duplicate IDs.
        :return: Validated and sorted values dictionary
        """
        sections = values.get("sections", [])

        # Ensure at least one section is present
        if not sections:
            raise ValueError("At least one ground section must be present.")

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
            duplicate_ids = [
                id for id in section_ids if section_ids.count(id) > 1]
            raise ValueError(
                f"Each section must have a unique section_id. Duplicate IDs found: {set(duplicate_ids)}")

        # First section's depth_from should be 0
        if sorted_sections[0].depth_from != 0:
            raise ValueError("Ground must start at depth_from = 0.0")

        values["sections"] = sorted_sections
        return values

    def add_section(self, new_section: GroundSection) -> None:
        """
        Adds a new ground section to the model, ensuring proper depth continuity.

        :param new_section: The new ground section to add
        :raises ValueError: If the section's depth doesn't align with existing sections
                           or if its section_id is not unique
        """
        # Check if this is the first section (special case)
        if not self.sections:
            if new_section.depth_from != 0:
                raise ValueError(
                    "First ground section must start at depth_from = 0.0")
            self.sections = [new_section]
            self.ground_depth = new_section.depth_to
            self.number_of_sections = 1
            return

        # Check for unique section_id
        section_ids = [sec.section_id for sec in self.sections]
        if new_section.section_id in section_ids:
            raise ValueError(
                f"Section ID '{new_section.section_id}' already exists. Section IDs must be unique.")

        # Determine where to insert the new section
        if new_section.depth_from == 0:
            # New section starts at the top
            if new_section.depth_to != self.sections[0].depth_from:
                raise ValueError(
                    f"New section ending at {new_section.depth_to}m must connect with existing section starting at {self.sections[0].depth_from}m")
            self.sections.insert(0, new_section)
        elif new_section.depth_from == self.ground_depth:
            # New section extends the ground model deeper
            if self.sections[-1].depth_to != new_section.depth_from:
                raise ValueError(
                    f"New section must start at current maximum depth {self.ground_depth}m")
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
                        raise ValueError(
                            f"New section must connect with existing sections")
                    break
                elif section.depth_to == new_section.depth_from:
                    # Insert after this section
                    insertion_index = i + 1
                    # Check if new section connects to next section
                    if i < len(self.sections) - 1 and self.sections[i+1].depth_from != new_section.depth_to:
                        raise ValueError(
                            f"New section must connect with existing sections")
                    break

            if insertion_index is None:
                raise ValueError(
                    f"New section with depth_from={new_section.depth_from}m and depth_to={new_section.depth_to}m doesn't align with existing sections")

            self.sections.insert(insertion_index, new_section)

        # Re-sort the sections to ensure proper order
        self.sections.sort(key=lambda sec: sec.depth_from)

        # Update ground_depth and number_of_sections
        self.number_of_sections = len(self.sections)
        if self.sections:
            self.ground_depth = self.sections[-1].depth_to

    def calculate_stresses(self) -> None:
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

        :raises ValueError: If unit weights are not constant values
        """
        cumulative_total_stress = 0.0
        cumulative_effective_stress = 0.0

        for i, section in enumerate(self.sections):
            # Get unit weights for current section
            unit_weight_bulk = section.get_property('unit_weight_bulk')
            unit_weight_sat = section.get_property('unit_weight_sat')

            # Check for valid unit weights
            if isinstance(unit_weight_bulk, (dict, str)) or isinstance(unit_weight_sat, (dict, str)):
                raise ValueError(
                    f"Section {section.section_id}: Stress calculation requires constant unit weight values, not gradients or expressions")

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
                effective_stress_increment_to_mid = (
                    unit_weight_sat - self.unit_weight_water) * distance_to_mid
            else:
                # Water table within this segment (between top and midpoint)
                above_water_distance = self.gwt - depth_from
                below_water_distance = mid_depth - self.gwt

                # Above water contribution
                total_stress_increment_to_mid = unit_weight_bulk * above_water_distance
                effective_stress_increment_to_mid = unit_weight_bulk * above_water_distance

                # Below water contribution
                total_stress_increment_to_mid += unit_weight_sat * below_water_distance
                effective_stress_increment_to_mid += (
                    unit_weight_sat - self.unit_weight_water) * below_water_distance

            # --- Calculate stress increment to bottom ---

            # If the water table is within this section, handle differently
            if self.gwt > depth_to:
                # Bottom is above water table - use bulk unit weight
                total_stress_increment_to_bottom = unit_weight_bulk * distance_to_bottom
                effective_stress_increment_to_bottom = unit_weight_bulk * distance_to_bottom
            elif self.gwt <= depth_from:
                # Section entirely below water table - use saturated unit weight
                total_stress_increment_to_bottom = unit_weight_sat * distance_to_bottom
                effective_stress_increment_to_bottom = (
                    unit_weight_sat - self.unit_weight_water) * distance_to_bottom
            else:
                # Water table within this section
                above_water_distance = self.gwt - depth_from
                below_water_distance = depth_to - self.gwt

                # Above water contribution
                total_stress_increment_to_bottom = unit_weight_bulk * above_water_distance
                effective_stress_increment_to_bottom = unit_weight_bulk * above_water_distance

                # Below water contribution
                total_stress_increment_to_bottom += unit_weight_sat * below_water_distance
                effective_stress_increment_to_bottom += (
                    unit_weight_sat - self.unit_weight_water) * below_water_distance

            # Update cumulative stresses at midpoint
            total_stress_at_mid = cumulative_total_stress + total_stress_increment_to_mid
            effective_stress_at_mid = cumulative_effective_stress + \
                effective_stress_increment_to_mid

            # Update cumulative stresses at bottom (for next layer)
            cumulative_total_stress += total_stress_increment_to_bottom
            cumulative_effective_stress += effective_stress_increment_to_bottom

            # Store the stresses in the section's properties
            section.set_property(
                'stress_vertical_total_mid', total_stress_at_mid)
            section.set_property(
                'stress_vertical_effective_mid', effective_stress_at_mid)
            section.set_property(
                'stress_vertical_total_bottom', cumulative_total_stress)
            section.set_property(
                'stress_vertical_effective_bottom', cumulative_effective_stress)

    def refine_sections(self, depth_increment: float = 0.1) -> 'Ground':
        """
        Divides each ground section into multiple finer sections with the specified depth increments.

        :param depth_increment: Depth increment for dividing sections, must be positive.
        :return: A new Ground object with refined sections.
        :raises ValueError: If depth_increment is not positive.
        """
        if depth_increment <= 0:
            raise ValueError("Depth increment must be a positive number.")

        refined_sections = []
        section_counter = 1

        for section in self.sections:
            current_depth = section.depth_from
            while current_depth < section.depth_to:
                next_depth = min(
                    current_depth + depth_increment, section.depth_to)

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
                    rel_top = (refined_section.depth_from -
                               original_section.depth_from) / orig_thickness
                    rel_bottom = (refined_section.depth_to -
                                  original_section.depth_from) / orig_thickness

                    # Linear interpolation
                    new_top = orig_top + rel_top * (orig_bottom - orig_top)
                    new_bottom = orig_top + rel_bottom * \
                        (orig_bottom - orig_top)

                    # Update the parameter
                    refined_section.params[param_name] = {
                        'top': new_top, 'bottom': new_bottom}

            # Update props that have top/bottom gradients
            for prop_name, prop_value in refined_section.props.items():
                if isinstance(prop_value, dict) and 'top' in prop_value and 'bottom' in prop_value:
                    orig_top = prop_value['top']
                    orig_bottom = prop_value['bottom']

                    # Calculate relative position in the original section
                    rel_top = (refined_section.depth_from -
                               original_section.depth_from) / orig_thickness
                    rel_bottom = (refined_section.depth_to -
                                  original_section.depth_from) / orig_thickness

                    # Linear interpolation
                    new_top = orig_top + rel_top * (orig_bottom - orig_top)
                    new_bottom = orig_top + rel_bottom * \
                        (orig_bottom - orig_top)

                    # Update the property
                    refined_section.props[prop_name] = {
                        'top': new_top, 'bottom': new_bottom}

        return refined_ground

    def get_section_at_depth(self, depth: float) -> Optional[GroundSection]:
        """
        Retrieves the ground section at a specific depth.

        :param depth: The depth to query in meters
        :return: The section at the specified depth or None if no section exists
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

        :param depth: The depth to check in meters
        :return: True if depth is below water table, False otherwise
        """
        return depth >= self.gwt

    def get_property_profile(self, property_name: str) -> List[Tuple[float, Any]]:
        """
        Creates a depth profile of a specific soil property.

        :param property_name: Name of the soil property to profile
        :return: List of (depth, value) pairs for the property
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


def create_test_model() -> Ground:
    """
    Create a test ground model with two soil layers.

    :return: Ground model for testing
    """
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
                # Gradient from top to bottom (kPa)
                "cohesion": {'top': 20.0, 'bottom': 30.0},
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
        return Ground(
            gwt=3.0,  # Groundwater table at 3m depth
            sections=[clay_section, sand_section],
            name="Test Site Ground Model"
        )
    except ValueError as e:
        print(f"Error creating test model: {str(e)}")
        raise


def run_tests() -> None:
    """
    Run a series of tests on the Ground and GroundSection classes.
    """
    try:
        # Create test model
        print("Creating test ground model...")
        ground_model = create_test_model()

        # Display basic ground model information
        print(f"Ground model: {ground_model.name}")
        print(f"Ground depth: {ground_model.ground_depth}m")
        print(f"Water table depth: {ground_model.gwt}m")
        print(f"Number of sections: {ground_model.number_of_sections}")

        print("\nGround sections:")
        for i, section in enumerate(ground_model.sections):
            print(
                f"  Section {i+1}: {section.name} ({section.depth_from}m to {section.depth_to}m)")
            print(f"    - Behaviour: {section.behaviour}")
            print(f"    - Main ground unit: {section.main_ground_unit}")
            print(
                f"    - Unit weight bulk: {section.get_property('unit_weight_bulk')} kN/m³")
            print(
                f"    - Unit weight saturated: {section.get_property('unit_weight_sat')} kN/m³")

        # Test calculation of vertical stresses
        print("\nCalculating vertical stresses...")
        ground_model.calculate_stresses()
        print("Vertical stresses calculated:")
        for i, section in enumerate(ground_model.sections):
            print(f"  Section {i+1}: {section.name}")
            print(
                f"    - Total stress at midpoint: {section.props.get('stress_vertical_total_mid'):.2f} kPa")
            print(
                f"    - Effective stress at midpoint: {section.props.get('stress_vertical_effective_mid'):.2f} kPa")
            print(
                f"    - Total stress at bottom: {section.props.get('stress_vertical_total_bottom'):.2f} kPa")
            print(
                f"    - Effective stress at bottom: {section.props.get('stress_vertical_effective_bottom'):.2f} kPa")

        # Test refining the ground model
        print("\nTesting section refinement...")
        refined_model = ground_model.refine_sections(depth_increment=1.0)
        print(f"Original model had {ground_model.number_of_sections} sections")
        print(f"Refined model has {refined_model.number_of_sections} sections")

        # Test getting section at specific depth
        test_depth = 7.5
        print(f"\nTesting get_section_at_depth({test_depth})...")
        section = ground_model.get_section_at_depth(test_depth)
        if section:
            print(f"Found section: {section.name} at depth {test_depth}m")
        else:
            print(f"No section found at depth {test_depth}m")

        # Test adding a new section
        print("\nTesting add_section()...")
        try:
            new_section = GroundSection(
                section_id="rock1",
                depth_from=10.0,
                depth_to=15.0,
                behaviour="rock",
                main_ground_unit="rock",
                name="Weathered Rock Layer",
                params={
                    "unit_weight_bulk": 22.0,
                    "unit_weight_sat": 23.0,
                    "cohesion": 500.0,
                    "angle_friction": 45.0
                }
            )

            ground_model.add_section(new_section)
            print(
                f"Added new section. Ground model now has {ground_model.number_of_sections} sections")
            print(f"New ground depth: {ground_model.ground_depth}m")
        except ValueError as e:
            print(f"Error adding section: {str(e)}")

        # Test error handling for invalid section
        print("\nTesting error handling for invalid section...")
        try:
            invalid_section = GroundSection(
                section_id="invalid1",
                depth_from=20.0,  # Gap in continuity
                depth_to=25.0,
                behaviour="cohesive",
                main_ground_unit="clay",
                params={
                    "unit_weight_bulk": 18.0,
                    "unit_weight_sat": 19.0,
                    "cohesion": 25.0
                }
            )

            ground_model.add_section(invalid_section)
        except ValueError as e:
            print(f"Successfully caught error: {str(e)}")

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_tests()
