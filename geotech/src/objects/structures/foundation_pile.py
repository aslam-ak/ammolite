import os
import sys
from typing import Dict, List, Literal, Optional, Union

from pydantic import (BaseModel, Field, ValidationError, field_validator,
                      model_validator)

# Set up the root path and ensure it's in the system path
root_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), R"..\..\..\.."))
if root_path not in sys.path:
    sys.path.append(root_path)

from utils.src.common import (calculate_cross_sectional_area,
                              calculate_perimeter)


class PileSection(BaseModel):
    """
    Represents a pile section with geometric properties and calculations.

    :param shape: The shape of the pile (rectangle, square, circle, h_section, c_section, l_section, custom).
    :param depth_from: Starting depth of the pile section (meters).
    :param depth_to: Ending depth of the pile section (meters).
    :param section_id: Unique identifier for the pile section.
    :param name: Optional name of the pile section.
    :param outer_diameter: Outer diameter of the pile (meters) for circular shapes.
    :param outer_width: Outer width of the pile (meters) for rectangular shapes.
    :param outer_breadth: Outer breadth of the pile (meters) for rectangular shapes.
    :param inner_diameter: Inner diameter (meters) for hollow circular piles.
    :param inner_width: Inner width (meters) for hollow rectangular piles.
    :param inner_breadth: Inner breadth (meters) for hollow rectangular piles.
    :param flange_width: Width of the flange for H/C/L-section piles (meters).
    :param flange_thickness: Thickness of the flange for H/C/L-section piles (meters).
    :param web_height: Height of the web for H/C/L-section piles (meters).
    :param web_thickness: Thickness of the web for H/C/L-section piles (meters).
    :param cross_section: Indicates whether the section is solid or hollow.
    :param material: Material of the pile section (timber, steel, concrete).
    :param unit_weight: Unit weight of the material (kN/m³).
    :param custom_area: Custom cross-sectional area for custom shapes (m²).
    :param custom_perimeter: Custom perimeter for custom shapes (m).
    """
    shape: str
    depth_from: float = Field(default=0.0, ge=0, metadata={'unit': 'm'})
    depth_to: float = Field(gt=0, metadata={'unit': 'm'})
    section_id: str = Field(default='default_id')
    name: Optional[str] = Field(default='unnamed pile section')
    outer_diameter: Optional[float] = Field(
        default=None, gt=0, metadata={'unit': 'm'})
    outer_width: Optional[float] = Field(
        default=None, gt=0, metadata={'unit': 'm'})
    outer_breadth: Optional[float] = Field(
        default=None, gt=0, metadata={'unit': 'm'})
    inner_diameter: Optional[float] = Field(
        default=None, gt=0, metadata={'unit': 'm'})
    inner_width: Optional[float] = Field(
        default=None, gt=0, metadata={'unit': 'm'})
    inner_breadth: Optional[float] = Field(
        default=None, gt=0, metadata={'unit': 'm'})
    flange_width: Optional[float] = Field(
        default=None, gt=0, metadata={'unit': 'm'})
    flange_thickness: Optional[float] = Field(
        default=None, gt=0, metadata={'unit': 'm'})
    web_height: Optional[float] = Field(
        default=None, gt=0, metadata={'unit': 'm'})
    web_thickness: Optional[float] = Field(
        default=None, gt=0, metadata={'unit': 'm'})
    cross_section: Literal["solid", "hollow"] = Field(default='solid')
    material: Literal["timber", "steel", "concrete"] = Field(default='steel')
    unit_weight: Optional[float] = Field(
        default=None, gt=0, metadata={'unit': 'kN/m³'})
    custom_area: Optional[float] = Field(
        default=None, gt=0, metadata={'unit': 'm²'})
    custom_perimeter: Optional[float] = Field(
        default=None, gt=0, metadata={'unit': 'm'})

    @property
    def total_cross_sectional_area(self) -> float:
        """
        Calculate the total cross-sectional area based on outer dimensions.

        :return: Total cross-sectional area in m²
        """
        if self.shape == "custom" and self.custom_area is not None:
            return self.custom_area

        try:
            if self.shape in ["h_section", "c_section", "l_section"]:
                return calculate_cross_sectional_area(
                    self.shape, "total",
                    flange_width=self.flange_width,
                    flange_thickness=self.flange_thickness,
                    web_height=self.web_height,
                    web_thickness=self.web_thickness)

            return calculate_cross_sectional_area(
                self.shape, "total",
                outer_diameter=self.outer_diameter,
                outer_width=self.outer_width,
                outer_breadth=self.outer_breadth,
                cross_section=self.cross_section)
        except ValueError as e:
            raise ValueError(
                f"Error calculating total cross-sectional area: {e}")

    @property
    def inner_cross_sectional_area(self) -> float:
        """
        Calculate the hollow portion area if applicable, always 0 for solid, H, C, or L sections.

        :return: Inner cross-sectional area in m²
        """
        if self.shape in ["custom", "h_section", "c_section", "l_section"]:
            return 0  # These sections are always solid

        try:
            return calculate_cross_sectional_area(
                self.shape, "inner",
                inner_diameter=self.inner_diameter,
                inner_width=self.inner_width,
                inner_breadth=self.inner_breadth,
                cross_section=self.cross_section)
        except ValueError as e:
            raise ValueError(
                f"Error calculating inner cross-sectional area: {e}")

    @property
    def bearing_area(self) -> float:
        """
        Calculate the bearing area (annular area for hollow sections).

        :return: Bearing area in m²
        """
        if self.shape == "custom" and self.custom_area is not None:
            return self.custom_area

        if self.shape in ["h_section", "c_section", "l_section"]:
            return self.total_cross_sectional_area  # These sections are always solid

        try:
            return calculate_cross_sectional_area(
                self.shape, "annular",
                outer_diameter=self.outer_diameter,
                outer_width=self.outer_width,
                outer_breadth=self.outer_breadth,
                inner_diameter=self.inner_diameter,
                inner_width=self.inner_width,
                inner_breadth=self.inner_breadth,
                cross_section=self.cross_section)
        except ValueError as e:
            raise ValueError(f"Error calculating bearing area: {e}")

    @property
    def outer_perimeter(self) -> float:
        """
        Calculate the outer perimeter.

        :return: Outer perimeter in m
        """
        if self.shape == "custom" and self.custom_perimeter is not None:
            return self.custom_perimeter

        try:
            if self.shape in ["h_section", "c_section", "l_section"]:
                return calculate_perimeter(
                    self.shape, "outer",
                    flange_width=self.flange_width,
                    flange_thickness=self.flange_thickness,
                    web_height=self.web_height,
                    web_thickness=self.web_thickness)

            return calculate_perimeter(
                self.shape, "outer",
                outer_diameter=self.outer_diameter,
                outer_width=self.outer_width,
                outer_breadth=self.outer_breadth,
                cross_section=self.cross_section)
        except ValueError as e:
            raise ValueError(f"Error calculating outer perimeter: {e}")

    @property
    def inner_perimeter(self) -> float:
        """
        Calculate the inner perimeter if hollow, always 0 for solid, H, C, or L sections.

        :return: Inner perimeter in m
        """
        if self.shape in ["custom", "h_section", "c_section", "l_section"]:
            return 0  # These sections are always solid

        try:
            return calculate_perimeter(
                self.shape, "inner",
                inner_diameter=self.inner_diameter,
                inner_width=self.inner_width,
                inner_breadth=self.inner_breadth,
                cross_section=self.cross_section)
        except ValueError as e:
            raise ValueError(f"Error calculating inner perimeter: {e}")

    @property
    def section_height(self) -> float:
        """
        Calculate the height of this section.

        :return: Section height in m
        """
        return self.depth_to - self.depth_from

    @property
    def outer_lateral_surface_area(self) -> float:
        """
        Calculate outer lateral surface area.

        :return: Outer lateral surface area in m²
        """
        return self.outer_perimeter * self.section_height

    @property
    def inner_lateral_surface_area(self) -> float:
        """
        Calculate inner lateral surface area if hollow, otherwise return 0.

        :return: Inner lateral surface area in m²
        """
        return self.inner_perimeter * self.section_height

    @property
    def section_volume(self) -> float:
        """
        Calculate the volume of the pile section.

        :return: Section volume in m³
        """
        return self.bearing_area * self.section_height

    @property
    def section_mass(self) -> float:
        """
        Calculate the mass of the pile section in kN.

        :return: Section mass in kN
        """
        if self.unit_weight is None:
            return 0
        return self.section_volume * self.unit_weight

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: str) -> str:
        """
        Validate the shape is one of the allowed values.

        :param value: Shape value to validate
        :return: Validated shape value
        :raises ValueError: If the shape is invalid
        """
        valid_shapes = {"rectangle", "square", "circle",
                        "h_section", "c_section", "l_section", "custom"}
        if not isinstance(value, str) or value.lower() not in valid_shapes:
            raise ValueError(
                f"Invalid shape '{value}', must be one of {valid_shapes}")
        return value.lower()

    @field_validator('section_id', mode='before')
    def ensure_string_section_id(cls, value) -> str:
        """
        Ensure the section ID is a string.

        :param value: Section ID value to validate
        :return: Section ID as string
        """
        return str(value)

    @model_validator(mode="before")
    @classmethod
    def validate_depths(cls, values: Dict) -> Dict:
        """
        Ensure depth_from is less than depth_to.

        :param values: Model values dictionary
        :return: Validated values dictionary
        :raises ValueError: If depth_from is greater than or equal to depth_to
        """
        depth_from, depth_to = values.get("depth_from"), values.get("depth_to")
        if depth_from is not None and depth_to is not None and depth_from >= depth_to:
            raise ValueError(
                f"depth_from ({depth_from}) must be less than depth_to ({depth_to})")
        return values

    @model_validator(mode="after")
    def validate_dimensions(self) -> "PileSection":
        """
        Ensure proper dimensions based on the shape.

        :return: Validated PileSection instance
        :raises ValueError: If dimensions are invalid for the specified shape
        """
        # Set default unit weights if not provided
        if self.unit_weight is None:
            if self.material == "concrete":
                self.unit_weight = 24.0  # kN/m³
            elif self.material == "steel":
                self.unit_weight = 78.5  # kN/m³
            elif self.material == "timber":
                self.unit_weight = 6.0   # kN/m³

        if self.shape == "rectangle":
            if self.outer_width is None or self.outer_breadth is None:
                raise ValueError(
                    f"{self.shape} piles must have both width and breadth")
            if self.outer_width <= self.outer_breadth:
                raise ValueError(
                    f"{self.shape} piles must have width > breadth")
        elif self.shape == 'square':
            if self.outer_width is None:
                raise ValueError(f"{self.shape} piles must have width")
            self.outer_breadth = self.outer_width
        elif self.shape == "circle":
            if self.outer_diameter is None:
                raise ValueError(f"{self.shape} piles must have diameter")
        elif self.shape in ["h_section", "c_section", "l_section"]:
            if None in (self.flange_width, self.flange_thickness, self.web_height, self.web_thickness):
                raise ValueError(
                    f"{self.shape} piles must have flange_width, flange_thickness, web_height, and web_thickness")
            # Force cross_section to be solid for H, C, and L sections
            self.cross_section = "solid"
        elif self.shape == "custom":
            if self.custom_area is None:
                raise ValueError("Custom piles must have custom_area defined")

        # Validate hollow sections
        if self.cross_section == "hollow":
            if self.shape == "circle":
                if self.inner_diameter is None:
                    raise ValueError(
                        "Hollow circular piles must have inner_diameter")
                if self.inner_diameter >= self.outer_diameter:
                    raise ValueError(
                        "inner_diameter must be less than outer_diameter")
            elif self.shape in ["rectangle", "square"]:
                if self.inner_width is None or self.inner_breadth is None:
                    raise ValueError(
                        "Hollow rectangular/square piles must have inner_width and inner_breadth")
                if self.inner_width >= self.outer_width or self.inner_breadth >= self.outer_breadth:
                    raise ValueError(
                        "Inner dimensions must be less than outer dimensions")

        return self


class Pile(BaseModel):
    """
    Represents a pile with multiple sections.

    :param pile_type: Type of the pile (e.g., driven, drilled, CFA).
    :param sections: List of pile sections.
    :param name: Optional name of the pile.
    :param shear_keys_present: Whether the pile has shear keys (defaults to True for drilled_and_grouted).
    """
    pile_type: Literal["driven", "drilled_and_grouted",
                       "bored"] = Field(alias="type")
    sections: List[PileSection] = Field()
    name: Optional[str] = Field(default='unnamed pile')
    shear_keys_present: Optional[bool] = Field(default=None)

    @property
    def pile_length(self) -> float:
        """
        Compute the total pile length based on the last section.

        :return: Total pile length in m
        """
        if not self.sections:
            return 0.0
        return max(section.depth_to for section in self.sections)

    @property
    def number_of_sections(self) -> int:
        """
        Compute the total number of sections.

        :return: Number of pile sections
        """
        return len(self.sections)

    @property
    def total_volume(self) -> float:
        """
        Compute the total volume of the pile.

        :return: Total volume in m³
        """
        return sum(section.section_volume for section in self.sections)

    @property
    def total_mass(self) -> float:
        """
        Compute the total mass of the pile in kN.

        :return: Total mass in kN
        """
        return sum(section.section_mass for section in self.sections)

    @model_validator(mode="after")
    def validate_sections(self) -> "Pile":
        """
        Ensure sections are contiguous and ordered by depth.

        :return: Validated Pile instance
        :raises ValueError: If sections are not contiguous
        """
        if not self.sections:
            raise ValueError("Pile must have at least one section")

        # Ensure sorted by depth
        self.sections.sort(key=lambda sec: sec.depth_from)

        # Check first section starts at surface
        if self.sections[0].depth_from != 0:
            raise ValueError(
                f"First pile section must start at depth 0, not {self.sections[0].depth_from}")

        # Check sections are contiguous
        for i in range(len(self.sections) - 1):
            # Using epsilon for float comparison
            if abs(self.sections[i].depth_to - self.sections[i + 1].depth_from) > 1e-6:
                raise ValueError(
                    f"Non-contiguous sections: depth_to ({self.sections[i].depth_to}) "
                    f"does not match depth_from ({self.sections[i+1].depth_from})"
                )

        # Set default for shear_keys_present based on pile type
        if self.shear_keys_present is None:
            self.shear_keys_present = self.pile_type == "drilled_and_grouted"

        return self


def run_tests() -> None:
    """
    Run comprehensive tests for pile and pile section functionality.
    """
    print("=== RUNNING PILE STRUCTURE TESTS ===\n")

    # Test 1: Create a circular pile
    print("Test 1: Creating a simple circular pile")
    try:
        section1 = PileSection(
            shape="circle",
            depth_from=0.0,
            depth_to=10.0,
            section_id="C1",
            name="Upper Section",
            outer_diameter=0.6,
            material="steel"
        )

        print(f"  ✓ Created section: {section1.name}")
        print(
            f"    - Total area: {section1.total_cross_sectional_area:.4f} m²")
        print(f"    - Outer perimeter: {section1.outer_perimeter:.4f} m")
        print(f"    - Section volume: {section1.section_volume:.4f} m³")
        print(f"    - Section mass: {section1.section_mass:.2f} kN")
    except ValidationError as e:
        print(f"  ✗ Error: {e}")

    # Test 2: Create a hollow circular pile
    print("\nTest 2: Creating a hollow circular pile")
    try:
        hollow_section = PileSection(
            shape="circle",
            depth_from=0.0,
            depth_to=15.0,
            section_id="HC1",
            name="Hollow Circular Section",
            outer_diameter=0.8,
            inner_diameter=0.6,
            cross_section="hollow",
            material="steel"
        )

        print(f"  ✓ Created section: {hollow_section.name}")
        print(
            f"    - Total area: {hollow_section.total_cross_sectional_area:.4f} m²")
        print(
            f"    - Inner area: {hollow_section.inner_cross_sectional_area:.4f} m²")
        print(f"    - Bearing area: {hollow_section.bearing_area:.4f} m²")
        print(f"    - Outer perimeter: {hollow_section.outer_perimeter:.4f} m")
        print(f"    - Inner perimeter: {hollow_section.inner_perimeter:.4f} m")
    except ValidationError as e:
        print(f"  ✗ Error: {e}")

    # Test 3: Create an H-section pile
    print("\nTest 3: Creating an H-section pile")
    try:
        h_section = PileSection(
            shape="h_section",
            depth_from=0.0,
            depth_to=20.0,
            section_id="H1",
            name="H-Section",
            flange_width=0.3,
            flange_thickness=0.02,
            web_height=0.4,
            web_thickness=0.015,
            material="steel"
        )

        print(f"  ✓ Created section: {h_section.name}")
        print(
            f"    - Total area: {h_section.total_cross_sectional_area:.4f} m²")
        print(f"    - Outer perimeter: {h_section.outer_perimeter:.4f} m")
    except ValidationError as e:
        print(f"  ✗ Error: {e}")

    # Test 4: Create a complete pile with multiple sections
    print("\nTest 4: Creating a complete pile with multiple sections")
    try:
        # Upper section
        section1 = PileSection(
            shape="circle",
            depth_from=0.0,
            depth_to=10.0,
            section_id="P1-S1",
            name="Upper Section",
            outer_diameter=0.6,
            material="steel"
        )

        # Lower section
        section2 = PileSection(
            shape="circle",
            depth_from=10.0,
            depth_to=20.0,
            section_id="P1-S2",
            name="Lower Section",
            outer_diameter=0.5,
            material="steel"
        )

        # Create a pile with these sections
        pile = Pile(type='driven', sections=[
                    section1, section2], name="Test Pile")

        print(f"  ✓ Created pile: {pile.name}")
        print(f"    - Pile type: {pile.pile_type}")
        print(f"    - Pile length: {pile.pile_length} m")
        print(f"    - Number of sections: {pile.number_of_sections}")
        print(f"    - Total volume: {pile.total_volume:.4f} m³")
        print(f"    - Total mass: {pile.total_mass:.2f} kN")
        print(f"    - Shear keys present: {pile.shear_keys_present}")
    except ValidationError as e:
        print(f"  ✗ Error: {e}")

    # Test 5: Testing validation with invalid parameters
    print("\nTest 5: Testing validation with invalid parameters")
    try:
        # Try to create a pile with non-contiguous sections
        invalid_section1 = PileSection(
            shape="circle",
            depth_from=0.0,
            depth_to=10.0,
            section_id="IV1",
            outer_diameter=0.6
        )

        invalid_section2 = PileSection(
            shape="circle",
            depth_from=12.0,  # Gap between sections
            depth_to=20.0,
            section_id="IV2",
            outer_diameter=0.5
        )

        invalid_pile = Pile(type='driven', sections=[
                            invalid_section1, invalid_section2], name="Invalid Pile")
        print("  ✗ Failed: Should have raised validation error for non-contiguous sections")
    except ValidationError as e:
        print(
            f"  ✓ Correctly caught validation error: {str(e).split(':', 1)[0]}")

    # Test 6: Testing a custom shape
    print("\nTest 6: Testing a custom shape")
    try:
        custom_section = PileSection(
            shape="custom",
            depth_from=0.0,
            depth_to=15.0,
            section_id="CS1",
            name="Custom Section",
            custom_area=0.25,
            custom_perimeter=2.0,
            material="concrete"
        )

        print(f"  ✓ Created custom section: {custom_section.name}")
        print(
            f"    - Total area: {custom_section.total_cross_sectional_area:.4f} m²")
        print(f"    - Perimeter: {custom_section.outer_perimeter:.4f} m")
        print(
            f"    - Material: {custom_section.material} (unit weight: {custom_section.unit_weight} kN/m³)")
        print(f"    - Section mass: {custom_section.section_mass:.2f} kN")
    except ValidationError as e:
        print(f"  ✗ Error: {e}")

    print("\n=== TESTS COMPLETED ===")


if __name__ == "__main__":
    run_tests()
