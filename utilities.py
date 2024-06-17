# -*- coding: utf-8 -*-
"""
This module provides classes to model materials and geometric objects like 
cylinders and discs. It calculates various physical properties such as volume, 
mass, surface area, and moments of inertia.

Classes:
    Material: Represents a material with a name and density.
    Cylinder: Represents a hollow cylinder and calculates its properties.
    Collection: A class to hold a collection of objects.
"""

import pint
import numpy as np
from typing import List, Optional

ureg = pint.UnitRegistry()
ureg.default_format = '~P'  # Abbreviated unit names.
Q_ = ureg.Quantity

class Material:
    """Class representing a material with a name and density."""
    
    def __init__(self,
                 name: str,
                 density: pint.Quantity,
                 young_modulus: pint.Quantity) -> None:
        """
        Initialize a Material object.
        
        Args:
            name (str): The name of the material.
            density (pint.Quantity): The density of the material in kg/m^3.
            young_modulus (pint.Quantity): Young's modulus of the material in Pascals.
        """
        self.name = name
        self.density = density
        self.young_modulus = young_modulus

    def __repr__(self) -> str:
        """Return a detailed string representation of the instance."""
        return f"{self.__class__.__name__}({self.name})"
        
    def __str__(self) -> str:
        """
        Return a string representation of the material.
        
        Returns:
            str: A string that describes the material.
        """
        return (f"{self.__repr__()}\n"
                f"Density: {self.density}\n"
                f"Young's Modulus: {self.young_modulus}")


class Cylinder:
    """Class representing a hollow cylinder."""
    
    def __init__(self,
                 outer_radius: pint.Quantity,
                 inner_radius: pint.Quantity,
                 length: pint.Quantity,
                 material: Material) -> None:
        """
        Initialize a Cylinder object.
        
        Args:
            outer_radius (pint.Quantity): The outer radius of the cylinder in meters.
            inner_radius (pint.Quantity): The inner radius of the hollow part of the
                cylinder in meters.
            length (pint.Quantity): The length of the cylinder in meters.
            material (Material): The material of the cylinder.
        
        Remarks:
            The properties mass and volume, once computed, these values are cached to
            improve performance if they are called frequently.
        """
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.length = length
        self.material = material
        self._volume: Optional[pint.Quantity] = None
        self._mass: Optional[pint.Quantity] = None

    @property
    def volume(self) -> pint.Quantity:
        """
        Calculate the volume of the hollow cylinder.
        
        Returns:
            pint.Quantity: Volume of the hollow cylinder in cubic meters.
        """
        if self._volume is None:
            outer_volume = np.pi * (self.outer_radius ** 2) * self.length
            inner_volume = np.pi * (self.inner_radius ** 2) * self.length
            self._volume = outer_volume - inner_volume
        return self._volume

    @property
    def mass(self) -> pint.Quantity:
        """
        Calculate the mass of the hollow cylinder.
        
        Returns:
            pint.Quantity: Mass of the hollow cylinder in kilograms.
        """
        if self._mass is None:
            self._mass = self.volume * self.material.density
        return self._mass
    
    @property
    def M(self) -> np.ndarray:
        """
        Get the mass matrix.
        
        Returns:
            np.ndarray: A diagonal matrix with mass and moment of inertia x and z.
        """
        return np.diag([self.mass.m,
                        self.mass.m,
                        self.I_x.m,
                        self.I_z.m])

    @property
    def surface_area(self) -> pint.Quantity:
        """
        Calculate the surface area of the hollow cylinder.
        
        Returns:
            pint.Quantity: Surface area of the hollow cylinder in square meters.
        """
        outer_side_area = 2 * np.pi * self.outer_radius * self.length
        inner_side_area = 2 * np.pi * self.inner_radius * self.length
        top_bottom_area = 2 * (np.pi * (self.outer_radius ** 2) 
                               - np.pi * (self.inner_radius ** 2))
        
        return outer_side_area + inner_side_area + top_bottom_area
    
    @property
    def cross_sectional_area(self) -> pint.Quantity:
        """
        Calculate the cross-sectional area of the hollow cylinder.

        Returns:
            pint.Quantity: Cross-sectional area of the hollow cylinder in square meters.
        """
        outer_area = np.pi * self.outer_radius ** 2
        hollow_area = np.pi * self.inner_radius ** 2
        return outer_area - hollow_area
    
    @property
    def Ix(self) -> pint.Quantity:
        """
        Calculate the area moment of inertia or second moment of area (rotational inertia)
        about the x-axis.
        
        Returns:
            pint.Quantity: Area moment of inertia about the x-axis (radial orientation).
        """
        return np.pi / 4 * (self.outer_radius**4 - self.inner_radius**4)
     
    @property
    def Iz(self) -> pint.Quantity:
        """
        Calculate the area moment of inertia or second moment of area (rotational inertia)
        about the z-axis.
        
        Returns:
            pint.Quantity: Area moment of inertia about the z-axis (radial orientation).
        """
        return self.Ix
    
    @property
    def J(self) -> pint.Quantity:
        """
        Calculate the polar moment of inertia or second polar moment of area (rotational inertia) about its center of mass.
        
        Returns:
            pint.Quantity: Area moment of inertia about the origin.
        """
        return self.Ix + self.Iz
    
    @property
    def I_x(self) -> pint.Quantity:
        """
        Calculate the mass moment of inertia (rotational inertia) about its center of mass.
        
        Returns:
            pint.Quantity: Mass moment of inertia about the x-axis (radial orientation).
        """
        return 1 / 12 * self.mass * (3 * self.outer_radius ** 2
                                     + 3 * self.inner_radius ** 2
                                     + self.length ** 2)
    
    @property
    def I_y(self) -> pint.Quantity:
        """
        Calculate the mass moment of inertia (rotational inertia) about its center of mass.
        
        Returns:
            pint.Quantity: Mass moment of inertia about the y-axis (longitudinal orientation).
        """
        return 1 / 2 * self.mass * (self.outer_radius ** 2
                                    + self.inner_radius ** 2)    
    
    @property
    def I_z(self) -> pint.Quantity:
        """
        Calculate the mass moment of inertia (rotational inertia) about its center of mass.
        
        Returns:
            pint.Quantity: Mass moment of inertia about the z-axis (radial orientation).
        """
        return self.I_x
    
    @property
    def I(self) -> np.ndarray:
        """
        Get the moment of inertia matrix.
        
        Returns:
            np.ndarray: A diagonal matrix with moment of inertia for each axis.
        """
        return np.diag([self.I_x.m,
                        self.I_y.m,
                        self.I_z.m])
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the collection."""
        return (f"{self.__class__.__name__}"
                f"(outer={self.outer_radius}, inner={self.inner_radius})")
        
    def __str__(self) -> str:
        """
        Return a string representation of the cylinder.
        
        Returns:
            str: A string that describes the cylinder.
        """
        return (f"{self.__repr__()}\n"
                f"Material: {self.material.name}\n"
                f"Outer Radius: {self.outer_radius}\n"
                f"Inner Radius: {self.inner_radius}\n"
                f"Length: {self.length}\n"
                f"Density: {self.material.density}\n"
                f"Volume: {self.volume:.4e~P}\n"
                f"Mass: {self.mass:.4f~P}\n"
                f"Cross-sectional Area: {self.cross_sectional_area:.4e~P}\n"
                f"Surface Area: {self.surface_area:.4e~P}\n"
                f"Area Moment of Inertia, x: {self.Ix:.4e~P}\n"
                f"Area Moment of Inertia, y: {self.Iz:.4e~P}\n"
                f"Polar Moment of Inertia, z: {self.J:.4e~P}\n"
                f"Mass Moment of Inertia, x: {self.I_x:.4e~P}\n"
                f"Mass Moment of Inertia, y: {self.I_y:.4e~P}\n"
                f"Mass Moment of Inertia, z: {self.I_z:.4e~P}")
    

class Collection:
    """A class to hold a collection of objects.

    Attributes:
        _items (list): A list to store the items in the collection.
    """

    def __init__(self, *args) -> None:
        """Initializes the collection with optional initial items.

        Args:
            *args: Variable length argument list. Can be a list, a tuple, or multiple objects.
        """
        self._items = []
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            self._items.extend(args[0])
        else:
            self._items.extend(args)

    def add_item(self, item) -> None:
        """Adds an item to the collection.

        Args:
            item: The item to be added to the collection.
        """
        self._items.append(item)

    def remove_item(self, item) -> None:
        """Removes an item from the collection.

        Args:
            item: The item to be removed from the collection.

        Raises:
            ValueError: If the item is not found in the collection.
        """
        if item in self._items:
            self._items.remove(item)
        else:
            raise ValueError("Item not found in the collection.")

    @property
    def items(self) -> List:
        """Returns all items in the collection.

        Returns:
            list: A list of all items in the collection.
        """
        return self._items

    def find_item(self, item) -> bool:
        """Checks if an item exists in the collection.

        Args:
            item: The item to be checked.

        Returns:
            bool: True if the item is in the collection, False otherwise.
        """
        return item in self._items

    def __len__(self) -> int:
        """Returns the number of items in the collection.

        Returns:
            int: The number of items in the collection.
        """
        return len(self._items)

    def __getitem__(self, index: int):
        """Gets the item at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            The item at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        return self._items[index]

    def __repr__(self) -> str:
        """Returns a string representation of the collection.

        Returns:
            str: A string representation of the collection.
        """
        return f"Collection, {len(self._items)} items ({self._items})"

    def __str__(self) -> str:
        """Returns a user-friendly string representation of the collection.

        Returns:
            str: A user-friendly string representation of the collection.
        """
        string = ""
        for item in self._items:
            string += item.__str__() + "\n\n"
        return (f"{self.__repr__()}\n\n{string}")
    
if __name__ == '__main__':
    # Example of Material Class
    sae_4140_steel = Material(name='SAE 4140 Steel',
                              density=Q_(7850, 'kg/m^3'),
                              young_modulus=Q_(2e11, "Pa"))
    print(sae_4140_steel, '\n')

    # Example of Cylinder Class
    cylinder1 = Cylinder(outer_radius=Q_(0.5, 'm'),
                         inner_radius=Q_(0.2, 'm'),
                         length=Q_(1.0, 'm'),
                         material=sae_4140_steel)
    cylinder2 = Cylinder(outer_radius=Q_(0.3, 'm'),
                         inner_radius=Q_(0.1, 'm'),
                         length=Q_(0.8, 'm'),
                         material=sae_4140_steel)
    cylinder3 = Cylinder(outer_radius=Q_(0.6, 'm'),
                         inner_radius=Q_(0.2, 'm'),
                         length=Q_(1.2, 'm'),
                         material=sae_4140_steel)
    print(cylinder1, '\n')
    print(cylinder2, '\n')
    print(cylinder3, '\n')

    # Example of Collection Class
    collection = Collection(cylinder1, cylinder2)
    print(collection, '\n')

    # Adding an item to the collection
    collection.add_item(cylinder3)
    print("After adding cylinder3:\n", collection, '\n')

    # Removing an item from the collection
    collection.remove_item(cylinder2)
    print("After removing cylinder2:\n", collection, '\n')

    # Checking if an item is in the collection
    print("Is cylinder2 in the collection?", collection.find_item(cylinder2))
    print("Is cylinder3 in the collection?", collection.find_item(cylinder3), '\n')

    # Accessing items by index
    print("Item at index 0:", collection[0], '\n')

    # Iterating through the collection
    print("Iterating through the collection:")
    for item in collection:
        print(item, '\n')