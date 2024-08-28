# -*- coding: utf-8 -*-
"""
rotor_analysis
==============

A package for performing rotor dynamics analysis.

Modules
-------
rotordynamics
    Contains classes to model a spinning rotor with a mass attached, including:
    - Disc: Represents a disc and calculates its kinetic energy.
    - Shaft: Represents a shaft.
    - Rotor: Represents a rotor assembly with a shaft and discs.
utilities
    Provides classes to model materials and geometric objects like cylinders 
    and discs, including:
    - Material: Represents a material with a name and density.
    - Cylinder: Represents a hollow cylinder and calculates its properties.
    - Collection: A class to hold a collection of objects.
results
    Manages the storage and retrieval of analysis results. (Under construction)

Example
-------
>>> from rotor_analysis import rotordynamics, utilities
>>> steel = utilities.Material(name='Steel',
                               density=utilities.Q_(7800, 'kg/m^3'),
                               young_modulus=utilities.Q_(2e11, 'Pa'))
>>> shaft = rotordynamics.Shaft(outer_radius=utilities.Q_(0.01, 'm'),
                                inner_radius=utilities.Q_(0.0, 'm'),
                                length=utilities.Q_(0.4, 'm'),
                                material=steel)
>>> disc = rotordynamics.Disc(outer_radius=utilities.Q_(0.150, 'm'),
                              inner_radius=utilities.Q_(0.010, 'm'),
                              length=utilities.Q_(0.030, 'm'),
                              material=steel,
                              coordinate=utilities.Q_(0.1, 'm'))
>>> rotor = rotordynamics.Rotor(shaft,
                                disc,
                                material='steel',
                                max_speed=5000)
>>> print(rotor)
Rotor (1 Shaft, 1 Disc(s))
"""

__author__ = "Cristofer Antoni Souza Costa"
__version__ = "0.1.0"
__email__ = "cristofercosta@yahoo.com.br"
__status__ = "Prototype"

from .rotordynamics import Disc, Shaft, Rotor
from .utilities import Material, Q_
from .results import (
    add_secondary_yaxis,
    campbell_diagram_axial_forces,
    create_video_from_frames,
    FFT,
    interactive_orbit,
    interactive_orbit_campbell,
    interactive_orbit_campbell_async,
    interactive_orbit_fixed_speed,
    plot_vibration_amplitude,
    save_orbit_frames,
    update_circle_and_sine,
)


__all__ = [
    "add_secondary_yaxis",
    "campbell_diagram_axial_forces",
    "create_video_from_frames",
    "Disc",
    "FFT",
    "interactive_orbit",
    "interactive_orbit_campbell",
    "interactive_orbit_campbell_async",
    "interactive_orbit_fixed_speed",
    "Material",
    "plot_vibration_amplitude",
    "Q_",
    "Rotor",
    "save_orbit_frames",
    "Shaft",
    "update_circle_and_sine",
]
