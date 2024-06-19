# -*- coding: utf-8 -*-
"""
This module provides classes to model a spinning rotor with a mass attached.

Classes:
    Disc: Represents a disc, inheriting from Cylinder, and calculates its 
        kinetic energy.
    Shaft: Represents a shaft, inheriting from Cylinder.
    Rotor: Represents a rotor assembly with a shaft and discs.
"""

import pint
import numpy as np
from scipy.optimize import fsolve
import plotly.graph_objects as go
import plotly.offline as pyo
from typing import List, Tuple
import utilities as ut

Material = ut.Material
Q_ = ut.Q_

class Disc(ut.Cylinder):
    """Class representing a disc, inheriting from Cylinder."""
    
    def __init__(self,
                 outer_radius: pint.Quantity,
                 inner_radius: pint.Quantity,
                 length: pint.Quantity,
                 material: ut.Material,
                 coordinate: pint.Quantity) -> None:
        """
        Initialize a Disc object.
        
        Args:
            outer_radius (pint.Quantity): The outer radius of the disc in meters.
            inner_radius (pint.Quantity): The inner radius of the hollow part of the disc in meters.
            length (pint.Quantity): The length of the disc in meters.
            material (Material): The material of the disc.
            coordinate (pint.Quantity): The position of the disc along the shaft in meters.
        """
        super().__init__(outer_radius, inner_radius, length, material)
        self.coordinate = coordinate


class Shaft(ut.Cylinder):
    """Class representing a shaft, inheriting from Cylinder."""
    pass


class Rotor:
    def __init__(self,
                 shaft: Shaft,
                 *discs: Disc,
                 **properties) -> None:
        paramenters = ["axial_force"]
        """
        Initialize a Rotor object.
        
        Args:
            shaft (Shaft): The shaft of the rotor.
            *discs (Disc): Variable length argument list of discs.
            **properties: Additional properties of the rotor.
        """
        self._shaft = shaft
        if len(discs) == 1 and isinstance(discs[0], (list, tuple)):
            self._discs = ut.Collection(discs[0])
        else:
            self._discs = ut.Collection(discs)
        for key, value in properties.items():
            setattr(self, key, value)

    @property
    def shaft(self) -> Shaft:
        """Return the shaft of the rotor."""
        return self._shaft

    @property
    def discs(self) -> ut.Collection:
        """Return the discs of the rotor."""
        return self._discs
    
    @property
    def length(self) -> pint.Quantity:
        """Return the length of the shaft."""
        return self.shaft.length
    
    @property
    def mass(self) -> float:
        """
        Calculate the mass of the rotor.
        
        Returns:
            float: The mass of the rotor.

        Reference:
            Rotordynamics Prediction in Engineering, Second Edition, by Michel 
            Lalanne and Guy Ferraris equation 2.12.
        """
        m_0 = 0.0
        for disc in self.discs:
            m_0 += disc.mass.m * self.f(disc.coordinate.m)**2 + disc.I_x.m * self.g(disc.coordinate.m)**2
        
        shaft = self.shaft
        m_1 = (shaft.material.density.m * shaft.cross_sectional_area.m * 
               self.int_f_sqr(shaft.length.m) +
               shaft.material.density.m * shaft.Ix.m * self.int_g_sqr(shaft.length.m))

        return m_0 + m_1
    
    @property
    def M(self) -> np.ndarray:
        """
        Get the mass matrix.
        
        Returns:
            np.ndarray: Mass matrix.
        """
        mass = np.array((
            [self.mass, 0],
            [0, self.mass]
        ))
        return mass
    
    @property
    def M_inverse(self) -> np.ndarray:
        """
        Get the inverse of the mass matrix.
        
        Returns:
            np.ndarray: Mass matrix.
        """
        mass = np.array((
            [1/self.mass, 0],
            [0, 1/self.mass]
        ))
        return mass
    
    @property
    def a(self) -> float:
        """
        Calculate the parameter 'a' for the rotor dynamics.
        
        Returns:
            float: The parameter 'a'.

        Reference:
            Rotordynamics Prediction in Engineering, Second Edition, by Michel 
            Lalanne and Guy Ferraris equation 2.12.
        """
        a_0 = 0.0
        for disc in self.discs:
            a_0 += disc.I_y.m * self.g(disc.coordinate.m)**2
        
        shaft = self.shaft
        a_1 = 2 * shaft.material.density.m * shaft.Ix.m * self.int_g_sqr(shaft.length.m)
        return a_0 + a_1
    
    @property
    def G(self) -> np.ndarray:
        """
        Get the gyroscopic matrix.
        
        Returns:
            np.ndarray: Gyroscopic matrix.
        """
        gyro = np.array((
            [0, -self.a],
            [self.a, 0]
        ))
        return gyro
    
    @property
    def stiffness(self) -> float:
        """
        Calculate the stiffness of the rotor.
        
        Returns:
            float: The stiffness of the rotor.
        """
        shaft = self.shaft
        k_0 = shaft.material.young_modulus.m * shaft.Ix.m * self.int_h_sqr(shaft.length.m)

        # Axial force
        if hasattr(self, 'axial_force'):
            F0 = self.axial_force.to('N').m
            k_1 = F0 * self.int_g_sqr(shaft.length.m)
            return k_0 + k_1
        return k_0

    @property
    def K(self) -> np.ndarray:
        """
        Get the stiffness matrix.
        
        Returns:
            np.ndarray: Stiffness matrix.
        """
        K = np.array((
            [self.stiffness, 0],
            [0, self.stiffness]
        ))
        return K
    
    def A(self, speed):
        """
        This function returns the A matrix of the state-space representation.

        Args:
            speed (float): rotational speed in RPM.
        """
        zero_matrix = np.zeros_like(self.M)
        identity = np.eye(self.M.shape[0])

        omega = speed / 60 * 2 * np.pi
        bottom_block = np.hstack((-self.M_inverse @ (self.K), -self.M_inverse @ (omega * self.G)))

        top_block = np.hstack((zero_matrix, identity))
        A = np.vstack((top_block, bottom_block))
        return A
    
    def characteristic_eq(self,
                f: float,
                speed: float) -> float:
        """
        Characteristic equation function. Their roots are the eigenvalues.
        
        Args:
            f (float): Frequency in Hertz.
            speed (float): Rotational speed in RPM.
        
        Returns:
            float: The computed value.
        """
        comp = (self.mass**2 * (2 * np.pi * f)**4 - 
                (2 * self.stiffness * self.mass + self.a**2 * (speed / 60 * 2 * np.pi)**2) * (2 * np.pi * f)**2 +
                self.stiffness**2)
        return comp
    
    @property
    def omega_0(self) -> float:
        """
        Calculate the natural frequency of the rotor at 0 RPM.
        
        Returns:
            float: The natural frequency.
        """
        omega = fsolve(self.characteristic_eq, 10, args=0)
        return omega[0]
   
    def compute_roots(self,
              speed_range: np.ndarray = np.linspace(0, 9000, 101)) -> Tuple[List[float], List[float]]:
        """
        Calculate the forward and backward whirl speeds.
        
        Args:
            speed_range (np.ndarray): Array of rotational speeds in RPM.
        
        Returns:
            tuple[List[float], List[float]]: Forward and backward whirl speeds.
        """
        
        # Computing the eigenvalues
        omega_0 = self.omega_0
        roots_fw = [omega_0]
        roots_bw = [omega_0]
        
        for speed in speed_range[1:]:
            root_fw = fsolve(self.characteristic_eq, roots_fw[-1] + 1, args=(speed))
            root_bw = fsolve(self.characteristic_eq, roots_bw[-1] - 1, args=(speed))
            roots_fw.append(root_fw[0])
            roots_bw.append(root_bw[0])

        # Computing the critical speeds
        backward_diff = 1000
        backward_guess = 0
        forward_diff = 1000
        forward_guess = 0
        for i , j , k in zip(speed_range, roots_fw, roots_bw):
            if i > 0:
                if abs(j - i / 60) < forward_diff:
                    forward_diff = abs(j - i / 60)
                    forward_guess = i
                if abs(k - i / 60) < backward_diff:
                    backward_diff = abs(j - i / 60)
                    backward_guess = i
        
        from scipy.optimize import newton

        self.critical_speed_fw = Q_(newton(self.__func, forward_guess), "rpm")
        
        self.critical_speed_bw = Q_(newton(self.__func, backward_guess), "rpm")
        return roots_fw, roots_bw

    def __func(self, x):
        """Funtion to solve Newton and calculate the critical speeds"""    
        return self.characteristic_eq(x/60, x)
    
    def plot_Campbell(self,
                      speed_range: np.ndarray = np.linspace(0, 9000, 101),
                      return_data=False,
                      export_chart=False) -> None:
        """
        Plots a Campbell diagram.
        
        Args:
            speed_range (np.ndarray): Array of rotational speeds in RPM.
            export_chart (bool): Whether to export the chart to an HTML file.
            return_data (bool): Whether to return the chart data in a dict.
        
        Returns:
            None
        """
        fw, bw = self.compute_roots(speed_range)
        # Create traces
        trace0 = go.Scatter(
            x=speed_range,
            y=speed_range / 60,  # Converting RPM to Hertz
            mode='lines',
            name='Rotational Speed Frequency'
        )
        trace1 = go.Scatter(
            x=speed_range,
            y=fw,
            mode='lines',
            name='Forward'
        )
        trace2 = go.Scatter(
            x=speed_range,
            y=bw,
            mode='lines',
            name='Backward'
        )
        point_fw = go.Scatter(
            x=[self.critical_speed_fw.m],
            y=[self.critical_speed_fw.m / 60],
            mode='markers',
            marker={'size' : 10},
            hovertext=f'{self.critical_speed_fw:.0f~P}',
            name=f'Critical speed: {self.critical_speed_fw:.0f~P}'
        )
        point_bw = go.Scatter(
            x=[self.critical_speed_bw.m],
            y=[self.critical_speed_bw.m / 60],
            mode='markers',
            marker={'size' : 10},
            hovertext=f'{self.critical_speed_bw:.0f~P}',
            name=f'Critical speed: {self.critical_speed_bw:.0f~P}'
        )
        # Create the layout
        layout = go.Layout(
            title='Campbell Diagram',
            xaxis=dict(title='Rotational Speed (RPM)'),
            yaxis=dict(title='Natural Frequency (Hz)')
        )
        # Create the figure
        fig = go.Figure(data=[trace0, trace1, trace2, point_fw, point_bw], layout=layout)

        # Plot the figure
        if export_chart:
            pyo.plot(fig, filename='chart.html')
        fig.show()

        # Export the data
        if return_data:
            data = {
                'Speed' : list(speed_range),
                'Forward': fw,
                'Backward': bw
            }
            return data
      
    def f(self, y: float) -> float:
        """
        Displacement function.
        
        Args:
            y (float): Position along the shaft.
        
        Returns:
            float: Displacement.
        """
        return np.sin(np.pi * y / self.shaft.length.m)

    def int_f_sqr(self, y: float) -> float:
        """
        Integral of f**2.
        
        Args:
            y (float): Position along the shaft.
        
        Returns:
            float: Integral value.
        """
        return y / 2 - self.shaft.length.m * np.sin(2 * np.pi * y / self.shaft.length.m) / 4 / np.pi 

    def g(self, y: float) -> float:
        """
        Derivative of Displacement function.
        
        Args:
            y (float): Position along the shaft.
        
        Returns:
            float: Derivative value.
        """
        return np.pi / self.shaft.length.m * np.cos(np.pi * y / self.shaft.length.m)

    def int_g_sqr(self, y: float) -> float:
        """
        Integral g**2.
        
        Args:
            y (float): Position along the shaft.
        
        Returns:
            float: Integral value.
        """
        shaft = self.shaft
        return np.pi * (shaft.length.m * np.sin(2 * np.pi * y / shaft.length.m) + 2 * np.pi * y) / 4 / shaft.length.m**2

    def h(self, y: float) -> float:
        """
        Second Derivative of Displacement function.
        
        Args:
            y (float): Position along the shaft.
        
        Returns:
            float: Second derivative value.
        """
        return - (np.pi / self.shaft.length.m)**2 * np.sin(np.pi * y / self.shaft.length.m)

    def int_h_sqr(self, y: float) -> float:
        """
        Integral of h**2.
        
        Args:
            y (float): Position along the shaft.
        
        Returns:
            float: Integral value.
        """
        return np.pi**3 * (2 * np.pi * y - self.shaft.length.m * np.sin(2 * np.pi * y / self.shaft.length.m)) / 4 / self.shaft.length.m**4

    def plot_displacement_functions(self) -> None:
        """
        Plot the displacement function and its derivatives.
        
        Returns:
            None
        """
        x = np.linspace(0, self.shaft.length.m)
        trace = go.Scatter(
            x=x,
            y=self.f(np.linspace(0, self.shaft.length.m)),
            mode='lines',
            name='Displacement function'
        )
        trace1 = go.Scatter(
            x=x,
            y=self.g(np.linspace(0, self.shaft.length.m)),
            mode='lines',
            name='First derivative'
        )
        trace2 = go.Scatter(
            x=x,
            y=self.h(np.linspace(0, self.shaft.length.m)),
            mode='lines',
            name='Second derivative'
        )
        # Create the layout
        layout = go.Layout(
            title='Rotor Lateral Displacement Function',
            xaxis=dict(title=f'Length [{self.shaft.length.units}]'),
            yaxis=dict(title='Deflection [dimensionless]')
        )
        # Create the figure
        fig = go.Figure(data=[trace, trace1, trace2], layout=layout)
        
        fig.show()
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the rotor."""
        return f"{self.__class__.__name__} (1 Shaft, {len(self._discs)} Disc(s))"
    
    def __str__(self) -> str:
        """
        Return a string representation of the rotor.
        
        Returns:
            str: A string that describes the rotor.
        """
        return (f"{self.__repr__()}\n"
                f"\nShaft:\n"
                f"{self.shaft.__str__()}\n"
                f"\nDisc(s):\n"
                f"\n{self.discs.__str__()}")


if __name__ == '__main__':
    '''
    The numerical data were extracted from the second chapter of the book
    "Rotordynamics Prediction in Engineering, Second Edition" by Michel 
    Lalanne and Guy Ferraris.
    '''
    # Instantiate the Material Class
    steel = ut.Material(name='Steel',
                        density=Q_(7800, 'kg/m^3'),
                        young_modulus=Q_(2e11, "Pa"))
    print(steel, '\n')

    # Example of Shaft Class
    L = Q_(0.4, 'm')
    shaft = Shaft(outer_radius=Q_(0.01, 'm'),
                  inner_radius=Q_(0.0, 'm'),
                  length=L,
                  material=steel)
    print(shaft, '\n')

    # Example of Disc Class
    disc = Disc(outer_radius=Q_(0.150, 'm'),
                inner_radius=Q_(0.010, 'm'),
                length=Q_(0.030, 'm'),
                material=steel,
                coordinate=L/3)
    print(disc, '\n')
    
    # Create a Rotor
    # First option with two discs
    rotor = Rotor(shaft, (disc, disc), material="steel", max_speed=5000)
    print("First option to instantiate:", rotor.discs, sep='\n')

    # Second Option with two discs
    rotor = Rotor(shaft, disc, disc, material="steel", max_speed=5000)
    print("Second option to instantiate:", rotor.discs, sep='\n')

    # Third Option with two discs
    rotor = Rotor(shaft, [disc, disc], material="steel", max_speed=5000)
    print("Third option to instantiate:", rotor.discs, sep='\n')
    
    # object with additional properties
    rotor = Rotor(shaft, disc, max_speed=Q_(9000, 'rpm'))
    print(rotor)
    print(f"Natural frequency at 0 rpm: {rotor.omega_0:.3f} Hz")
    rotor.plot_Campbell()
