# -*- coding: utf-8 -*-
"""
This module is part of the rotor_analysis package and provides classes to model
a spinning rotor with a mass attached. It provides functions to plot the charts
and results of the model.

Functions
---------
campbell_diagram_axial_forces(data0, data1, data2, F0):
    Creates a Campbell diagram to compare the natural frequencies of three different
    rotor configurations under varying axial forces.

add_secondary_yaxis(fig, values, yaxis='y2', overlaying_axis='y'):
    Adds a secondary y-axis to an existing Plotly figure to display additional data.
"""

__author__ = "Cristofer Antoni Souza Costa"
__version__ = "0.0.1"
__email__ = "cristofercosta@yahoo.com.br"
__status__ = "Development"

import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import ipywidgets as widgets
import cv2

from typing import Any, Callable, Dict, Tuple
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from IPython.display import display

from .rotordynamics import Rotor



def campbell_diagram_axial_forces(
    data0: Dict[str, Any], data1: Dict[str, Any], data2: Dict[str, Any], F0: float
) -> None:
    """
    Creates a Campbell diagram to compare the natural frequencies of three different
    rotor configurations under varying axial forces.

    Parameters
    ----------
    data0 to data2 : dict
        Dictionary containing speed and frequency data for the no-load condition.
        Keys should include 'Speed', 'Forward', and 'Backward'.
    F0 : float
        Magnitude of the axial force applied in Newtons.

    Notes
    -----
    The function plots the Campbell diagram using Plotly, displaying the rotational speed
    vs. natural frequency for the forward and backward modes under no load, positive axial
    load, and negative axial load conditions. An interactive slider widget is provided
    to print the natural frequencies at a selected speed.
    """
    # Create traces
    speed_range = np.array(data0["Speed"])
    trace0 = go.Scatter(
        x=speed_range,
        y=speed_range / 60,  # Converting RPM to Hertz
        mode="lines",
        name="Rotational Speed Frequency",
    )
    trace1_line = go.scatter.Line(color="#EF553B")
    trace3_line = go.scatter.Line(color="#00CC96")
    trace5_line = go.scatter.Line(color="#AB63FA")

    trace1 = go.Scatter(
        x=speed_range,
        y=np.array(data0["Forward"]),
        line=trace1_line,
        mode="lines",
        name="Forward no load",
    )
    trace2 = go.Scatter(
        x=speed_range,
        y=np.array(data0["Backward"]),
        line=trace1_line,
        mode="lines",
        name="Backward no load",
    )
    trace3 = go.Scatter(
        x=speed_range,
        y=np.array(data1["Forward"]),
        line=trace3_line,
        mode="lines",
        name=f"Forward {F0} N",
    )
    trace4 = go.Scatter(
        x=speed_range,
        y=np.array(data1["Backward"]),
        line=trace3_line,
        mode="lines",
        name=f"Backward {+F0} N",
    )
    trace5 = go.Scatter(
        x=speed_range,
        y=np.array(data2["Forward"]),
        line=trace5_line,
        mode="lines",
        name=f"Forward {-F0} N",
    )
    trace6 = go.Scatter(
        x=speed_range,
        y=np.array(data2["Backward"]),
        line=trace5_line,
        mode="lines",
        name=f"Backward {-F0} N",
    )
    # Create the layout
    layout = go.Layout(
        title="Campbell Diagram",
        xaxis=dict(title="Rotational Speed (RPM)"),
        yaxis=dict(title="Natural Frequency (Hz)"),
    )
    # Create the figure
    fig = go.Figure(
        data=[trace0, trace1, trace2, trace3, trace4, trace5, trace6], layout=layout
    )

    # Legend configuration and size
    fig.update_layout(
        autosize=False,
        width=500 * 1.62,
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    return fig


def add_secondary_yaxis(
    fig: go.Figure, values: np.ndarray, yaxis: str = "y2", overlaying_axis: str = "y", title: str = ""
) -> None:
    """
    Add a secondary y-axis to an existing Plotly figure to display additional data.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The existing Plotly figure to which the secondary y-axis will be added.
    values : array-like
        The data values to be plotted on the secondary y-axis.
    yaxis : str, optional
        The identifier for the secondary y-axis. Default is 'y2'.
    overlaying_axis : str, optional
        The primary y-axis that the secondary y-axis will overlay. Default is 'y'.

    Notes
    -----
    This function modifies the provided Plotly figure by adding a secondary y-axis on the right side,
    configured to display the provided data values. The secondary y-axis uses a logarithmic scale and is
    labeled 'Amplitude (m)'. A trace for the secondary y-axis is added to the figure, representing the
    vibration amplitude.
    """
    # Add a secondary y-axis
    campbell_data = fig.to_dict()["data"]
    
    fig.update_layout(
        yaxis2=dict(
            title="Amplitude (m)",
            overlaying=overlaying_axis,
            type="log",
            side="right"
        ),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99),
        title="Campbell Diagram + Response",
    )

    # Add a trace for the secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=campbell_data[0]["x"],
            y=values,
            name="Vibration Amplitude " + title,
            yaxis=yaxis
        )
    )


def update_circle_and_sine(
    A1_func: Callable[..., float],
    A2_func: Callable[..., float],
    speed: float,
    theta: np.ndarray,
    *args: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update the circle and sine data based on the speed.

    Parameters
    ----------
    A1_func : function
        Function to calculate A1.
    A2_func : function
        Function to calculate A2.
    args: any
        Parameter as per A1 and A2 functions.
    speed : float
        The speed in RPM.
    theta : np.ndarray
        Array with the angle range to compute A1 and A2.

    Returns
    -------
    tuple
        x and y coordinates for the circle and sine plots.
    """
    A1_value = A1_func(speed, *args)
    A2_value = A2_func(speed, *args)

    q1_circle = A1_value * np.sin(theta)
    q2_circle = A2_value * np.cos(theta)
    return q1_circle, q2_circle


def interactive_orbit(
    A1_func,
    A2_func,
    *args: Any,
    initial_speed: float = 3200,
    max_amplitude: float = 20e-6,
) -> go.Figure:
    """
    Create a plot with a slider to vary speed from 0 to 9000 RPM.

    Parameters
    ----------
    A1_func : function
        Function to calculate A1.
    A2_func : function
        Function to calculate A2.
    initial_speed : float, optional
        Initial speed to plot the interactive orbit plot. Default is 3200.
    max_amplitude : float, optional
        Max amplitude of displacements q1 and q2. Default is 20e-6.

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive plot with orbit and time plots.
    """
    # Define the angle range to compute one revolution
    theta = np.linspace(0, 1.85 * np.pi, 50)

    # Get initial orbit and sine data
    q1_circle, q2_circle = update_circle_and_sine(
        A1_func, A2_func, initial_speed, theta, *args
    )

    # Define traces for the orbit and its starting point
    trace_orbit = go.Scatter(x=q1_circle, y=q2_circle, showlegend=False, name="Orbit")
    starting_point = go.Scatter(
        x=[q1_circle[0]],
        y=[q2_circle[0]],
        marker={"size": 10, "color": "#636EFA"},
        showlegend=False,
        name="Starting Point",
    )

    # Define traces for the cosine and sine plots
    q1_trace = go.Scatter(
        x=theta, y=q1_circle, mode="lines", name="q1", showlegend=False
    )
    q2_trace = go.Scatter(
        x=theta, y=q2_circle, mode="lines", name="q2", showlegend=False
    )

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=(f"Orbit @ {initial_speed} rpm", "q1 and q2")
    )

    # Add traces to the subplots
    fig.add_trace(trace_orbit, row=1, col=1)
    fig.add_trace(starting_point, row=1, col=1)
    fig.add_trace(q1_trace, row=1, col=2)
    fig.add_trace(q2_trace, row=1, col=2)
    fig.add_annotation(x=2, y=q1_circle[17], text="q1", row=1, col=2)
    fig.add_annotation(x=4, y=q2_circle[34], text="q2", row=1, col=2)

    # Set fixed axis ranges
    fig.update_xaxes(
        title="q1 (m)", range=[-1.2 * max_amplitude, 1.2 * max_amplitude], row=1, col=1
    )
    fig.update_yaxes(
        title="q2 (m)", range=[-1.2 * max_amplitude, 1.2 * max_amplitude], row=1, col=1
    )
    fig.update_xaxes(title="omega * t (Rad)", range=[0, 1.85 * np.pi], row=1, col=2)
    fig.update_yaxes(range=[-1.2 * max_amplitude, 1.2 * max_amplitude], row=1, col=2)

    fig.update_layout({"title": f"Orbit + Timeplot"})

    # Create slider steps
    steps = []
    for speed in range(0, 9001, 50):
        q1_circle, q2_circle = update_circle_and_sine(
            A1_func, A2_func, speed, theta, *args
        )
        # Adjusting the annotations for q1 and q2 in time representation
        annotations = fig.to_dict()["layout"]["annotations"]
        annotations[0]["text"] = f"Orbit @ {speed} rpm"
        annotations[2]["y"] = q1_circle[17]
        annotations[3]["y"] = q2_circle[34]
        step = dict(
            method="update",
            args=[
                {
                    "x": [q1_circle, [q1_circle[0]], theta, theta],
                    "y": [q2_circle, [q2_circle[0]], q1_circle, q2_circle],
                },
                {"annotations": annotations},
            ],
            label=f"{speed}",
        )
        steps.append(step)

    # Define the slider
    active_slider = 0 if initial_speed == 0 else int(initial_speed / 50)

    sliders = [
        dict(
            active=active_slider,
            currentvalue={"prefix": "Speed: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    # Update the layout with the slider
    fig.update_layout(autosize=False, width=500 * 1.62, height=500, sliders=sliders)
    return fig


def interactive_orbit_campbell(
    campbell: go.Figure,
    A1_func: Callable[..., float],
    A2_func: Callable[..., float],
    *args: Any,
    initial_speed: float = 3200,
    max_amplitude: float = 20e-6
) -> go.Figure:
    """
    Create a plot with a slider to vary speed from 0 to 9000 RPM.

    Parameters
    ----------
    A1_func : function
        Function to calculate A1.
    A2_func : function
        Function to calculate A2.
    initial_speed : float, optional
        Initial speed to plot the interactive orbit plot. Default is 3200.
    max_amplitude : float, optional
        Max amplitude of displacements q1 and q2. Default is 20e-6.

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive plot with orbit and time plots.
    """
    # Define the angle range to compute one revolution
    theta = np.linspace(0, 1.85 * np.pi)

    # Get initial orbit and sine data
    q1_circle, q2_circle = update_circle_and_sine(A1_func, A2_func, initial_speed, theta, *args)

    # Define traces for the orbit and starting point
    trace_circle = go.Scatter(
        x=q1_circle,
        y=q2_circle,
        showlegend=False,
        name='Orbit'
    )
    starting_point = go.Scatter(
        x=[q1_circle[0]],
        y=[q2_circle[0]],
        marker={'size': 10, 'color': '#636EFA'},
        showlegend=False,
        name='Starting Point'
    )

    # Define traces for the cosine and sine plots
    q1_trace = go.Scatter(
        x=theta,
        y=q1_circle,
        mode='lines',
        name='q1',
        showlegend=False
    )
    q2_trace = go.Scatter(
        x=theta,
        y=q2_circle,
        mode='lines',
        name='q2',
        showlegend=False
    )

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2, "secondary_y": True}, None]],
        vertical_spacing=0.12,
        row_heights=[0.40, 0.60],
        subplot_titles=(f"Orbit @ {initial_speed} rpm", "q1 and q2", "Campbell + Vibration Amplitude")
    )

    # Add traces to the subplots
    fig.add_trace(trace_circle, row=1, col=1)
    fig.add_trace(starting_point, row=1, col=1)
    fig.add_trace(q1_trace, row=1, col=2)
    fig.add_trace(q2_trace, row=1, col=2)
    fig.add_annotation(x=2, y=q1_circle[17], text='q1', row=1, col=2)
    fig.add_annotation(x=4, y=q2_circle[34], text='q2', row=1, col=2)

    # Adding Campbell data
    campbell = campbell.to_dict()['data']
    for data in campbell[:-1]:
        fig.add_trace(data, row=2, col=1)

    fig.add_trace(campbell[-1], secondary_y=True, row=2, col=1)

    fig.update_layout(
        yaxis3=dict(
            title='Frequency (Hz)',
            anchor='x3',
            domain=[0.0, 0.51],
            range=[0, campbell[0]['y'][-1]]
        ),
        yaxis4=dict(
            title='Amplitude (m)',
            overlaying='y3',
            type='log',
            side='right',
        )
    )

    # Add vertical dashed gray line
    fig.add_shape(
        type="line",
        x0=initial_speed,
        x1=initial_speed,
        y0=0,
        y1=campbell[0]['y'][-1],
        line=dict(color="gray", width=2, dash="dot"),
        row=2,
        col=1,
        name='Vertical Line'
    )

    max_amplitude = max(np.max(np.abs(q1_circle)), np.max(np.abs(q2_circle)))

    # Set fixed axis ranges
    fig.update_xaxes(title='q1 (m)', range=[-1.2 * max_amplitude, 1.2 * max_amplitude], row=1, col=1)
    fig.update_yaxes(title='q2 (m)', range=[-1.2 * max_amplitude, 1.2 * max_amplitude], row=1, col=1)
    fig.update_xaxes(title='omega * t (Rad)', range=[0, 1.85 * np.pi], row=1, col=2)
    fig.update_yaxes(range=[-1.2 * max_amplitude, 1.2 * max_amplitude], row=1, col=2)
    
    fig.update_layout({'title': 'Rotor Analysis'})

    # Create slider steps
    steps = []
    for speed in range(0, 9001, 50):
        layout_dict = fig.to_dict()['layout']

        q1_circle, q2_circle = update_circle_and_sine(A1_func, A2_func, speed, theta, *args)

        # Adjusting the y and x axis scales for orbit and timebase plot
        max_amplitude = max(np.max(np.abs(q1_circle)), np.max(np.abs(q2_circle)))
        yaxis = layout_dict['yaxis']
        yaxis['range'] = [-1.2 * max_amplitude, 1.2 * max_amplitude]
        xaxis = layout_dict['xaxis']
        xaxis['range'] = [-1.2 * max_amplitude, 1.2 * max_amplitude]
        yaxis2 = layout_dict['yaxis2']
        yaxis2['range'] = [-1.2 * max_amplitude, 1.2 * max_amplitude]

        # Adjusting the annotations for q1 and q2 in time representation
        annotations = layout_dict['annotations']
        annotations[0]['text'] = f"Orbit @ {speed} rpm"
        annotations[3]['y'] = q1_circle[17]
        annotations[4]['y'] = q2_circle[34]

        # Adjusting the shapes to speed
        shapes = layout_dict['shapes']
        shapes[0]['x0'] = speed
        shapes[0]['x1'] = speed

        step = dict(
            method='update',
            args=[{'x': [q1_circle, [q1_circle[0]], theta, theta,
                         campbell[0]['x'], campbell[1]['x'], campbell[2]['x'], campbell[3]['x'],
                         campbell[4]['x'], campbell[5]['x']],
                   'y': [q2_circle, [q2_circle[0]], q1_circle, q2_circle,
                         campbell[0]['y'], campbell[1]['y'], campbell[2]['y'], campbell[3]['y'],
                         campbell[4]['y'], campbell[5]['y']]},
                  {'annotations': annotations,
                   'yaxis': yaxis,
                   'xaxis': xaxis,
                   'yaxis2': yaxis2,
                   'shapes': shapes}],
            label=f'{speed}'
        )
        steps.append(step)

    # Define the slider
    active_slider = 0 if initial_speed == 0 else int(initial_speed / 50)
        
    sliders = [dict(
        active=active_slider,
        currentvalue={"prefix": "Speed: ", "suffix": " RPM"},
        pad={"t": 15},
        steps=steps,
        len=.96,
    )]

    # Update the layout with the slider
    fig.update_layout(
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.93),
        autosize=False,
        width=750,
        height=800,
        sliders=sliders
    )

    return fig

def interactive_orbit_campbell_async(
    campbell: go.Figure,
    A1_func: Callable[..., float],
    A2_func: Callable[..., float],
    *args: Any,
    initial_speed: float = 3200,
    max_amplitude: float = 20e-6
) -> go.Figure:
    """
    Create a plot with a slider to vary speed from 0 to 9000 RPM.

    Parameters
    ----------
    A1_func : function
        Function to calculate A1.
    A2_func : function
        Function to calculate A2.
    initial_speed : float, optional
        Initial speed to plot the interactive orbit plot. Default is 3200.
    max_amplitude : float, optional
        Max amplitude of displacements q1 and q2. Default is 20e-6.

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive plot with orbit and time plots.
    """
    # Define the angle range to compute one revolution
    theta = np.linspace(0, 1.85 * np.pi)

    # Get initial orbit and sine data
    q1_circle, q2_circle = update_circle_and_sine(A1_func, A2_func, initial_speed, theta, *args)

    # Define traces for the orbit and starting point
    trace_circle = go.Scatter(
        x=q1_circle,
        y=q2_circle,
        showlegend=False,
        name='Orbit'
    )
    starting_point = go.Scatter(
        x=[q1_circle[0]],
        y=[q2_circle[0]],
        marker={'size': 10, 'color': '#636EFA'},
        showlegend=False,
        name='Starting Point'
    )

    # Define traces for the cosine and sine plots
    q1_trace = go.Scatter(
        x=theta,
        y=q1_circle,
        mode='lines',
        name='q1',
        showlegend=False
    )
    q2_trace = go.Scatter(
        x=theta,
        y=q2_circle,
        mode='lines',
        name='q2',
        showlegend=False
    )

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2, "secondary_y": True}, None]],
        vertical_spacing=0.12,
        row_heights=[0.40, 0.60],
        subplot_titles=(f"Orbit @ {initial_speed} rpm", "q1 and q2", "Campbell + Vibration Amplitude")
    )

    # Add traces to the subplots
    fig.add_trace(trace_circle, row=1, col=1)
    fig.add_trace(starting_point, row=1, col=1)
    fig.add_trace(q1_trace, row=1, col=2)
    fig.add_trace(q2_trace, row=1, col=2)
    fig.add_annotation(x=2, y=q1_circle[17], text='q1', row=1, col=2)
    fig.add_annotation(x=4, y=q2_circle[34], text='q2', row=1, col=2)

    # Adding Campbell data
    campbell = campbell.to_dict()['data']
    for data in campbell[:-1]:
        fig.add_trace(data, row=2, col=1)

    fig.add_trace(campbell[-1], secondary_y=True, row=2, col=1)

    fig.update_layout(
        yaxis3=dict(
            title='Frequency (Hz)',
            anchor='x3',
            domain=[0.0, 0.51],
            range=[0, campbell[0]['y'][-1]]
        ),
        yaxis4=dict(
            title='Amplitude (m)',
            overlaying='y3',
            type='log',
            side='right',
        )
    )

    # Add vertical dashed gray line
    fig.add_shape(
        type="line",
        x0=initial_speed,
        x1=initial_speed,
        y0=0,
        y1=campbell[0]['y'][-1],
        line=dict(color="gray", width=2, dash="dot"),
        row=2,
        col=1,
        name='Vertical Line'
    )

    # Set fixed axis ranges
    fig.update_xaxes(title='q1 (m)', range=[-1.2 * max_amplitude, 1.2 * max_amplitude], row=1, col=1)
    fig.update_yaxes(title='q2 (m)', range=[-1.2 * max_amplitude, 1.2 * max_amplitude], row=1, col=1)
    fig.update_xaxes(title='omega * t (Rad)', range=[0, 1.85 * np.pi], row=1, col=2)
    fig.update_yaxes(range=[-1.2 * max_amplitude, 1.2 * max_amplitude], row=1, col=2)
    
    fig.update_layout({'title': 'Rotor Analysis'})

    # Create slider steps
    steps = []
    for speed in range(0, 9001, 50):
        q1_circle, q2_circle = update_circle_and_sine(A1_func, A2_func, speed, theta, *args)

        layout_dict = fig.to_dict()['layout']
        # Adjusting the annotations for q1 and q2 in time representation
        annotations = layout_dict['annotations']
        annotations[0]['text'] = f"Orbit @ {speed} rpm"
        annotations[3]['y'] = q1_circle[17]
        annotations[4]['y'] = q2_circle[34]

        # Adjusting the shapes to speed
        shapes = layout_dict['shapes']
        shapes[0]['x0'] = speed
        shapes[0]['x1'] = speed

        step = dict(
            method='update',
            args=[{'x': [q1_circle, [q1_circle[0]], theta, theta,
                         campbell[0]['x'], campbell[1]['x'], campbell[2]['x'], campbell[3]['x'],
                         campbell[4]['x'], campbell[5]['x'], campbell[6]['x'], campbell[7]['x']],
                   'y': [q2_circle, [q2_circle[0]], q1_circle, q2_circle,
                         campbell[0]['y'], campbell[1]['y'], campbell[2]['y'], campbell[3]['y'],
                         campbell[4]['y'], campbell[5]['y'], campbell[6]['y'], campbell[7]['y']]},
                  {'annotations': annotations,
                   'shapes': shapes}],
            label=f'{speed}'
        )
        steps.append(step)

    # Define the slider
    active_slider = 0 if initial_speed == 0 else int(initial_speed / 50)
        
    sliders = [dict(
        active=active_slider,
        currentvalue={"prefix": "Speed: ", "suffix": " RPM"},
        pad={"t": 15},
        steps=steps,
        len=.96,
    )]

    # Update the layout with the slider
    fig.update_layout(
        legend=dict(yanchor="top", y=0.51, xanchor="left", x=0.01),
        autosize=False,
        width=750,
        height=800,
        sliders=sliders
    )

    return fig


def plot_vibration_amplitude(A1_function,
                             A2_function=None,
                             start_freq=0,
                             end_freq=80,
                             num_points=8001,
                             speed=4000,
                             some_param=1) -> None:
    """
    Plots the vibration amplitude response of a system over a specified frequency range.

    Parameters
    ----------
    A1_function : callable
        A function that calculates the amplitude based on speed, some_param, and frequency.
        It should accept three arguments: speed, some_param, and frequency, and return the amplitude.
    A2_function : callable, optional
        An optional second function to calculate a second set of amplitudes. It should have the same
        signature as A1_function. Default is None.
    start_freq : float, optional
        The starting frequency of the range. Default is 0.
    end_freq : float, optional
        The ending frequency of the range. Default is 80.
    num_points : int, optional
        The number of points in the frequency range to evaluate. Default is 8001.
    speed : float, optional
        The speed parameter to pass to the amplitude functions. Default is 4000.
    some_param : float, optional
        An additional parameter to pass to the amplitude functions. Default is 1.

    Returns
    -------
    None
        This function does not return any value. It displays a plot of the vibration amplitude.

    Examples
    --------
    >>> def amplitude_function(speed, param, freq):
    >>>     # Example amplitude function
    >>>     return speed * param / (freq**2 + 1)
    >>> plot_vibration_amplitude(amplitude_function, start_freq=0, end_freq=50, speed=3000, some_param=2)
    """
    
    freq_range = np.linspace(start_freq, end_freq, num_points)
    
    valuesA1 = [abs(A1_function(speed, some_param, freq)) for freq in freq_range]
    trace0 = go.Scatter(
        x=freq_range,
        y=valuesA1,
        mode='lines',
        name=f'Excitation Frequency ({A1_function.__name__})'
    )
    
    if A2_function:
        valuesA2 = [abs(A2_function(speed, some_param, freq)) for freq in freq_range]
        trace1 = go.Scatter(
            x=freq_range,
            y=valuesA2,
            mode='lines',
            name=f'Excitation Frequency ({A2_function.__name__})'
        )

        # Create the layout
        layout = go.Layout(
            title=f'Harmonic Force Fixed in Space Response ({A1_function.__name__}, {A2_function.__name__})',
            xaxis=dict(title='Excitation Frequency'),
            yaxis=dict(title='Vibration Amplitude (m)')
        )
        # Create the figure
        fig = go.Figure(data=[trace0, trace1], layout=layout)
    else:
        # Create the layout
        layout = go.Layout(
            title=f'Harmonic Force Fixed in Space Response ({A1_function.__name__})',
            xaxis=dict(title='Excitation Frequency'),
            yaxis=dict(title='A1 Vibration Amplitude (m)')
        )

        # Create the figure
        fig = go.Figure(data=[trace0], layout=layout)
    
    fig.update_layout(
        autosize=False,
        width=500 * 1.62,
        height=500,
        legend=dict(yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01)
    )
    fig.update_yaxes(type="log")
    return fig



def interactive_orbit_fixed_speed(
    vibration_response: go.Figure,
    A1_func: Callable[..., float],
    A2_func: Callable[..., float],
    speed: float,
    *args: Any
) -> go.Figure:
    """
    Create a plot with a slider to vary frequency from 0 to 80 Hz.

    Parameters
    ----------
    vibration_response : plotly.graph_objects.Figure
        The figure containing the vibration response data.
    A1_func : Callable[..., float]
        Function to calculate A1.
    A2_func : Callable[..., float]
        Function to calculate A2.
    speed : float
        Speed to plot the interactive orbit plot.
    *args : Any
        Additional arguments, with the first argument expected to be a force in Newtons and
        second argument expected to be the initial frequency.

    Returns
    -------
    plotly.graph_objects.Figure
        The interactive plot with orbit and time plots.

    Notes
    -----
    This function defines traces for the orbit, cosine, and sine plots, creates subplots,
    and adds vibration response data. It also creates a slider to vary the frequency 
    from 0 to 80 Hz and updates the plot accordingly.
    """
    # Define the angle range to compute one revolution
    theta = np.linspace(0, 1.85 * np.pi)

    # Get initial orbit and sine data
    q1_circle, q2_circle = update_circle_and_sine(A1_func, A2_func, speed, theta, *args)

    # Define traces for the orbit and starting point
    trace_orbit = go.Scatter(
        x=q1_circle,
        y=q2_circle,
        showlegend=False,
        name='Orbit'
    )
    starting_point = go.Scatter(
        x=[q1_circle[0]],
        y=[q2_circle[0]],
        marker={'size': 10, 'color': '#636EFA'},
        showlegend=False,
        name='Starting Point'
    )

    # Define traces for the cosine and sine plots
    q1_trace = go.Scatter(
        x=theta,
        y=q1_circle,
        mode='lines',
        name='q1',
        showlegend=False
    )
    q2_trace = go.Scatter(
        x=theta,
        y=q2_circle,
        mode='lines',
        name='q2',
        showlegend=False
    )

    initial_frequency = args[1]

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}],
               [{"colspan": 2}, None]],
        vertical_spacing=0.12,
        row_heights=[0.40, 0.60],
        subplot_titles=(f"Orbit @ {speed} RPM", "q1 and q2", "Vibration Amplitude")
    )

    # Add traces to the subplots
    fig.add_trace(trace_orbit, row=1, col=1)
    fig.add_trace(starting_point, row=1, col=1)
    fig.add_trace(q1_trace, row=1, col=2)
    fig.add_trace(q2_trace, row=1, col=2)
    fig.add_annotation(x=2, y=q1_circle[17], text='q1', row=1, col=2)
    fig.add_annotation(x=4, y=q2_circle[34], text='q2', row=1, col=2)

    # Adding vibration response data
    vibration_data = vibration_response.to_dict()['data']

    for data in vibration_data:
        fig.add_trace(data, row=2, col=1)

    fig.update_layout(
        yaxis3=dict(
            title='Amplitude (m)',
            type='log',
            range=[-9, -1]  # log range: 10^-9, 10^-1
        )
    )

    # Add vertical dashed gray line
    fig.add_shape(
        type="line",
        x0=initial_frequency,
        x1=initial_frequency,
        y0=0,
        y1=1e-1,
        line=dict(color="gray", width=2, dash="dot"),
        row=2,
        col=1,
        name='Vertical Line'
    )

    max_amplitude = max(np.max(np.abs(q1_circle)), np.max(np.abs(q2_circle)))

    # Set fixed axis ranges
    fig.update_xaxes(title='q1 (m)', range=[-1.2 * max_amplitude, 1.2 * max_amplitude], row=1, col=1)
    fig.update_yaxes(title='q2 (m)', range=[-1.2 * max_amplitude, 1.2 * max_amplitude], row=1, col=1)
    fig.update_xaxes(title='omega * t (Rad)', range=[0, 1.85 * np.pi], row=1, col=2)
    fig.update_yaxes(range=[-1.2 * max_amplitude, 1.2 * max_amplitude], row=1, col=2)

    fig.update_layout({'title': 'Rotor Analysis - Harmonic Force Fixed in Space'})

    # Create slider steps
    steps = []
    for frequency in range(0, 81, 1):
        layout_dict = fig.to_dict()['layout']

        q1_circle, q2_circle = update_circle_and_sine(A1_func, A2_func, speed, theta, args[0], frequency)

        # Adjusting the y and x axis scales for orbit and timebase plot
        max_amplitude = max(np.max(np.abs(q1_circle)), np.max(np.abs(q2_circle)))
        yaxis = layout_dict['yaxis']
        yaxis['range'] = [-1.2 * max_amplitude, 1.2 * max_amplitude]
        xaxis = layout_dict['xaxis']
        xaxis['range'] = [-1.2 * max_amplitude, 1.2 * max_amplitude]
        yaxis2 = layout_dict['yaxis2']
        yaxis2['range'] = [-1.2 * max_amplitude, 1.2 * max_amplitude]

        # Adjusting the annotations for q1 and q2 in time representation
        annotations = layout_dict['annotations']
        annotations[0]['text'] = f"Orbit @ {speed} RPM"
        annotations[3]['y'] = q1_circle[17]
        annotations[4]['y'] = q2_circle[34]

        # Adjusting the shapes to speed
        shapes = layout_dict['shapes']
        shapes[0]['x0'] = frequency
        shapes[0]['x1'] = frequency

        step = dict(
            method='update',
            args=[{'x': [q1_circle, [q1_circle[0]], theta, theta,
                         vibration_data[0]['x'], vibration_data[1]['x']],
                   'y': [q2_circle, [q2_circle[0]], q1_circle, q2_circle,
                         vibration_data[0]['y'], vibration_data[1]['y']]},
                  {'annotations': annotations,
                   'yaxis': yaxis,
                   'xaxis': xaxis,
                   'yaxis2': yaxis2,
                   'shapes': shapes}],
            label=f'{frequency}'
        )
        steps.append(step)

    # Define the slider
    active_slider = initial_frequency

    sliders = [dict(
        active=active_slider,
        currentvalue={"prefix": "Excitation frequency: ", "suffix": " Hz"},
        pad={"t": 15},
        steps=steps,
        len=.96,
    )]

    # Update the layout with the slider
    fig.update_layout(
        legend=dict(yanchor="top", y=0.51, xanchor="left", x=0.01),
        autosize=False,
        width=750,
        height=800,
        sliders=sliders
    )

    return fig


def save_orbit_frames(
    vibration_response: go.Figure,
    A1_func: Callable[..., float],
    A2_func: Callable[..., float],
    speed: float,
    *args: Any,
    output_dir='frames'
) -> go.Figure:
    """
    Generate and save frames for each frequency from 0 to 80 Hz.

    Parameters
    ----------
    vibration_response : go.Figure
        A Plotly figure object containing the vibration response data.
    A1_func : Callable[..., float]
        A function to calculate the amplitude A1.
    A2_func : Callable[..., float]
        A function to calculate the amplitude A2.
    speed : float
        The speed of the rotor in RPM.
    *args : Any
        Additional arguments to pass to the functions A1_func and A2_func.
    output_dir : str, optional
        The directory where the frames will be saved (default is 'frames').

    Returns
    -------
    go.Figure
        The Plotly figure object with the vibration response data.
    
    Notes
    -----
    This function generates frames for frequencies ranging from 0 to 80 Hz,
    in steps of 0.1 Hz. For each frequency, it calculates the orbit and sine
    data, plots the orbit, time response, and vibration amplitude, and saves
    the plot as an image file.
    
    Examples
    --------
    >>> import plotly.graph_objs as go
    >>> fig = go.Figure()
    >>> def A1_func(*args): return 1.0
    >>> def A2_func(*args): return 1.0
    >>> save_orbit_frames(fig, A1_func, A2_func, 3000)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define the angle range
    theta = np.linspace(0, 1.85 * np.pi, 1000)

    # Adding vibration response data
    vibration_data = vibration_response.to_dict()['data']

    F0 = args[0]
    for f in range(1, 801, 1):  # Frequencies from 0 to 80 Hz
        # Get circle (orbit) and sine data for the current frequency
        q1_circle, q2_circle = update_circle_and_sine(A1_func, A2_func, speed, theta, F0, f/10)

        # Create the figure and axis objects
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 2)
        ax_orbit = fig.add_subplot(gs[0, 0])
        ax_time = fig.add_subplot(gs[0, 1], sharey=ax_orbit)
        ax_vibration = fig.add_subplot(gs[1, :])

        # Plot orbit circle data
        ax_orbit.set_title(f'Orbit @ {speed} RPM')
        orbit_line, = ax_orbit.plot(q1_circle, q2_circle, label='Orbit')
        ax_orbit.plot(q1_circle[0], q2_circle[0], 'o', markersize=10, color='#636EFA', label='Starting Point')
        ax_orbit.set_xlabel('q1 (m)')
        ax_orbit.set_ylabel('q2 (m)')

        # Plot sine data
        ax_time.set_title('q1 and q2')
        q1_line, = ax_time.plot(theta, q1_circle, label='q1')
        q2_line, = ax_time.plot(theta, q2_circle, label='q2')
        ax_time.legend()
        ax_time.set_xlabel('omega * t (Rad)')
        # ax_time.set_ylabel('Amplitude (m)')

        # Vibration amplitude
        ax_vibration.set_title('Vibration Amplitude')
        ax_vibration.plot(vibration_data[0]['x'], vibration_data[0]['y'], label='Vibration A1')
        ax_vibration.plot(vibration_data[1]['x'], vibration_data[1]['y'], label='Vibration A2')
        ax_vibration.plot([f/10, f/10], [1e-9, 1e-1], '--', label='Excitation\nFrequency')
        # vibration_line, = ax_vibration.plot([initial_frequency, initial_frequency], [0, max(vibration_response[1])], 'k--', label='Frequency Line')
        ax_vibration.set_xlabel('Frequency (Hz)')
        ax_vibration.set_ylabel('Amplitude (m)')
        ax_vibration.set_yscale('log')
        ax_vibration.set_ylim(1e-9, 1e-1)   # log range: 10^-9, 10^-1
        ax_vibration.set_xlim(0, 80)

        # Calculate the limits for q1 and q2 to maintain the aspect ratio
        q1_min, q1_max = min(q1_circle), max(q1_circle)
        q2_min, q2_max = min(q2_circle), max(q2_circle)
        q_min = min(q1_min, q2_min)
        q_max = max(q1_max, q2_max)
        padding = 0.1 * (q_max - q_min)

        # Set axis labels and ranges
        ax_orbit.set_xlim(q_min - padding, q_max + padding)
        ax_orbit.set_ylim(q_min - padding, q_max + padding)
        ax_orbit.set_aspect('equal', 'box')
        ax_time.set_xlim(0, 1.85 * np.pi)
        ax_time.set_ylim(q_min - padding, q_max + padding)

        ax_vibration.legend()

        # Save the frame
        frame_filename = os.path.join(output_dir, f'frame_{f:03d}.png')
        plt.suptitle(f'Rotor Analysis - Harmonic Force Fixed in Space\nExcitation Frequency: {f/10} Hz')
        plt.savefig(frame_filename)
        plt.close(fig)

    print(f"Frames saved in directory: {output_dir}")


def create_video_from_frames(image_folder: str, video_name: str, frame_rate: int = 20, codec: str = 'DIVX') -> None:
    """
    Create a video from the saved images in the specified folder.

    Parameters
    ----------
    image_folder : str
        The folder containing the images to be used for the video.
    video_name : str
        The name of the output video file.
    frame_rate : int, optional
        The frame rate of the output video (default is 20).
    codec : str, optional
        The codec to be used for the video (default is 'DIVX').

    Returns
    -------
    None

    Examples
    --------
    >>> create_video_from_frames('frames', 'charts_video.avi')
    """
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    if not images:
        raise ValueError("No images found in the specified folder")

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*codec), frame_rate, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    
    
    
def FFT(signal1, signal2, signal_length):
    """
    Compute the Fourier Transform of two signals and plot the results.

    Parameters
    ----------
    signal1 : np.ndarray
        First input signal.
    signal2 : np.ndarray
        Second input signal.
    signal_length : float
        Duration of the signals in seconds.

    Returns
    -------
    None
    
    Remarks:
    --------
    Sampling Rate (Sampling Frequency): This is the rate at which the continuous
    signal is sampled to produce the discrete signal. It is usually denoted by fs
    and measured in samples per second or Hertz (Hz). For example, if a signal is 
    sampled 1000 times per second, the sampling rate is 1000 Hz.
    """
    samples = len(signal1)
    fs = samples / signal_length
    t = np.linspace(0, signal_length, samples)
    
    FT1 = fft(signal1)
    freqs1 = fftfreq(len(signal1), 1/fs)

    FT2 = fft(signal2)
    freqs2 = fftfreq(len(signal2), 1/fs)
    
    # Plot the original signals
    plt.figure(figsize=(14, 6))
    
    plt.subplot(2, 2, 1)
    plt.plot(t, signal1)
    plt.title("Original Signal 1")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.subplot(2, 2, 3)
    plt.plot(t, signal2)
    plt.title("Original Signal 2")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    
    # Plot the magnitude of the Fourier Transforms
    plt.subplot(2, 2, 2)
    plt.plot(freqs1, np.abs(FT1))
    plt.title("Fourier Transform (FFT1)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.yscale('log')
    
    plt.subplot(2, 2, 4)
    plt.plot(freqs2, np.abs(FT2))
    plt.title("Fourier Transform (FFT2)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.yscale('log')
    
    print(f'Samples:\t\t{samples}')
    print(f'Duration:\t\t{signal_length} s')
    print(f'Sampling Frequency:\t{fs} Hz')
    
    # Display the plots
    plt.tight_layout()
    plt.show()