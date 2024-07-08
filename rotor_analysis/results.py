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

import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets

from plotly.subplots import make_subplots
from typing import Any, Callable, Dict, Tuple
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
    fig.show()

    # Define a function to update the output based on the slider value
    def update_output(index: int) -> None:
        print(f"Axial load comparison {data0['Speed'][index]:g} rpm")
        print(f"-----------------------------")
        print(f"No load:\t{data0['Forward'][index]:.3f} Hz")
        print(f"Positive:\t{data1['Forward'][index]:.3f} Hz")
        print(f"Negative:\t{data2['Forward'][index]:.3f} Hz")

    # Create a slider widget
    index_slider = widgets.IntSlider(
        value=33, min=0, max=len(data0["Speed"]) - 1, step=1, description="Sample"
    )

    # Define an interactive output
    output = widgets.interactive_output(update_output, {"index": index_slider})

    # Display the slider and the output
    display(index_slider, output)


def add_secondary_yaxis(
    fig: go.Figure, values: np.ndarray, yaxis: str = "y2", overlaying_axis: str = "y"
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
            title="Amplitude (m)", overlaying=overlaying_axis, type="log", side="right"
        ),
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        title="Campbell Diagram + Unbalance Response",
    )

    # Add a trace for the secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=campbell_data[0]["x"], y=values, name="Vibration Amplitude", yaxis=yaxis
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
                         campbell[4]['x'], campbell[5]['x']],
                   'y': [q2_circle, [q2_circle[0]], q1_circle, q2_circle,
                         campbell[0]['y'], campbell[1]['y'], campbell[2]['y'], campbell[3]['y'],
                         campbell[4]['y'], campbell[5]['y']]},
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
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.93),
        autosize=False,
        width=750,
        height=800,
        sliders=sliders
    )

    return fig
