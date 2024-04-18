# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:57:45 2024

@author: EMKA
"""
import plotly.graph_objects as go
import numpy as np

def plot_vectors(*vectors, origin = (0,0,0)):
    """
    Plot 3D vectors.

    Parameters:
        *vectors: Variable number of numpy arrays representing vectors.
        origin: Tuple representing the origin point for the vectors.
    """
    data = []
    colors = ['rgb(255,0,0)',
              'rgb(0,255,0)',
              'rgb(0,0,255)']
    
    for i, vector in enumerate(vectors):
        # Add a line trace for the vector
        data.append(
            go.Scatter3d(x=[origin[0], vector.item(0)],
                         y=[origin[1], vector.item(1)],
                         z=[origin[2], vector.item(2)],
                         mode='lines',
                         marker={'color': colors[i]},
                         line={'dash': 'solid',
                               'color': colors[i], # Use predefined color for the vector
                               'width': 5},
                         name="v%s" % i)
        )
        # Add a cone at the end of the vector to represent direction
        data.append(
            go.Cone(x=[vector.item(0)],
                    y=[vector.item(1)],
                    z=[vector.item(2)],
                    u=[vector.item(0)-0.1*vector.item(0)], # Compute cone direction
                    v=[vector.item(1)-0.1*vector.item(1)],
                    w=[vector.item(2)-0.1*vector.item(2)],
                    anchor="tail",
                    colorscale=[[0, colors[i]], [1, colors[i]]], # Use same color as the vector
                    showscale=False)
        )

    # Create figure
    fig = go.Figure(data=data)

    scene=dict(camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)), #the default values are 1.25, 1.25, 1.25
           xaxis=dict(),
           yaxis=dict(),
           zaxis=dict(),
           aspectmode="manual", #this string can be 'data', 'cube', 'auto', 'manual'
           #a custom aspectratio is defined as follows:
           aspectratio=dict(x=1, y=1, z=1)
           )

    fig.update_scenes(scene)

    fig.show()
