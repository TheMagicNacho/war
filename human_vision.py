

import numpy as np
from manifold import Mainfold
import plotly.graph_objects as go
from plotly.subplots import make_subplots

"""
Humans, in their hubris, wish to gaze upon the manifold and see the entities within.
But humans are weak, limited and foolish.
This class takes the manifold and allows humans to see within.
"""
class HumanVision:
    def plot_2d(self, manifold: Mainfold):
        # Extract positions and colors from entities
        positions = np.array([pos for _, pos in manifold.entities.values()])
        colors = [entity.properties.get("color", "gray") for entity, _ in manifold.entities.values()]

        # Create a scatter plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode='markers',
            marker=dict(size=10, color=colors),
            name="Entities"
        ))

        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = manifold.radius * np.cos(theta)
        circle_y = manifold.radius * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            line=dict(color='black', dash='dash'),
            name="Manifold Boundary"
        ))

        fig.update_layout(
            title="2D Manifold Visualization",
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=True
        )
        fig.show()

    def plot_3d(self, manifold: Mainfold):
        # Extract positions and colors from entities
        positions = np.array([pos for _, pos in manifold.entities.values()])
        colors = [entity.properties.get("color", "gray") for entity, _ in manifold.entities.values()]

        # Create a 3D scatter plot
        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(size=5, color=colors),
            name="Entities"
        ))
        # Create a sphere to represent the manifold boundary
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = manifold.radius * np.outer(np.cos(u), np.sin(v))
        y = manifold.radius * np.outer(np.sin(u), np.sin(v))
        z = manifold.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        fig.add_trace(go.Surface(
            x=x,
            y=y,
            z=z,
            opacity=0.3,
            colorscale='Viridis',
            showscale=False,
            name="Manifold Boundary"
        ))
        fig.update_layout(
            title="3D Manifold Visualization",
            scene=dict(
                xaxis_title="X-axis",
                yaxis_title="Y-axis",
                zaxis_title="Z-axis",
                aspectmode='data'
            ),
            showlegend=True
        )
        fig.show()

    def plot_hypersphere(self, manifold: Mainfold):
        # Extract entity positions and properties
        positions = np.array([pos for _, pos in manifold.entities.values()])
        colors = [entity.properties.get("color", "gray") for entity, _ in manifold.entities.values()]
        entity_names = [entity.common_name for entity, _ in manifold.entities.values()]
        
        # Function for stereographic projection from 4D to 3D
        def stereographic_projection(points_4d, angle):
            """Project 4D points onto 3D using stereographic projection with 4D rotation"""
            rotated_points = np.zeros_like(points_4d)
            for i, point in enumerate(points_4d):
                x, y, z, w = point
                # Apply rotation in xw-plane (one of many possible 4D rotations)
                rotated_points[i, 0] = x * np.cos(angle) - w * np.sin(angle)
                rotated_points[i, 1] = y
                rotated_points[i, 2] = z
                rotated_points[i, 3] = x * np.sin(angle) + w * np.cos(angle)
            
            # Stereographic projection from north pole (0,0,0,1)
            projected_points = np.zeros((len(rotated_points), 3))
            for i, point in enumerate(rotated_points):
                x, y, z, w = point
                # Avoid division by zero when w=1
                denom = 1.0 - w
                if abs(denom) < 1e-10:
                    # Point very close to north pole projects to "infinity"
                    # Use a large but finite value
                    factor = 1e10
                else:
                    factor = 1.0 / denom
                
                projected_points[i] = [x * factor, y * factor, z * factor]
            
            return projected_points
        
        # Generate points on the hypersphere surface (3-sphere)
        # We'll use a parametrization that gives a good distribution of points
        points_per_dim = 15  # Reduce for better performance
        theta = np.linspace(0, np.pi, points_per_dim)
        phi = np.linspace(0, 2*np.pi, points_per_dim)
        chi = np.linspace(0, np.pi, points_per_dim)
        
        # Generate a subset of points on the hypersphere for visualization
        hypersphere_points = []
        for t in theta[::3]:  # Use stride to reduce point count
            for p in phi[::3]:
                for c in chi[::3]:
                    x = manifold.radius * np.sin(c) * np.sin(p) * np.sin(t)
                    y = manifold.radius * np.sin(c) * np.sin(p) * np.cos(t)
                    z = manifold.radius * np.sin(c) * np.cos(p)
                    w = manifold.radius * np.cos(c)
                    hypersphere_points.append([x, y, z, w])
        
        hypersphere_points = np.array(hypersphere_points)
        
        # Create frames for animation
        frames = []
        angles = np.linspace(0, 2*np.pi, 40)  # 40 frames for one full rotation
        
        for i, angle in enumerate(angles):
            projected_entities = stereographic_projection(positions, angle)
            projected_sphere = stereographic_projection(hypersphere_points, angle)
            
            frames.append(
                go.Frame(
                    data=[
                        # Entity points
                        go.Scatter3d(
                            x=projected_entities[:, 0], 
                            y=projected_entities[:, 1], 
                            z=projected_entities[:, 2],
                            mode='markers+text',
                            marker=dict(size=10, color=colors),
                            text=entity_names,
                            name="Entities",
                            hoverinfo="text"
                        ),
                        # Hypersphere surface
                        go.Scatter3d(
                            x=projected_sphere[:, 0],
                            y=projected_sphere[:, 1],
                            z=projected_sphere[:, 2],
                            mode='markers',
                            marker=dict(size=3, color='lightblue', opacity=0.3),
                            name="Hypersphere",
                            hoverinfo="none"
                        )
                    ],
                    name=f"frame{i}"
                )
            )
        
        # Initial projection (angle=0)
        initial_projected_entities = stereographic_projection(positions, 0)
        initial_projected_sphere = stereographic_projection(hypersphere_points, 0)
        
        # Create figure
        fig = go.Figure(
            data=[
                # Entity points
                go.Scatter3d(
                    x=initial_projected_entities[:, 0],
                    y=initial_projected_entities[:, 1], 
                    z=initial_projected_entities[:, 2],
                    mode='markers+text',
                    marker=dict(size=10, color=colors),
                    text=entity_names,
                    name="Entities",
                    hoverinfo="text"
                ),
                # Hypersphere surface
                go.Scatter3d(
                    x=initial_projected_sphere[:, 0],
                    y=initial_projected_sphere[:, 1],
                    z=initial_projected_sphere[:, 2],
                    mode='markers',
                    marker=dict(size=3, color='lightblue', opacity=0.3),
                    name="Hypersphere",
                    hoverinfo="none"
                )
            ],
            frames=frames
        )
        
        fig.update_layout(
            title="4D Hypersphere (Stereographic Projection)",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='data'
            ),
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 10},
                'x': 0.1,
                'y': 0
            }],
            sliders=[{
                'steps': [
                    {
                        'method': 'animate',
                        'label': f'{angle:.2f}',
                        'args': [
                            [f'frame{i}'],
                            {'frame': {'duration': 100, 'redraw': True}, 'mode': 'immediate'}
                        ]
                    }
                    for i, angle in enumerate(angles)
                ],
                'active': 0,
                'currentvalue': {
                    'prefix': 'Angle: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 300},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0
            }]
        )
        
        fig.show()

    def visualize_manifold(self, manifold: Mainfold):
        if manifold.dimensions == 2:
            self.plot_2d(manifold)
        elif manifold.dimensions == 3:
            self.plot_3d(manifold)
        else:
            self.plot_hypersphere(manifold)
