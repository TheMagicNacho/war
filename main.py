from uuid import uuid4 as uuid
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np

"""
Within the manifold all life is an entity. And all entities possess energy.
The entity only stores the mass, but the energy is derived from the mass.
Setting the radius declares the size of the rigid body. We can then calculate the volume of the entity, and the density of the entity.
An entity exists, it does not perseve nor does it move.
The mind exists as the super ego.
"""
class Entity:
    def __init__(self, id: uuid, common_name: str, mass: float, radius: float, properties: dict):
        self.id = id
        self.common_name = common_name
        self.mass = mass
        self.radius = radius
        self.properties = properties

"""
The manifold. The fabric of the universe.
The manifold is a closed space, meaning that the entities cannot leave the manifold.
The dimensionality of the manifold is defined on instanciation, and is always a unit sphere where each dimension is orthogonal.

dimensions: int # The number of dimensions of the manifold.
radius: float # the radius of a dimension. The implication is then that the manifold is a unit sphere.
"""
class Mainfold:
    def __init__(self, dimensions: int, radius: float = 1.0):
        self.dimensions: int = dimensions
        self.entities: dict = {}
        self.radius: int = radius

    """
    Check if the position is valid for the manifold.
    return True if the position is valid, False otherwise.
    """
    def is_position_valid(self, position: tuple) -> bool:
        if len(position) != self.dimensions:
            return False
        for i in range(self.dimensions):
            if position[i] < 0 or position[i] > self.radius:
                return False
        return True

    """
    Add an entity to the manifold.
    position: tuple # The position of the entity in the manifold.
    entity: Entity # The entity to add.
    return True if the entity was added, False otherwise.
    """
    def add_entity(self, entity: Entity, position: tuple) -> bool:
        if not self.is_position_valid(position):
            print(f"Position {position} is out of bounds for manifold of radius {self.radius}")
            return False
        if entity.id in self.entities:
            print(f"Entity with id {entity.id} already exists in the manifold.")
            return False
        self.entities[entity.id] = (entity, position)
        print(f"Entity {entity.common_name} added at position {position}")
        return True

    def remove_entity(self, entity_id: str):
        if entity_id not in self.entities:
            print(f"Entity with id {entity_id} does not exist in the manifold")
            return False
        try:
            del self.entities[entity_id]
            print(f"Entity with id {entity_id} removed from the manifold")
            return True
        except KeyError:
            print(f"Error removing entity with id {entity_id}")
            return False



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
        """
        Visualize a high-dimensional manifold (n > 3) using stereographic projection
        and interactive sliders for quaternion-based rotation angles (wx, wy, wz).
        Also draws lines along each axis to help visualize the boundary of the hypersphere.
        """
        import plotly.graph_objects as go
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        positions = np.array([pos for _, pos in manifold.entities.values()])
        colors = [entity.properties.get("color", "gray") for entity, _ in manifold.entities.values()]

        n = manifold.dimensions
        if n <= 3 or positions.shape[0] == 0:
            print("Hypersphere visualization requires more than 3 dimensions and at least one entity.")
            return

        # Pad positions to 4D if needed
        if positions.shape[1] < 4:
            positions = np.pad(positions, ((0, 0), (0, 4 - positions.shape[1])), 'constant')
        elif positions.shape[1] > 4:
            from sklearn.decomposition import PCA
            positions = PCA(n_components=4).fit_transform(positions)

        # Stereographic projection from 4D to 3D using quaternion rotation
        def stereographic_project(points, wx=0, wy=0, wz=0):
            rot = R.from_euler('xyz', [wx, wy, wz])
            pts_rot = points.copy()
            pts_rot[:, :3] = rot.apply(pts_rot[:, :3])
            w = pts_rot[:, 3:4]
            denom = 1 - w / manifold.radius
            denom[denom == 0] = 1e-8
            projected = manifold.radius * pts_rot[:, :3] / denom
            return projected

        # Generate axis lines in 4D (from -radius to +radius along each axis)
        axis_lines_4d = []
        for axis in range(4):
            for sign in [-1, 1]:
                pt1 = np.zeros(4)
                pt2 = np.zeros(4)
                pt1[axis] = -manifold.radius
                pt2[axis] = manifold.radius
                # For each axis, draw a line from -radius to +radius, all other coords 0
                axis_lines_4d.append((pt1, pt2))

        # Project axis lines to 3D for initial frame
        axis_lines_3d = []
        for pt1, pt2 in axis_lines_4d:
            proj_line = stereographic_project(np.stack([pt1, pt2]), 0, 0, 0)
            axis_lines_3d.append(proj_line)

        # Slider values
        slider_steps = 20
        angles = np.linspace(0, 2 * np.pi, slider_steps)

        # Precompute all frames for animation (for each combination of wx, wy, wz)
        frames = []
        for i, wx in enumerate(angles):
            for j, wy in enumerate(angles):
                for k, wz in enumerate(angles):
                    if i == 0 and j == 0 and k == 0:
                        continue  # skip initial frame
                    proj = stereographic_project(positions, wx, wy, wz)
                    # Project axis lines for this rotation
                    axis_lines_proj = []
                    for pt1, pt2 in axis_lines_4d:
                        axis_lines_proj.append(stereographic_project(np.stack([pt1, pt2]), wx, wy, wz))
                    frame_data = [
                        go.Scatter3d(
                            x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
                            mode='markers',
                            marker=dict(size=7, color=colors),
                            name="Entities"
                        )
                    ]
                    # Add axis lines to frame
                    for idx, line in enumerate(axis_lines_proj):
                        frame_data.append(
                            go.Scatter3d(
                                x=line[:, 0], y=line[:, 1], z=line[:, 2],
                                mode='lines',
                                line=dict(color='black', width=2, dash='dash'),
                                name=f"Axis {idx//2 + 1}" if idx % 2 == 0 else None,
                                showlegend=(idx % 2 == 0)
                            )
                        )
                    frames.append(go.Frame(
                        data=frame_data,
                        name=f"wx={wx:.2f},wy={wy:.2f},wz={wz:.2f}"
                    ))

        # Initial projection
        proj = stereographic_project(positions, 0, 0, 0)

        # Prepare initial axis lines
        axis_lines_traces = []
        for idx, line in enumerate(axis_lines_3d):
            axis_lines_traces.append(
                go.Scatter3d(
                    x=line[:, 0], y=line[:, 1], z=line[:, 2],
                    mode='lines',
                    line=dict(color='black', width=2, dash='dash'),
                    name=f"Axis {idx//2 + 1}" if idx % 2 == 0 else None,
                    showlegend=(idx % 2 == 0)
                )
            )

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
                    mode='markers',
                    marker=dict(size=7, color=colors),
                    name="Entities"
                )
            ] + axis_lines_traces,
            layout=go.Layout(
                title=f"{n}D Manifold Stereographic Projection (Quaternion Rotation)",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode='data'
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        buttons=[
                            dict(label="Play", method="animate", args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]),
                            dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                        ]
                    )
                ]
            ),
            frames=frames
        )

        # Add sliders for wx, wy, wz
        sliders = [
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [f"wx={wx:.2f},wy={0:.2f},wz={0:.2f}"],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}
                        ],
                        label=f"{wx:.2f}"
                    ) for wx in angles
                ],
                active=0,
                currentvalue={"prefix": "wx: "}
            ),
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [f"wx={0:.2f},wy={wy:.2f},wz={0:.2f}"],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}
                        ],
                        label=f"{wy:.2f}"
                    ) for wy in angles
                ],
                active=0,
                currentvalue={"prefix": "wy: "}
            ),
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [f"wx={0:.2f},wy={0:.2f},wz={wz:.2f}"],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}
                        ],
                        label=f"{wz:.2f}"
                    ) for wz in angles
                ],
                active=0,
                currentvalue={"prefix": "wz: "}
            )
        ]
        fig.update_layout(sliders=sliders)
        fig.show()

    def visualize_manifold(self, manifold: Mainfold):
        if manifold.dimensions == 2:
            self.plot_2d(manifold)
        elif manifold.dimensions == 3:
            self.plot_3d(manifold)
        else:
            self.plot_hypersphere(manifold)


### Example usage
# manifold = Mainfold(dimensions=5, size=1.0)
hv = HumanVision();

e1 = Entity(id=uuid(), common_name="Entity 1", mass=10.0, radius=5.0, properties={"color": "red"})
e2 = Entity(id=uuid(), common_name="Entity 2", mass=20.0, radius=10.0, properties={"color": "blue"})


## # 2D Example
# m2 = Mainfold(dimensions=2, radius=1.0)
# m2.add_entity(e1, (0.5, 0.6))
# m2.add_entity(e2, (0.0, 0.0))

# hv.visualize_manifold(m2)



# ## # 3D Example
# m3 = Mainfold(dimensions=3, radius=1.0)
# m3.add_entity(e1, (0.5, 0.6, 0.7))
# m3.add_entity(e2, (0.0, 0.0, 0.0))
# hv.visualize_manifold(m3)


## # 4D Example
m4 = Mainfold(dimensions=4, radius=1.0)
m4.add_entity(e1, (0.5, 0.6, 0.7, 0.8))
m4.add_entity(e2, (0.0, 0.0, 0.0, 0.0))
hv.visualize_manifold(m4)

