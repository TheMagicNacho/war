from uuid import uuid4 as uuid
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import numpy as np

from entity import Entity
from human_vision import HumanVision
from manifold import Mainfold





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

