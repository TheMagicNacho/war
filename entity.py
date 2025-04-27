from uuid import uuid4 as uuid

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