
"""
The manifold. The fabric of the universe.
The manifold is a closed space, meaning that the entities cannot leave the manifold.
The dimensionality of the manifold is defined on instanciation, and is always a unit sphere where each dimension is orthogonal.

dimensions: int # The number of dimensions of the manifold.
radius: float # the radius of a dimension. The implication is then that the manifold is a unit sphere.
"""
from entity import Entity


class Mainfold:
    def __init__(self, dimensions: int, radius: float = 1.0):
        if dimensions < 1:
            raise ValueError("Dimensions must be greater than 0")
        if dimensions > 4:
            raise ValueError("Dimensions must be less than or equal to 4")
        

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
        if len(position) != self.dimensions:
            raise ValueError(f"Position must be of length {self.dimensions}")


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

