class Arena:
    def __init__(
        self,
        spawnCoords: list,
        status: str,
        name: str,
        map_name: str = "classic",
        bounds: list = None,
        goal_zones: list = None,
        reset_coords: list = None,
    ):
        self.spawnCoords = []
        for i in range(len(spawnCoords)):
            self.spawnCoords.append(spawnCoords[i])
        self.status = status
        self.name = name
        self.map_name = map_name
        self.bounds = bounds or []
        self.goal_zones = goal_zones or []
        self.reset_coords = reset_coords or []
