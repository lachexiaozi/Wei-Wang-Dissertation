"""Obstacle layout 3 configuration.

This initial version intentionally matches obstacle_layout_1. Its coordinates
and dimensions can be changed independently when layout 3 is designed.
"""

LAYOUT_ID = 3
LAYOUT_NAME = "Layout 3"

RIGHT_LANE_Y = 2.415
CENTRE_LANE_Y = 6.2
LEFT_LANE_Y = 10.415

# (result dictionary key, actor number, location, yaw in degrees)
QCAR_OBSTACLES = (
    ("Qcar2", 2, [35, CENTRE_LANE_Y, 2], 0.0),
    ("Qcar3", 3, [45, LEFT_LANE_Y, 2], 0.0),
    ("Qcar4", 4, [55, RIGHT_LANE_Y, 2], 0.0),
)

CUBE_ROTATION = [0, 0, 0]
# (actor number, centre location, x/y/z scale)
CUBE_OBSTACLES = (
    (0, [50, 9.415, 1.5], [1, 1, 2]),
    (1, [40, 6.2, 1.5], [1, 1, 4]),
)

# (location, rotation in degrees, scale, configuration)
TRAFFIC_CONE_OBSTACLES = (
    ([33.788, 7.2, 2], [0, 0, 180], [1, 1, 1], 1),
    ([50, 2.415, 2], [0, 0, 180], [1, 1, 1], 1),
)


def main():
    """Spawn layout 3 directly for debugging."""
    from qcar2 import run_obstacle_layout

    run_obstacle_layout(LAYOUT_ID)


if __name__ == "__main__":
    main()
