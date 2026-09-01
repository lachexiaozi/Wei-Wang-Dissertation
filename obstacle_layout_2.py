"""Obstacle layout 2 configuration.

This initial version intentionally matches obstacle_layout_1. Its coordinates
and dimensions can be changed independently when layout 2 is designed.
"""

LAYOUT_ID = 2
LAYOUT_NAME = "Layout 2"

RIGHT_LANE_Y = 2.415
CENTRE_LANE_Y = 6.2
LEFT_LANE_Y = 10.415

# (result dictionary key, actor number, location, yaw in degrees)
QCAR_OBSTACLES = (
    ("Qcar2", 2, [35, RIGHT_LANE_Y, 2], 0.0),
    ("Qcar3", 3, [45, CENTRE_LANE_Y, 2], 0.0),
    ("Qcar4", 4, [55, LEFT_LANE_Y, 2], 0.0),
)

CUBE_ROTATION = [0, 0, 0]
# (actor number, centre location, x/y/z scale)
CUBE_OBSTACLES = (
    (0, [30, 2.415, 1.5], [1, 1, 2]),
    (1, [41, 7.7, 1.5], [1, 1, 4]),
)

# (location, rotation in degrees, scale, configuration)
TRAFFIC_CONE_OBSTACLES = (
    ([40, 3, 2], [0, 0, 180], [1, 1, 1], 1),
    ([50, 5, 2], [0, 0, 180], [1, 1, 1], 1),
)


def main():
    """Spawn layout 2 directly for debugging."""
    from qcar2 import run_obstacle_layout

    run_obstacle_layout(LAYOUT_ID)


if __name__ == "__main__":
    main()
