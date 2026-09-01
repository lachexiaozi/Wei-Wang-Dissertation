"""Layout 1: near, middle, and far static obstacle groups.

The ego QCar is spawned separately at [20, 6.2, 2]. Looking forward from the
ego vehicle, increasing y moves obstacles towards the left side of the road
and increasing x moves them farther away.
"""

LAYOUT_ID = 1
LAYOUT_NAME = "Layout 1"

RIGHT_LANE_Y = 2.415
CENTRE_LANE_Y = 6.2
LEFT_LANE_Y = 10.415

# (result dictionary key, actor number, location, yaw in degrees)
QCAR_OBSTACLES = (
    ("Qcar2", 2, [35, LEFT_LANE_Y, 2], 0.0),
    ("Qcar3", 3, [45, RIGHT_LANE_Y, 2], 0.0),
    ("Qcar4", 4, [55, CENTRE_LANE_Y, 2], 0.0),
)

CUBE_ROTATION = [0, 0, 0]
# (actor number, centre location, x/y/z scale)
CUBE_OBSTACLES = (
    # Far group: tall cube on the right, intersecting the LiDAR plane.
    (1, [40.5, LEFT_LANE_Y, 1.5], [1, 1, 4]),
    (2, [40.5, RIGHT_LANE_Y, 1.5], [1, 1, 4]),
)

def main():
    from qcar2 import run_obstacle_layout

    run_obstacle_layout(LAYOUT_ID)


if __name__ == "__main__":
    main()
