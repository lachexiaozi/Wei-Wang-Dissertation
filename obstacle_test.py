"""Create the six-QCar bearing-calibration scene directly in QLabs.

QCar1 is the stationary ego/sensor vehicle. QCar2-QCar6 are stationary
obstacles. This module does not depend on the experiment layouts in qcar2.py.
"""

from __future__ import annotations

from qvl.qcar2 import QLabsQCar2
from qvl.qlabs import QuanserInteractiveLabs




LAYOUT_ID = 4
LAYOUT_NAME = "Layout Test"

# (result dictionary key, actor number, location, yaw in degrees)
QCAR_SCENE = (
    ("Qcar1", 1, [20, 6.2, 2], 0.0),
    ("Qcar2", 2, [35, 2.415, 2], 0.0),
    ("Qcar3", 3, [35, 4.1, 2], 0.0),
    ("Qcar4", 4, [35, 6.2, 2], 0.0),
    ("Qcar5", 5, [35, 8.3, 2], 0.0),
    ("Qcar6", 6, [35, 10.415, 2], 0.0),
)

# Obstacles-only view retained for other possible scene consumers.
QCAR_OBSTACLES = QCAR_SCENE[1:]


def _spawn_stationary_qcar(
    qlabs: QuanserInteractiveLabs,
    actor_number: int,
    location: list[float],
    yaw_degrees: float,
) -> QLabsQCar2:
    qcar = QLabsQCar2(qlabs)
    status = qcar.spawn_id_degrees(
        actorNumber=actor_number,
        location=location,
        rotation=[0, 0, yaw_degrees],
        waitForConfirmation=True,
    )
    if status != 0:
        raise RuntimeError(
            f"Unable to spawn QCar actor {actor_number}; QLabs status={status}"
        )

    qcar.set_led_strip_uniform(color=[1, 0, 0])
    qcar.set_velocity_and_request_state_degrees(
        forward=0,
        turn=0,
        headlights=False,
        leftTurnSignal=False,
        rightTurnSignal=False,
        brakeSignal=True,
        reverseSignal=False,
    )
    return qcar


def spawn_obstacle_test_scene(
    qlabs: QuanserInteractiveLabs,
    destroy_existing: bool = True,
) -> dict[str, QLabsQCar2]:
    """Spawn QCar1-QCar4 and return their live QLabs handles."""
    if qlabs is None:
        raise ValueError("An open QLabs connection is required")
    if destroy_existing:
        qlabs.destroy_all_spawned_actors()

    qcars: dict[str, QLabsQCar2] = {}
    for name, actor_number, location, yaw_degrees in QCAR_SCENE:
        qcars[name] = _spawn_stationary_qcar(
            qlabs,
            actor_number,
            location,
            yaw_degrees,
        )

    print(
        f"Spawned {LAYOUT_NAME}: QCar1 ego vehicle and "
        f"{len(QCAR_OBSTACLES)} obstacle QCars."
    )
    return qcars


def main() -> int:
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    if not qlabs.open("localhost"):
        print("Unable to connect to QLabs")
        return 1

    try:
        spawn_obstacle_test_scene(qlabs)
    finally:
        qlabs.close()
    print("Obstacle test scene is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
