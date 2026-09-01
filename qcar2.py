"""Spawn the dissertation QCar scene and record synchronized sensor data.

The public ``spawnqcar2`` function is designed to be imported by the weather
scenario scripts. Importing this module has no side effects.
"""

import time

import cv2

from qvl.basic_shape import QLabsBasicShape
from qvl.free_camera import QLabsFreeCamera
from qvl.qcar2 import QLabsQCar2
from qvl.qlabs import QuanserInteractiveLabs
from qvl.traffic_cone import QLabsTrafficCone

import obstacle_layout_1
import obstacle_layout_2
import obstacle_layout_3
from sensor_data import LidarDisplay, SensorBatchRecorder


OBSTACLE_LAYOUTS = {
    1: obstacle_layout_1,
    2: obstacle_layout_2,
    3: obstacle_layout_3,
}
EGO_QCAR_LOCATION = [20.0, 6.2, 2.0]


def select_fusion_mode():
    """Ask whether this experiment uses dynamic or fixed sensor weights."""
    print("\nAvailable fusion modes:")
    print("  1 - Dynamic Weight")
    print("  2 - Fixed Weight")

    choices = {"1": "adaptive", "2": "fixed"}
    while True:
        selection = input("Select fusion mode [1-2]: ").strip()
        if selection in choices:
            return choices[selection]
        print("Invalid selection. Please enter 1 or 2.")


def _resolve_fusion_mode(fusion_mode, prompt_for_mode):
    """Return the internal fusion mode, prompting only when requested."""
    if fusion_mode is None:
        return select_fusion_mode() if prompt_for_mode else "adaptive"

    normalized = str(fusion_mode).strip().lower()
    aliases = {
        "1": "adaptive",
        "dynamic": "adaptive",
        "dynamic weight": "adaptive",
        "adaptive": "adaptive",
        "2": "fixed",
        "fixed": "fixed",
        "fixed weight": "fixed",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "fusion_mode must be 1/'adaptive' or 2/'fixed'"
        ) from exc


def select_obstacle_layout():
    """Ask the terminal user which configured obstacle layout to run."""
    print("\nAvailable obstacle layouts:")
    for layout_number, layout in OBSTACLE_LAYOUTS.items():
        print(f"  {layout_number} - {layout.LAYOUT_NAME}")

    while True:
        selection = input("Select obstacle layout [1-3]: ").strip()
        try:
            layout_number = int(selection)
        except ValueError:
            layout_number = None

        if layout_number in OBSTACLE_LAYOUTS:
            return layout_number

        print("Invalid layout. Please enter 1, 2, or 3.")


def _resolve_obstacle_layout(layout_number, prompt_for_layout):
    """Return one layout module, prompting only when requested."""
    if layout_number is None:
        layout_number = select_obstacle_layout() if prompt_for_layout else 1

    try:
        layout_number = int(layout_number)
    except (TypeError, ValueError) as exc:
        raise ValueError("layout_number must be 1, 2, or 3") from exc

    if layout_number not in OBSTACLE_LAYOUTS:
        raise ValueError("layout_number must be 1, 2, or 3")

    return OBSTACLE_LAYOUTS[layout_number]


def _spawn_stationary_qcar(
    qlabs,
    actor_number,
    location,
    yaw_degrees=0.0,
):
    """Spawn one stationary QCar with a configurable yaw orientation."""
    qcar = QLabsQCar2(qlabs)
    qcar.spawn_id_degrees(
        actorNumber=actor_number,
        location=location,
        rotation=[0, 0, yaw_degrees],
        waitForConfirmation=True,
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


def _spawn_cube_obstacles(qlabs, layout):
    """Spawn the configured static cube obstacles after QLabs is connected."""
    cubes = []
    for actor_number, location, scale in layout.CUBE_OBSTACLES:
        cube = QLabsBasicShape(qlabs)
        cube.spawn_id(
            actorNumber=actor_number,
            location=location,
            rotation=layout.CUBE_ROTATION,
            scale=scale,
            configuration=cube.SHAPE_CUBE,
            waitForConfirmation=True,
        )
        cubes.append(cube)
    return cubes


def _spawn_traffic_cone_obstacles(qlabs, layout):
    """Spawn all traffic cones defined by the selected layout."""
    cones = []
    # Traffic cones are optional. A layout may omit the configuration entirely
    # (or define it as an empty tuple) when no cones are needed.
    for location, rotation, scale, configuration in getattr(
        layout, "TRAFFIC_CONE_OBSTACLES", ()
    ):
        cone = QLabsTrafficCone(qlabs)
        cone.spawn_degrees(
            location=location,
            rotation=rotation,
            scale=scale,
            configuration=configuration,
            waitForConfirmation=True,
        )
        cones.append(cone)
    return cones


def _run_lidar_preview(qcar, sample_points=400, square_size=100):
    """Display live QCar1 LiDAR data without creating experiment files."""
    lidar_display = LidarDisplay(square_size=square_size)
    failure_reported = False
    print("LiDAR preview active. Close the LiDAR window or press Ctrl+C to stop.")

    try:
        while lidar_display.is_open():
            success, angles, distances = qcar.get_lidar(
                samplePoints=sample_points
            )
            if success and angles is not None and distances is not None:
                lidar_display.update(angles, distances)
                failure_reported = False
            else:
                lidar_display.process_events()
                if not failure_reported:
                    print("Warning: unable to acquire LiDAR preview data.")
                    failure_reported = True
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("LiDAR preview stopped.")
    finally:
        lidar_display.close()


def spawnqcar2(
    qlabs,
    destroy_existing=True,
    spawn_obstacles=True,
    update_callback=None,
    initial_weather="unknown",
    scenario_name="unknown_scenario",
    record_data=True,
    preview_lidar=False,
    layout_number=None,
    max_frames=None,
    fusion_mode=None,
):

    if qlabs is None:
        raise ValueError("An open qlabs connection is required")

    # For all six interactive weather scripts, choose the fusion method first
    # and the obstacle layout second.  Explicit API arguments skip the prompt.
    fusion_mode = _resolve_fusion_mode(
        fusion_mode,
        prompt_for_mode=record_data,
    )
    print(
        "Selected fusion mode: "
        + ("Dynamic Weight" if fusion_mode == "adaptive" else "Fixed Weight")
    )

    layout = _resolve_obstacle_layout(
        layout_number,
        prompt_for_layout=spawn_obstacles,
    )
    if spawn_obstacles:
        print(
            f"Selected obstacle layout: {layout.LAYOUT_ID} "
            f"({layout.LAYOUT_NAME})"
        )

    if destroy_existing:
        qlabs.destroy_all_spawned_actors()

    if record_data:
        cv2.startWindowThread()

    overhead_camera = QLabsFreeCamera(qlabs)
    overhead_camera.spawn_id(
        actorNumber=1,
        location=[0, 0, 6.074],
        rotation=[0, 0, 0],
    )

    qcar1 = _spawn_stationary_qcar(qlabs, 1, EGO_QCAR_LOCATION)
    qcars = {"Qcar1": qcar1}
    for name, actor_number, location, yaw_degrees in layout.QCAR_OBSTACLES:
        qcars[name] = _spawn_stationary_qcar(
            qlabs,
            actor_number,
            location,
            yaw_degrees=yaw_degrees,
        )
    overhead_camera.possess()

    if spawn_obstacles:
        _spawn_cube_obstacles(qlabs, layout)
        _spawn_traffic_cone_obstacles(qlabs, layout)

    if not record_data:
        if preview_lidar:
            _run_lidar_preview(qcar1)
        return qcars

    recorder_class = SensorBatchRecorder
    if fusion_mode == "fixed":
        # Import only when requested so all existing scenario scripts keep the
        # adaptive recorder and its current behaviour unchanged.
        from sensor_fixed_weight import SensorBatchRecorder as recorder_class

    recorder = recorder_class(
        scenario_name=scenario_name,
        layout_id=layout.LAYOUT_ID,
        max_frames=max_frames,
    )
    current_weather = str(initial_weather)

    try:
        while True:
            if update_callback is not None:
                callback_result = update_callback()
                if callback_result is False:
                    break
                if callback_result is not None:
                    current_weather = str(callback_result)

            if not recorder.update(qcar1, weather=current_weather):
                break

            # Avoid a busy loop between the recorder's scheduled acquisitions.
            time.sleep(0.005)
    finally:
        recorder.close()

    return qcars


def main():
    """Spawn the QCar scene directly without recording experiment data."""
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    if not qlabs.open("localhost"):
        print("Unable to connect to QLabs")
        return

    print("Connected")
    try:
        spawnqcar2(
            qlabs=qlabs,
            initial_weather="unknown",
            record_data=False,
            preview_lidar=True,
        )
    finally:
        qlabs.close()
        print("Done!")


def run_obstacle_layout(layout_number, preview_lidar=False):
    """Connect to QLabs and spawn one obstacle layout for standalone debugging."""
    qlabs = QuanserInteractiveLabs()
    print(f"Connecting to QLabs for obstacle layout {layout_number}...")
    if not qlabs.open("localhost"):
        print("Unable to connect to QLabs")
        return False

    print("Connected")
    try:
        spawnqcar2(
            qlabs=qlabs,
            record_data=False,
            preview_lidar=preview_lidar,
            layout_number=layout_number,
        )
    finally:
        qlabs.close()
        print(f"Obstacle layout {layout_number} spawned. Connection closed.")

    return True


if __name__ == "__main__":
    main()
