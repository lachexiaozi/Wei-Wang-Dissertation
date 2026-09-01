# This scenario includes gradual weather transitions.
# clear skies -> light snow -> snow -> blizzard -> snow -> light snow -> clear

import time
from pathlib import Path

from qvl.environment_outdoors import QLabsEnvironmentOutdoors
from qvl.qlabs import QuanserInteractiveLabs
from qvl.system import QLabsSystem

from qcar2 import spawnqcar2


def main():
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    if not qlabs.open("localhost"):
        print("Unable to connect to QLabs")
        return

    print("Connected")
    system = QLabsSystem(qlabs)
    environment = QLabsEnvironmentOutdoors(qlabs)

    stages = (
        (environment.CLEAR_SKIES, "clear_skies", "Clear Skies", 5),
        (environment.LIGHT_SNOW, "light_snow", "Light Snow", 15),
        (environment.SNOW, "snow", "Snow", 15),
        (environment.BLIZZARD, "blizzard", "Blizzard", 15),
        (environment.SNOW, "snow", "Snow", 15),
        (environment.LIGHT_SNOW, "light_snow", "Light Snow", 15),
        (environment.CLEAR_SKIES, "clear_skies", "Clear Skies", 5),
    )
    start_time = None
    active_stage = None

    def update_weather():
        nonlocal start_time, active_stage
        now = time.monotonic()
        if start_time is None:
            start_time = now

        elapsed_time = now - start_time
        stage_index = None
        stage_end_time = 0
        for index, (*_stage_details, duration) in enumerate(stages):
            stage_end_time += duration
            if elapsed_time < stage_end_time:
                stage_index = index
                break

        if stage_index is None:
            return False

        if stage_index != active_stage:
            preset, _label, title, _duration = stages[stage_index]
            environment.set_weather_preset(preset)
            system.set_title_string(f"Current Weather: {title}")
            active_stage = stage_index

        return stages[stage_index][1]

    try:
        spawnqcar2(
            qlabs=qlabs,
            destroy_existing=True,
            spawn_obstacles=True,
            update_callback=update_weather,
            initial_weather="clear_skies",
            scenario_name=Path(__file__).stem,
        )
    finally:
        qlabs.close()
        print("Done!")


if __name__ == "__main__":
    main()
