# This scenario include an abrupt weather transitions.
# clear skies -> foggy -> clear skies

from qvl.qlabs import QuanserInteractiveLabs
from qvl.free_camera import QLabsFreeCamera
from qvl.environment_outdoors import QLabsEnvironmentOutdoors
from qvl.system import QLabsSystem

from qcar2 import spawnqcar2

import time
from pathlib import Path



def main():
    # Creates a server connection with Quanser Interactive Labs and manages
    # the communications
    qlabs = QuanserInteractiveLabs()

    # Ensure that QLabs is running on your local machine
    print("Connecting to QLabs...")
    if (not qlabs.open("localhost")):
        print("Unable to connect to QLabs")
        return    

    print("Connected")

    hSystem = QLabsSystem(qlabs)

    ### Outdoor Environment
    hEnvironmentOutdoors2 = QLabsEnvironmentOutdoors(qlabs)

    hEnvironmentOutdoors2.set_weather_preset(hEnvironmentOutdoors2.CLEAR_SKIES)
    hSystem.set_title_string('Current Weather: Clear Skies')

    transition_start_time = None
    weather_stage = "clear_before_fog"

    def update_weather():
        """Run a 5 s clear, 30 s foggy, 5 s clear weather sequence."""
        nonlocal transition_start_time, weather_stage

        current_time = time.monotonic()
        if transition_start_time is None:
            transition_start_time = current_time

        elapsed_time = current_time - transition_start_time

        if elapsed_time >= 40:
            return False

        if elapsed_time >= 35 and weather_stage != "clear_after_fog":
            hEnvironmentOutdoors2.set_weather_preset(
                hEnvironmentOutdoors2.CLEAR_SKIES
            )
            hSystem.set_title_string('Current Weather: Clear Skies')
            weather_stage = "clear_after_fog"
        elif elapsed_time >= 5 and weather_stage == "clear_before_fog":
            hEnvironmentOutdoors2.set_weather_preset(hEnvironmentOutdoors2.FOGGY)
            hSystem.set_title_string('Current Weather: Foggy')
            weather_stage = "foggy"

        if weather_stage == "foggy":
            return "foggy"
        return "clear_skies"

    # Spawn QCar2 and collect synchronized sensor data throughout the weather
    # transition. Front CSI brightness is printed for every complete batch.
    spawnqcar2(
        qlabs=qlabs,
        destroy_existing=True,
        spawn_obstacles=True,
        update_callback=update_weather,
        initial_weather="clear_skies",
        scenario_name=Path(__file__).stem,
    )

    print("\n\n------------------------------ Communications --------------------------------\n")

if __name__ == "__main__":
    main()
