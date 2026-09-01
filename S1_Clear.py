# This scenario include baseline weather.
# clear skies

from qvl.qlabs import QuanserInteractiveLabs
from qvl.free_camera import QLabsFreeCamera
from qvl.environment_outdoors import QLabsEnvironmentOutdoors
from qvl.system import QLabsSystem

from qcar2 import spawnqcar2

import time
import os
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
    hSystem.set_title_string('Current Weather: Clear skies')

    hqcar2 = spawnqcar2(
        qlabs=qlabs,
        destroy_existing=True,
        spawn_obstacles=True,
        initial_weather="clear_skies",
        scenario_name=Path(__file__).stem,
        max_frames=200,
    )

    print("\n\n------------------------------ Communications --------------------------------\n")

if __name__ == "__main__":
    main()
