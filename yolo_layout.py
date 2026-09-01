import sys

from qvl.qlabs import QuanserInteractiveLabs
from qvl.free_camera import QLabsFreeCamera
from qvl.basic_shape import QLabsBasicShape
from qvl.qcar2 import QLabsQCar2
from qvl.traffic_cone import QLabsTrafficCone
from qvl.environment_outdoors import QLabsEnvironmentOutdoors

import time
import math
import numpy as np
import cv2
import os
import re

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from qvl.system import QLabsSystem


OUTPUT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_train")


def get_next_image_number(output_directory):
    """Return the next shared CSI/RGB image number, starting from 4."""
    # Ignore old zero-padded names such as _000001.jpg.  The new sequence uses
    # _2.jpg, _3.jpg, ... and the next run must start at _4.jpg.
    image_pattern = re.compile(r"^QCar1_(?:Front_CSI|RGB)_([1-9]\d*)\.jpg$")
    existing_numbers = []

    for filename in os.listdir(output_directory):
        match = image_pattern.match(filename)
        if match:
            existing_numbers.append(int(match.group(1)))

    return max(existing_numbers + [3]) + 1


def capture_qcar1_images(qcar, output_directory=OUTPUT_DIRECTORY):
    """Capture and save one Front CSI image and one RGB image from QCar1."""
    os.makedirs(output_directory, exist_ok=True)

    print(f"Saving QCar1 camera images to: {output_directory}")

    csi_success, csi_image = qcar.get_image(camera=qcar.CAMERA_CSI_FRONT)
    rgb_success, rgb_image = qcar.get_image(camera=qcar.CAMERA_RGB)

    if not csi_success or not rgb_success:
        failed_cameras = []
        if not csi_success:
            failed_cameras.append("Front CSI")
        if not rgb_success:
            failed_cameras.append("RGB")
        raise RuntimeError(
            f"Camera acquisition failed: {', '.join(failed_cameras)}"
        )

    image_number = get_next_image_number(output_directory)
    csi_path = os.path.join(
        output_directory, f"QCar1_Front_CSI_{image_number}.jpg"
    )
    rgb_path = os.path.join(output_directory, f"QCar1_RGB_{image_number}.jpg")

    if not cv2.imwrite(csi_path, csi_image):
        raise OSError(f"Unable to save image: {csi_path}")
    if not cv2.imwrite(rgb_path, rgb_image):
        raise OSError(f"Unable to save image: {rgb_path}")

    print(f"Saved QCar1 camera image pair {image_number}.")


def main():
    os.system('cls')

    #Communications with qlabs

    qlabs = QuanserInteractiveLabs()
    cv2.startWindowThread()

    print("Connecting to QLabs...")
    if (not qlabs.open("localhost")):
        print("Unable to connect to QLabs")
        return    

    print("Connected")

    qlabs.destroy_all_spawned_actors()

    # Use hSystem to set the tutorial title in the upper left of the qlabs window 
    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string('QCar Tutorial')

    ### QCar

    hCameraQCars = QLabsFreeCamera(qlabs)
    hCameraQCars.spawn_id(actorNumber=1, location=[-15.075, 26.703, 6.074], rotation=[0, 0.564, -1.586])
    hCameraQCars.possess()

    print("\n\n---QCar---")


    #spawn a QCar with degrees
    hQCar1 = QLabsQCar2(qlabs)
    hQCar1.spawn_id_degrees(actorNumber=1, location=[0,6.2,2], rotation=[0,0,0], waitForConfirmation=True)

    hQCar2 = QLabsQCar2(qlabs)
    hQCar2.spawn_id_degrees(actorNumber=2, location=[16.6,9.4,2], rotation=[0,0,0], waitForConfirmation=True)

    hQCar3 = QLabsQCar2(qlabs)
    hQCar3.spawn_id_degrees(actorNumber=3, location=[20.8,8.9,2], rotation=[0,0,0], waitForConfirmation=True)



    cone = QLabsTrafficCone(qlabs)
    cone1 = QLabsTrafficCone(qlabs)
    cone.spawn(location=[14.8,4.8,2], rotation=[0,0,math.pi], scale=[1,1,1], configuration=1, waitForConfirmation=True)
    cone1.spawn(location=[12.3,9.7,2], rotation=[0,0,math.pi], scale=[1,1,1], configuration=1, waitForConfirmation=True)


    cube0 = QLabsBasicShape(qlabs)
    cube1 = QLabsBasicShape(qlabs)
    cube0.spawn_id(actorNumber=4, location=[10.2,3.8,1.5], rotation=[0,0,0], scale=[1,1,4], configuration=cube0.SHAPE_CUBE, waitForConfirmation=True)
    cube1.spawn_id(actorNumber=5, location=[19.0,3.3,1.5], rotation=[0,0,0], scale=[1,1,2], configuration=cube0.SHAPE_CUBE, waitForConfirmation=True)

    capture_qcar1_images(hQCar1)


main()
