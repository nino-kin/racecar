#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""This script contains a class for handling ultrasonic sensors using Raspberry Pi GPIO pins, as well as a main function
to measure distances and logs. The script uses the RPi.GPIO library for GPIO pin control and numpy for data handling.

Modules imported:
- time
- RPi.GPIO as GPIO
- config
- numpy as np

Classes:
- Ultrasonic: A class for handling ultrasonic sensor measurements.

Functions:
- measure(): Measures distance using the ultrasonic sensor.
- main(): The main function to initialize sensors, record measurements, and save data.
"""

import time

import numpy as np
import RPi.GPIO as GPIO

import config


class Ultrasonic:
    """A class to handle ultrasonic sensor measurements.

    Attributes:
    - name (str): Name of the ultrasonic sensor.
    - trig (int): GPIO pin number for the trigger.
    - echo (int): GPIO pin number for the echo.
    - records (np.ndarray): Array to store past measurements.
    - dis (int): Distance measured.
    """

    def __init__(self, name: str):
        """Initializes the Ultrasonic sensor with the specified name.

        Args:
            name (str): Name of the ultrasonic sensor.
        """
        self.name = name
        self.trig = config.ultrasonics_dict_trig[name]
        self.echo = config.ultrasonics_dict_echo[name]
        self.records = np.zeros(config.ultrasonics_Nrecords)
        self.dis = 0

    def measure(self) -> int:
        """Measures the distance using the ultrasonic sensor.

        Returns:
        - int: Measured distance.
        """
        self.dis = 0
        sigoff = 0
        sigon = 0
        GPIO.output(self.trig, GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(self.trig, GPIO.LOW)
        starttime = time.perf_counter()
        while GPIO.input(self.echo) == GPIO.LOW:
            sigoff = time.perf_counter()
            if sigoff - starttime > 0.02:
                break
        while GPIO.input(self.echo) == GPIO.HIGH:
            sigon = time.perf_counter()
            if sigon - sigoff > 0.02:
                break
        # time * sound speed / 2 (round trip)
        d = int((sigon - sigoff) * 340000 / 2)
        # Ignore measurements greater than 2m
        if d > 2000:
            self.dis = 2000
        # Replace negative noise values with the last valid measurement
        elif d < 0:
            print(f"@{self.name}, a noise occurred, using the last value")
            self.dis = self.records[0]
        else:
            self.dis = d
        # Insert the new measurement at the beginning of the records array and remove the last entry
        self.records = np.insert(self.records, 0, self.dis)
        self.records = np.delete(self.records, -1)
        return self.dis


if __name__ == "__main__":
    GPIO.setwarnings(False)
    # Set GPIO pin numbering mode
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(config.e_list, GPIO.IN)
    GPIO.setup(config.t_list, GPIO.OUT, initial=GPIO.LOW)

    # Set up the ultrasonic sensors
    ultrasonics = []
    config.ultrasonics_list = ["RrLH", "FrLH", "Fr", "FrRH", "RrRH"]
    for name in config.ultrasonics_list:
        ultrasonics.append(Ultrasonic(name))

    print("Using the following ultrasonic sensors:")
    print(" ", ultrasonics)

    # Create an array to record data
    d = np.zeros(len(ultrasonics))
    d_stack = np.zeros(len(ultrasonics) + 1)
    # Start time for recording
    start_time = time.perf_counter()
    # Number of measurements
    sampling_times = config.sampling_times
    # Target sample rate, adjust if using multiple sensors
    sampling_cycle = config.sampling_cycle / len(ultrasonics)

    # Pause until Enter is pressed to start measurements
    print("Enter the number of measurements, press Enter to start")
    print(f"Default if only Enter is pressed: {sampling_times}")
    # Confirm input
    while True:
        sampling_times = input()
        if sampling_times.isnumeric() and int(sampling_times) > 0:
            break
        elif sampling_times == "":
            sampling_times = config.sampling_times
            break
        else:
            print("Enter an integer greater than 0...")
    print(f"Starting {sampling_times} measurements!")
    # Convert input to int
    sampling_times = int(sampling_times)

    try:
        for i in range(sampling_times):
            message = ""
            for j in range(len(ultrasonics)):
                dis = ultrasonics[j].measure()
                # Record distance data in array
                d[j] = dis
                # Modify for display
                message += f"{config.ultrasonics_list[j]}:{dis}, "
                time.sleep(sampling_cycle)
            d_stack = np.vstack((d_stack, np.insert(d, 0, time.perf_counter() - start_time)))
            print(message)
        GPIO.cleanup()
        np.savetxt(config.record_filename, d_stack, fmt="%.3e")
        # Calculate and display average distance over time
        print("Number of measurements: ", sampling_times)
        print("Average distance:", np.round(np.mean(d_stack[:, 1:], axis=0), 0))
        print(
            "Average measurement time per sensor (seconds):",
            round(
                (time.perf_counter() - start_time) / sampling_times / len(ultrasonics),
                2,
            ),
        )
        print(f"Saved record to {config.record_filename}")

    except KeyboardInterrupt:
        np.savetxt(config.record_filename, d_stack, fmt="%.3e")
        print("Stop!")
        GPIO.cleanup()
