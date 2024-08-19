#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import numpy as np
import pygame

import config


class Joystick:
    """A class to manage joystick functionality.

    This class handles joystick initialization, state updates, and mode switching.
    """

    def __init__(self, dev_fn=config.JOYSTICK_DEVICE_FILE):
        """Initialize the Joystick class.

        Args:
            dev_fn (str): The joystick device file name. Defaults to config.JOYSTICK_DEVICE_FILE.

        Attributes:
            HAVE_CONTROLLER (bool): Whether a joystick is connected.
            stick_left (int): Axis number for the left stick.
            stick_right (int): Axis number for the right stick.
            button_Y, button_X, button_A, button_B, button_S (int): Button numbers.
            steer (float): Steering value.
            accel, accel1, accel2 (float): Acceleration values.
            breaking (int): Braking value.
            mode (list): List of operation modes.
            recording (bool): Whether recording is active.
        """
        self.HAVE_CONTROLLER = True
        self.stick_left = config.JOYSTICK_AXIS_LEFT
        self.stick_right = config.JOYSTICK_AXIS_RIGHT
        self.button_Y = config.JOYSTICK_Y
        self.button_X = config.JOYSTICK_X
        self.button_A = config.JOYSTICK_A
        self.button_B = config.JOYSTICK_B
        self.button_S = config.JOYSTICK_S
        self.steer = 0.0
        self.accel = 0.0
        self.accel1 = 0.0
        self.accel2 = 0.0
        self.breaking = 0
        self.mode = ["auto", "auto_str", "user"]
        self.recording = True

        # Initialize pygame
        pygame.init()
        # Initialize joystick module
        pygame.joystick.init()

        try:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print("Joystick name:", self.joystick.get_name())
            print("Number of buttons:", self.joystick.get_numbuttons())
        except pygame.error:
            self.HAVE_CONTROLLER = False
            print("No joystick connected. Disabling joystick functionality.")

    def poll(self):
        """Update the joystick state.

        This method processes joystick events and updates steering, acceleration, and braking values.
        It also handles mode switching and recording toggle.
        """
        for e in pygame.event.get():
            self.steer = round(self.joystick.get_axis(self.stick_left), 2)
            self.accel = round(self.joystick.get_axis(self.stick_right), 2)
            self.accel1 = self.joystick.get_button(self.button_A)
            self.accel2 = self.joystick.get_button(self.button_B)
            self.breaking = self.joystick.get_button(self.button_X)
            if self.joystick.get_button(self.button_S):
                self.mode = np.roll(self.mode, 1)
                print(" mode:", self.mode[0])
            if self.joystick.get_button(self.button_Y):
                self.recording = not self.recording


if __name__ == "__main__":
    joystick = Joystick()
    while True:
        for e in pygame.event.get():
            print(e)
