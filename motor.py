#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import time

import Adafruit_PCA9685

import config


class Motor:
    """A class to control motor and steering for an RC car.

    This class manages PWM signals for throttle and steering control.
    """

    def __init__(self):
        """Initialize the Motor class.

        Sets up the PWM controller and loads configuration values.
        """
        self.pwm = Adafruit_PCA9685.PCA9685(address=0x40)
        self.pwm.set_pwm_freq(60)
        self.CHANNEL_STEERING = config.CHANNEL_STEERING
        self.CHANNEL_THROTTLE = config.CHANNEL_THROTTLE
        self.STEERING_CENTER_PWM = config.STEERING_CENTER_PWM
        self.STEERING_WIDTH_PWM = config.STEERING_WIDTH_PWM
        self.STEERING_RIGHT_PWM = config.STEERING_RIGHT_PWM
        self.STEERING_LEFT_PWM = config.STEERING_LEFT_PWM
        self.THROTTLE_STOPPED_PWM = config.THROTTLE_STOPPED_PWM
        self.THROTTLE_WIDTH_PWM = config.THROTTLE_WIDTH_PWM
        self.THROTTLE_FORWARD_PWM = config.THROTTLE_FORWARD_PWM
        self.THROTTLE_REVERSE_PWM = config.THROTTLE_REVERSE_PWM

    def set_throttle_pwm_duty(self, duty):
        """Set the throttle PWM duty cycle.

        Args:
            duty (int): The duty cycle for throttle (-100 to 100).
        """
        if duty >= 0:
            throttle_pwm = int(
                self.THROTTLE_STOPPED_PWM
                + (self.THROTTLE_FORWARD_PWM - self.THROTTLE_STOPPED_PWM) * duty / 100
            )
        else:
            throttle_pwm = int(
                self.THROTTLE_STOPPED_PWM
                + (self.THROTTLE_REVERSE_PWM - self.THROTTLE_STOPPED_PWM)
                * abs(duty)
                / 100
            )

        self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, throttle_pwm)

    def set_steer_pwm_duty(self, duty):
        """Set the steering PWM duty cycle.

        Args:
            duty (int): The duty cycle for steering (-100 to 100).
        """
        if duty >= 0:
            steer_pwm = int(
                self.STEERING_CENTER_PWM
                + (self.STEERING_RIGHT_PWM - self.STEERING_CENTER_PWM) * duty / 100
            )
        else:
            steer_pwm = int(
                self.STEERING_CENTER_PWM
                + (self.STEERING_LEFT_PWM - self.STEERING_CENTER_PWM) * abs(duty) / 100
            )
        steer_pwm = self.limit_steer_PWM(steer_pwm)
        self.pwm.set_pwm(self.CHANNEL_STEERING, 0, steer_pwm)

    def limit_steer_PWM(self, steer_pwm):
        """Limit the steering PWM to safe values.

        Args:
            steer_pwm (int): The proposed steering PWM value.

        Returns:
            int: The limited steering PWM value.
        """
        if steer_pwm > config.STEERING_RIGHT_PWM_LIMIT:
            print(
                f"\n[WARN] Please set maximum value to {config.STEERING_RIGHT_PWM_LIMIT} to avoid damage!\n"
            )
            return config.STEERING_RIGHT_PWM_LIMIT
        elif steer_pwm < config.STEERING_LEFT_PWM_LIMIT:
            print(
                f"\n[WARN] Please set minimum value to {config.STEERING_LEFT_PWM_LIMIT} to avoid damage!\n"
            )
            return config.STEERING_LEFT_PWM_LIMIT
        else:
            return steer_pwm

    def adjust_steering(self):
        """Interactive method to adjust steering center position.

        Returns:
            tuple: The right, center, and left steering PWM values.
        """
        print("========================================")
        print(" Steering adjustment, set center position")
        print("========================================")
        while True:
            print("Enter PWM value, e.g., 390")
            print("Press Enter when center value is set")
            print("Caution: Continuous noise may indicate potential damage")
            ad = input()
            if ad == "e" or ad == "":
                self.STEERING_RIGHT_PWM = (
                    self.STEERING_CENTER_PWM + self.STEERING_WIDTH_PWM
                )
                self.STEERING_LEFT_PWM = (
                    self.STEERING_CENTER_PWM - self.STEERING_WIDTH_PWM
                )
                break
            self.STEERING_CENTER_PWM = int(ad)
            self.limit_steer_PWM(self.STEERING_CENTER_PWM)
            self.pwm.set_pwm(self.CHANNEL_STEERING, 0, self.STEERING_CENTER_PWM)
        print("")
        return self.STEERING_RIGHT_PWM, self.STEERING_CENTER_PWM, self.STEERING_LEFT_PWM

    def adjust_throttle(self):
        """Interactive method to adjust throttle neutral position.

        Returns:
            tuple: The forward, stopped, and reverse throttle PWM values.
        """
        print("========================================")
        print(" Throttle adjustment, set neutral position")
        print("========================================")
        while True:
            print("Enter PWM value, e.g., 390")
            print("Press Enter when neutral value is set")
            ad = input()
            if ad == "e" or ad == "":
                self.THROTTLE_FORWARD_PWM = (
                    self.THROTTLE_STOPPED_PWM + self.THROTTLE_WIDTH_PWM
                )
                self.THROTTLE_REVERSE_PWM = (
                    self.THROTTLE_STOPPED_PWM - self.THROTTLE_WIDTH_PWM
                )
                break
            self.THROTTLE_STOPPED_PWM = int(ad)
            self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, self.THROTTLE_STOPPED_PWM)
        print("")
        return (
            self.THROTTLE_FORWARD_PWM,
            self.THROTTLE_STOPPED_PWM,
            self.THROTTLE_REVERSE_PWM,
        )

    def writetofile(self, path):
        """Write the current PWM settings to a file.

        Args:
            path (str): The file path to write the settings to.
        """
        with open(path, "w") as f:
            f.write(f"STEERING_RIGHT_PWM = {self.STEERING_RIGHT_PWM}\n")
            f.write(f"STEERING_CENTER_PWM = {self.STEERING_CENTER_PWM}\n")
            f.write(f"STEERING_LEFT_PWM = {self.STEERING_LEFT_PWM}\n")
            f.write(f"THROTTLE_FORWARD_PWM = {self.THROTTLE_FORWARD_PWM}\n")
            f.write(f"THROTTLE_STOPPED_PWM = {self.THROTTLE_STOPPED_PWM}\n")
            f.write(f"THROTTLE_REVERSE_PWM = {self.THROTTLE_REVERSE_PWM}\n")

    def breaking(self):
        """Perform a breaking sequence.

        This method rapidly alternates between stop and reverse to simulate breaking.
        """
        print("Breaking!!!")
        self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, self.THROTTLE_STOPPED_PWM)
        time.sleep(0.05)
        self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, self.THROTTLE_REVERSE_PWM)
        time.sleep(0.05)
        self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, self.THROTTLE_STOPPED_PWM)
        time.sleep(0.05)
        self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, self.THROTTLE_REVERSE_PWM)
        time.sleep(0.05)
        self.pwm.set_pwm(self.CHANNEL_THROTTLE, 0, self.THROTTLE_STOPPED_PWM)


if __name__ == "__main__":
    try:
        motor = Motor()
        motor.set_throttle_pwm_duty(0)
        motor.set_steer_pwm_duty(0)

        (
            STEERING_RIGHT_PWM,
            STEERING_CENTER_PWM,
            STEERING_LEFT_PWM,
        ) = motor.adjust_steering()
        (
            THROTTLE_FORWARD_PWM,
            THROTTLE_STOPPED_PWM,
            THROTTLE_REVERSE_PWM,
        ) = motor.adjust_throttle()
        print(
            "---Enter the following values in config.py.\nFine-tune values while driving."
        )
        print(f"STEERING_RIGHT_PWM = {STEERING_RIGHT_PWM}")
        print(f"STEERING_CENTER_PWM = {STEERING_CENTER_PWM}")
        print(f"STEERING_LEFT_PWM = {STEERING_LEFT_PWM}")
        print(f"THROTTLE_FORWARD_PWM = {THROTTLE_FORWARD_PWM}")
        print(f"THROTTLE_STOPPED_PWM = {THROTTLE_STOPPED_PWM}")
        print(f"THROTTLE_REVERSE_PWM = {THROTTLE_REVERSE_PWM}")
        print(
            "---Enter the above values in config.py.\nFine-tune values while driving."
        )
        motor.set_throttle_pwm_duty(0)
        motor.set_steer_pwm_duty(0)

    except KeyboardInterrupt:
        print("Stopped by user")
        motor.set_steer_pwm_duty(config.NEUTRAL)
        motor.set_throttle_pwm_duty(config.STOP)
