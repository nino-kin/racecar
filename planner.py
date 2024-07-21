#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import config
import time
if config.HAVE_NN:
    import torch.tensor
    from train_pytorch import denormalize_motor, normalize_ultrasonics

class Planner:
    """
    A class for planning and controlling the movement of a robot.

    This class implements various methods for navigation using ultrasonic sensors,
    including simple obstacle avoidance, wall following, and neural network-based control.
    """

    def __init__(self, name):
        """
        Initialize the Planner.

        Args:
            name (str): The name of the planner.
        """
        self.name = name
        self.steer_pwm_duty = 0
        self.throttle_pwm_duty = 0
        # Detection distance settings
        self.DETECTION_DISTANCE_Fr = config.DETECTION_DISTANCE_Fr
        self.DETECTION_DISTANCE_RL = config.DETECTION_DISTANCE_RL
        # Other detection distance settings
        self.DETECTION_DISTANCE_STOP = config.DETECTION_DISTANCE_STOP
        self.DETECTION_DISTANCE_BACK = config.DETECTION_DISTANCE_BACK
        self.DETECTION_DISTANCE_TARGET = config.DETECTION_DISTANCE_TARGET
        self.DETECTION_DISTANCE_RANGE = config.DETECTION_DISTANCE_RANGE
        # PID parameters
        self.K_P = config.K_P
        self.K_I = config.K_I
        self.K_D = config.K_D
        self.min_dis = 0

        # Decision flags
        self.flag_stop = False
        self.flag_back = False
        # Output message
        self.message = ""
        # Past control value records
        self.records_steer_pwm_duty = np.zeros(config.motor_Nrecords)
        self.records_throttle_pwm_duty = np.zeros(config.motor_Nrecords)

    def Back(self, ultrasonic_Fr, ultrasonic_FrRH, ultrasonic_FrLH):
        """
        Determine if the robot should move backwards based on front sensor readings.

        Args:
            ultrasonic_Fr (Ultrasonic): Front ultrasonic sensor.
            ultrasonic_FrRH (Ultrasonic): Front-right ultrasonic sensor.
            ultrasonic_FrLH (Ultrasonic): Front-left ultrasonic sensor.
        """
        times = 3
        if min(max(ultrasonic_Fr.records[:times]), max(ultrasonic_FrRH.records[:times]), max(ultrasonic_FrLH.records[:times])) < self.DETECTION_DISTANCE_BACK:
            self.flag_back = True
            print("Backing up")
        elif max(ultrasonic_Fr.records[:times]) > self.DETECTION_DISTANCE_BACK:
            self.flag_back = False

    def Stop(self, ultrasonic_Fr):
        """
        Determine if the robot should stop based on front sensor reading.

        Args:
            ultrasonic_Fr (Ultrasonic): Front ultrasonic sensor.
        """
        times = 3
        if max(ultrasonic_Fr.records[0:times-1]) < self.DETECTION_DISTANCE_STOP:
            self.flag_stop = True
            print("Stopping")

    def Right_Left_3(self, dis_FrLH, dis_Fr, dis_FrRH):
        """
        Determine movement based on three front sensors.

        Args:
            dis_FrLH (float): Distance from front-left sensor.
            dis_Fr (float): Distance from front sensor.
            dis_FrRH (float): Distance from front-right sensor.

        Returns:
            tuple: Steering and throttle PWM duty cycles.
        """
        if dis_Fr < self.DETECTION_DISTANCE_Fr or dis_FrLH < self.DETECTION_DISTANCE_RL or dis_FrRH < self.DETECTION_DISTANCE_RL:
            if dis_FrLH < dis_FrRH:
                self.steer_pwm_duty = config.RIGHT
                self.throttle_pwm_duty = config.FORWARD_C
                self.message = "Turning right"
            else:
                self.steer_pwm_duty = config.LEFT
                self.throttle_pwm_duty = config.FORWARD_C
                self.message = "Turning left"
        else:
            self.steer_pwm_duty = config.NUTRAL
            self.throttle_pwm_duty = config.FORWARD_S
            self.message = "Moving straight"

        if config.print_plan_result:
            print(self.message)
        return self.steer_pwm_duty, self.throttle_pwm_duty

    def Right_Left_3_Records(self, dis_FrLH, dis_Fr, dis_FrRH):
        """
        Determine movement based on three front sensors with smoothing.

        Args:
            dis_FrLH (float): Distance from front-left sensor.
            dis_Fr (float): Distance from front sensor.
            dis_FrRH (float): Distance from front-right sensor.

        Returns:
            tuple: Smoothed steering and throttle PWM duty cycles.
        """
        self.steer_pwm_duty, self.throttle_pwm_duty = self.Right_Left_3(dis_FrLH, dis_Fr, dis_FrRH)

        self.records_steer_pwm_duty = np.insert(self.records_steer_pwm_duty, 0, self.steer_pwm_duty)
        self.records_steer_pwm_duty = np.delete(self.records_steer_pwm_duty, -1)
        self.records_throttle_pwm_duty = np.insert(self.records_throttle_pwm_duty, 0, self.throttle_pwm_duty)
        self.records_throttle_pwm_duty = np.delete(self.records_throttle_pwm_duty, -1)

        if config.print_plan_result:
            print(self.message)
        return np.mean(self.records_steer_pwm_duty), np.mean(self.records_throttle_pwm_duty)

    def RightHand(self, dis_FrRH, dis_RrRH):
        """
        Implement right-hand wall following.

        Args:
            dis_FrRH (float): Distance from front-right sensor.
            dis_RrRH (float): Distance from rear-right sensor.

        Returns:
            tuple: Steering and throttle PWM duty cycles.
        """
        # The vehicle is a large distance from the wall on the right side
        if dis_FrRH > self.DETECTION_DISTANCE_TARGET + self.DETECTION_DISTANCE_RANGE and dis_RrRH > self.DETECTION_DISTANCE_TARGET + self.DETECTION_DISTANCE_RANGE:
            self.steer_pwm_duty = config.RIGHT
            self.throttle_pwm_duty = config.FORWARD_C
            self.message = "Turning right"
        # The vehicle is approaching the wall on the right.
        elif dis_FrRH < self.DETECTION_DISTANCE_TARGET - self.DETECTION_DISTANCE_RANGE or dis_RrRH < self.DETECTION_DISTANCE_TARGET - self.DETECTION_DISTANCE_RANGE:
            self.steer_pwm_duty = config.LEFT
            self.throttle_pwm_duty = config.FORWARD_C
            self.message = "Turning left"
        else:
            self.steer_pwm_duty = config.NUTRAL
            self.throttle_pwm_duty = config.FORWARD_S
            self.message = "Moving straight"

        if config.print_plan_result:
            print(self.message)
        return self.steer_pwm_duty, self.throttle_pwm_duty

    def LeftHand(self, dis_FrLH, dis_RrLH):
        """
        Implement left-hand wall following.

        Args:
            dis_FrLH (float): Distance from front-left sensor.
            dis_RrLH (float): Distance from rear-left sensor.

        Returns:
            tuple: Steering and throttle PWM duty cycles.
        """
        if dis_FrLH > self.DETECTION_DISTANCE_TARGET + self.DETECTION_DISTANCE_RANGE and dis_RrLH > self.DETECTION_DISTANCE_TARGET + self.DETECTION_DISTANCE_RANGE:
            self.steer_pwm_duty = config.LEFT
            self.throttle_pwm_duty = config.FORWARD_C
            self.message = "Turning left"
        elif dis_FrLH < self.DETECTION_DISTANCE_TARGET - self.DETECTION_DISTANCE_RANGE or dis_RrLH < self.DETECTION_DISTANCE_TARGET - self.DETECTION_DISTANCE_RANGE:
            self.steer_pwm_duty = config.RIGHT
            self.throttle_pwm_duty = config.FORWARD_C
            self.message = "Turning right"
        else:
            self.steer_pwm_duty = config.NUTRAL
            self.throttle_pwm_duty = config.FORWARD_S
            self.message = "Moving straight"

        if config.print_plan_result:
            print(self.message)
        return self.steer_pwm_duty, self.throttle_pwm_duty

    def RightHand_PID(self, ultrasonic_FrRH, ultrasonic_RrRH, t=0, integral_delta_dis=0, min_dis=config.DETECTION_DISTANCE_TARGET):
        """
        Implement right-hand wall following with PID control.

        Args:
            ultrasonic_FrRH (Ultrasonic): Front-right ultrasonic sensor.
            ultrasonic_RrRH (Ultrasonic): Rear-right ultrasonic sensor.
            t (float): Current time.
            integral_delta_dis (float): Integral of distance error.
            min_dis (float): Minimum distance to wall.

        Returns:
            tuple: PID-adjusted steering and throttle PWM duty cycles.
        """
        t_before = t
        t = time.perf_counter()
        delta_t = t - t_before
        # 右手法最小距離更新
        min_dis_before = min_dis
        min_dis = min(ultrasonic_FrRH.dis, ultrasonic_RrRH.dis)
        # 目標値までの差更新
        delta_dis = min_dis - self.DETECTION_DISTANCE_TARGET
        # 目標値までの差積分更新
        integral_delta_dis += delta_dis
        # 速度更新
        v = (min_dis - min_dis_before) / delta_t
        # PID制御でステア値更新
        steer_pwm_duty_pid = self.K_P * delta_dis - self.K_D * v + self.K_I * integral_delta_dis
        ### -100~100に収めて正の割合化
        steer_pwm_duty_pid = abs(max(-100, min(100, steer_pwm_duty_pid)) / 100)

        ## モーターへ出力を返す
        if config.print_plan_result:
            print("output * PID:{:3.1f}, [P:{:3.1f}, I:{:3.1f}, D:{:3.1f}]".format(steer_pwm_duty_pid, self.K_P*delta_dis, self.K_D*v, self.K_I*integral_delta_dis))
        self.steer_pwm_duty, self.throttle_pwm_duty = self.RightHand(ultrasonic_FrRH.dis, ultrasonic_RrRH.dis)
        return steer_pwm_duty_pid * self.steer_pwm_duty, self.throttle_pwm_duty

    def LeftHand_PID(self, ultrasonic_FrLH, ultrasonic_RrLH, t=0, integral_delta_dis=0, min_dis=config.DETECTION_DISTANCE_TARGET):
        """
        Implement left-hand wall following with PID control.

        Args:
            ultrasonic_FrLH (Ultrasonic): Front-left ultrasonic sensor.
            ultrasonic_RrLH (Ultrasonic): Rear-left ultrasonic sensor.
            t (float): Current time.
            integral_delta_dis (float): Integral of distance error.
            min_dis (float): Minimum distance to wall.

        Returns:
            tuple: PID-adjusted steering and throttle PWM duty cycles.
        """
        t_before = t
        t = time.perf_counter()
        delta_t = t - t_before
        # 右手法最小距離更新
        min_dis_before = min_dis
        min_dis = min(ultrasonic_FrLH.dis, ultrasonic_RrLH.dis)
        # 目標値までの差更新
        delta_dis = min_dis - self.DETECTION_DISTANCE_TARGET
        # 目標値までの差積分更新
        integral_delta_dis += delta_dis
        # 速度更新
        v = (min_dis - min_dis_before) / delta_t
        # PID制御でステア値更新
        steer_pwm_duty_pid = self.K_P * delta_dis - self.K_D * v + self.K_I * integral_delta_dis
        ### -100~100に収めて正の割合化
        steer_pwm_duty_pid = abs(max(-100, min(100, steer_pwm_duty_pid)) / 100)

        ## モーターへ出力を返す
        if config.print_plan_result:
            print("output * PID:{:3.1f}, [P:{:3.1f}, I:{:3.1f}, D:{:3.1f}]".format(steer_pwm_duty_pid, self.K_P*delta_dis, self.K_D*v, self.K_I*integral_delta_dis))
        self.steer_pwm_duty, self.throttle_pwm_duty = self.LeftHand(ultrasonic_FrLH.dis, ultrasonic_RrLH.dis)
        return steer_pwm_duty_pid * self.steer_pwm_duty, self.throttle_pwm_duty

    # Neural Netを用いた走行
    if config.HAVE_NN:
        # train_pytorch.py内ので正規化処理を用いる
        def NN(self, model, *args):
            """
            Use a neural network model for navigation.

            Args:
                model: The neural network model.
                *args: Ultrasonic sensor values.

            Returns:
                tuple: Steering and throttle PWM duty cycles predicted by the neural network.
            """
            ultrasonic_values = args
            input = normalize_ultrasonics(torch.tensor(ultrasonic_values, dtype=torch.float32).unsqueeze(0))
            output = denormalize_motor(model.predict(model, input).squeeze(0))
            self.steer_pwm_duty = int(output[0])
            self.throttle_pwm_duty = int(output[1])

            ## モーターへ出力を返す
            return self.steer_pwm_duty, self.throttle_pwm_duty
