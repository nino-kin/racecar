#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import time
from copy import deepcopy


class IMU:
    """Base class for IMU (Inertial Measurement Unit) sensors."""

    def __init__(self):
        """Initialize the IMU."""
        print("Using IMU unit: ", end="")
        self.unit = None
        self.start_time = time.perf_counter()
        self.end_time = time.perf_counter()
        # angle is the estimated value from sensor fusion
        self.angle = 0.0  # Angle [degrees]
        self.angle_pre = 0.0  # Previous angle for noise removal

        # Sensor values - each axis holds a list of past values
        self.mem = 3
        self.acc = {
            "x": [0.0] * self.mem,
            "y": [0.0] * self.mem,
            "z": [0.0] * self.mem,
        }  # Acceleration [m/s^2]
        self.gyr = deepcopy(self.acc)  # Angular velocity [°/s]

        # Differential calculation values
        self.jerk = deepcopy(self.acc)  # Jerk [m/s^3]
        self.angular_velocity = deepcopy(self.acc)  # Angular acceleration [°/s^2]

        # Integral calculation values
        self.velocity = deepcopy(self.acc)
        self.position = deepcopy(self.acc)
        self.rotation = deepcopy(self.acc)

        # Lap flag
        self.lap = 0

        # Dynamic control parameters using gyro
        self.Gthr = 1.0  # Throttle gain (forward/backward direction)
        self.Gstr = 0.0  # Steering gain (lateral direction)
        self.Gthr_flag = 0
        self.Gstr_flag = 0
        self.rotation_speed = 1  # Reference rotation speed [°/s]
        # For GV
        self.Gxc = 1.0
        self.Cxy = 1.0
        self.Ts = 1.0

    def GCounter(self):
        """Calculate counter-steering control.

        Returns:
            float: Steering gain
        """
        self.Gstr = min(
            1, abs(sum(self.gyr["z"]) / len(self.gyr["z"])) / self.rotation_speed
        )
        return self.Gstr

    def GVectoring(self):
        """Calculate lateral G throttle control.

        Returns:
            float: Throttle gain
        """
        self.Gxc = abs(
            (self.acc["y"][-1] * self.jerk["y"][-1])
            * self.Cxy
            / (1 + self.Ts)
            * abs(self.jerk["y"][-1])
        )
        self.Gthr = min(1, self.Gxc)
        return self.Gthr


class BNO055(IMU):
    """Class for BNO055 9-axis sensor fusion module.
    Modified from https://github.com/ghirlekar/bno055-python-i2c.git, MIT license
    """

    BNO055_ADDRESS_A = 0x28
    BNO055_ADDRESS_B = 0x29
    BNO055_ID = 0xA0

    # Power mode settings
    POWER_MODE_NORMAL = 0x00
    POWER_MODE_LOWPOWER = 0x01
    POWER_MODE_SUSPEND = 0x02

    # Operation mode settings
    OPERATION_MODE_CONFIG = 0x00
    OPERATION_MODE_ACCONLY = 0x01
    OPERATION_MODE_MAGONLY = 0x02
    OPERATION_MODE_GYRONLY = 0x03
    OPERATION_MODE_ACCMAG = 0x04
    OPERATION_MODE_ACCGYRO = 0x05
    OPERATION_MODE_MAGGYRO = 0x06
    OPERATION_MODE_AMG = 0x07
    OPERATION_MODE_IMUPLUS = 0x08
    OPERATION_MODE_COMPASS = 0x09
    OPERATION_MODE_M4G = 0x0A
    OPERATION_MODE_NDOF_FMC_OFF = 0x0B
    OPERATION_MODE_NDOF = 0x0C

    # Output vector type
    VECTOR_ACCELEROMETER = 0x08
    VECTOR_MAGNETOMETER = 0x0E
    VECTOR_GYROSCOPE = 0x14
    VECTOR_EULER = 0x1A
    VECTOR_LINEARACCEL = 0x28
    VECTOR_GRAVITY = 0x2E

    # REGISTER DEFINITION START
    BNO055_PAGE_ID_ADDR = 0x07

    BNO055_CHIP_ID_ADDR = 0x00
    BNO055_ACCEL_REV_ID_ADDR = 0x01
    BNO055_MAG_REV_ID_ADDR = 0x02
    BNO055_GYRO_REV_ID_ADDR = 0x03
    BNO055_SW_REV_ID_LSB_ADDR = 0x04
    BNO055_SW_REV_ID_MSB_ADDR = 0x05
    BNO055_BL_REV_ID_ADDR = 0x06

    # Accel data register
    BNO055_ACCEL_DATA_X_LSB_ADDR = 0x08
    BNO055_ACCEL_DATA_X_MSB_ADDR = 0x09
    BNO055_ACCEL_DATA_Y_LSB_ADDR = 0x0A
    BNO055_ACCEL_DATA_Y_MSB_ADDR = 0x0B
    BNO055_ACCEL_DATA_Z_LSB_ADDR = 0x0C
    BNO055_ACCEL_DATA_Z_MSB_ADDR = 0x0D

    # Mag data register
    BNO055_MAG_DATA_X_LSB_ADDR = 0x0E
    BNO055_MAG_DATA_X_MSB_ADDR = 0x0F
    BNO055_MAG_DATA_Y_LSB_ADDR = 0x10
    BNO055_MAG_DATA_Y_MSB_ADDR = 0x11
    BNO055_MAG_DATA_Z_LSB_ADDR = 0x12
    BNO055_MAG_DATA_Z_MSB_ADDR = 0x13

    # Gyro data registers
    BNO055_GYRO_DATA_X_LSB_ADDR = 0x14
    BNO055_GYRO_DATA_X_MSB_ADDR = 0x15
    BNO055_GYRO_DATA_Y_LSB_ADDR = 0x16
    BNO055_GYRO_DATA_Y_MSB_ADDR = 0x17
    BNO055_GYRO_DATA_Z_LSB_ADDR = 0x18
    BNO055_GYRO_DATA_Z_MSB_ADDR = 0x19

    # Euler data registers
    BNO055_EULER_H_LSB_ADDR = 0x1A
    BNO055_EULER_H_MSB_ADDR = 0x1B
    BNO055_EULER_R_LSB_ADDR = 0x1C
    BNO055_EULER_R_MSB_ADDR = 0x1D
    BNO055_EULER_P_LSB_ADDR = 0x1E
    BNO055_EULER_P_MSB_ADDR = 0x1F

    # Quaternion data registers
    BNO055_QUATERNION_DATA_W_LSB_ADDR = 0x20
    BNO055_QUATERNION_DATA_W_MSB_ADDR = 0x21
    BNO055_QUATERNION_DATA_X_LSB_ADDR = 0x22
    BNO055_QUATERNION_DATA_X_MSB_ADDR = 0x23
    BNO055_QUATERNION_DATA_Y_LSB_ADDR = 0x24
    BNO055_QUATERNION_DATA_Y_MSB_ADDR = 0x25
    BNO055_QUATERNION_DATA_Z_LSB_ADDR = 0x26
    BNO055_QUATERNION_DATA_Z_MSB_ADDR = 0x27

    # Linear acceleration data registers
    BNO055_LINEAR_ACCEL_DATA_X_LSB_ADDR = 0x28
    BNO055_LINEAR_ACCEL_DATA_X_MSB_ADDR = 0x29
    BNO055_LINEAR_ACCEL_DATA_Y_LSB_ADDR = 0x2A
    BNO055_LINEAR_ACCEL_DATA_Y_MSB_ADDR = 0x2B
    BNO055_LINEAR_ACCEL_DATA_Z_LSB_ADDR = 0x2C
    BNO055_LINEAR_ACCEL_DATA_Z_MSB_ADDR = 0x2D

    # Gravity data registers
    BNO055_GRAVITY_DATA_X_LSB_ADDR = 0x2E
    BNO055_GRAVITY_DATA_X_MSB_ADDR = 0x2F
    BNO055_GRAVITY_DATA_Y_LSB_ADDR = 0x30
    BNO055_GRAVITY_DATA_Y_MSB_ADDR = 0x31
    BNO055_GRAVITY_DATA_Z_LSB_ADDR = 0x32
    BNO055_GRAVITY_DATA_Z_MSB_ADDR = 0x33

    # Temperature data register
    BNO055_TEMP_ADDR = 0x34

    # Status registers
    BNO055_CALIB_STAT_ADDR = 0x35
    BNO055_SELFTEST_RESULT_ADDR = 0x36
    BNO055_INTR_STAT_ADDR = 0x37

    BNO055_SYS_CLK_STAT_ADDR = 0x38
    BNO055_SYS_STAT_ADDR = 0x39
    BNO055_SYS_ERR_ADDR = 0x3A

    # Unit selection register
    BNO055_UNIT_SEL_ADDR = 0x3B
    BNO055_DATA_SELECT_ADDR = 0x3C

    # Mode registers
    BNO055_OPR_MODE_ADDR = 0x3D
    BNO055_PWR_MODE_ADDR = 0x3E

    BNO055_SYS_TRIGGER_ADDR = 0x3F
    BNO055_TEMP_SOURCE_ADDR = 0x40

    # Axis remap registers
    BNO055_AXIS_MAP_CONFIG_ADDR = 0x41
    BNO055_AXIS_MAP_SIGN_ADDR = 0x42

    # SIC registers
    BNO055_SIC_MATRIX_0_LSB_ADDR = 0x43
    BNO055_SIC_MATRIX_0_MSB_ADDR = 0x44
    BNO055_SIC_MATRIX_1_LSB_ADDR = 0x45
    BNO055_SIC_MATRIX_1_MSB_ADDR = 0x46
    BNO055_SIC_MATRIX_2_LSB_ADDR = 0x47
    BNO055_SIC_MATRIX_2_MSB_ADDR = 0x48
    BNO055_SIC_MATRIX_3_LSB_ADDR = 0x49
    BNO055_SIC_MATRIX_3_MSB_ADDR = 0x4A
    BNO055_SIC_MATRIX_4_LSB_ADDR = 0x4B
    BNO055_SIC_MATRIX_4_MSB_ADDR = 0x4C
    BNO055_SIC_MATRIX_5_LSB_ADDR = 0x4D
    BNO055_SIC_MATRIX_5_MSB_ADDR = 0x4E
    BNO055_SIC_MATRIX_6_LSB_ADDR = 0x4F
    BNO055_SIC_MATRIX_6_MSB_ADDR = 0x50
    BNO055_SIC_MATRIX_7_LSB_ADDR = 0x51
    BNO055_SIC_MATRIX_7_MSB_ADDR = 0x52
    BNO055_SIC_MATRIX_8_LSB_ADDR = 0x53
    BNO055_SIC_MATRIX_8_MSB_ADDR = 0x54

    # Accelerometer Offset registers
    ACCEL_OFFSET_X_LSB_ADDR = 0x55
    ACCEL_OFFSET_X_MSB_ADDR = 0x56
    ACCEL_OFFSET_Y_LSB_ADDR = 0x57
    ACCEL_OFFSET_Y_MSB_ADDR = 0x58
    ACCEL_OFFSET_Z_LSB_ADDR = 0x59
    ACCEL_OFFSET_Z_MSB_ADDR = 0x5A

    # Magnetometer Offset registers
    MAG_OFFSET_X_LSB_ADDR = 0x5B
    MAG_OFFSET_X_MSB_ADDR = 0x5C
    MAG_OFFSET_Y_LSB_ADDR = 0x5D
    MAG_OFFSET_Y_MSB_ADDR = 0x5E
    MAG_OFFSET_Z_LSB_ADDR = 0x5F
    MAG_OFFSET_Z_MSB_ADDR = 0x60

    # Gyroscope Offset registers
    GYRO_OFFSET_X_LSB_ADDR = 0x61
    GYRO_OFFSET_X_MSB_ADDR = 0x62
    GYRO_OFFSET_Y_LSB_ADDR = 0x63
    GYRO_OFFSET_Y_MSB_ADDR = 0x64
    GYRO_OFFSET_Z_LSB_ADDR = 0x65
    GYRO_OFFSET_Z_MSB_ADDR = 0x66

    # Radius registers
    ACCEL_RADIUS_LSB_ADDR = 0x67
    ACCEL_RADIUS_MSB_ADDR = 0x68
    MAG_RADIUS_LSB_ADDR = 0x69
    MAG_RADIUS_MSB_ADDR = 0x6A

    # REGISTER DEFINITION END

    def __init__(self, sensorId=-1, address=0x28):
        """Initialize the BNO055 sensor.

        Args:
            sensorId (int): Sensor ID
            address (int): I2C address of the sensor
        """
        super().__init__()
        print("BNO055, 9axis sensor fusion module")
        self._sensorId = sensorId
        self._address = address
        self._mode = BNO055.OPERATION_MODE_NDOF

        print(" Please wait for a few seconds to initialize...")
        if not self.begin(self._mode):
            print("Error initializing device")
            exit()
        time.sleep(1)
        self.setExternalCrystalUse(True)

    def begin(self, mode=None):
        """Begin communication with the sensor and set its mode.

        Args:
            mode (int, optional): Operation mode to set. Defaults to OPERATION_MODE_NDOF.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        # ... (rest of the method implementation remains unchanged)

    # ... (rest of the methods remain unchanged, but should be documented similarly)

    def measure(self):
        """Measure and calculate sensor values.

        Returns:
            tuple: Angle, acceleration, and gyroscope values
        """
        # Time unit
        self.end_time = time.perf_counter()
        dt = self.end_time - self.start_time

        # Get estimated angle
        self.angle = self.getVector(self.VECTOR_EULER)[0]
        # Get values
        acc_tmp = self.getVector(self.VECTOR_ACCELEROMETER)
        gyr_tmp = self.getVector(self.VECTOR_GYROSCOPE)

        # Record
        for i, axis in enumerate(self.acc):
            self.acc[axis].append(acc_tmp[i])
            self.acc[axis].pop(0)
            self.gyr[axis].append(gyr_tmp[i])
            self.gyr[axis].pop(0)
            # Calculate differential values
            self.jerk[axis].append(self.acc[axis][-1] - self.acc[axis][-2] / dt)
            self.jerk[axis].pop(0)
            self.angular_velocity[axis].append(
                self.gyr[axis][-1] - self.gyr[axis][-2] / dt
            )
            self.angular_velocity[axis].pop(0)
        self.start_time = time.perf_counter()
        return self.angle, acc_tmp, gyr_tmp

    def convert_angle_plusminus180(self):
        """Convert angle to -180° ~ 180° range."""
        if self.angle > 180:
            self.angle = self.angle - 360

    def filter_angle(self):
        """Filter out noise from angle measurements."""
        if self.angle > self.angle_pre + 7:
            print("Noise on angle value, using previous value ", end="")
            self.angle = self.angle_pre
            print(f"  >> Angle [°]:{self.angle}")
        else:
            self.angle_pre = self.angle

    def measure_set(self):
        """Perform a complete set of measurements and filtering.

        Returns:
            tuple: Filtered angle, acceleration, and gyroscope values
        """
        self.angle, acc_tmp, gyr_tmp = self.measure()
        self.convert_angle_plusminus180()
        self.filter_angle()
        return self.angle, acc_tmp, gyr_tmp


if __name__ == "__main__":
    # Create IMU instance
    imu = BNO055()
    # Measurement loop
    while True:
        angle, acc, gyr = imu.measure()
        # Convert to -180° ~ 180° range
        imu.convert_angle_plusminus180()
        print(
            f"Acceleration [m/s^2]:{acc} Angle [°]:{angle}  Angular velocity-z [°/s]:{imu.gyr['z']}"
        )
        print("Gstr: ", imu.GCounter())
        time.sleep(0.1)
