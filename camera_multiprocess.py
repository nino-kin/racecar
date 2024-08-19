#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Capture camera data."""

from __future__ import annotations

import ctypes
import multiprocessing
import multiprocessing.sharedctypes
import multiprocessing.synchronize
import signal
from time import perf_counter
from typing import cast

import cv2
import numpy as np


def _update(
    args: tuple,
    buffer: ctypes.Array[ctypes.c_uint8],
    ready: multiprocessing.synchronize.Event,
    cancel: multiprocessing.synchronize.Event,
) -> None:
    """Function responsible for video capture and buffer updating.

    Args:
        args (tuple): Arguments for video capture
        buffer (ctypes.Array[ctypes.c_uint8]): Shared memory buffer
        ready (multiprocessing.synchronize.Event): Event indicating readiness
        cancel (multiprocessing.synchronize.Event): Event indicating cancellation

    Raises:
        IOError: If opening the capture device fails
    """
    # Set up signal handler for Ctrl+C during capture
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Open capture device
    video_capture = cv2.VideoCapture(*args)
    if not video_capture.isOpened():
        raise IOError()

    try:
        # Capture loop
        while not cancel.is_set():
            # Get frame
            ret, img = cast("tuple[bool, cv2.Mat]", video_capture.read())
            if not ret:
                continue

            # Prepare for buffer update
            ready.clear()
            memoryview(buffer).cast("B")[:] = memoryview(img).cast("B")[:]
            ready.set()

    finally:
        # Release capture device
        video_capture.release()


def _get_information(args: tuple) -> None:
    """Function responsible for getting video information.

    Args:
        args (tuple): Arguments for video capture

    Returns:
        tuple: Frame shape (height, width, channels)

    Raises:
        IOError: If opening the capture device fails
    """
    # Open capture device
    video_capture = cv2.VideoCapture(*args)
    if not video_capture.isOpened():
        raise IOError()

    try:
        # Get the first frame
        ret, img = cast("tuple[bool, cv2.Mat]", video_capture.read())
        if not ret:
            raise IOError()

        # Return frame shape
        return img.shape

    finally:
        # Release capture device
        video_capture.release()


class VideoCaptureWrapper:
    """A wrapper class for video capture."""

    def __init__(self, *args) -> None:
        """Initialize VideoCaptureWrapper.

        Args:
            *args: Arguments for video capture
        """
        self.currentframe = None

        # Get capture device information
        self.__shape = _get_information(args)
        height, width, channels = self.__shape

        # Create shared buffer
        self.__buffer = multiprocessing.sharedctypes.RawArray(ctypes.c_uint8, height * width * channels)

        # Create synchronization events
        self.__ready = multiprocessing.Event()
        self.__cancel = multiprocessing.Event()

        # Start capture process
        self.__enqueue = multiprocessing.Process(
            target=_update,
            args=(args, self.__buffer, self.__ready, self.__cancel),
            daemon=True,
        )
        self.__enqueue.start()

        self.__released = cast(bool, False)

    def read(self):
        """Get a frame.

        Returns:
            tuple: (bool, np.ndarray) Success flag and frame image
        """
        self.__ready.wait()
        self.currentframe = np.reshape(self.__buffer, self.__shape).copy()
        return cast(bool, True), self.currentframe

    def release(self) -> None:
        """Release the capture."""
        if self.__released:
            return

        self.__cancel.set()
        self.__enqueue.join()
        self.__released = True

    def __del__(self):
        """Destructor."""
        try:
            self.release()
        except:
            pass

    def save(self, img, ts, steer, throttle, image_dir):
        """Save an image.

        Args:
            img (np.ndarray): Image to save
            ts (float): Timestamp
            steer (float): Steering value
            throttle (float): Throttle value
            image_dir (str): Directory to save the image

        Returns:
            np.ndarray: Saved image
        """
        try:
            cv2.imwrite(
                image_dir + "/" + str(ts)[:13] + "_" + str(steer) + "_" + str(throttle) + ".jpg",
                img,
            )
            return img
        except:
            print("Cannot save image!")
            pass


if __name__ == "__main__":
    print(" Note: If the camera is already running with red LED on, resource busy, please restart!\n")
    # Create VideoCaptureWrapper using camera
    video_capture = VideoCaptureWrapper(0)
    _, img = video_capture.read()
    img = cv2.resize(img, (160, 120))

    try:
        # Main loop
        while True:
            start_time = perf_counter()
            _, img = video_capture.read()
            print("fps:" + str(round(1 / (perf_counter() - start_time))))
            try:
                pass
            # In case of no local screen output
            except:
                pass

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt, camera stopping.")
        pass

    # Release capture
    video_capture.release()
