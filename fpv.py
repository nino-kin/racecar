#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from flask import Flask, render_template, Response
from markupsafe import escape
import cv2
import camera_multiprocess

app = Flask(__name__)
data_sh = []

# "/" を呼び出したときには、indexが表示される。
@app.route('/')
def index():
    """
    Route handler for the root URL ("/").
    Renders the index.html template.

    Returns:
        str: Rendered HTML content
    """
    return render_template('index.html')

def gen(camera):
    """
    Generator function for video streaming.

    Args:
        camera (VideoCamera): Instance of VideoCamera class

    Yields:
        bytes: JPEG image data for video streaming
    """
    global data_sh
    while True:
        frame = camera.get_frame_multi()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# returnではなくジェネレーターのyieldで逐次出力。
# Generatorとして働くためにgenとの関数名にしている
# Content-Type（送り返すファイルの種類として）multipart/x-mixed-replace を利用。
# HTTP応答によりサーバーが任意のタイミングで複数の文書を返し、紙芝居的にレンダリングを切り替えさせるもの。

@app.route('/video_feed')
def video_feed():
    """
    Route handler for video feed.
    Returns a Response object with the video stream.

    Returns:
        Response: Flask Response object with video stream data
    """
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run(*args, **kwargs):
    """
    Run the Flask application.

    Args:
        *args: Variable length argument list
        **kwargs: Arbitrary keyword arguments
    """
    global data_sh
    data_sh.append(args)
    print("print data_sh:", data_sh)
    app.run(**kwargs)

class VideoCamera(object):
    """
    Video camera class for capturing and processing video frames.
    """

    def __init__(self):
        """
        Initialize the VideoCamera object.
        """
        #self.video = cv2.VideoCapture(0)
        self.video = camera_multiprocess.VideoCaptureWrapper(0)

    def __del__(self):
        """
        Release the video capture resource.
        """
        self.video.release()

    def get_frame(self):
        """
        Capture a frame from the video feed and return it as JPEG bytes.

        Returns:
            bytes: JPEG encoded image data
        """
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

        # read()は、二つの値を返すので、success, imageの2つ変数で受けています。
        # OpencVはデフォルトでは raw imagesなので JPEGに変換
        # ファイルに保存する場合はimwriteを使用、メモリ上に格納したい時はimencodeを使用
        # cv2.imencode() は numpy.ndarray() を返すので .tobytes() で bytes 型に変換
    def get_frame_multi(self):
        """
        Capture a frame from the video feed and return it as JPEG bytes.
        This method is used for multiprocessing.

        Returns:
            bytes: JPEG encoded image data
        """
        success, image = self.video.read()
        #image = cv2.resize(image, (160, 120))
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

if __name__ == '__main__':
    # 0.0.0.0はすべてのアクセスを受け付けます。
    app.run(host='0.0.0.0', debug=True)
