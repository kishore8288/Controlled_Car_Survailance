from flask import Flask, Response

import cv2

from picamera2 import Picamera2

import time



app = Flask(__name__)



# Initialize the camera

picam = None



def initialize_camera():

    global picam

    try:

        picam = Picamera2()

        picam.start()

        time.sleep(2)  # Give the camera some time to adjust

    except Exception as e:

        print(f"Failed to initialize the camera: {e}")



initialize_camera()



def generate():

    while True:

        try:

            # Capture an image

            frame = picam.capture_array()



            # Convert the color from RGB to BGR

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)



            # Encode the frame as JPEG

            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()



            # Yield the frame in a format suitable for streaming

            yield (b'--frame\r\n'

                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:

            print(f"Error capturing image: {e}")

            break



@app.route('/video_feed')

def video_feed():

    return Response(generate(),

                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/')

def index():

    return "Video streaming is running. Go to /video_feed to see the live feed."



if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)
