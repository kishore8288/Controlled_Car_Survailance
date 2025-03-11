from flask import Flask, render_template, request
import RPi.GPIO as GPIO
import time
import threading
import Adafruit_DHT

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.models as models

Transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])

# Loading Model :
efficientnet = models.efficientnet_b0(pretrained = True)
num_features = efficientnet.classifier[1].in_features
efficientnet.classifier[1] = nn.Linear(num_features,32)
efficientnet.classifier.fc = nn.Linear(32,7)

efficientnet.load_state_dict(torch.load("/home/pi/Desktop/efficientnet_weights_64.pth"))

d = d = {'Crown gall(Agrobacterium tumefacins)':[['Remove infected plants, avoid injuring stems, and sterilize pruning tools.'],['Copper-based sprays, Agrobacterium radiobacter (a biological control agent)']],
     'Healthy' : [['Congrats'],['No further assistance needed']],
     'Rust(phragmidium species)':[['Prune infected parts, avoid wetting foliage, and use fungicides if necessary.'],['Copper-based sprays, sulfur fungicides, or mancozeb.']],
     'black rust':[['Remove infected leaves, improve air circulation, and use fungicides.'],['Neem oil, copper-based sprays, chlorothalonil, mancozeb, or myclobutanil.']],
     'botrysis blight':[['Remove dead flowers, ensure proper spacing, and use fungicides if needed.'],['Captan, thiophanate-methyl, or chlorothalonil.']],
     'phodosphera ponnosa':[['Ensure good airflow, avoid overhead watering, and apply fungicides.'],['Sulfur-based sprays, potassium bicarbonate, myclobutanil, or triforine.']],
     'roserosette':[['Remove and destroy infected plants; control mite populations.'],['Horticultural oils, insecticidal soap, or abamectin-based miticides.']]
    }

diseases = list()
for disease in d.keys():
    diseases.append(disease)

# Live stream Runnning :
url = 'http://192.168.214.142:5000/video_feed'

cam = cv.VideoCapture(url)

app = Flask(__name__)

# GPIO pin setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor pins for the bot
IN1 = 17
IN2 = 18
IN3 = 22
IN4 = 23
ENA = 27
ENB = 24

# Servo motor pin
SERVO_PIN = 5

# Moisture sensor pin
MOISTURE_SENSOR_PIN = 4

# DHT11 sensor setup
DHT_SENSOR = Adafruit_DHT.DHT11
DHT_PIN = 21  # GPIO pin for DHT11

# PWM frequency for the servo
PWM_FREQ = 50

# Duty cycle values for the servo
DUTY_CYCLE_0 = 2.5  # 0 degrees
DUTY_CYCLE_180 = 7.5  # 180 degrees

# Speed control for the servo (smaller step = slower movement)
STEP_DELAY = 0.1  # Time delay between each step (in seconds)
STEP_SIZE = 0.1  # Increment in duty cycle per step

# Set up GPIO pins for the bot
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Set up moisture sensor pin
GPIO.setup(MOISTURE_SENSOR_PIN, GPIO.IN)

# Set up PWM for the bot motors
pwm_a = GPIO.PWM(ENA, 100)  # 100 Hz frequency
pwm_b = GPIO.PWM(ENB, 100)
pwm_a.start(0)  # Start with 0% duty cycle
pwm_b.start(0)

# Set up PWM for the servo motor
servo_pwm = GPIO.PWM(SERVO_PIN, PWM_FREQ)
servo_pwm.start(DUTY_CYCLE_0)  # Start at 0 degrees

# Bot control functions
def forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(50)
    pwm_b.ChangeDutyCycle(50)

def backward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(50)
    pwm_b.ChangeDutyCycle(50)

def left():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(20)
    pwm_b.ChangeDutyCycle(20)

def right():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(20)
    pwm_b.ChangeDutyCycle(20)

def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

# Servo control function
def move_servo_slowly(start_duty, end_duty):
    if start_duty < end_duty:
        step = STEP_SIZE  # Move forward
    else:
        step = -STEP_SIZE  # Move backward

    for duty in range(int(start_duty * 10), int(end_duty * 10), int(step * 10)):
        servo_pwm.ChangeDutyCycle(duty / 10)
        time.sleep(STEP_DELAY)

# Moisture sensor callback function
def moisture_callback(channel):
    if GPIO.input(channel):
        print("Water Detected!")
    else:
        print("No Water Detected!")

# Set up moisture sensor event detection
GPIO.add_event_detect(MOISTURE_SENSOR_PIN, GPIO.BOTH, bouncetime=300)
GPIO.add_event_callback(MOISTURE_SENSOR_PIN, moisture_callback)

# DHT11 sensor reading function
def read_dht_sensor():
    while True:
        humidity, temperature = Adafruit_DHT.read(DHT_SENSOR, DHT_PIN)
        if humidity is not None and temperature is not None:
            print(f"Temperature: {temperature:.1f}°C, Humidity: {humidity:.1f}%")
        else:
            print("Failed to retrieve data from DHT11 sensor. Retrying...")
        time.sleep(2)  # Wait 2 seconds before the next reading

# Web interface routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    ret, frame = cam.read()
    if ret:
        frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = Transform(frame)
        frame = frame.unsqueeze(0)
        output = efficientnet(frame)
        pred = torch.argmax(output,1)
        disease = diseases[pred.item()]
        frame = cv.putText(frame,disease,(10,60),cv.FONT_HARSHEY_SCRIPT_SIMPLEX,2,(0,0,0),2,cv.LINE_AA)
        cv.imshow("Frame",frame)
        cv.waitKey(2000)
        print('Image captured successfully!')
        print(f"Status : {disease} \n Pesticides Needed : {d[disease][1]} \n Suggestions : {d[disease][0]}")
    else:
        return 'Failed to capture image.'

@app.route('/control', methods=['POST'])
def control():
    command = request.form['command']
    if command == 'forward':
        forward()
    elif command == 'backward':
        backward()
    elif command == 'left':
        left()
    elif command == 'right':
        right()
    elif command == 'stop':
        stop()
    elif command == 'servo_180':
        move_servo_slowly(DUTY_CYCLE_0, DUTY_CYCLE_180)
    elif command == 'servo_0':
        move_servo_slowly(DUTY_CYCLE_180, DUTY_CYCLE_0)
        time.sleep(5)  # Stay at 0 degrees for 5 seconds
    return ('', 204)

# Cleanup function
def cleanup():
    pwm_a.stop()
    pwm_b.stop()
    servo_pwm.stop()
    GPIO.cleanup()

# Run moisture sensor in a separate thread
def moisture_sensor_loop():
    while True:
        time.sleep(0.1)  # Small delay to avoid CPU overload

if __name__ == '__main__':
    try:
        # Start moisture sensor thread
        moisture_thread = threading.Thread(target=moisture_sensor_loop)
        moisture_thread.daemon = True  # Daemonize thread to exit when the main program exits
        moisture_thread.start()

        # Start DHT11 sensor thread
        dht_thread = threading.Thread(target=read_dht_sensor)
        dht_thread.daemon = True  # Daemonize thread to exit when the main program exits
        dht_thread.start()

        # Start Flask app
        app.run(debug = True,host='192.168.214.124', port=5000)
    except KeyboardInterrupt:
        cleanup()
