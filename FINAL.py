import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
import serial
import pynmea2
from fuzzywuzzy import fuzz
import time
import board
import adafruit_lis3dh
import RPi.GPIO as GPIO

# Code adapted from: 
# Canu, S., n.d. Distance Detection with Depth Camera. [Online] 
# Available at: https://pysource.com/2021/03/11/distance-detection-with-depth-camera-intel-realsense-d435i/ [Accessed 31 May 2022]. 
# agenis, 2021. how to set the minimum possible reading distance, from the python script (L515) #8244. [Online] 
# Available at: https://github.com/IntelRealSense/librealsense/issues/8244 [Accessed 31 May 2022]

class DepthCamera:
    
    def __init__(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()
        # Create config
        config = rs.config()
        
        # Configure pipeline to stream depth and color streams
        # Depth stream resolution has to be lowered if not USB3
        config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)
    
    def min_distance(self):
        # Get active profiles
        profile = self.get_active_profile()
        sensor = profile.get_device().query_sensors()[0] # 0 for depth, 1 for cam
        # Changes the device settings so that distances under 25cm can be detected
        sensor.set_option(rs.option.min_distance, 0)

    def get_frame(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # If no frames are detected, then return None
        if not depth_frame or not color_frame:
            return None, None, None
        # Otherwise return depth_frame, depth_image, color_image
        return depth_frame, depth_image, color_image
    
    def release(self):
        # Stops pipeline stream
        self.pipeline.stop()

# Code adapted from:
# Ansari, A., 2022. Real-Time Edge Detection using OpenCV in Python | Canny edge detection method. [Online] 
# Available at: https://www.geeksforgeeks.org/real-time-edge-detection-using-opencv-python [Accessed 31 May 2022].

def canny_edge_detection(color_image):      
    # Converting BGR to HSV
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    
    # define range of red color in HSV
    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])
    
    # Create a red HSV color boundary and 
    # Threshold HSV image
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(color_image, color_image, mask= mask)
    
    # Finds edges in the input image image and marks them in the output map edges
    return cv2.Canny(color_image, 100, 200)

# Code adapted from:
# OpenCV, n.d. Hough Line Transform. [Online] 
# Available at: https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html [Accessed 31 May 2022].

def hough_lines(edges, color_image, depth_frame):
    lines = cv2.HoughLinesP(
        edges, # Input edge image
        1, # Distance resolution in pixels
        np.pi/2, # Angle resolution in radians, only allows vertical and horizontal lines                
        threshold=50, # Min number of votes for valid line
        minLineLength=200, # Min allowed length of line
        maxLineGap=50 # Max allowed gap between line for joining them
        )
    
    # Finds lines
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            x1 = l[0]
            y1 = l[1]
            x2 = l[2]
            y2 = l[3]
            # Only accepts horizontal lines that appear on the bottom half
            if (y1 - y2 == 0) and (y1 and y2 >= 320):
                return [x1, y1, x2, y2]
    else:      
        return None

def find_x_and_y(distance):
    # Convert to radians
    theta = 59 * np.pi/180
    # Distance is halved, step doesnt need to extend all the way
    # Convert to mm and round
    x = np.round((distance/2) * np.sin(theta)*10**3)
    y = np.round((distance/2) * np.cos(theta)*10**3)
    return x,y

def import_train_geocodes():
    # Import train geocodes
    station_df = pd.read_csv('train_geocodes.csv', usecols=[1,2,3])
    # Converts columns lat and long from object to float64 so maths is possible
    station_df['Lat'] = pd.to_numeric(station_df['Lat'], errors = 'coerce')
    station_df['Long'] = pd.to_numeric(station_df['Long'], errors = 'coerce')
    return station_df

def import_platform_distances():
    # Imports and simplifies excel data
    platform_df = pd.read_excel('T1037 July13 NGD Platforms V03 copy.xlsx', 1, header= 1, names= [ 'Stations', 'X', 'Y'], usecols='C,K,O', nrows=5752)
    return platform_df

# Code adapted from:
# Dunnington, D., 2016. Using PySerial, Pynmea2, and Raspberry Pi to Log NMEA Output. [Online] 
# Available at: https://dewey.dunnington.ca/post/2016/using-pyserial-pynmea2-and-raspberry-pi-to-log-nmea-output/ [Accessed 31 05 2022].
def get_location():
    # Get port information
    port = serial.Serial('COM3', timeout=1)
    # Limit how many sentences are printed
    for i in range(0,14):
        # Read the sentence
        nmea = port.readline().decode('ascii', errors = 'replace')
        # If the sentence contains GPRMC
        if 'GPRMC' in nmea:
            # print(nmea)
            # Parse the sentence
            text = pynmea2.parse(nmea)
            # print(repr(text))
    # Close port
    port.close()
    return text.latitude, text.longitude

def get_station_name(lat, long, station_df):
    # Subtracts the step longitude from each of the station longitudes
    # Result goes into a new DataFrame or series
    long_df = station_df['Long']-long
    # Reduces entries to those between -0.1 and 0.1 and sorts them
    station_names = long_df[(-0.1 <= long_df) & (long_df <= 0.1)].sort_values()
    # Finds the same stations in station_names and gets the lat info
    lat_df = station_df.iloc[station_names.index]['Lat']
    # Substract lat from each entry in lat_df, gets the absolute value and sorts them
    # Gets the first entry of the series, and uses its index to find all the data from station_df
    station_name = station_df.iloc[(lat_df-lat).abs().sort_values()[:1].index]
    # Gets the station name
    station_name = station_name.iloc[0]['Station']
    return station_name

def get_platform_distance(platform_df, station_name):
    # Creates empty array
    arr = np.array([])

    # For each entry in the column stations in the spreadsheet
    for i in platform_df['Stations']:
        # Compare station_name to i and give a score
        score = fuzz.partial_ratio(station_name, i)
        # Append the score to the array
        arr = np.append(arr, score)

    # Find the index of the station that has the highest score
    max_score = np.where(arr == np.amax(arr))[0][0]
    # Find platform offset (X), for the entry that has an index of max score
    # Round the value then halve it
    X = np.around(platform_df['X'][max_score])/2
    # Find platform height (Y), for the entry that has an index of max score
    # Round the value then halve it
    Y = np.around(platform_df['Y'][max_score])/2
    return X, Y

# Calculate max acceleration and velocity according to equations
def am_vm(x):
    theta = 2*np.pi*x/0.003
    s = 0.8125*theta
    vm = s / 3
    am = 2*vm/0.5
    return am, vm

# Calculate velocities for region 1
def one(am, vm):
    # Create empty array
    v = np.asarray([])
    # Create array of time steps
    arr = np.arange(0.05,0.3,0.05).round(2)
    # For every time step
    for i in arr:
        # Calculate velocity from eq
        vel = (am/0.5)*i**2
        # append to array
        v = np.append(v, vel)
    return v

# Calculate velocities for region 2      
def two(am,vm):
    # Create empty array
    v = np.asarray([])
    # Create array of time steps
    arr = np.arange(0.05,0.3,0.05).round(2)
    # For every time step
    for i in arr:
        # Calculate velocity from eq
        vel = ((-am/0.5)*i**2) + (am*i) + (vm/2)
        # append to array
        v = np.append(v, vel)
    return v

# Calculate velocities for region 4  
def four(am,vm):
    # Create empty array
    v = np.asarray([])
    # Create array of time steps
    arr = np.arange(0.05,0.3,0.05).round(2)
    # For every time step
    for i in arr:
        # Calculate velocity from eq
        vel = ((-am/0.5)*i**2) + (vm)
        # append to array
        v = np.append(v, vel)
    return v

# Calculate velocities for region 5        
def five(am,vm):
    # Create empty array
    v = np.asarray([])
    # Create array of time steps
    arr = np.arange(0.05,0.25,0.05).round(2)
    # For every time step
    for i in arr:
        # Calculate velocity from eq
        vel = ((am/0.5)*i**2) - (am*i) + (vm/2)
        # append to array
        v = np.append(v, vel)
    return v

# Determine pulse time period
def period(v):
    # If v is 0, then T is 0
    if v == 0:
        T = 0
    # If v not 0
    else:
        # find pulse frequency
        rpm = (v*60)/(2*np.pi)
        f = (rpm*360)/(0.18*60)
        # Find period
        T = 1/f
    return T

# Initialise camera
camera = DepthCamera()
# Initialise I2C
i2c = board.I2C()
# Set up accelerometer
lis3dh = adafruit_lis3dh.LIS3DH_I2C(i2c)
# Set dwell time
dwell = 10
# GPIO pin for direction output
direction = 16
# GPIO pin for step output
step = 26
# Set pin numbering mode
GPIO.setmode(GPIO.BCM)
# Set as outputs
GPIO.setup(direction, GPIO.OUT)
GPIO.setup(step, GPIO.OUT)
# Turn off run time warnings
GPIO.setwarnings(False)

while True:
    # Set up accelerometer
    x,y,z = lis3dh.acceleration
    # Runs while train is stopped
    if np.absolute(x) <= 0.32 and np.absolute(y) <= 0.32:
        # Get camera frames, needed for functions
        depth_frame, depth_image, color_image = camera.get_frame()
        # Perform canny edge detection on the color_image
        edges = canny_edge_detection(color_image)
        # Array of x1, x2, y1 and y2 obtained from hough transform
        arr = hough_lines(edges, color_image, depth_image)
        # Runs when there is a line
        if arr is not None:
            #Take note of start time
            start = time.time()
            # Draws line according to hough coordinates
            cv2.line(color_image, (arr[0], arr[1]), (arr[2], arr[3]), (0,0,255), 1, cv2.LINE_AA)  
            # Since resolution is halved, y coordinate must scaled for depth camera
            avg_y = (arr[1] // 2) 
            # Scales x like the y coordinate but also averaged to get middle of line
            avg_x = (arr[0]+arr[3])//4
            # Get distance at point avg_x, avg_y using the depth camera
            distance = depth_frame.get_distance(avg_x,avg_y)
            distances = np.empty(10)
            # Only stores suitable distances into the array
            # Needs to be revised, needs a minimum distance
            if distance < 0.375:
                # Array takes 10 inputs
                for i in range(0,10):
                    distances[i] = distance
            # Calculates average of all 10 outputs
            mean_distance = np.mean(distances)           
            # Finds extension distance through function
            x, y = find_x_and_y(mean_distance)
            # Take note of end time
            end = time.time()
            # Print information for testing purposes
            print('lidar:')
            print(x, y)
            print(start, end)
            if x >= 150:
                # Call functions
                am, vm = am_vm(x)
                v1 = one(am, vm)
                v2 = two(am,vm)
                v3 = vm
                v4 = four(am,vm)
                v5 = five(am,vm)
                
                # Create dictionary
                d = {1:v1, 2:v2, 3:v3, 4:v4, 5:v5}
                
                # Turn motor clockwise
                GPIO.output(direction, GPIO.HIGH)
                # Take note of time
                a = time.time()
                # Run through values in dictionary
                for k in range(1,6):
                    # If not region 3:
                    if k != 3:
                        # Create empty array
                        times = np.asarray([])
                        # For every entry in the corresponding array
                        for i in d[k]:
                            # Run function period with argument i 
                            T = period(i)
                            # Append each value of T into the array times
                            times = np.append(times, T)
                        # print('times; ',k, times)
                        # For every entry in times
                        for t in times:
                            # Set how long the loop can run for
                            timeout = time.time() + 0.05
                            # let the loop run for 0.05s
                            while time.time() <= timeout:
                                # Send high for T/2
                                GPIO.output(step, GPIO.HIGH)
                                time.sleep(T/2)
                                # Send low for T/2
                                GPIO.output(step, GPIO.LOW)
                                time.sleep(T/2)
                    # If region 3
                    else:
                        # Constant velocity
                        T = period(d[3])
                        #print('T:', T)
                        #Set how long the loop can run for
                        timeout = time.time() + 2
                        #let the loop run for 2s
                        while time.time() <= timeout:
                            # Send high for T/2
                            GPIO.output(step, GPIO.HIGH)
                            time.sleep(T/2)
                            # Send low for T/2
                            GPIO.output(step, GPIO.LOW)
                            time.sleep(T/2)                
                # Note time
                b = time.time()
                # Wait for however long the dwell time is
                time.sleep(dwell)
                #Note time
                c = time.time()
                # Set direction to anti-clockwise
                GPIO.output(direction, GPIO.LOW)
                # Run through values in dictionary
                for k in range(1,6):
                    # If not region 3:
                    if k != 3:
                        # Create empty array
                        times = np.asarray([])
                        # For every entry in the corresponding array
                        for i in d[k]:
                            # Run function period with argument i 
                            T = period(i)
                            # Append each value of T into the array times
                            times = np.append(times, T)
                        # print('times; ',k, times)
                        # For every entry in times
                        for t in times:
                            # Set how long the loop can run for
                            timeout = time.time() + 0.05
                            # let the loop run for 0.05s
                            while time.time() <= timeout:
                                # Send high for T/2
                                GPIO.output(step, GPIO.HIGH)
                                time.sleep(T/2)
                                # Send low for T/2
                                GPIO.output(step, GPIO.LOW)
                                time.sleep(T/2)
                    # If region 3
                    else:
                        # Constant velocity
                        T = period(d[3])
                        #print('T:', T)
                        #Set how long the loop can run for
                        timeout = time.time() + 2
                        #let the loop run for 2s
                        while time.time() <= timeout:
                            # Send high for T/2
                            GPIO.output(step, GPIO.HIGH)
                            time.sleep(T/2)
                            # Send low for T/2
                            GPIO.output(step, GPIO.LOW)
                            time.sleep(T/2)                
                # Note time
                d = time.time()
                # Print for info
                print(a,b,c,d)
                # Ensures the code only runs once during the duration the train is stopped
                # For testing this has been made to be 3x the dwell time
                time.sleep(3 * dwell)
        # Runs if arr is empty
        elif arr is None:
            # Take note of start time
            start = time.time()
            # get lat and long of step
            #lat, long = get_location()
            #Coordinates of bagshot station
            lat = 51.36446
            long = -0.68871
            lat = np.round(lat, 4)
            long = np.round(long, 4)
            # Imports all station lat long
            station_df = import_train_geocodes()
            # Imports RSSB spreadsheet
            platform_df = import_platform_distances()
            # Finds station name using GPS output
            station_name = get_station_name(lat, long, station_df)
            # Uses station name to get platform data from RSSB
            x, y = get_platform_distance(platform_df, station_name)
            # Take note of end time
            end = time.time()
            # Print info for testing
            print('table:')
            print(x,y)
            print(start, end)
            if x >= 150:
                # Call functions
                am, vm = am_vm(x)
                v1 = one(am, vm)
                v2 = two(am,vm)
                v3 = vm
                v4 = four(am,vm)
                v5 = five(am,vm)
                
                # Create dictionary
                d = {1:v1, 2:v2, 3:v3, 4:v4, 5:v5}
                
                # Motor direction: clockwise
                GPIO.output(direction, GPIO.HIGH)
                # Note time
                a = time.time()
                # Run through values in dictionary
                for k in range(1,6):
                    # If not region 3:
                    if k != 3:
                        # Create empty array
                        times = np.asarray([])
                        # For every entry in the corresponding array
                        for i in d[k]:
                            # Run function period with argument i 
                            T = period(i)
                            # Append each value of T into the array times
                            times = np.append(times, T)
                        # print('times; ',k, times)
                        # For every entry in times
                        for t in times:
                            # Set how long the loop can run for
                            timeout = time.time() + 0.05
                            # let the loop run for 0.05s
                            while time.time() <= timeout:
                                # Send high for T/2
                                GPIO.output(step, GPIO.HIGH)
                                time.sleep(T/2)
                                # Send low for T/2
                                GPIO.output(step, GPIO.LOW)
                                time.sleep(T/2)
                    # If region 3
                    else:
                        # Constant velocity
                        T = period(d[3])
                        #print('T:', T)
                        #Set how long the loop can run for
                        timeout = time.time() + 2
                        #let the loop run for 2s
                        while time.time() <= timeout:
                            # Send high for T/2
                            GPIO.output(step, GPIO.HIGH)
                            time.sleep(T/2)
                            # Send low for T/2
                            GPIO.output(step, GPIO.LOW)
                            time.sleep(T/2)                
                # Note time
                b = time.time()
                # Wait for however long the dwell time is
                time.sleep(dwell)
                # Note time
                c = time.time()
                # Set direction to anti-clockwise
                GPIO.output(direction, GPIO.LOW)
                # Run through values in dictionary
                for k in range(1,6):
                    # If not region 3:
                    if k != 3:
                        # Create empty array
                        times = np.asarray([])
                        # For every entry in the corresponding array
                        for i in d[k]:
                            # Run function period with argument i 
                            T = period(i)
                            # Append each value of T into the array times
                            times = np.append(times, T)
                        # print('times; ',k, times)
                        # For every entry in times
                        for t in times:
                            # Set how long the loop can run for
                            timeout = time.time() + 0.05
                            # let the loop run for 0.05s
                            while time.time() <= timeout:
                                # Send high for T/2
                                GPIO.output(step, GPIO.HIGH)
                                time.sleep(T/2)
                                # Send low for T/2
                                GPIO.output(step, GPIO.LOW)
                                time.sleep(T/2)
                    # If region 3
                    else:
                        # Constant velocity
                        T = period(d[3])
                        #print('T:', T)
                        #Set how long the loop can run for
                        timeout = time.time() + 2
                        #let the loop run for 2s
                        while time.time() <= timeout:
                            # Send high for T/2
                            GPIO.output(step, GPIO.HIGH)
                            time.sleep(T/2)
                            # Send low for T/2
                            GPIO.output(step, GPIO.LOW)
                            time.sleep(T/2)
                # Note time            
                d = time.time()
                # Print for info
                print(a,b,c,d)
                # Ensures the code only runs once during the duration the train is stopped
                # For testing this has been made to be 3x the dwell time
                time.sleep(3 * dwell)

