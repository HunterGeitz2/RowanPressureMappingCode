# Emerson Pummill (Modified by Brandon Dang)
# 2/14/2025

# This program is used to collect data from our prototype.
# Ensure that there is a "slopes.csv" file in this folder... this will be verification that the sensors have been calibrated.
# Both the number of sensors and their locations are variable, so these values must be confirmed before running this program:

# Number of sensors
#n =  

# Location of sensors
# Starting at the top left corner of the board and working from left -> right then down
# Each location should have a label, use these values to fill out the array, and add 0x4 before the middle value:
# EX. if we have two sensors at 4xAx0 and 5X8x3, the array should be:
# locations = [
# [4, 0x4A, 0], [5, 0x48, 3]
# ]

locations = [
    [1,0x4A,0],[1,0x4A,1],[1,0x4B,0],[1,0x4B,1],[1,0x48,0],[1,0x48,1],
    [1,0x4A,2],[1,0x4A,3],[1,0x4B,2],[1,0x4B,3],[1,0x48,2],[1,0x48,3],
    [1,0x49,0],[1,0x49,1],[3,0x48,0],[3,0x48,1],[2,0x4A,0],[2,0x4A,1],
    [1,0x49,2],[1,0x49,3],[3,0x48,2],[3,0x48,3],[2,0x4A,2],[2,0x4A,3],
    [2,0x4B,0],[2,0x4B,1],[2,0x48,0],[2,0x48,1],[2,0x49,0],[2,0x49,1],
    [2,0x4B,2],[2,0x4B,3],[2,0x48,2],[2,0x48,3],[2,0x49,2],[2,0x49,3]
    #[1,0x49,3]
]


import numpy as np            
import pandas as pd
import time
import csv
import board
import adafruit_tca9548a
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from scipy import signal
from pandas import read_csv

#collect data
def collect_data(chan, add, p) :
    val = AnalogIn(ADS.ADS1115(tca[chan], address=add), eval(f"ADS.P{p}")).value
    #print(val)
    return val


                                                                                                                                                                                                                                                                                                 # Create I2C bus as normal
i2c = board.I2C()

# Create the TCA9548A object and give it the I2C bus
tca = adafruit_tca9548a.TCA9548A(i2c)
    
# Create the lowpass filter
b,a = signal.butter(1,150/(860/2),'low')

#collect data from all sensors
values = []
all_values = []
temp = []
cal_data = np.array(read_csv('//home//pi//Desktop//prototype//Calibration Data.csv'))
#matrix = np.zeros((6,6))
#Take & store the original value of each sensor
for i, loc in enumerate(locations):
    temp.append(collect_data(loc[0], loc[1], loc[2]))

#Take the inital, store it, use it in collect_data
while True:
    try:
        values = []
        for i, loc in enumerate(locations):
            val = collect_data(loc[0], loc[1], loc[2])
            # Convert raw value to force using baseline
            #force = (val - temp[i]) / (-28.078)
            #force = 5*(np.sqrt(82905393489-5502400*(val - temp[i]))-24617) / (27512)-47.5 #THIS ONE IS AN ABRITRATY FORCE CURVE, ONLY FOR DEBUGGING
            force = (val-cal_data[i,0])/((cal_data[i,1]-cal_data[i,0])/cal_data[i,2])
            if force < 0:
                force = 0
            #force = val
            #print(i)    
            values.append(round(force,3))
            
        print(values)
        # Convert flat list to 6×6 matrix
        matrix = np.array(values).reshape(6, 6)

        # ✅ Save matrix to 6x6_data.csv for Flask to read
        np.savetxt("/home/pi/Desktop/prototype/6x6_data.csv", matrix, delimiter=",")

        #print("Matrix written to 6x6_data.csv:")
        print(matrix)

        time.sleep(.1)  # delay between readings

    except KeyboardInterrupt:
        print("Stopping sensor data collection.")
        break
