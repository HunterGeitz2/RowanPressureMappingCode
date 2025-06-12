import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from pandas import read_csv

fs = 860
data = read_csv('//home//pi//Desktop//prototype//data.csv')
data_filtered = read_csv('//home//pi//Desktop//prototype//PlotResults.py')

n = len(data) # total number of samples
T = n/fs        # seconds
t = np.linspace(0, T, n, endpoint=False)

data_filtered = np.array(data_filtered)
data = np.array(data)
filtered_avg = []
data_avg = []
for i in range(len(y)):
    temp = y[i] - np.average(filtered)
    filtered_avg.append(temp)
for i in range(len(data)):
    temp = data[i] - np.average(data)
    data_avg.append(temp)
plt.subplot(2, 1, 1)
plt.plot(t, data, 'b-', label='data')
#plt.subplot(3, 1, 3)
plt.plot(t, data_filtered, 'g-', label='filtered data')
plt.title("Filter Results")
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)

plt.subplot(2, 1, 2)
plt.plot(t, data_avg, 'b-', label='data')
#plt.subplot(3, 1, 3)
plt.plot(t, filtered_avg, 'g-', label='filtered data')
plt.title("Overlayed Outputs")
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()