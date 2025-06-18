Low resolution array that osu will be sending data from is a 6x6 arrary matrix and data is saved as a csv where each line is a 36 array of data representing each value recorded from the 6x6 sensors. 
The high resolution or commercial array is 32x32 (will get this size confirmed) and the data will also likely be saved as a csv with each line being a single array of all the sensor values.
We preferrably want as much data to train the model as possible, the information above will be needed for generating our own data for training the model. 
OSU will send multiple csv's each one representing a different person, and will also involved the person moving around (this will be shown in the hundreds of frames).
