import os
import time

PI_USER = "pi"
PI_IP = "169.254.75.9"
PI_CSV_PATH = "/home/pi/Desktop/prototype/6x6_data.csv"

# 👇 Update this path to where your app.py is
#LOCAL_CSV_PATH = r"C:\Users\Atrc2\OneDrive\Desktop\Pressure_Sensing\OSU-Pressure-Mapping-Research-Code\Realtime_UI_SRGAN\6x6_data.csv"
LOCAL_CSV_PATH = r"C:\prototype-code\OSU-Pressure-Mapping-Research-Code\Realtime_UI_SRGAN"

while True:
    print("⏳ Syncing 6x6_data.csv from Raspberry Pi...")
    result = os.system(f"scp {PI_USER}@{PI_IP}:{PI_CSV_PATH} \"{LOCAL_CSV_PATH}\"")
    
    if result == 0:
        print(" Synced successfully!")
    else:
        print(" Sync failed. Check SSH or Pi status.")
    
    time.sleep(1)  # Sync every second
AS