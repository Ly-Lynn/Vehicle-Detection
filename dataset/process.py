import os
import shutil

# Define paths
daytime = r'D:\codePJ\BKAI\Track1\train\daytime'
nighttime = r'D:\codePJ\BKAI\Track1\train\nighttime'
new_day = r'D:\codePJ\BKAI\Track1\train\new_day'

# Get lists of files
day_list = os.listdir(daytime)
night_list = os.listdir(nighttime)

# Ensure the new_day directory exists
os.makedirs(new_day, exist_ok=True)

# Move files from daytime to new_day if not in night_list
for file in day_list:
    if file not in night_list:
        src_path = os.path.join(daytime, file)
        dest_path = os.path.join(new_day, file)
        shutil.move(src_path, dest_path)
        print(f"Moved: {file} to {new_day}")
