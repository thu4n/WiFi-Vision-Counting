#!/usr/bin/env python

import os
import subprocess
from sound_alert import end_sound

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Global variables
KIND = ['0_Raw', '1_Processed']
DATE = '05-10-2024'
TYPE = ['CSI_Packages', 'Image_Frames']
LOCAL_DIRECTORY = ['csi_main_0', 'cv_main']

spec_date = DATE

# Raw Images
spec_kind = KIND[0]
spec_dir = LOCAL_DIRECTORY[1]
spec_type = TYPE[1]

# Raw CSI
# spec_kind = KIND[0]
# spec_dir = LOCAL_DIRECTORY[0]
# spec_type = TYPE[0]

SYNC_DIRS = [
   {
      'src': f"{spec_dir}",  # Update this to your Windows path
      'dst': f"UIT_Graduation_Thesis/Dataset/{spec_kind}/{spec_date}/{spec_type}",
      'log': os.path.join(script_dir, f'sync_{spec_date}_{spec_type}_log.txt')
   }
]
RCLONE_RETRIES = 3
RCLONE_CHECKERS = 4
RCLONE_TRANSFERS = 8
RCLONE_STATS_INTERVAL = '10s'
RCLONE_CHUNK_SIZE = '64M'
RCLONE_UPLOAD_CUTOFF = '512M'

# Synchronize each directory to Google Drive
for sync_entry in SYNC_DIRS:
   # Backup the previous sync log file
   log_file = sync_entry['log']

   #
   # Sync the next directory to Google Drive.
   #
   # Use the rclone 'copy' command instead of 'sync'
   # to prevent accidental deletion on the remote if
   # the source directory is not mounted.
   #
   # A 64M chunk-size is used for performance purposes.
   # Google recommends as large a chunk size as possible.
   # Rclone will use the following amount of RAM at run-time
   # (8MB chunks by default; not high enough)...
   #
   #    RAM = (chunk-size * num-transfers)
   #
   # So our command will use larger chunk sizes (more RAM)...
   #
   #    RAM = 0.5 GB = (64MB * 8 transfers)
   #
   # For more details...
   #
   #    https://github.com/ncw/rclone/issues/397
   #

   # Build the shell command
   cmd = """rclone --verbose \
         --retries {retries} \
         --checkers {checkers} \
         --transfers {transfers} \
         --stats {stats_interval} \
         --drive-chunk-size {chunk_size} \
         --drive-upload-cutoff {upload_cutoff} \
         copy --no-update-modtime {src} ggdrive:{dst}"""

   # Build the command arg values
   args = {
      'retries': RCLONE_RETRIES,
      'checkers': RCLONE_CHECKERS,
      'transfers': RCLONE_TRANSFERS,
      'stats_interval': RCLONE_STATS_INTERVAL,
      'chunk_size': RCLONE_CHUNK_SIZE,
      'upload_cutoff': RCLONE_UPLOAD_CUTOFF,
      'src': sync_entry['src'],
      'dst': sync_entry['dst']
   }
   
   uploaded_file_count = 0  # Track the number of uploaded files

   # Open the log file for writing
   with open(log_file, 'a') as log:
      # Execute the shell command and log the output
      process = subprocess.Popen(cmd.format(**args), 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, 
                                 shell=True)
      for line in process.stdout:
         decoded_line = line.decode()
         print(decoded_line.strip())

         # Check if the line contains 'Copied (New)' (indicates a file upload)
         if 'Copied (new)' in decoded_line:
            uploaded_file_count += 1
            log.write(decoded_line)

      process.wait()  # Wait for the process to finish

   # Print the result summary
   print(f"Files uploaded to Google Drive: {uploaded_file_count}")

end_sound()