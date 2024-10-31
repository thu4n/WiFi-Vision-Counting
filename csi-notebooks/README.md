# Guide

Make sure your data format is according to the ESP32 CSI Tool (add link later)

Run the `0_Raw_CSI_Process.ipynb` first to get the amplitude values as well as null filter. Then, run the `1_Amp_CSI_Process.ipynb` to apply segmentation for your data into 600-packet bundle of CSI. Lastly is feature extraction for each bundle in `2_CSI_Feature_Extraction.ipynb`.

To-do: Add more comments.