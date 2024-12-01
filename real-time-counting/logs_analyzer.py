import re
import os
import numpy as np

def parse_log_line(line):
    """Parses a log line and extracts relevant information.

    Args:
        line: The log line to parse.

    Returns:
        A tuple containing:
            - Timestamp
            - Event type
            - CPU usage (if available)
            - Memory usage (if available)
            - Other relevant information (e.g., CV count, CSI count)
    """

    timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line)
    if timestamp_match:
        timestamp = timestamp_match.group(1)

    event_type_match = re.search(r"- INFO - (.*)", line)
    if event_type_match:
        event_type = event_type_match.group(1)

    cpu_usage_match = re.search(r"CPU usage: (\d+\.\d+)%", line)
    if cpu_usage_match:
        cpu_usage = float(cpu_usage_match.group(1))
    else:
        cpu_usage = None

    memory_usage_match = re.search(r"Memory usage: (\d+\.\d+)%", line)
    if memory_usage_match:
        memory_usage = float(memory_usage_match.group(1))
    else:
        memory_usage = None

    cv_count_match = re.search(r"CV count: (\d+)", line)
    if cv_count_match:
        cv_count = int(cv_count_match.group(1))
    else:
        cv_count = None

    csi_count_match = re.search(r"CSI count: (\d+\.\d+)", line)
    if csi_count_match:
        csi_count = float(csi_count_match.group(1))
    else:
        csi_count = None

    combined_coount_match = re.search(r"Combined CV \+ CSI count: (\d+\.\d+)", line)
    if combined_coount_match:
        combined_count = float(combined_coount_match.group(1))
    else:
        combined_count = None

    return timestamp, event_type, cpu_usage, memory_usage, cv_count, csi_count, combined_count

log_data_list = []

directory_path = "logs/"
for filename in os.listdir(directory_path):
    filepath = os.path.join(directory_path, filename)
    if os.path.isfile(filepath):
        log_data  = {
                "filename": filename,
                "timestamps": [],
                "event_types": [],
                "cpu_usages": [],
                "memory_usages": [],
                "cv_count": [],
                "csi_count": [],
                "combined_count": []
        }

        with open(filepath, 'r') as file:
            for line in file:
                timestamp, event_type, cpu_usage, memory_usage, cv_count, csi_count, combined_count = parse_log_line(line)
                log_data["timestamps"].append(timestamp)
                log_data["event_types"].append(event_type)
                log_data["cpu_usages"].append(cpu_usage)
                log_data["memory_usages"].append(memory_usage)
                if cv_count is not None:
                    log_data["cv_count"].append(cv_count)
                if csi_count is not None:
                    log_data["csi_count"].append(csi_count)
                if combined_count is not None:
                    log_data["combined_count"].append(combined_count)

        log_data_list.append(log_data)

true_count = 0
prefix = "Normal Environment"

for log_data in log_data_list:
    if true_count > 5:
        true_count = 0
        prefix = "Dark Environment"

    total_count = len(log_data["combined_count"])
    print(f"===== {prefix} - {true_count} People Count - Total Prediction: {total_count} =====")

    cv_count_array = np.array(log_data["cv_count"])
    csi_count_array = np.array(log_data["csi_count"])
    combined_count_array = np.array(log_data["combined_count"])

    mae_csi_count = np.median(np.abs(csi_count_array - true_count))
    mae_combined_count = np.median(np.abs(combined_count_array - true_count))

    cv_wrong_count = cv_count_array[cv_count_array != true_count]
    combined_count_rounded_array = np.round(combined_count_array)
    combined_wrong_count = combined_count_rounded_array[combined_count_rounded_array != true_count]

    print("Wrong Prediction (Vision Only): ",len(cv_wrong_count))
    print("Wrong Prediction (Combined): ", len(combined_wrong_count))
    print("MAE CSI Count: ", mae_csi_count)
    print("MAE Combined Count: ", mae_combined_count)

    true_count += 1