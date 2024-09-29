#!/usr/bin/python3
import os
import re

def remove_first_line(file_path):
    # remove first line from csv file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    with open(file_path, 'w') as f:
        f.writelines(lines[1:])

def remove_last_line(file_path):
    # remove last line from csv file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    with open(file_path, 'w') as f:
        f.writelines(lines[:-1])

def add_header(file_path, header):
    # add header to csv file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    with open(file_path, 'w') as f:
        f.write(header + "\n")
        f.writelines(lines)

def remove_error_lines(file):
    # remove line that has commar split less than 27
    with open(file, 'r') as f:
        lines = f.readlines()
    with open(file, 'w') as f:
        for line in lines:
            if len(line.split(",")) == 27:
                f.write(line)

def print_error_lines(file):
    # check if line has '--', if yes, print the line
    with open(file, 'r') as f:
        lines = f.readlines()
    err=0
    for line in lines:
        if "--" in line:
            err+=1
            print(line)
    print("Total error lines: ", err)

def fix_error_lines(file):
    # check if line has '--', if yes, replace with '-'
    with open(file, 'r') as f:
        lines = f.readlines()
    fix=0
    with open(file, 'w') as f:
        for line in lines:
            if "--" in line:
                line = line.replace("--", "-")
                fix+=1
            f.write(line)
    print("Total fixed lines: ", fix)

def remove_column(oldfile, newfile):
    # remove last column from csv file, copy to new file
    with open(oldfile, 'r') as f:
        lines = f.readlines()
    with open(newfile, 'w') as f:
        for line in lines:
            f.write(','.join(line.split(',')[:-1]) + '\n')

def regex_solution(file):
    # count lines with regex '\[(.*)\]' in line
    with open(file, 'r') as f:
        lines = f.readlines()
    count=0
    for line in lines[1:]:
        if re.search(r'\[(.*)\]', line) == None:
            print(f"Index: {lines.index(line)}", line)
            count+=1
    print("Total lines with regex: ", count)


# given folder path
folder_path = "csi_dev_0"

# list all files
fl = os.listdir(folder_path)

# header for csv files
hd = "type,role,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,real_time_set,real_timestamp,len,CSI_DATA,machine_timestamp"

f = "p1_0_people.csv"
fp = folder_path + "/" + f

print(regex_solution(fp))