#!/usr/bin/env python3

# TODO convert_to_csv() combines data from different topics to a single csv, and adds ground truth. This is experiment
#  dependent at the moment and you will likely want to change it.

import shutil
import os
from bagpy import bagreader
import logging
from sensor_data import SensorData
import pandas as pd
import re

class ProcessBag:
    def __init__(self, base_path = os.getcwd(),
                 source_dir_name = 'test',
                 destination_dir_name = 'test_process_bag',
                 topic_list = ['/hub_0/sensor_0','/hub_0/sensor_1']):
        self.source_dir = base_path + '\\' + source_dir_name
        self.destination_dir = base_path + '\\' + 'processed_data' + '\\' + destination_dir_name
        self.topic_list = topic_list
        self.run()

    def run(self):
        # Make the destination dir if it doesn't exist.
        if not os.path.exists(self.destination_dir):
            os.makedirs(self.destination_dir)
        os.chdir(self.source_dir)
        # For every filename string in the source directory.
        for i, filename in enumerate(os.listdir(self.source_dir)):
            os.chdir(self.source_dir)
            if filename.endswith(".bag"):
                self.convert_to_csv(filename)
            else:
                continue

    def convert_to_csv(self,filename):
        """Converts bag files to csv files and moves to destination dir."""
        b = bagreader(filename)
        # Save a csv file for each topic in the topic list inside a new folder.
        for topic in self.topic_list:
            # Create a csv from the topic name.
            csv_filename = b.message_by_topic([topic])
        # TODO combine_tactile_sensor_data() adds ground truth values, but this is experiment dependent at the moment.
        self.combine_tactile_sensor_data(csv_filename)
        self.move_files(filename)

    def move_files(self,filename):
        filename_base = filename.removesuffix('.bag')
        topic_folder_path = self.source_dir + '\\' + filename_base
        # Move the folder to a the destination folder.
        try:
            _ = shutil.move(topic_folder_path, self.destination_dir)
        except Exception as e:
            logging.error('{}\nFolder or file not moved.'.format(e))
        try:
            _ = shutil.move(self.source_dir + '\\' + filename, self.destination_dir + '\\' + filename_base)
        except:
            pass

    def combine_tactile_sensor_data(self, csv_filename):
        """Combine the sensor data into a single csv, and add ground truth values."""
        csv_filename = csv_filename.split("/")
        self.csv_filename_path = os.getcwd() + '\\' + csv_filename[0]
        sensor_data = SensorData()
        sensor_data.prepare_data(self.csv_filename_path)

        self.csv_list = []
        self.add_colm_headers(camera_ground_truth = True)
        self.add_sensor_data(sensor_data,camera_ground_truth = True)

    def find_pitch_roll(self):
        roll = float(re.findall("roll_.\d*",self.csv_filename_path)[0][5:])
        pitch = float(re.findall("pitch_.\d*",self.csv_filename_path)[0][6:])
        return roll,pitch

    def add_sensor_data(self,sensor_data, camera_ground_truth = True):
        self.ground_truth_index = 0
        # Gather data
        sensor0_data = sensor_data.tactile_sensor_0_data.timeseries_data
        sensor1_data = sensor_data.tactile_sensor_1_data.timeseries_data
        roll,pitch = self.find_pitch_roll()
        # Manipulate data and create the final csv.
        starting_time = sensor0_data[0].time
        # data series is not usually exactly the same length. Take the smaller of the two.
        series_len = len(sensor0_data)
        if len(sensor1_data) < series_len:
            series_len = len(sensor1_data)
        for i in range(series_len):
            colms = []
            sensor0_sample = sensor0_data[i]
            sensor1_sample = sensor1_data[i]
            sample_time = sensor0_sample.time - starting_time
            colms.append(sample_time)
            self.append_sensor_data(colms, sensor0_sample)
            self.append_sensor_data(colms, sensor1_sample)
            colms.append(roll)
            colms.append(pitch)
            self.csv_list.append(colms)
        df = pd.DataFrame(self.csv_list)
        df.to_csv('combined_tactile.csv', index=False, header=False)

    def get_df(self, dir_path, target_filename=None):
        os.chdir(dir_path)
        return pd.read_csv(target_filename)

    def append_sensor_data(self,colms,sensor_sample):
        colms.append(sensor_sample.global_forces.x)
        colms.append(sensor_sample.global_forces.y)
        colms.append(sensor_sample.global_forces.z)
        for pillar_num in range(9):
            colms.append(sensor_sample.pillars[pillar_num].forces.x)
            colms.append(sensor_sample.pillars[pillar_num].forces.y)
            colms.append(sensor_sample.pillars[pillar_num].forces.z)
            colms.append(sensor_sample.pillars[pillar_num].deflections.x)
            colms.append(sensor_sample.pillars[pillar_num].deflections.y)
            colms.append(sensor_sample.pillars[pillar_num].deflections.z)
            colms.append(sensor_sample.pillars[pillar_num].contact)

    def add_colm_headers(self,camera_ground_truth):
        colms = []
        colms.append('time')
        # sensor 0
        colms.append('sen0 gfx')
        colms.append('sen0 gfy')
        colms.append('sen0 gfz')
        for pillar_num in range(9):
            colms.append('sen0 pil' + str(pillar_num) + ' fx')
            colms.append('sen0 pil' + str(pillar_num) + ' fy')
            colms.append('sen0 pil' + str(pillar_num) + ' fz')
            colms.append('sen0 pil' + str(pillar_num) + ' dx')
            colms.append('sen0 pil' + str(pillar_num) + ' dy')
            colms.append('sen0 pil' + str(pillar_num) + ' dz')
            colms.append('sen0 pil' + str(pillar_num) + ' contact')
        # sensor 1
        colms.append('sen1 gfx')
        colms.append('sen1 gfy')
        colms.append('sen1 gfz')
        for pillar_num in range(9):
            colms.append('sen1 pil' + str(pillar_num) + ' fx')
            colms.append('sen1 pil' + str(pillar_num) + ' fy')
            colms.append('sen1 pil' + str(pillar_num) + ' fz')
            colms.append('sen1 pil' + str(pillar_num) + ' dx')
            colms.append('sen1 pil' + str(pillar_num) + ' dy')
            colms.append('sen1 pil' + str(pillar_num) + ' dz')
            colms.append('sen1 pil' + str(pillar_num) + ' contact')
        self.csv_list.append(colms)
        if camera_ground_truth:
            colms.append('ground_truth_roll')
            colms.append('ground_truth_pitch')

    def append_ground_truth(self,colms,ground_truth_df,sample_time,starting_time):
        for i in range(self.ground_truth_index,len(ground_truth_df)):
            gt_time = ground_truth_df['Time'][i] - starting_time
            if sample_time <= gt_time: break
        colms.append(ground_truth_df['list_0'][i]) # pos
        colms.append(ground_truth_df['list_1'][i]) # ang
        self.ground_truth_index = i

def test(destination_dir_name='test_process_bag', topic_list=['/hub_0/sensor_0', '/hub_0/sensor_1']):
    csv_dir_path = os.getcwd() + '\\' + 'processed_data' + '\\' + destination_dir_name
    _ = ProcessBag(destination_dir_name=destination_dir_name, topic_list=topic_list)

def main(destination_dir_name='test_process_bag', topic_list=['/hub_0/sensor_0', '/hub_0/sensor_1']):
    """Process all the .bag files into csv files. Save in the specified destination directory within the processed_data directory."""
    csv_dir_path = os.getcwd() + '\\' + 'processed_data' + '\\' + destination_dir_name
    _ = ProcessBag(destination_dir_name=destination_dir_name, topic_list=topic_list)

if __name__ == '__main__':
    main(destination_dir_name='cable_tension_vec', topic_list=['/hub_0/sensor_0', '/hub_0/sensor_1'])