#!/usr/bin/env python3
"""
A container for contactile sensors data.

Contactile sensor 0 is the left sensor if you consider the side with the plugs the top.

SensorData class serves as a container for all sensor data in a specified directory that contains ROS topic cvs data
files (produced from bag files). Stores the data in the multiple forms so as to be compatible with different matplotlib
functions.

Each sensor has a dictionary, such as self.tactile_sensor_0_data, which holds the data in different forms.

self.tactile_sensor_0_data['quiver_frames'] contains data for matplotlib quiver plots:
quiver_frames: 1st dim == frame num, 2nd dim == (xforce grid, yforce grid, zforce grid)
Each grid is 2 dim array for which the 1st dim is row, 2nd dim is column of pillars.
Example: self.data[10][1][2][3] is a single force at [11th frame][yforce grid][pillar row 2][pillar column 3]
"""

import os
import pandas as pd
import numpy as np
from collections import namedtuple

Tactile_Sensor_NT =  namedtuple("Tactile_Sensor_NT", "timeseries_data quiver_frames heatmap_frames")
Tactile_Sample_NT =  namedtuple("Tactile_Sample_NT", "time pillars global_forces")
Pillar_NT =  namedtuple("Pillar_NT", "contact forces deflections")
XYZ_NT =  namedtuple("XYZ_NT", "x y z")

class SensorData:
    def __init__(self):
        # Used if the filename below is found in the directory, establish the corresponding function to run.
        self.topic_function_bindings = {'hub_0-sensor_0.csv':self.tactile_data,
                                        'hub_0-sensor_1.csv':self.tactile_data}

    def prepare_data(self, csv_data_path = os.getcwd()+'\\'+'processed_data'+'\\'+'test_data'+'\\'+'grasp_release_0degree_0offset_2022-03-27-22-00-10'):
        """Looks in the path for tactile sensor data csv files and makes creates this sensor data object that contains
        all the data so it is easy to access."""
        self.csv_dir = csv_data_path
        self.file_list = self.get_file_names()

        # Go through all of the topic functions, and do the associated function if the csv file for the topic is there.
        for topic_csv_filename in self.topic_function_bindings.keys():
            if topic_csv_filename not in self.file_list:
                # Set the associated data storage to None. Ex: self.tactile_sensor_0_data = None
                self.topic_function_bindings[topic_csv_filename](topic_csv_filename,set_data_to_none=True)
            elif topic_csv_filename in self.file_list:
                # Put the data in the in the data structure.
                dataframe = pd.read_csv(topic_csv_filename)
                self.topic_data = dataframe.to_numpy()
                self.topic_function_bindings[topic_csv_filename](topic_csv_filename,set_data_to_none=False)

    def tactile_data(self,topic,set_data_to_none=False):
        sensor_num = int(topic[len(topic)-5])
        if set_data_to_none and sensor_num == 0:
            self.tactile_sensor_0_data = None
            return
        elif set_data_to_none and sensor_num == 1:
            self.tactile_sensor_1_data = None
            return
        quiver_frames, heatmap_frames = self.get_tactile_frames()
        timeseries_data = self.get_tactile_data()
        data = Tactile_Sensor_NT(timeseries_data, quiver_frames, heatmap_frames)
        if sensor_num == 0:
            self.tactile_sensor_0_data = data
        else:
            self.tactile_sensor_1_data = data


    def get_tactile_data(self):
        """Returns all of the tactile data as a list of sample named tuples."""
        data = []
        for sample in self.topic_data:
            time = sample[0]
            pillars = self.process_pillar_data(sample[6])
            global_forces = XYZ_NT(x=sample[7],y=sample[8],z=sample[9])
            sample_named_tuple = Tactile_Sample_NT(time,pillars, global_forces)
            data.append(sample_named_tuple)
        return data


    def process_pillar_data(self, pillar_data):
        "String of pillar data as an argument. Returns a list of  to different types and return in data structures."
        pillars = []
        pillar_data = pillar_data.strip('[]')
        pillar_data = pillar_data.split(',')
        for i, pillar in enumerate(pillar_data):
            pillar_data = pillar.split('\n')
            dx = pillar_data[7].split(':')
            dy = pillar_data[8].split(':')
            dz = pillar_data[9].split(':')
            deflections = XYZ_NT(x=float(dx[1]), y=float(dy[1]), z=float(dz[1]))
            fx = pillar_data[10].split(':')
            fy = pillar_data[11].split(':')
            fz = pillar_data[12].split(':')
            forces = XYZ_NT(x=float(fx[1]), y=float(fy[1]), z=float(fz[1]))
            contact = eval(pillar_data[13].split(':')[1])
            pillar_named_tuple = Pillar_NT(contact, forces, deflections)
            pillars.append(pillar_named_tuple)
        return pillars

    def get_tactile_frames(self):
        """1st dim == frame num, 2nd dim == xforce grid(0), yforce grid(1), zforce grid(2)
        Each grid is 2 dim array. 1st dim is row, 2nd dim is column of pillars."""
        quiver_frames = []
        heatmap_frames = []
        for sample in self.topic_data:
            pillar_data = sample[6]
            pillar_data = pillar_data.strip('[]')
            pillar_data = pillar_data.split(',')
            xgrid, ygrid, zgrid = self.get_xyz_grids(pillar_data)
            quiver_frames.append([xgrid, ygrid, zgrid])
            heatmap_frames.append(zgrid)
        return quiver_frames,heatmap_frames

    def get_xyz_grids(self,pillar_data):
        xgrid = np.empty(0)
        ygrid = np.empty(0)
        zgrid = np.empty(0)
        for i, pillar in enumerate(pillar_data):
            pillar_list = pillar.split('\n')
            fx = pillar_list[10].split(':')
            fx = float(fx[1])
            fy = pillar_list[11].split(':')
            fy = float(fy[1])
            fz = pillar_list[12].split(':')
            fz = float(fz[1])
            xgrid = np.append(xgrid, fx)
            ygrid = np.append(ygrid, fy)
            zgrid = np.append(zgrid, fz)
        xgrid = (np.flip(xgrid)).reshape(3, 3)
        ygrid = (np.flip(ygrid)).reshape(3, 3)
        zgrid = (np.flip(zgrid)).reshape(3, 3)
        return xgrid, ygrid, zgrid

    def get_file_names(self):
        # Change working dir to the right directory in case it got switched.
        os.chdir(self.csv_dir)
        return os.listdir(self.csv_dir)


def main():
    data_inst = SensorData()
    data_inst.prepare_data()
    print(data_inst.tactile_sensor_0_data)

if __name__ == '__main__':
    main()