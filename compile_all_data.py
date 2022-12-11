#!/usr/bin/env python3
"""
This does further data processing, which is likely experiment dependent and you will want to modify.
"""

import shutil
import os
from bagpy import bagreader
import logging
import pandas as pd
import numpy as np
import re
import visualizations as viz
from sklearn.model_selection import train_test_split


pd.set_option("display.max_rows", None, "display.max_columns", None)


class AllData2:
    def __init__(self, data_dir_name, filename):
        self.data_dir_path = 'C:\\Users\\kylem\\OneDrive\\Documents\\GitHub\\Contactile_Data' + '\\' + 'processed_data' + '\\' + data_dir_name
        self.saved_filename = filename
        # Sample numbers for the 4 different grasps.
        self.grasp_sample_nums = [4492, 6493, 8489, 10489, 12350]
        self.data = pd.DataFrame()
        self.colm_names = ['gfx', 'gfy', 'gfz',
                           'pil0_fx', 'pil0_fy', 'pil0_fz', 'pil0_contact', 'pil1_fx', 'pil1_fy', 'pil1_fz', 'pil1_contact',
                           'pil2_fx', 'pil2_fy', 'pil2_fz', 'pil2_contact', 'pil3_fx', 'pil3_fy', 'pil3_fz', 'pil3_contact',
                           'pil4_fx', 'pil4_fy', 'pil4_fz', 'pil4_contact', 'pil5_fx', 'pil5_fy', 'pil5_fz', 'pil5_contact',
                           'pil6_fx', 'pil6_fy', 'pil6_fz', 'pil6_contact', 'pil7_fx', 'pil7_fy', 'pil7_fz', 'pil7_contact',
                           'pil8_fx', 'pil8_fy', 'pil8_fz', 'pil8_contact']
        self.run()

    def run(self):
        # For every directory
        for filename in os.listdir(self.data_dir_path):
            self.filename = filename
            self.append_all_samples()
        self.save_data()

    def append_all_samples(self):
        """Uses data from a single experimental run and appends all of the direct and mirrored samples from that
        experimental run to self.data."""
        try:
            exp_data = pd.read_csv(self.data_dir_path + '\\' + self.filename + '\\' + 'combined_tactile.csv')
        except:
            print("\nSkipping file: {}".format(self.filename))
            return

        print("{}\nData size: {}".format(self.filename,exp_data.shape))
        for sample_num in self.grasp_sample_nums:
            single_full_sample = exp_data.iloc[[sample_num]]
            single_full_sample = single_full_sample.drop(single_full_sample.columns[[0]], axis=1)
            self.data = self.data.append(single_full_sample, ignore_index=True)

    def save_data(self,test_perc=30):
        df = pd.DataFrame(self.data)
        df = df.sample(frac=1) # Randomize
        df = df.replace([True], 1)
        df = df.replace([False], 0)
        os.chdir(self.data_dir_path)
        df.to_csv(self.saved_filename+'.csv', index=False)
        len_output = 2
        len_input = len(list(df)) - len_output
        inputs, outputs = df.iloc[:,0:len_input], df.iloc[:,len_input:]
        inputs.to_csv(self.saved_filename + '_X_train_all_data.csv', index=False)
        outputs.to_csv(self.saved_filename + '_y_train_all_data.csv', index=False)

        end_of_filename = '_'+str(10)+'-'+str(90)+'.csv'
        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 10/100, random_state = 42)
        X_train.to_csv(self.saved_filename + '_X_train'+end_of_filename, index=False)
        X_test.to_csv(self.saved_filename + '_X_test'+end_of_filename, index=False)
        y_train.to_csv(self.saved_filename + '_y_train'+end_of_filename, index=False)
        y_test.to_csv(self.saved_filename + '_y_test'+end_of_filename, index=False)

        train_perc = 100 - test_perc
        end_of_filename = '_'+str(test_perc)+'-'+str(train_perc)+'.csv'
        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size = test_perc/100, random_state = 42)
        X_train.to_csv(self.saved_filename + '_X_train'+end_of_filename, index=False)
        X_test.to_csv(self.saved_filename + '_X_test'+end_of_filename, index=False)
        y_train.to_csv(self.saved_filename + '_y_train'+end_of_filename, index=False)
        y_test.to_csv(self.saved_filename + '_y_test'+end_of_filename, index=False)





class AllData:
    def __init__(self, data_dir_name, filename, mirrors = False):
        self.data_dir_path = 'C:\\Users\\kylem\\OneDrive\\Documents\\GitHub\\Contactile_Data' + '\\' + 'processed_data' + '\\' + data_dir_name
        self.mirrors = mirrors
        self.saved_filename = filename
        # Sample numbers for the 4 different grasps.
        self.grasp_sample_nums = [2130,3390,4630,6000]
        self.data = pd.DataFrame()
        self.set_drop_colms()
        self.colm_names = ['gfx', 'gfy', 'gfz',
                           'pil0_fx', 'pil0_fy', 'pil0_fz', 'pil0_contact', 'pil1_fx', 'pil1_fy', 'pil1_fz', 'pil1_contact',
                           'pil2_fx', 'pil2_fy', 'pil2_fz', 'pil2_contact', 'pil3_fx', 'pil3_fy', 'pil3_fz', 'pil3_contact',
                           'pil4_fx', 'pil4_fy', 'pil4_fz', 'pil4_contact', 'pil5_fx', 'pil5_fy', 'pil5_fz', 'pil5_contact',
                           'pil6_fx', 'pil6_fy', 'pil6_fz', 'pil6_contact', 'pil7_fx', 'pil7_fy', 'pil7_fz', 'pil7_contact',
                           'pil8_fx', 'pil8_fy', 'pil8_fz', 'pil8_contact']
        self.pillar_map = {0:3, 1:7, 2:11, 3:15, 4:19, 5:23, 6:27, 7:31, 8:35}
        self.run()

    def run(self):
        # For every directory
        for filename in os.listdir(self.data_dir_path):
            self.filename = filename
            # os.chdir(self.data_dir_path + '\\' + filename)
            self.append_all_samples()
        self.save_data()

    def append_all_samples(self):
        """Uses data from a single experimental run and appends all of the direct and mirrored samples from that
        experimental run to self.data."""
        try:
            exp_data = pd.read_csv(self.data_dir_path + '\\' + self.filename + '\\' + 'combined_tactile.csv')
        except:
            print("\nSkipping file: {}".format(self.filename))
            return

        for sample_num in self.grasp_sample_nums:
            single_full_sample = exp_data.iloc[[sample_num]]
            # Sensor 0
            sen0_sample = self.append_labeled_sample(single_full_sample, sensor_num = 0)
            if self.mirrors and sen0_sample is not None: self.append_mirrors(sen0_sample)
            # Sensor 1
            sen1_sample = self.append_labeled_sample(single_full_sample, sensor_num=1)
            if self.mirrors and sen1_sample is not None: self.append_mirrors(sen1_sample)

    def append_labeled_sample(self, single_full_sample, sensor_num):
        assert(sensor_num == 0 or sensor_num == 1)
        # Remove any 999's which were incorrectly captured data.
        if 999 in single_full_sample.values: return None
        # Remove any extremely off pos samples. This can be considered an incorrect outlier.
        if float(single_full_sample["ground_truth_pos"]) < -50 or float(single_full_sample["ground_truth_pos"]) > 50: return None
        if sensor_num == 0:
            sensor_sample = single_full_sample.loc[:, 'sen0 gfx':'sen0 pil8 contact']
        elif sensor_num == 1:
            sensor_sample = single_full_sample.loc[:, 'sen1 gfx':'sen1 pil8 contact']
        sensor_sample.drop(sensor_sample.columns[self.drop_colm_indexes], axis=1, inplace=True)
        sensor_sample.columns = self.colm_names
        # Append ground truth columns.
        sensor_sample['position'], sensor_sample['angle'] = self.find_pose(single_full_sample,sensor_num=sensor_num)
        # Append and return complete sample.
        self.data = self.data.append(sensor_sample, ignore_index=True)
        return sensor_sample

    def find_pose(self, single_full_sample, sensor_num=0):
        # self.find_regex_pose()
        self.gt_pos = float(single_full_sample['ground_truth_pos'])
        self.gt_ang = float(single_full_sample['ground_truth_ang'])
        # Sensor 1 is an x axis mirror of sensor 0.
        if sensor_num == 1:
            self.gt_ang = -self.gt_ang
        return self.gt_pos,self.gt_ang

    def find_regex_pose(self):
        """Finds and sets the pose in the filename and sets values to be used to find the actual pose."""
        match = re.match('.+?(?=mm)', self.filename, flags=0)
        self.regex_pos = float(match[0]) - 1.5

        fn_segs = self.filename.split('_')
        match = re.match('\d+', fn_segs[1], flags=0)
        self.regex_ang = float(match[0])

    def set_drop_colms(self):
        self.drop_colm_indexes = [6,7,8]
        for i in range(8):
            next1 = self.drop_colm_indexes[-3] + 7
            next2 = self.drop_colm_indexes[-2] + 7
            next3 = self.drop_colm_indexes[-1] + 7
            self.drop_colm_indexes += [next1,next2,next3]

    def save_data(self,test_perc=30):
        df = pd.DataFrame(self.data)
        df = df.sample(frac=1) # Randomize
        df = df.replace([True], 1)
        df = df.replace([False], 0)
        os.chdir(self.data_dir_path)
        df.to_csv(self.saved_filename+'.csv', index=False)
        len_output = 2
        len_input = len(list(df)) - len_output
        inputs, outputs = df.iloc[:,0:len_input], df.iloc[:,len_input:]
        inputs.to_csv(self.saved_filename + '_X_train_all_data.csv', index=False)
        outputs.to_csv(self.saved_filename + '_y_train_all_data.csv', index=False)

        end_of_filename = '_'+str(10)+'-'+str(90)+'.csv'
        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size = 10/100, random_state = 42)
        X_train.to_csv(self.saved_filename + '_X_train'+end_of_filename, index=False)
        X_test.to_csv(self.saved_filename + '_X_test'+end_of_filename, index=False)
        y_train.to_csv(self.saved_filename + '_y_train'+end_of_filename, index=False)
        y_test.to_csv(self.saved_filename + '_y_test'+end_of_filename, index=False)

        train_perc = 100 - test_perc
        end_of_filename = '_'+str(test_perc)+'-'+str(train_perc)+'.csv'
        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size = test_perc/100, random_state = 42)
        X_train.to_csv(self.saved_filename + '_X_train'+end_of_filename, index=False)
        X_test.to_csv(self.saved_filename + '_X_test'+end_of_filename, index=False)
        y_train.to_csv(self.saved_filename + '_y_train'+end_of_filename, index=False)
        y_test.to_csv(self.saved_filename + '_y_test'+end_of_filename, index=False)

    ####################################################################################################################
    # Data mirroring functions.
    ####################################################################################################################

    def append_mirrors(self,sample):
        x_mirror = self.mirror_sample(sample, axis = 'x')
        y_mirror = self.mirror_sample(sample, axis = 'y')
        xy_mirror = self.mirror_sample(sample, axis = 'xy')
        self.data = self.data.append(x_mirror, ignore_index=True)
        self.data = self.data.append(y_mirror, ignore_index=True)
        self.data = self.data.append(xy_mirror, ignore_index=True)

    def mirror_sample(self, sample, axis):
        assert(axis=='x' or axis =='y' or axis=='xy')
        mirror = sample.copy(deep=True)
        mirror = self.mirror_pillars(mirror, sample, axis)
        mirror = self.mirror_forces(mirror,axis)
        mirror = self.mirror_label(mirror,axis)
        return mirror

    def mirror_pillars(self,mirror, original, axis):
        if axis == 'x':
            mirror = self.swap_pillars(mirror, original, pillar_num=6, with_pillar=0)
            mirror = self.swap_pillars(mirror, original, pillar_num=7, with_pillar=1)
            mirror = self.swap_pillars(mirror, original, pillar_num=8, with_pillar=2)
        elif axis == 'y':
            mirror = self.swap_pillars(mirror, original, pillar_num=0, with_pillar=2)
            mirror = self.swap_pillars(mirror, original, pillar_num=3, with_pillar=5)
            mirror = self.swap_pillars(mirror, original, pillar_num=6, with_pillar=8)
        elif axis == 'xy':
            mirror = self.swap_pillars(mirror, original, pillar_num=6, with_pillar=2)
            mirror = self.swap_pillars(mirror, original, pillar_num=3, with_pillar=5)
            mirror = self.swap_pillars(mirror, original, pillar_num=0, with_pillar=8)
            mirror = self.swap_pillars(mirror, original, pillar_num=1, with_pillar=7)
        return mirror

    def swap_pillars(self, mirror, original, pillar_num = 0, with_pillar = 0):
        """Swaps pillar values from sample to mirror - x,y,z and contact values. Returns mirrored dataframe."""
        mirror.iloc[:,self.pillar_map[pillar_num]] = original.iloc[:,self.pillar_map[with_pillar]]
        mirror.iloc[:, self.pillar_map[pillar_num] + 1] = original.iloc[:, self.pillar_map[with_pillar] + 1]
        mirror.iloc[:, self.pillar_map[pillar_num] + 2] = original.iloc[:, self.pillar_map[with_pillar] + 2]
        mirror.iloc[:, self.pillar_map[pillar_num] + 3] = original.iloc[:, self.pillar_map[with_pillar] + 3]
        mirror.iloc[:, self.pillar_map[with_pillar]] = original.iloc[:, self.pillar_map[pillar_num]]
        mirror.iloc[:, self.pillar_map[with_pillar] + 1] = original.iloc[:, self.pillar_map[pillar_num] + 1]
        mirror.iloc[:, self.pillar_map[with_pillar] + 2] = original.iloc[:, self.pillar_map[pillar_num] + 2]
        mirror.iloc[:, self.pillar_map[with_pillar] + 3] = original.iloc[:, self.pillar_map[pillar_num] + 3]
        return mirror

    def mirror_forces(self, mirror, axis):
        if axis == 'x':
            mirror = self.mirror_forces_x(mirror)
        elif axis == 'y':
            mirror = self.mirror_forces_y(mirror)
        elif axis == 'xy':
            mirror = self.mirror_forces_x(mirror)
            mirror = self.mirror_forces_y(mirror)
        return mirror

    def mirror_forces_x(self, mirror):
        mirror.pil0_fy = -mirror.pil0_fy
        mirror.pil1_fy = -mirror.pil1_fy
        mirror.pil2_fy = -mirror.pil2_fy
        mirror.pil3_fy = -mirror.pil3_fy
        mirror.pil4_fy = -mirror.pil4_fy
        mirror.pil5_fy = -mirror.pil5_fy
        mirror.pil6_fy = -mirror.pil6_fy
        mirror.pil7_fy = -mirror.pil7_fy
        mirror.pil8_fy = -mirror.pil8_fy
        return mirror

    def mirror_forces_y(self, mirror):
        mirror.pil0_fx = -mirror.pil0_fx
        mirror.pil1_fx = -mirror.pil1_fx
        mirror.pil2_fx = -mirror.pil2_fx
        mirror.pil3_fx = -mirror.pil3_fx
        mirror.pil4_fx = -mirror.pil4_fx
        mirror.pil5_fx = -mirror.pil5_fx
        mirror.pil6_fx = -mirror.pil6_fx
        mirror.pil7_fx = -mirror.pil7_fx
        mirror.pil8_fx = -mirror.pil8_fx
        return mirror

    def mirror_label(self,mirror,axis):
        if axis == 'x':
            mirror.angle = float(-mirror.angle)
        elif axis == 'y':
            mirror.angle = float(-mirror.angle)
            mirror.position = float(-mirror.position)
        elif axis == 'xy':
            mirror.position = float(-mirror.position)
        return mirror


    ####################################################################################################################
    # Testing functions
    ####################################################################################################################

    def check_mirror(self,m, s, axis='x'):
        """Checks that the values of the mirror and sample are equivalent in the correct positions.
        m: mirror, s: sample (original)"""
        assert(axis=='x' or axis =='y' or axis=='xy')
        assert(float(m.gfz) > 0.5)
        if axis == 'x':
            assert (float(m.position) == float(s.position) and float(m.angle) == float(-s.angle))
            assert ( float(m.pil6_fx) == float(s.pil0_fx) and float(m.pil6_fy) == float(-s.pil0_fy) and float(m.pil6_fz) == float(s.pil0_fz) and int(m.pil6_contact) == int(s.pil0_contact) )
            assert ( float(m.pil0_fx) == float(s.pil6_fx) and float(m.pil0_fy) == float(-s.pil6_fy) and float(m.pil0_fz) == float(s.pil6_fz) and int(m.pil0_contact) == int(s.pil6_contact))

            assert ( float(m.pil7_fx) == float(s.pil1_fx) and float(m.pil7_fy) == float(-s.pil1_fy) and float(m.pil7_fz) == float(s.pil1_fz) and int(m.pil7_contact) == int(s.pil1_contact))
            assert ( float(m.pil1_fx) == float(s.pil7_fx) and float(m.pil1_fy) == float(-s.pil7_fy) and float(m.pil1_fz) == float(s.pil7_fz) and int(m.pil1_contact) == int(s.pil7_contact))

            assert ( float(m.pil8_fx) == float(s.pil2_fx) and float(m.pil8_fy) == float(-s.pil2_fy) and float(m.pil8_fz) == float(s.pil2_fz) and int(m.pil8_contact) == int(s.pil2_contact))
            assert ( float(m.pil2_fx) == float(s.pil8_fx) and float(m.pil2_fy) == float(-s.pil8_fy) and float(m.pil2_fz) == float(s.pil8_fz) and int(m.pil2_contact) == int(s.pil8_contact))
        elif axis == 'y':
            assert (float(m.position) == float(-s.position) and float(m.angle) == float(-s.angle))
            assert (float(m.pil6_fx) == float(-s.pil8_fx) and float(m.pil6_fy) == float(s.pil8_fy) and float(m.pil6_fz) == float(s.pil8_fz) and int(m.pil6_contact) == int(s.pil8_contact))
            assert (float(m.pil8_fx) == float(-s.pil6_fx) and float(m.pil8_fy) == float(s.pil6_fy) and float(m.pil8_fz) == float(s.pil6_fz) and int(m.pil8_contact) == int(s.pil6_contact))

            assert (float(m.pil3_fx) == float(-s.pil5_fx) and float(m.pil3_fy) == float(s.pil5_fy) and float(m.pil3_fz) == float(s.pil5_fz) and int(m.pil3_contact) == int(s.pil5_contact) )
            assert (float(m.pil5_fx) == float(-s.pil3_fx) and float(m.pil5_fy) == float(s.pil3_fy) and float(m.pil5_fz) == float(s.pil3_fz) and int(m.pil5_contact) == int(s.pil3_contact))

            assert (float(m.pil0_fx) == float(-s.pil2_fx) and float(m.pil0_fy) == float(s.pil2_fy) and float(m.pil0_fz) == float(s.pil2_fz) and int(m.pil0_contact) == int(s.pil2_contact))
            assert (float(m.pil2_fx) == float(-s.pil0_fx) and float(m.pil2_fy) == float(s.pil0_fy) and float(m.pil2_fz) == float(s.pil0_fz) and int(m.pil2_contact) == int(s.pil0_contact))
        elif axis == 'xy':
            assert (float(m.position) == float(-s.position) and float(m.angle) == float(s.angle))
            assert (float(m.pil6_fx) == float(-s.pil2_fx) and float(m.pil6_fy) == float(-s.pil2_fy) and float(m.pil6_fz) == float(s.pil2_fz) and int(m.pil6_contact) == int(s.pil2_contact))
            assert (float(m.pil2_fx) == float(-s.pil6_fx) and float(m.pil2_fy) == float(-s.pil6_fy) and float(m.pil2_fz) == float(s.pil6_fz) and int(m.pil2_contact) == int(s.pil6_contact))

            assert (float(m.pil3_fx) == float(-s.pil5_fx) and float(m.pil3_fy) == float(-s.pil5_fy) and float(m.pil3_fz) == float(s.pil5_fz) and int(m.pil3_contact) == int(s.pil5_contact) )
            assert (float(m.pil5_fx) == float(-s.pil3_fx) and float(m.pil5_fy) == float(-s.pil3_fy) and float(m.pil5_fz) == float(s.pil3_fz) and int(m.pil5_contact) == int(s.pil3_contact))

            assert (float(m.pil7_fx) == float(-s.pil1_fx) and float(m.pil7_fy) == float(-s.pil1_fy) and float(m.pil7_fz) == float(s.pil1_fz) and int(m.pil7_contact) == int(s.pil1_contact))
            assert (float(m.pil1_fx) == float(-s.pil7_fx) and float(m.pil1_fy) == float(-s.pil7_fy) and float(m.pil1_fz) == float(s.pil7_fz) and int(m.pil1_contact) == int(s.pil7_contact))

            assert (float(m.pil0_fx) == float(-s.pil8_fx) and float(m.pil0_fy) == float(-s.pil8_fy) and float(m.pil0_fz) == float(s.pil8_fz) and int(m.pil0_contact) == int(s.pil8_contact))
            assert (float(m.pil8_fx) == float(-s.pil0_fx) and float(m.pil8_fy) == float(-s.pil0_fy) and float(m.pil8_fz) == float(s.pil0_fz) and int(m.pil8_contact) == int(s.pil0_contact))

    def test(self):
        """Just testing to make sure the mirror samples were created correctly."""
        data = pd.read_csv(self.data_dir_path + '\\' + self.saved_filename)
        assert(len(data)%4 == 0)
        num_samples_quads = len(data)/4
        assert(num_samples_quads.is_integer())
        for sample_num in range(int(num_samples_quads)):
            orig_sample = data.iloc[[0]]
            x_mirror = data.iloc[[1]]
            y_mirror = data.iloc[[2]]
            xy_mirror = data.iloc[[3]]
            self.check_mirror(x_mirror, orig_sample, axis='x')
            self.check_mirror(y_mirror, orig_sample, axis='y')
            self.check_mirror(xy_mirror, orig_sample, axis='xy')
            data = data.drop(data.index[:4])

        print("\nPASSED TEST\n")

def main():
    data_dir_name = 'cable_pose_with_cam'
    saved_filename = 'cam_pose_not_mirrored'
    all_data = AllData(data_dir_name,saved_filename,mirrors = False)
    # all_data.test()


if __name__ == '__main__':
    main()
