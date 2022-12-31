#!/usr/bin/env python3

from process_bag import ProcessBag
from compile_all_data import AllData,AllData2
import visualizations as viz

# Where processed (.bag to csv) files will be placed. It will auto create if the directory doesn't exist.
destination_dir_name='cable_tension_vec'
topic_list = ['/hub_0/sensor_0', '/hub_0/sensor_1']

def main():
    # Take all of the .bag files in the bag folder, process the topics in topic list, and put in destination dir, inside
    # of the "processed_data" directory.
    # TODO The ground truth in this is experiment dependent. You will want to update this, probably.
    _ = ProcessBag(source_dir_name = 'bag_data',destination_dir_name=destination_dir_name, topic_list=topic_list)

    # Take the csv data files and do more data processing. This likely changes per experiment and you will want to
    # implement your own data processing...
    # all_data = AllData2(destination_dir_name,destination_dir_name)

def test():
    main()

if __name__ == "__main__":
    main()




