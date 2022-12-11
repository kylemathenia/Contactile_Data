"""Classes and functions to vizualize the data."""

from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib
import numpy as np
import sys
from sensor_data import SensorData


class QuiverAnimation:
    def __init__(self,data,filename="quiver",hz=500,save=False):
        self.data=data.quiver_frames
        self.save = save
        self.filename = filename
        self.hz = hz

    def run(self):
        # 2 dim arrays for x and y. These are the static locations of the arrows on the plot.
        # mgrid takes the format: np.mgrid[x_range_low:x_range_high:num_spots,y_range_low:y_range_high:num_spots]
        X, Y = np.mgrid[:10:3j,:10:3j]
        # U and V are grids where each corresponding spot has x and y values that define the arrows.
        U = self.data[0][0]
        V = self.data[0][1]
        # Single color values for every spot. This would be the z value.
        C = self.data[0][2]

        # Initialize the quiver
        fig, ax = plt.subplots(1,1)
        # C argument can be used for the colors.
        # See all the cmaps at: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        norm = matplotlib.colors.Normalize(vmin=0, vmax=6) # This sets the range for the colors.
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
        # Q = ax.quiver(X, Y, U, V, C, pivot='mid', units='inches', norm=norm, cmap='plasma')
        Q = ax.quiver(X, Y, U, V, C, pivot='mid', units='inches', norm=norm, cmap='coolwarm',width = .05, scale=1 / 0.5)
        ax.set_xlim(-3, 13)
        ax.set_ylim(-3, 13)

        # You need to set blit=False, or the first set of arrows never gets cleared on subsequent frames.
        # https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.animation.FuncAnimation.html
        if self.save:
            anim = animation.FuncAnimation(fig, self.update_quiver, fargs=(Q,), interval=1000 / self.hz, frames=len(self.data),
                                           blit=False,save_count=sys.maxsize)
            anim.save(self.filename + '.mp4')
        else:
            anim = animation.FuncAnimation(fig, self.update_quiver, fargs=(Q,), interval=1000 / self.hz,
                                           frames=len(self.data),blit=False)
            fig.tight_layout()
            plt.show()

    def update_quiver(self,i, Q):
        """updates the horizontal and vertical vector components by a
        fixed increment on each frame
        """
        U = self.data[i][0]
        V = self.data[i][1]
        C = self.data[i][2]
        Q.set_UVC(U,V,C)
        return Q,



class QuiverHeatmapAnimation:
    def __init__(self,data,max_force = 10,filename="quiver_heatmap",hz=500,save=False):
        self.data=data.quiver_frames
        self.save = save
        self.filename = filename
        self.max_force = max_force
        self.hz = hz

    def run(self):
        fig, ax = plt.subplots(1,1)


        # Heatmap
        self.im = plt.imshow(self.data[0][2], cmap='YlOrRd', interpolation='nearest', animated=True, vmin=0,
                             vmax=self.max_force)

        #Quiver
        # 2 dim arrays for x and y. These are the static locations of the arrows on the plot.
        # mgrid takes the format: np.mgrid[x_range_low:x_range_high:num_spots,y_range_low:y_range_high:num_spots]
        X, Y = np.mgrid[:2:3j,:2:3j]
        # U and V are grids where each corresponding spot has x and y values that define the arrows.
        U = self.data[0][1] # X values
        V = self.data[0][0] # Y values
        self.Q = ax.quiver(X, Y, U, V, pivot='mid', units='inches',  color = 'dodgerblue', cmap='YlOrRd',width = .05, scale=1 / 0.5)
        ax.set_xlim(-.5, 2.5)
        ax.set_ylim(-.5, 2.5)

        if self.save:
            anim = animation.FuncAnimation(fig, self.update_quiver, interval=1000 / self.hz, frames=len(self.data),
                                           blit=False,save_count=sys.maxsize)
            anim.save(self.filename + '.mp4')
        else:
            anim = animation.FuncAnimation(fig, self.update_quiver, interval=1000 / self.hz, frames=len(self.data),blit=False)
            plt.show()

    def update_quiver(self,i):
        self.im.set_array((self.data[i][2].transpose()))
        U = self.data[i][1]
        V = self.data[i][0]
        self.Q.set_UVC(U,V)
        return None,



class HeatmapAnimation:
    def __init__(self,data,max_force = 1,filename="heatmap",hz=500,save=False):
        self.data = data.heatmap_frames
        self.max_force = max_force
        self.save = save
        self.filename = filename
        self.hz = hz

    def run(self):
        fig = plt.figure()
        self.im = plt.imshow(self.data[0], cmap='coolwarm', interpolation='nearest',animated=True,vmin=-self.max_force,vmax=self.max_force)
        if self.save:
            anim = animation.FuncAnimation(fig, self.update, frames=len(self.data), interval=1000 / self.hz, blit=True,
                                           save_count=sys.maxsize)
            anim.save(self.filename + '.mp4')
        else:
            anim = animation.FuncAnimation(fig, self.update, frames=len(self.data), interval=1000 / self.hz, blit=True)
        plt.show()

    def update(self,i):
        self.im.set_array((self.data[i]))
        return self.im,




class PoseAnimation:
    def __init__(self,data,filename="pose_animation",hz=500,save=False):
        self.data = data.tactile_sensor_0_data.timeseries_data
        self.contact_frames = self.create_contact_frames()
        self.point_frames = self.create_point_frames()
        self.save=save
        self.hz = hz
        self.filename = filename

    def create_contact_frames(self):
        frames = []
        for i in range(len(self.data)):
            frame = []
            for pillar_num in range(9):
                frame.append(int(self.data[i].pillars[pillar_num].contact))
            frames.append(np.reshape(frame,[3,3]))
        return frames

    def create_point_frames(self):
        frames = []
        for i in range(len(self.data)):
            frame = {}
            xlist=[]
            ylist=[]
            for pillar_num in range(9):
                contact = self.data[i].pillars[pillar_num].contact
                if not contact:
                    continue
                # TODO make this a hash map...
                elif pillar_num == 0:
                    x = 0
                    y = 0
                elif pillar_num == 1:
                    x = 1
                    y = 0
                elif pillar_num == 2:
                    x = 2
                    y = 0
                elif pillar_num == 3:
                    x = 0
                    y = 1
                elif pillar_num == 4:
                    x = 1
                    y = 1
                elif pillar_num == 5:
                    x = 2
                    y = 1
                elif pillar_num == 6:
                    x = 0
                    y = 2
                elif pillar_num == 7:
                    x = 1
                    y = 2
                elif pillar_num == 8:
                    x = 2
                    y = 2
                xlist.append(x)
                ylist.append(y)
            frame['xlist'] = xlist
            frame['ylist'] = ylist
            frames.append(frame)
        return frames

    def run(self):
        fig, ax = plt.subplots(1, 1)
        # Initial heatmap
        self.im = plt.imshow(self.contact_frames[0], cmap='YlOrRd', interpolation='nearest',animated=True,vmin=0,vmax=1)
        self.line, = plt.plot([0, 1], [0, 1], linewidth=4, color="g")

        if self.save:
            anim = animation.FuncAnimation(fig, self.update, frames=len(self.contact_frames), interval=1000 / self.hz, blit=True,save_count=sys.maxsize)
            anim.save(self.filename + '.mp4')
        else:
            anim = animation.FuncAnimation(fig, self.update, frames=len(self.contact_frames), interval=1000 / self.hz, blit=False)
        plt.show()

    def update(self,i):
        x = self.point_frames[i]['xlist']
        y = self.point_frames[i]['ylist']
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            # print(x)
            # print(p(x))
            self.line.set_xdata(x)
            self.line.set_ydata(p(x))
        else:
            self.line.set_xdata(x)
            self.line.set_ydata(y)
        self.im.set_array((self.contact_frames[i]))
        return self.im,



def main():
    filepath = r'C:\Users\kylem\OneDrive\Documents\GitHub\Contactile_Data_Viz\processed_data\cable_pose_data\3.5mm_25deg_2022-04-30-19-15-31'
    data = SensorData()
    data.prepare_data(filepath)
    pose_animation = PoseAnimation(data,save=True)
    pose_animation.run()

if __name__ == '__main__':
    main()