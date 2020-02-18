import time
from mmt.tracker import StereoFeatureTracker
from mmt.utils import load_stereo_views
import numpy as np


class GUIStereoTracker(StereoFeatureTracker):
    verbose = True
    
    def __init__(self):
        view1, view2 = load_stereo_views('Stereo_calibration_2409_0_1.npz')
        super().__init__(view1, view2)

        self.frame = None # Current frame, as an image.
        
    def get_pose(self, frame):
        f1 = frame[:, 640:]
        f2 = frame[:, :640]
        pose, _ = self.process_frame(f1, f2, verbose=False)
        if self.verbose:
            print('Pose [Rx Ry Rz x y z]:\n', pose)
        return pose

    def calculate_pose(self):
        frame = self.frame
        return self.get_pose(frame)

    
class DummyTracker:
    verbose = True
    latency = 0.5
    max_val = 10

    def __init__(self):
        self.count1 = 0
        self.count2 = 0
        self.pose = np.zeros(6)

        self.add = True

        self.send_ready = False
        self.active = False
        self.frame = None

    #     self._latency = latency

    # @property
    # def latency(self):
    #     return self.latency

    # @property
    # def max_val(self):
    #     return self.max_val

    def get_pose(self, frame):
        time.sleep(self.latency)
        axis = self.count1 % 6
        axis_value = self.set_value(self.count2 % self.max_val + 1)
        self.pose[axis] = axis_value
        pose = self.pose

        if self.verbose:
            print('Pose [Rx Ry Rz x y z]:\n', pose)

        self.count2 += 1
        if (self.count2 % self.max_val) == 0:
            self.count1 += 1

        if (self.count2 % self.max_val) == 0 and (self.count1 % 6) == 0:
            self.switch_add()

        return pose

    def calculate_pose(self):
        frame = self.frame
        return self.get_pose(frame)

    def set_value(self, value):
        if self.add:
            retval = value
        else:
            retval = self.max_val - value
        return retval

    def switch_add(self):
        if self.add:
            self.add = False
        else:
            self.add = True


if __name__ == '__main__':
    tracker = DummyTracker(latency=0.1)

    while(True):
        pose = tracker.get_pose(None)
        print(pose)
