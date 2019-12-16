import time
from mmt.motion_tracker import StereoFeatureTracker
from mmt.utils import load_stereo_views
import numpy as np


class GUIStereoTracker(StereoFeatureTracker):
    def __init__(self):
        view1, view2 = load_stereo_views('Stereo_calibration_2409_0_1.npz')
        super().__init__(view1, view2)

        self.verbose = True
        
    def get_pose(self, frame):
        f1 = frame[:, 640:]
        f2 = frame[:, :640]
        pose, _ = self.process_frame(f1, f2, verbose=False)
        if self.verbose:
            print('Pose [Rx Ry Rz x y z]:\n', pose)
        return pose

    
class DummyTracker:
    def __init__(self, latency=0.5):
        self.count1 = 0
        self.count2 = 0
        self.pose = np.zeros(6)
        self.latency = latency

        self.add = True

        self.verbose = True

    def get_pose(self, frame):
        time.sleep(self.latency)
        # self.pose = np.zeros(6)
        axis = self.count1 % 6
        axis_value = self.set_value(self.count2 % 10 + 1)
        self.pose[axis] = axis_value
        pose = self.pose.copy()

        if self.verbose:
            print('Pose [Rx Ry Rz x y z]:\n', pose)

        self.count2 += 1
        if (self.count2 % 10) == 0:
            self.count1 += 1

        if (self.count2 % 10) == 0 and (self.count1 % 6) == 0:
            self.switch_add()

        return pose

    def set_value(self, value):
        if self.add:
            retval = value
        else:
            retval = 10 - value
        return retval

    def switch_add(self):
        if self.add:
            self.add = False
        else:
            self.add = True


if __name__ == '__main__':
    tracker = DummyTracker(latency=0.5)

    while(True):
        pose = tracker.get_pose(None)
        # print(pose)