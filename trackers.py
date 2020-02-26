import time
import logging
from mmt.tracker import StereoFeatureTracker
from mmt.utils import load_stereo_views, vec2mat, mat2vec
import numpy as np


class TrackingWrapper:
    """Wrapper class for motion trackers."""

    def __init__(self, tracker=None, cross_calibration=None, outputlog=None):
        if tracker:
            self.tracker = tracker
        else:
            self.tracker = DummyTracker()

        if cross_calibration:
            self.xcal = np.load(cross_calibration)
        else:
            self.xcal = np.eye(4)
        self.xcal_inv = np.linalg.inv(self.xcal)

        self.frame = None
        self.initial_position = np.eye(4)
        self.table_loc = 0

        self.logger = logging.getLogger(__name__)
        self.formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)

        if outputlog:
            self.file_handler = logging.FileHandler(outputlog)
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)

    def calculate_pose(self):
        frame = self.frame
        pose = self.tracker.get_pose(frame)
        posemat = vec2mat(pose)
        outpose = self.xcal.dot(posemat).dot(self.xcal_inv).dot(self.initial_position)
        self.logger.info(('Pose [Rx Ry Rz x y z]:\n', mat2vec(outpose)))
        return outpose

    def update_initial_pos(self, new_pos):
        self.initial_position = new_pos

    def update_xcal(self, new_xcal):
        self.xcal = new_xcal
        self.xcal_inv = np.linalg.inv(new_xcal)

    def update_table(self, new_val):
        if np.abs(new_val) > 1e-2:
            T_table = np.eye(4)
            T_table[2,3] = new_val - self.table_loc
            new_xcal = T_table.dot(self.xcal)
            self.update_xcal(new_xcal)


class GUIStereoTracker(StereoFeatureTracker):
    verbose = True
    
    def __init__(self, cross_calibration=None):
        view1, view2 = load_stereo_views('Stereo_calibration_2409_0_1.npz')
        super().__init__(view1, view2)
        if cross_calibration:
            self.cross_calibration = np.load(cross_calibration)
        else:
            self.cross_calibration = np.eye(4)

        self.frame = None # Current frame, as an image.
        self.initial_position = None
        
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
