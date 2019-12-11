from mmt.motion_tracker import StereoFeatureTracker
from mmt.utils import load_stereo_views


class GUIStereoTracker(StereoFeatureTracker):
    def __init__(self):
        view1, view2 = load_stereo_views('Stereo_calibration_2409_0_1.npz')
        super().__init__(view1, view2)
        
    def get_pose(self, frame):
        f1 = frame[:, 640:]
        f2 = frame[:, :640]
        pose, _ = self.process_frame(f1, f2, verbose=False)
        return pose

    