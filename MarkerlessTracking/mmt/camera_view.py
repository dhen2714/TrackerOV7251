import cv2
import numpy as np
import time
from sklearn.preprocessing import normalize


def time_method(method):
    def timer(*args, **kwargs):
        start = time.perf_counter()
        method(*args, **kwargs)
        timeTaken = time.perf_counter() - start
        return timeTaken
    return timer


class CameraView:
    """
    CameraView contains information about a given view, such as its associated
    projection matrix, distortion coefficients and image.
    """

    detectorType = cv2.xfeatures2d.SIFT_create()
    descriptorType = cv2.xfeatures2d.SIFT_create()
    frame_number = 0

    def __init__(self, P=np.ones((3, 4)), fc=np.zeros(2),
                 pp=np.zeros(2), kk=np.zeros(3), kp=np.zeros(2), mask=None):
        # print('Initialize camera view.')
        self.P = P    # Projection matrix
        self.fc = fc  # 2 element focal length
        self.pp = pp  # Principal point
        self.kk = kk  # Radial distortion, 2 or 3 coefficients
        self.kp = kp  # Tangential distortion
        self.mask = mask

        # Attributes below change with every call of process_frame()
        self.image = None
        self.keys = np.array([])  # Keypoints for the current image
        self.descriptors = np.array([])  # Descriptors associated with keypoints
        self.key_coords = np.array([])  # Pixel coordinates of keypoints
        self.n_keys = 0  # Number of keypoints found in current image

    def process_frame(self, image):
        """
        Updates the current image for the camera view, finds keypoints and
        associated descriptors, and undistorts the pixel coordinates of the
        keypoints.
        """
        loadTime = self.update_view(image)
        detDesTime = self.find_keypoints()
        correctDistTime = self.correct_distortion()
        return loadTime, detDesTime, correctDistTime

    @time_method
    def update_view(self, image):
        self.image = image  # Image is a numpy array.

    @time_method
    def find_keypoints(self):
        """
        Get keypoints and descriptors for the image currently stored in view.
        """
        assert self.image is not None, 'No image loaded into view!'

        # If detector and descriptor are same (e.g. SIFT and SIFT), the
        # detectAndCompute() method is preferred for efficiency over separate
        # detect() and compute() steps.
        if self.detectorType.__class__ == self.descriptorType.__class__:
            self.keys, self.descriptors = self.detectorType.detectAndCompute(
                self.image, self.mask)
        else:
            self.keys = self.detectorType.detect(self.image, self.mask)
            self.descriptors = self.descriptorType.compute(self.image, self.keys)

        # normalize throws ValueError if descriptors is None.
        try:
            self.descriptors = (self.descriptors)
        except ValueError:
            pass

        if self.descriptors is None:
            self.descriptors = np.array([])

        self.n_keys = len(self.keys)
        # Set key_coords as n_keys*2 numpy array of pixel coordinates.
        self.key_coords = np.zeros((self.n_keys, 2), dtype=np.float64)
        self.key_coords = np.array([self.keys[i].pt
                                    for i in range(self.n_keys)])

    @time_method
    def correct_distortion_BC(self):
        """
        Corrects distortion using Brown's (or Brown-Conrady) distortion model.
        This is the same method used by OpenCV.
        """
        # May be 2 or 3 radial coefficients depending on calibration method
        if len(self.kk) == 3:
            k3 = self.kk[2]
        else:
            k3 = 0

        ud = self.key_coords[:, 0]
        vd = self.key_coords[:, 1]
        xn = (ud - self.pp[0])/self.fc[0]  # Normalise points
        yn = (vd - self.pp[1])/self.fc[1]
        r2 = xn*xn + yn*yn
        r4 = r2*r2
        r6 = r4*r2

        k_radial = 1 + self.kk[0]*r2 + self.kk[1]*r4 + k3*r6
        x = xn*k_radial + 2*self.kp[0]*xn*yn + self.kp[1]*(r2 + 2*xn*xn)
        y = yn*k_radial + self.kp[0]*(r2 + 2*yn*yn) + 2*self.kp[1]*xn*yn

        x = self.fc[0]*x + self.pp[0]  # Convert back to pix coords
        y = self.fc[1]*y + self.pp[1]
        self.key_coords = np.array([x, y]).T

    @time_method
    def correct_distortion(self):
        """
        Correct for distortion by adjusting the attribute key_coords.
        """
        # Two or three radial coefficients depending on calibration method.
        if len(self.kk) == 3:
            k3 = self.kk[2]
        else:
            k3 = 0

        if self.n_keys != 0:
            ud = self.key_coords[:, 0]
            vd = self.key_coords[:, 1]
            xn = (ud - self.pp[0])/self.fc[0]  # Normalise points
            yn = (vd - self.pp[1])/self.fc[1]
            x = xn
            y = yn

            for i in range(20):
                r2 = x*x + y*y
                r4 = r2*r2
                r6 = r4*r2
                k_radial = 1 + self.kk[0]*r2 + self.kk[1]*r4 + k3*r6
                delta_x = 2*self.kp[0]*x*y + self.kp[1]*(r2 + 2*x*x)
                delta_y = 2*self.kp[1]*x*y + self.kp[0]*(r2 + 2*y*y)
                x = (xn - delta_x)/k_radial
                y = (yn - delta_y)/k_radial

            x = self.fc[0]*x + self.pp[0]  # Undo normalisation
            y = self.fc[1]*y + self.pp[1]

            self.key_coords = np.array([x, y]).T

    @classmethod
    def set_detAndDes(cls, features):
        """
        Sets both detector and descriptor for all views. features is an opencv
        features object,
        e.g. cv2.xfeatures2d.SURF_create()
        """
        cls.detectorType = features
        cls.descriptorType = features

    @classmethod
    def set_detector(cls, det):
        """
        Sets detector type for all views. det is an opencv features object,
        e.g cv2.xfeatures2d.SURF_create()
        """
        cls.detectorType = det

    @classmethod
    def set_descriptor(cls, des):
        """
        Sets descriptor type for all views. des is an opencv features object,
        e.g cv2.xfeatures2d.SURF_create()
        """
        cls.descriptorType = des

    @classmethod
    def update_frame_number(cls):
        cls.frame_number += 1


class MatchHandler:
    """
    Handles process of matching two arrays of descriptors.
    """
    ratioTest = True
    distRatio = 0.6
    binaryDesc = False

    def __init__(self):
        self.matches = []
        if ratioTest:
            bf = cv2.BFMatcher()
        else:
            if binaryDesc:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def __call__(self, descriptors1, descriptors2):
        self.matches = []
        if self.ratioTest:
            match = bf.knnMatch(descriptors1, descriptors2)
            for m, n in match:
                if m.distance < self.distRatio*n.distance:
                    self.matches.append(m)
        else:
            matches = bf.match(descriptors1, descriptors2)

        matches = self.remove_duplicates(np.array(matches))
        in1 = np.array([matches[i].queryIdx
                        for i in range(len(matches))], dtype='int')
        in2 = np.array([matches[i].trainIdx
                        for i in range(len(matches))], dtype='int')
        return in1, in2

    @staticmethod
    def remove_duplicates(matches):
        matchIndices = np.array([(matches[i].queryIdx, matches[i].trainIdx)
                                 for i in range(len(matches))])
        if matchIndices.size:
            _, countsT = np.unique(matchIndices[:, 1], return_counts=True)
        else:
            return matches
        return matches[np.where(countsT == 1)[0]]

    @classmethod
    def set_ratioTest(cls, val=True, ratio=0.6):
        cls.ratioTest = val
        if val:
            cls.distratio = ratio

    @classmethod
    def set_binaryDesc(cls, val=False):
        cls.binaryDesc = val

    pass


"""Testing"""
if __name__ == "__main__":
    print("Hello world!")
    derp = np.zeros((3, 3))
    if derp is not None:
        print("Herp")
