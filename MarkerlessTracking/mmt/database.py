"""
LandmarkDatabase class used by StereoFeatureTracker
"""
import pandas as pd
import numpy as np


class LandmarkDatabase:
    """
    Database of 3D landmark positions and their descriptors.
    Attributes:
        landmarks     - 3D position of landmarks
        descriptors   - landmark feature vectors
        miss_counts   - number of consecutive frames in which landmark not seen
        miss_thresh   - if miss_count > miss_thresh, landmark is pruned. If
                        miss_thresh is None, no landmark pruning is done
    """

    def __init__(self, X, descriptors, miss_thresh=None):
        self.landmarks = X
        self.descriptors = descriptors
        self.miss_counts = np.zeros(len(X), dtype=int)
        self.miss_thresh = miss_thresh

    def __len__(self):
        return len(self.landmarks)

    def update(self, X, descriptors, landmarks_used=None):
        if len(self) == 0:
            self.landmarks = X
            self.descriptors = descriptors
            self.miss_counts = np.zeros(len(X), dtype=int)
        else:
            self.landmarks = np.vstack((self.landmarks, X))
            self.descriptors = np.vstack((self.descriptors, descriptors))
            self.miss_counts = np.concatenate((self.miss_counts,
                                               np.zeros(len(X))))

    def trim(self, indices):
        """Removes entries with given indices from database"""
        self.landmarks = np.delete(self.landmarks, indices, axis=0)
        self.descriptors = np.delete(self.descriptors, indices, axis=0)
        self.miss_counts = np.delete(self.miss_counts, indices)

    def trim_unused(self, miss_thresh=None):
        """Trims landmarks from database that have not been used recently."""
        if miss_thresh is None and self.miss_thresh is not None:
            miss_thresh = self.miss_thresh
            self.trim(np.where(self.miss_counts >= miss_thresh)[0])
        elif miss_thresh is not None:
            self.trim(np.where(self.miss_counts >= miss_thresh)[0])

    def update_landmark_usage(self, landmark_indices):
        """
        Updates miss counts for entries in database. If indices are in the input
        array landmark_indices, then their corresponding miss count is set to 0.
        Otherwise, miss count is incremented by 1.

        Database entries with miss counts greater than miss threshold are then
        removed.
        """
        mask_used = np.zeros(len(self), dtype=bool)
        mask_unused = np.ones(len(self), dtype=bool)
        mask_used[landmark_indices] = True
        mask_unused[landmark_indices] = False

        self.miss_counts[mask_used] = 0
        self.miss_counts[mask_unused] += 1
        if self.miss_thresh is not None:
            self.trim(np.where(self.miss_counts >= self.miss_thresh)[0])

    @property
    def dataframe(self):
        """Returns database as a pandas Dataframe"""
        raw_data = [[X, m, desc] for X, m, desc in zip(
                    self.landmarks, self.miss_counts, self.descriptors)]
        df = pd.DataFrame(data=raw_data,
                          columns=['Position', 'Miss count', 'Descriptor'])
        return df

    def save(self, output_filename):
        """
        Saves database as a dataframe, serialised in pickle format.
        """
        # raw_data = [[X, m, desc] for X, m, desc in zip(
        #             self.landmarks, self.miss_counts, self.descriptors)]
        self.dataframe.to_pickle(output_filename)
        # pd.DataFrame(data=raw_data,
        #              columns=['Position',
        #                       'Miss count',
        #                       'Descriptor']).to_pickle(output_filename)
