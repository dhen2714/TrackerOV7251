import socket
import asyncore
import struct
import sys
import time
import logging
import numpy as np


def mdot(*args):
    """
    Matrix multiplication for more than 2 arrays.
    """
    ret = args[0]
    for a in args[1:]:
        ret = np.dot(ret, a)
    return ret


def vec2mat(Rx, Ry, Rz, x, y, z):
    """
    Converts a six vector represenation of motion to a 4x4 matrix.
    Assumes yaw, pitch, roll are in degrees.
    """
    ax = np.radians(Rx)
    ay = np.radians(Ry)
    az = np.radians(Rz)
    t1 = np.array([[1, 0, 0, 0],
                    [0, np.cos(ax), -np.sin(ax), 0],
                    [0, np.sin(ax), np.cos(ax), 0],
                    [0, 0, 0, 1]])
    t2 = np.array([[np.cos(ay), 0, np.sin(ay), 0],
                    [0, 1, 0, 0],
                    [-np.sin(ay), 0, np.cos(ay), 0],
                    [0, 0, 0, 1]])
    t3 = np.array([[np.cos(az), -np.sin(az), 0, 0],
                    [np.sin(az), np.cos(az), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    tr = np.array([[1, 0, 0, x],
                    [0, 1, 0, y],
                    
                    [0, 0, 1, z],
                    [0, 0, 0, 1]])
    return mdot(tr, t3, t2, t1)


def random_pose(rot=5, tr=10):
    """
    Create random 4x4 homogeneous matrix representing a random rigid body
    pose. Max range of rotations and translations specified by rot and tr.
    Rotations and translations are uniformly sampled within ranges given.
    """
    yaw = rot*2*(np.random.rand() - 0.5)
    pitch = rot*2*(np.random.rand() - 0.5)
    roll = rot*2*(np.random.rand() - 0.5)
    x = tr*2*(np.random.rand() - 0.5)
    y = tr*2*(np.random.rand() - 0.5)
    z = tr*2*(np.random.rand() - 0.5)
    return vec2mat(yaw, pitch, roll, x, y, z)


def encode_pose_est(pose):
    """
    packet size = 60 bytes (15 integers)
 
    buf[0] -> packet_type (just set to 0)
    /** Packet type for optical motion correction */
    #define RTMOCO_PACKET_TYPE_OPTICAL 0
    /** Packet type for PACE motion correction - for future use */
    #define RTMOCO_PACKET_TYPE_PACE 1
    /** Packet type for wireless marker tracking - for future use */
    #define RTMOCO_PACKET_TYPE_MTRACK 2
    /** Packet type for real-time phase correction for DWI - for future use */
    #define RTMOCO_PACKET_TYPE_DWIPHASECORR 3
     
    buf[1] -> status bitmask
    /** Status message bitmask defines */
    #define    RTMOCOSTATUSMESSAGE_MARKER_NOT_FOUND        (1u << 0)
    #define    RTMOCOSTATUSMESSAGE_MARKER_TOO_CLOSE        (1u << 1)
    #define    RTMOCOSTATUSMESSAGE_MARKER_TOO_FAR            (1u << 2)
    #define    RTMOCOSTATUSMESSAGE_PLANAR_POSE_MODE        (1u << 3)
    #define    RTMOCOSTATUSMESSAGE_HIGH_VELOCITY            (1u << 4)
    #define    RTMOCOSTATUSMESSAGE_ILL_COND                (1u << 5)
    #define    RTMOCOSTATUSMESSAGE_INSUFFICIENT_POINTS     (1u << 6)
    #define    RTMOCOSTATUSMESSAGE_TOO_LEFT                (1u << 7)
    #define    RTMOCOSTATUSMESSAGE_TOO_RIGHT                (1u << 8)
    #define    RTMOCOSTATUSMESSAGE_TOO_SUPERIOR            (1u << 9)
    #define    RTMOCOSTATUSMESSAGE_TOO_INFERIOR              (1u << 10)
    #define    RTMOCOSTATUSMESSAGE_REFERENCE_POS_NOT_SET    (1u << 11)
    #define    RTMOCOSTATUSMESSAGE_SYSTEM_UNCALIBRATED        (1u << 12)
     
    buf[2] -> seq_type (also set to 0)
    /** Default imaging boffset */
    #define RTMOCO_SEQ_TYPE_IMAGING 0
    /** Wireless marker tracking projection sequence */
    #define RTMOCO_SEQ_TYPE_MTRACK 1
    /** Pause sequence */
    #define RTMOCO_SEQ_TYPE_PAUSE 2
     
    buf[3 to 14] -> 12 pose estimate, the first 9 are the 3x3 rotation matrix, in order of first row-second row - third row, the final 3 is the translation vector in mm.
    """
    b = np.zeros(15, dtype=np.float32)
    b[0] = 0 # Packet type
    b[1] = 0 # Status bitmask
    b[2] = 0 # Seq type
    b[3], b[4], b[5] = pose[0, :3]
    b[6], b[7], b[8] = pose[1, :3]
    b[9], b[10], b[11] = pose[2, :3]
    b[12], b[13], b[14] = pose[:3, 3]
    buf = struct.pack('f' * 15, *b)
    return buf


class AsyncoreClientUDP(asyncore.dispatcher):

    def __init__(self, server, port, tracker, outputlog=None):
        self.server = server
        self.port = port
        self.tracker = tracker

        asyncore.dispatcher.__init__(self)
        self.create_socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_size = 40

        self.recv_errcount = 0

        # For logging purposes.
        self.logger = logging.getLogger(__name__)

        # self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)

        if outputlog:
            self.file_handler = logging.FileHandler(outputlog)
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)

    def writable(self):
        # print('Check')
        return self.tracker.active

    def handle_write(self):
        pose = self.tracker.calculate_pose()
        buf = encode_pose_est(vec2mat(*pose))
        sent = self.socket.sendto(buf, (self.server, self.port))
        self.logger.info('Sent {} bytes to {}'.format(sent, self.server))

    # Once a "connection" is made do this stuff.
    def handle_connect(self):
        self.logger.info('Connected')

    # If a "connection" is closed do this stuff.
    def handle_close(self):
        """Implied by a read event with no data available."""
        self.logger.info('No data received...')

    def readable(self):
        return self.tracker.active

    # If a message has arrived, process it.
    def handle_read(self):

        try:
            data = self.recv(self.recv_size)
            while True:
                # This is to flush out the buffer, in case the scanner is
                # faster than the tracker.
                data = self.recv(self.recv_size)
        except Exception as e:
            self.logger.debug(e)
            self.recv_errcount += 1
        
        try:
            b = struct.unpack('i'*10, data)
            self.logger.debug(b[0])
            self.logger.info('Received {} bytes from {}'.format(len(data), 
                self.server))
        except Exception as e:
            self.logger.debug('Error decoding.')
            self.logger.debug(e)


if __name__ == '__main__':
    from trackers import GUIStereoTracker, DummyTracker
    tracker = DummyTracker
    remote_address = 'localhost'
    port = 4950
    client = AsyncoreClientUDP(remote_address, port, tracker)
    asyncore.loop()
