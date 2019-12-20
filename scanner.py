import socket
import struct
import sys
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


class UDPConnection:

    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.scanner_add = '10.0.1.3'
        self.port = 4950

        # Send/receive package size in bytes
        self.send_size = 60
        self.rcv_size = 40

        self.socket.bind((self.scanner_add, self.port))

    def sendto(self, data):
        sent = self.socket.sendto(data, (self.scanner_add, self.port))
        return sent

    def receive(self):
        data, add = self.socket.recvfrom(self.rcv_size)
        return data, add

    def send_pose(self, data):
        mat = vec2mat(*data)
        buf = self.encode_motion_est(mat)
        return self.sendto(buf)

    def encode_pose_est(self, motion):
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
        b[3], b[4], b[5] = motion[0, :3]
        b[6], b[7], b[8] = motion[1, :3]
        b[9], b[10], b[11] = motion[2, :3]
        b[12], b[13], b[14] = motion[:3, 3]
        buf = struct.pack('iii' + 'f' * 12, *b)
        return buf

    def decode_scanner_packet(self, data):
        """
        size = 40 bytes (10 integers)
        buf[0] -> frame number (for logging purposes - can be ignored)
        buf[1] -> slice number (for logging purposes - can be ignored)
        buf[2] -> table location (is used for adjusting scanner camera calibration when table position changes. Make necessary changes to your transformation matrix when you receive this.)
        buf[3] -> initial position (set the current or next pose estimate as your reference point and register everything else to this position)
        Rest are reserved.
        """
        b = np.zeros(10, dtype=np.int32)
        return


if __name__ == '__main__':
    import numpy as np
    motion = np.array([4,5,14,2,0,5])
    connection = UDPConnection()
    for i in range(5):

        connection.send_pose(motion)
        connection.receive()