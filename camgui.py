import os
import asyncore
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QMainWindow, 
    QWidget, 
    QPushButton, 
    QVBoxLayout, 
    QApplication, 
    QSlider, 
    QLabel, 
    QLineEdit, 
    QFormLayout, 
    QInputDialog, 
    QSlider
)
from PyQt5.QtGui import QImage, QPixmap
from tracker_client import AsyncoreClientUDP


class ExposureControlWidget(QWidget):
    """
    Get max exposure, min exposure for given camera
    """
    def __init__(self, camera=None, tick_interval=1):
        super().__init__()
        self.camera = camera
        self.tick_interval = tick_interval
        
        if self.camera:
            self.max = self.camera.max_exposure
            self.min = self.camera.min_exposure
        else:
            self.max = 10
            self.min = 0

        self.label = QLabel()
        self.label.setText('Exposure control:')
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(self.min)
        self.slider.setMaximum(self.max)
        self.slider.setValue(self.max)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(self.tick_interval)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self.change_exposure)

    def change_exposure(self):
        if self.camera:
            exp = self.slider.value()
            self.camera.set_exposure(exp)


class ImageDisplayWidget(QLabel):
    """
    Displays frame from camera stream.
    If no camera connected, displays blank image.
    """
    def __init__(self, camera=None):
        super().__init__()
        self.camera = camera

        if self.camera:
            self.width = self.camera.width
            self.height = self.camera.height
            self.channels = self.camera.channels
            self.bitdepth = self.camera.bitdepth
            self.qformat = self.camera.qformat
        else:
            self.width = 640
            self.height = 480
            self.channels = 1
            self.bitdepth = 8
            self.qformat = QImage.Format_Grayscale8

        nullimg = np.zeros((self.height, self.width, self.channels))
        qimg = QImage(nullimg, self.width, self.height, self.qformat)
        self.setPixmap(QPixmap.fromImage(qimg))

    def update_image(self, qimage):
        self.setPixmap(QPixmap.fromImage(qimage))


class StartWindow(QMainWindow):
    """Main GUI window."""
    def __init__(self, camera=None, trackertype=None, remote_address='', 
        port=4950):
        super().__init__()
        self.camera = camera

        self.central_widget = QWidget()
        self.image_display = ImageDisplayWidget(self.camera)
        # Button to save a single camera image.
        self.button_frame = QPushButton('Acquire single frame', 
                                        self.central_widget)
        # Button to toggle motion tracking.
        self.button_track = QPushButton('Toggle tracking', self.central_widget)
        # Button to acquire multiple frames.
        self.button_frames = QPushButton('Acquire frames', self.central_widget)
        # Slider for exposure control.
        self.widget_exp = ExposureControlWidget(self.camera)

        # Add save directory option.
        self.widget_savedir = QWidget()
        self.layout_savedir = QFormLayout(self.widget_savedir)
        self.line_edit_savedir = QLineEdit()
        self.label_savedir = QLabel()
        self.button_savedir = QPushButton('Directory for saved images:')
        self.layout_savedir.addRow(self.button_savedir, self.label_savedir)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.button_frame)
        self.layout.addWidget(self.button_frames)
        self.layout.addWidget(self.widget_savedir)
        self.layout.addWidget(self.widget_exp)
        self.layout.addWidget(self.image_display)
        self.layout.addWidget(self.button_track)
        self.setCentralWidget(self.central_widget)

        # Button clicked triggers.
        self.button_frame.clicked.connect(self.save_frame)
        self.button_frames.clicked.connect(self.save_frames)
        self.button_savedir.clicked.connect(self.get_savedir)
        self.button_track.clicked.connect(self.toggle_tracking)

        # Tracker client thread will start when tracking is toggled.
        self.trackerclient_thread = None
        self.remote_address = remote_address
        self.port = port

        # Thread for displaying camera feed.
        self.movie_thread = MovieThread(self.camera)
        self.movie_thread.send_frame.connect(self.update_frame)
        self.movie_thread.start()

        self.savedir = None # Directory to write images to.
        self.write = False # If writing images to disc.
        self.write_num = 0 # Number of images to write.

        self.trackertype = trackertype
        if self.trackertype:
            self.tracker = trackertype()
        else:
            self.tracker = None
        self.track = False # Set to True if tracking is on.

    @pyqtSlot(np.ndarray, int)
    def update_frame(self, frame, timestamp):
        """
        Pulls latest frame from the camera stream, updates image display.
        Writes to disk if option toggled.
        """
        qimage = QImage(frame, self.camera.width, self.camera.height, 
                        self.camera.qformat)
        self.image_display.update_image(qimage)
        if self.tracker:
            self.tracker.frame = frame
        
        if self.write and self.write_num:
            outname = str(timestamp) + self.camera.savefmt
            if self.savedir:
                outpath = os.path.join(self.savedir, outname)
            else:
                outpath = outname
            # Color has been converted to RGB on the camera end.
            if self.camera.savefmt == '.png':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if cv2.imwrite(outpath, frame):
                print('Image saved:', outpath)
            else:
                print('Could not save to', outpath)
            self.write_num -= 1
        else:
            self.write = False

    def update_write(self, write_num, savedir=None):
        """
        Updates number of frames to write, activated by 'write frames' button.
        """
        self.savedir = savedir
        if self.write:
            print('Already writing!')
        else:
            self.write_num = write_num
            self.write = True

    def save_frame(self):
        """Calls update_write() for a single frame."""
        self.update_write(1, self.savedir)

    def save_frames(self):
        """Calss update_write() for multiple frames."""
        numframes, ret = QInputDialog.getInt(self, 'Input', 
            'Enter number of frames to write (>0):')
        if ret and numframes > 0:
            self.update_write(numframes, self.savedir)

    def get_savedir(self):
        """Input for changing the directory to which images are saved."""
        savedir, ret = QInputDialog.getText(self, 'Input', 'Enter path to \
            directory in which images are to be saved:')
        if ret:
            self.savedir = str(savedir)
            self.line_edit_savedir.setText(str(savedir))
            self.label_savedir.setText(str(savedir))

    def change_exposure(self):
        """Triggered by moving exposure control slider."""
        exp = self.widget_exp.slider.value()
        self.camera.set_exposure(exp)

    def toggle_tracking(self):
        """Toggle tracking on/off. Starts/ends tracking thread."""
        if self.tracker:
            if self.tracker.active:
                self.tracker.active = False
                if self.trackerclient_thread:
                    self.trackerclient_thread.thread_close()
                # Reset tracker after not in use.
                self.tracker = self.trackertype()
            else:
                self.tracker.active = True
                print('Tracker turned on')
                self.trackerclient_thread = TrackerClientThread(
                    self.tracker, 
                    self.remote_address, 
                    self.port
                )
                self.trackerclient_thread.start()
        else:
            print('No tracker available!')

    def closeEvent(self, event):
        """
        Override of closeEvent, closes down all threads.
        """
        self.movie_thread.thread_close()
        if self.trackerclient_thread:
            self.trackerclient_thread.thread_close()
        event.accept() # Let the window close.


class TrackerClientThread(QThread):
    def __init__(self, tracker, remote_address, port):
        super().__init__()
        self.tracker = tracker
        self.remote_address = remote_address
        self.port = port
        self.client = AsyncoreClientUDP(
            self.remote_address, 
            self.port, 
            self.tracker
        )
        print('Tracking thread started.')

    def run(self):
        asyncore.loop()

    def thread_close(self):
        print('Stopping tracking thread, this may throw an exception...')
        asyncore.close_all()
        self.quit()
        self.wait()


class MovieThread(QThread):
    send_frame = pyqtSignal(np.ndarray, int)
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self._running = True

    def run(self):
        frame = np.zeros((480, 1280), dtype=np.uint8)
        while self._running:
            if self.camera:
                frame, t = self.camera.get_frame()

                self.send_frame.emit(frame, t)

    def thread_close(self):
        self._running = False
        if self.camera:
            self.camera.close()
        self.quit()
        self.wait()


if __name__ == '__main__':
    from cameras import Webcam, LIOV7251Stereo
    from trackers import GUIStereoTracker, DummyTracker
    from scanner import UDPConnection
    cam = LIOV7251Stereo('/dev/video0')
    app = QApplication([])

    tracker = GUIStereoTracker()
    # udp = UDPConnection()
    tracker.verbose = False
    window = StartWindow(cam, tracker, udp=None)
    window.show()
    app.exit(app.exec_())