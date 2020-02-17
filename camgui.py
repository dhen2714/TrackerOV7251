import os
import asyncore
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider, QLabel, QLineEdit, QFormLayout, QInputDialog, QSlider
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
    def __init__(self, camera=None, tracker=None, udp=None):
        super().__init__()
        self.camera = camera

        self.central_widget = QWidget()
        self.image_display = ImageDisplayWidget(self.camera)
        self.button_frame = QPushButton('Acquire single frame', self.central_widget)
        self.button_track = QPushButton('Toggle tracking', self.central_widget)
        self.button_frames = QPushButton('Acquire frames', self.central_widget)

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

        self.button_frame.clicked.connect(self.save_frame)
        self.button_frames.clicked.connect(self.save_frames)
        self.button_savedir.clicked.connect(self.get_savedir)

        self.trackerclient_thread = TrackerClientThread(tracker)
        self.trackerclient_thread.start()

        self.movie_thread = MovieThread(self.camera, tracker, udp)
        self.movie_thread.send_frame.connect(self.update_frame)
        self.button_track.clicked.connect(self.toggle_tracking)
        self.movie_thread.start()

        self.savedir = None
        self.write = False
        self.write_num = 0

        self.track = False
        self.tracker = tracker

    @pyqtSlot(np.ndarray, int)
    def update_frame(self, frame, timestamp):
        """
        Pulls latest frame from the camera stream, updates image display.
        Writes to disk if option toggled.
        """
        qimage = QImage(frame, self.camera.width, self.camera.height, self.camera.qformat)
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
            if cv2.imwrite(outpath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                print('Image saved:', outpath)
            else:
                print('Could not save to', outpath)
            self.write_num -= 1
        else:
            self.write = False

    def update_write(self, write_num, savedir=None):
        self.savedir = savedir
        if self.write:
            print('Already writing!')
        else:
            self.write_num = write_num
            self.write = True

    def save_frame(self):
        self.update_write(1, self.savedir)

    def save_frames(self):
        numframes, ret = QInputDialog.getInt(self, 'Input', 'Enter number of frames to write (>0):')
        if ret and numframes > 0:
            self.update_write(numframes, self.savedir)

    def get_savedir(self):
        savedir, ret = QInputDialog.getText(self, 'Input', 'Enter path to directory in which images are to be saved:')
        if ret:
            self.savedir = str(savedir)
            self.line_edit_savedir.setText(str(savedir))
            self.label_savedir.setText(str(savedir))

    def change_exposure(self):
        # exp = self.slider_exp.value()
        exp = self.widget_exp.slider.value()
        self.camera.set_exposure(exp)

    def toggle_tracking(self):
        print('Sisdf')
        if self.tracker:
            print('What the fuck')
            if self.tracker.active:
                self.tracker.active = False
            else:
                self.tracker.active = True
                print('Tracker turned on')

    def closeEvent(self, event):
        """
        Override of closeEvent, closes down all threads.
        """
        self.movie_thread.stop_running()
        self.movie_thread.quit()
        self.movie_thread.wait()
        event.accept() # Let the window close.


class TrackerClientThread(QThread):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        print('Tracking thread started.')

    def run(self):
        remote_address = 'localhost'
        port = 4950
        client = AsyncoreClientUDP(remote_address, port, self.tracker)
        asyncore.loop()


class MovieThread(QThread):
    send_frame = pyqtSignal(np.ndarray, int)
    def __init__(self, camera, tracker=None, udp=None):
        super().__init__()
        self.camera = camera
        self.track = False
        self.tracker = tracker
        self.udp = udp

        self._running = True

    def update_track(self):
        if self.track:
            self.track = False
        else:
            if self.tracker:
                self.track = True
            else:
                print('No tracker available.')

    def run(self):
        frame = np.zeros((480, 1280), dtype=np.uint8)
        while self._running:
            if self.camera:
                frame, t = self.camera.get_frame()

                self.send_frame.emit(frame, t)
                
            if self.track:
                pose = self.tracker.get_pose(frame)
                if self.udp:
                    sent = udp.send_pose(pose)
                    data, add = udp.receive()

    def stop_running(self):
        self._running = False
        if self.camera:
            self.camera.close()


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