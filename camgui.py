import os
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider, QLabel, QLineEdit, QFormLayout, QInputDialog, QSlider
from PyQt5.QtGui import QImage, QPixmap


class StartWindow(QMainWindow):
    def __init__(self, camera=None, tracker=None):
        super().__init__()
        self.camera = camera

        self.central_widget = QWidget()
        self.img_widget = QLabel()
        self.button_frame = QPushButton('Acquire single frame', self.central_widget)
        self.button_track = QPushButton('Toggle tracking', self.central_widget)
        self.button_frames = QPushButton('Acquire frames', self.central_widget)

        nullimg = np.zeros((480, 1280, 3), dtype=np.uint8)
        qimage = QImage(nullimg, 1280, 480, QImage.Format_Grayscale8)

        self.img_widget.setPixmap(QPixmap.fromImage(qimage))

        # Add save directory option.
        self.widget_savedir = QWidget()
        self.layout_savedir = QFormLayout(self.widget_savedir)
        self.line_edit_savedir = QLineEdit()
        self.label_savedir = QLabel()
        self.button_savedir = QPushButton('Directory for saved images:')
        self.layout_savedir.addRow(self.button_savedir, self.label_savedir)

        # Add slider for exposure control
        self.widget_exp = QWidget()
        self.layout_exp = QVBoxLayout(self.widget_exp)
        self.label_exp = QLabel()
        self.label_exp.setText(str('Exposure control:'))
        self.slider_exp = QSlider(Qt.Horizontal)
        self.slider_exp.setMinimum(0)
        self.slider_exp.setMaximum(31)
        self.slider_exp.setValue(31)
        self.slider_exp.setTickPosition(QSlider.TicksBelow)
        self.slider_exp.setTickInterval(1)
        self.layout_exp.addWidget(self.label_exp)
        self.layout_exp.addWidget(self.slider_exp)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.button_frame)
        self.layout.addWidget(self.button_frames)
        self.layout.addWidget(self.widget_savedir)
        self.layout.addWidget(self.widget_exp)
        self.layout.addWidget(self.img_widget)
        self.layout.addWidget(self.button_track)
        self.setCentralWidget(self.central_widget)

        self.button_frame.clicked.connect(self.get_frame)
        self.button_frames.clicked.connect(self.get_frames)
        self.button_savedir.clicked.connect(self.get_savedir)
        self.slider_exp.valueChanged.connect(self.change_exposure)

        self.movie_thread = MovieThread(self.camera, tracker)
        self.movie_thread.changePixmap.connect(self.update_img)
        self.button_track.clicked.connect(self.movie_thread.update_track)
        self.movie_thread.start()

        self.savedir = None

    @pyqtSlot(QImage)
    def update_img(self, image):
        self.img_widget.setPixmap(QPixmap.fromImage(image))

    def get_frame(self):
        self.movie_thread.update_write(1, self.savedir)

    def get_frames(self):
        numframes, ret = QInputDialog.getInt(self, 'Input', 'Enter number of frames to write (>0):')
        if ret and numframes > 0:
            self.movie_thread.update_write(numframes, self.savedir)

    def get_savedir(self):
        savedir, ret = QInputDialog.getText(self, 'Input', 'Enter path to directory in which images are to be saved:')
        if ret:
            self.savedir = str(savedir)
            # self.line_edit_savedir.setText(str(savedir))
            self.label_savedir.setText(str(savedir))

    def change_exposure(self):
        exp = self.slider_exp.value()
        self.camera.set_exposure(exp)


class MovieThread(QThread):
    changePixmap = pyqtSignal(QImage)
    def __init__(self, camera, tracker=None):
        super().__init__()
        self.camera = camera
        self.track = False
        self.write = False
        self.write_num = 0
        self.savedir = None
        self.tracker = tracker
        self.tracker_instance = None

    def update_track(self):
        if self.track:
            self.track = False
        else:
            if self.tracker:
                self.tracker_instance = self.tracker()
                self.track = True
            else:
                print('No tracker available.')

    def update_write(self, write_num, savedir=None):
        self.savedir = savedir
        if self.write:
            print('Already writing!')
        else:
            self.write_num = write_num
            self.write = True

    def run(self):
        while True:
            frame, t = self.camera.get_frame()

            if self.write and self.write_num:
                outname = '{}.pgm'.format(t)
                if self.savedir:
                    outpath = os.path.join(self.savedir, outname)
                else:
                    outpath = outname
                
                if cv2.imwrite(outpath, frame):
                    print('Image saved:', outpath)
                else:
                    print('Could not save to', outpath)
                self.write_num -= 1
            else:
                self.write = False

            if self.track:
                pose = self.tracker_instance.get_pose(frame)
                print('Pose [Rx Ry Rz x y z]:\n', pose)

            qimage = QImage(frame, 1280, 480, QImage.Format_Grayscale8)
            self.changePixmap.emit(qimage)


if __name__ == '__main__':
    from pyv4l2.camera import Camera
    cam = Camera('/dev/video0')
    app = QApplication([])
    from trackers import GUIStereoTracker
    tracker = None
    tracker = GUIStereoTracker
    window = StartWindow(cam, tracker)
    window.show()
    app.exit(app.exec_())