from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

requirements = [
    'setuptools>=18.0',
    'opencv-python<=3.4.2.17',
    'opencv-contrib-python<=3.4.2.17',
    'pandas',
    'sklearn'
    'PyQt5'
]

extensions = [
    Extension('pyv4l2.camera', ['Pyv4l2/pyv4l2/camera.pyx'],
        include_dirs=[numpy.get_include()],
        libraries=['v4l2']
    ),
    Extension('pyv4l2.controls', ['Pyv4l2/pyv4l2/controls.pyx'],
        include_dirs=[numpy.get_include()],
        libraries=['v4l2']
    ),
    Extension('pyv4l2.exceptions', ['Pyv4l2/pyv4l2/exceptions.py'])
]


setup(
    name='test',
    #version=__version__,
    setup_requires=['cython'],
    install_requires=requirements,
    packages=['Pyv4l2/pyv4l2', 'StereoFeatureTracking/mmt'],
    # packages=find_packages(),
    description='libv4l2 based frame grabber for OV580-OV7251',
    license='GNU Lesser General Public License v3 (LGPLv3)',
    ext_modules=cythonize(extensions, compiler_directives={'language_level' : '3'}),
    zip_safe=False,
    python_requires='<3.8'
)
