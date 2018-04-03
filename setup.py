import subprocess
import sys

from pip.req import parse_requirements
from setuptools import find_packages
from setuptools import setup


version = '1.2.2'


if sys.argv[-1] == 'release':
    commands = [
        'python setup.py sdist upload',
        'git tag v{0}'.format(version),
        'git push origin master --tag',
    ]
    for cmd in commands:
        subprocess.call(cmd, shell=True)
    sys.exit(0)


try:
    import cv2  # NOQA
except ImportError:
    print('Please install OpenCV.')
    quit(1)


install_requires = parse_requirements('requirements.txt', session=False)

setup(
    name='chainer-cyclegan',
    version=version,
    packages=find_packages(),
    install_requires=[str(ir.req) for ir in install_requires],
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    url='https://github.com/wkentaro/chainer-cyclegan',
    license='MIT',
)
