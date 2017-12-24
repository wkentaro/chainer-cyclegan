from pip.req import parse_requirements
from setuptools import find_packages
from setuptools import setup


version = '0.0.1'


install_requires = parse_requirements('requirements.txt', session=False)

setup(
    name='chainer_cyclegan',
    version=version,
    packages=find_packages(),
    install_requires=[str(ir.req) for ir in install_requires],
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    url='https://github.com/wkentaro/chainer-cyclegan',
    license='MIT',
)
