from setuptools import setup, find_packages

setup(name='smacc',
    version='1.0.0',
    description='Segment, Measure and AutoQC the midsagittal Corpus Callosum',
    url=' https://github.com/ShrutiGadewar/smacc',
    python_requires='==3.11',
    include_package_data=True,
    author='Shruti Gadewar',
    author_email='gadewar@usc.edu',
    license='MIT',
    entry_points={
        'console_scripts': [
           'smacc = smacc.main:run_smacc'
        ]
    }
)
