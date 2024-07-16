from setuptools import setup 

setup(
    name='npi-demo',
    version='1.2',
    description='Demo for the NPI engine. Converts images to readable strokes.',
    author='Daniel Park',
    license='Proprietary',
    packages=['npi_demo'],
    zip_safe=False,
    python_requires='>=3.6',
    classifiers=[
        'License :: Other/Proprietary License',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'opencv-python',
        'tqdm',
        'matplotlib',
        'pillow',
        'scikit-image',
        'torch',
        'torchvision'
    ],
    extras_require={
        'test': [
            'mock==4.0.2',
            'pytest==5.4.1',
            'pytest-testdox==1.2.1',
            'pytest-mock==3.0.0',
            'pytest-cov==2.10.1'
        ]
    }
)