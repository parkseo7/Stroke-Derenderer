from setuptools import setup 

setup(
    name='StrokeDerenderer',
    version='2.0',
    description='Converts images of handwritten lines to readable strokes.',
    author='Daniel Park',
    license='Proprietary',
    packages=['derenderer'],
    zip_safe=False,
    python_requires='>=3.10',
    classifiers=[
        'License :: Other/Proprietary License',
    ],
    install_requires=[
        'numpy==2.1.3',
        'pandas==2.2.3',
        'scipy==1.14.1',
        'opencv-python==4.10.0.84',
        'tqdm==4.67.0',
        'matplotlib==3.9.2',
        'pillow==11.0.0',
        'scikit-image==0.24.0',
        'pyyaml==6.0.2',
        'torch==2.5.1',
        'torchvision==0.20.1',
        'svgpathtools==1.6.1',
        'onnx==1.17.0',
        'onnxscript',
        'onnxruntime==1.20.0',
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