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
        'numpy',
        'pandas',
        'scipy',
        'opencv-python',
        'tqdm',
        'matplotlib',
        'pillow',
        'scikit-image',
        'torch',
        'torchvision',
        'svgpathtools',
        'onnx==1.16',
        'onnxscript',
        'onnxruntime==1.18',
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