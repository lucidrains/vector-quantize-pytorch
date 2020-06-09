from setuptools import setup, find_packages

setup(
  name = 'vector_quantize_pytorch',
  packages = find_packages(),
  version = '0.0.2',
  license='MIT',
  description = 'Simple Vector Quantization, in Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/vector-quantizer-pytorch',
  keywords = ['artificial intelligence', 'deep learning', 'pytorch'],
  install_requires=[
      'torch'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)