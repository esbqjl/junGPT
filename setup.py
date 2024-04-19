from setuptools import setup

setup(name='JUNGPT',
      version='0.0.1',
      author='Wenjun Zhang',
      packages=['gpt'],
      description='A PyTorch re-implementation of GPT',
      license='MIT',
      install_requires=[
            'torch',
      ],
)