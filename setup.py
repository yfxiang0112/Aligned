from setuptools import setup, find_packages

__author__ = "Yuanfang Xiang"
__version__ = "v1.0.0"

setup(name='aligned',
      author=__author__,
      author_email="yf.xiang@smail.nju.edu.cn",
      description="ALIGNED: Adaptive aLignment for Inconsistent Genetic kNowledgE and Data",
      long_description="A neuro-symbolic framework for genetic perturbation prediction that adaptively aligns data-driven learning with biological knowledge",
      version=__version__,
      packages=find_packages(),
      python_requires='>=3.8',
      license='CC BY 4.0',
      url='https://github.com/yfxiang0112/Aligned',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
      ])
