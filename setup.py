from setuptools import setup

setup(name='zenml',
      version='0.1',
      description='An easy-to-use data science framework',
      url='https://github.com/bobbywlindsey/zenml',
      author='Bobby Lindsey',
      author_email='bobby.w.lindsey@gmail.com',
      license='MIT',
      packages=['zenml'],
      install_requires=[
          'pandas',
          'sqlalchemy',
          'jaydebeapi',
          'termcolor',
          'pymssql',
          'requests',
          'gensim',
          'numpy',
          'scikit-learn',
          'scipy',
          'missingno',
          'nltk',
          'matplotlib'
      ],
    #   dependency_links=['http://github.com/user/repo/tarball/master#egg=package-1.0'],
      )