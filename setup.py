from distutils.core import setup

setup(
  name = 'mothe',         # How you named your package folder (MyLib)
  packages = ['mothe'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'mothe library facilitates detecting and tracking animals in heterogeneous ecological space by using a neural network to study collective behaviour.',   # Give a short description about your library
  author = 'YOUR NAME',                   # Type in your name
  author_email = 'your.email@domain.com',      # Type in your E-Mail
  url = 'https://github.com/user/reponame',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['object detection', 'object tracking', 'kalman filter', 'multiple object detection', 'dataset generation', 'computer vision', 'annotation', 'classification', 'CNN'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'yaml',
          'os',
          'sys',
          'csv',
          'io',
          'cv2',
          'numpy',
          'tensorflow',
          'keras',
          'h5py',
          'matplotlib.pyplot',
          'pandas',
          'scikit-learn',
          'scipy',
          'glob',
          'argparse',
          'filterpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Researchers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
