#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# for your packages to be recognized by python
d = generate_distutils_setup(
 packages=['mbot_nlu_bert', 'mbot_nlu_bert_ros'],
 package_dir={'mbot_nlu_bert': 'common/src/mbot_nlu_bert', 'mbot_nlu_bert_ros': 'ros/src/mbot_nlu_bert_ros'}
)

setup(**d)
