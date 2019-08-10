# Simple Python Animation Module
A basic, general-purpose module that provides classes and functions related to 3D animation and transforms.

## Purpose
This module was created as a general solution for retargeting 3D animations. Retargeting is the process of reusing an animation designed for one skeleton and applying it to a different skeleton. The module handles skeletons that may differ in bone orientation, but are similarly structured hierarchially.

Bipedal skeletons are a good example, as different games/applications define their bones differently but the structure is the same: i.e, feet, toes, arms, head, torso, etc.

## Usage
This was written for Python 3.0+. In addition, the following libraries have to be installed in order to use this module:
- [numpy](https://github.com/numpy/numpy)
- [transformations](https://pypi.org/project/transformations)
