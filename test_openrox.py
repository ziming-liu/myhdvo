#!/usr/bin/env python3
from ctypes import cdll

try:
    lib = cdll.LoadLibrary('/home/bosch_driving/hdvo/rox_odometry_module.so')
    print('✓ Library loaded successfully!')
    print('✓ ddo function found:', lib.ddo)
    print('\nOpenROX compilation and deployment successful!')
except Exception as e:
    print('✗ Error:', e)
