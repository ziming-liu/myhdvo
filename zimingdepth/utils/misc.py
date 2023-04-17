'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2021-02-26 22:07:42
LastEditors: Ziming Liu
LastEditTime: 2023-03-29 19:31:45
'''
import ctypes
import random
import string

random.seed(3407)

def get_random_string(length=15):
    """Get random string with letters and digits.

    Args:
        length (int): Length of random string. Default: 15.
    """
    return ''.join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length))


def get_thread_id():
    """Get current thread id."""
    # use ctype to find thread id
    thread_id = ctypes.CDLL('libc.so.6').syscall(186)
    return thread_id


def get_shm_dir():
    """Get shm dir for temporary usage."""
    return '/dev/shm'
