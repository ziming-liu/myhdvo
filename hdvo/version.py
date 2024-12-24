'''
Author: Ziming Liu
Date: 2021-02-26 22:07:42
LastEditors: Ziming Liu
LastEditTime: 2023-03-09 13:13:05
Team: ACENTAURI team, INRIA
Description: ...
Dependent packages: don't need any extral dependency
'''
 

__version__ = '0.0.1'


def parse_version_info(version_str):
    version_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info.append(int(patch_version[0]))
            version_info.append(f'rc{patch_version[1]}')
    return tuple(version_info)


version_info = parse_version_info(__version__)
