from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension,CUDAExtension
import os

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


version_file = 'hdvo/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            elif '@git+' in line:
                info['package'] = line
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == '__main__':
    include_dirs = os.path.dirname(os.path.abspath(__file__))

    setup(
        name='hdvo',
        version=get_version(),
        description='depth estimation Toolbox and Benchmark',
        long_description=readme(),
        maintainer='hdvo2 Authors',
        maintainer_email='ziming.liu@inria.fr',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        ext_modules=[
        #CUDAExtension('grad2_grid_sample', [
        #    os.path.join(include_dirs, 'hdvo/models/utils/cuda_gridsample_grad2/cuda_gridsample_grad2/gridsample_cuda.cpp'),    
        #    os.path.join(include_dirs, 'hdvo/models/utils/cuda_gridsample_grad2/cuda_gridsample_grad2/gridsample_kernel.cu'),
        #    ]),
        #CUDAExtension('esm_grid_sample', [
        #    os.path.join(include_dirs, 'hdvo/models/utils/esm_grid_sample/esm_grid_sample/esm_grid_sample_cuda.cpp'),    
        #    os.path.join(include_dirs, 'hdvo/models/utils/esm_grid_sample/esm_grid_sample/esm_grid_sample_kernel.cu'),
        #    ]),
        #CUDAExtension('fc_grid_sample', [
        #    os.path.join(include_dirs, 'hdvo/models/utils/esm_grid_sample/fc_grid_sample/fc_grid_sample_cuda.cpp'),    
        #    os.path.join(include_dirs, 'hdvo/models/utils/esm_grid_sample/fc_grid_sample/fc_grid_sample_kernel.cu'),
        #    ]),
        #CUDAExtension('ic_grid_sample', [
        #    os.path.join(include_dirs, 'hdvo/models/utils/esm_grid_sample/ic_grid_sample/ic_grid_sample_cuda.cpp'),    
        #    os.path.join(include_dirs, 'hdvo/models/utils/esm_grid_sample/ic_grid_sample/ic_grid_sample_kernel.cu'),
        #    ]),
        ],
        
        cmdclass={
            'build_ext': BuildExtension
        },
        keywords='stereo, mono depth',
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        zip_safe=False)
