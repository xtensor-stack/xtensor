from setuptools import setup
from glob import glob
import os

long_description = '''
`xtensor` is a C++ library meant for numerical analysis with multi-dimensional array expressions.

`xtensor` provides

- an extensible expression system enabling **lazy broadcasting**.
- an API following the idioms of the **C++ standard library**.
- tools to manipulate array expressions and build upon `xtensor`.

Containers of `xtensor` are inspired by `NumPy`_, the Python array programming library. **Adaptors** for existing data structures to be plugged into our expression system can easily be written. In fact, `xtensor` can be used to **process numpy data structures inplace** using Python's `buffer protocol`_. For more details on the numpy bindings, check out the xtensor-python_ project.

`xtensor` requires a modern C++ compiler supporting C++14. The following C++ compilers are supported:

- On Windows platforms, Visual C++ 2015 Update 2, or more recent
- On Unix platforms, gcc 4.9 or a recent version of Clang
'''

# Read version information in include/xtensor/xtensor_config.hp'
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'include', 'xtensor', 'xtensor_config.hpp')) as f:
    versions = [line for line in f.readlines() if line.startswith('#define XTENSOR_VERSION_')]
version = versions[0][30:-1] + '.' + versions[1][30:-1] + '.' + versions[2][30:-1]

setup(
    name='xtensor',
    version=version,
    description='Multi-dimensional arrays with broadcasting and lazy computing',
    long_description=long_description,
    url='https://github.com/QuantStack/xtensor',
    zip_safe=False,
    data_files=[
        ('include/xtensor', glob('include/xtensor/*.hpp')),
    ],
)
