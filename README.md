# xtensor

C++ tensor algebra.

## Building and Running the Tests

xtensor requires a recent compiler supporting C++14. We support:

 - Visual Studio 2015 Update 2, or more recent
 - gcc 4.9 or more recent
 - recent versions of clang

Building the tests requires the [GTest](https://github.com/google/googletest) testing framework and [cmake](https://cmake.org).

gtest and cmake are available as a packages for most linux distributions. They can also be installed with the `conda` package manager:

```bash
conda install -c conda-forge gtest cmake
```

Once `gtest` and `cmake` are installed, you can build and run the tests:

```bash
cd test
cmake .
make
./test_xtensor
```

In the context of continuous integration with Travis CI, tests are run in a `conda` environment, which can be activated with

```bash
cd test
conda env create -f ./test-environment.yml
source activate test-xtensor
cmake .
make
./test_xtensor
```

## Building the HTML Documentation

xtensor's documentation is built with three tools

 - [doxygen](http://www.doxygen.org)
 - [sphinx](http://www.sphinx-doc.org)
 - [breathe](https://breathe.readthedocs.io)

While doxygen must be installed separately, you can install breathe by typing

```bash
pip install breathe
``` 

Breathe can also be installed with `conda`

```bash
conda install -c conda-forge breathe
```

Finally, build the documentation with

```bash
make html
```

from the `docs` subdirectory.

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

This software is licensed under the BSD-3-Clause license. See the [LICENSE](LICENSE) file for details.
