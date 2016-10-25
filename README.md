# xtensor

C++ tensor algebra.

## Introduction

`xtensor` is a C++ library meant for numerical analysis with multi-dimensional array expressions.

`xtensor` provides

 - an extensible expression system enabling **lazy broadcasting**.
 - an API following the idioms of the **C++ standard library**.
 - tools to manipulate array expressions and build upon `xtensor`.

The implementation of the containers of `xtensor` is inspired by [NumPy](http://www.numpy.org), the Python array programming library. **Adaptors** for existing data structures to be plugged into our expression system can easily be written. In fact, `xtensor` can be used to **process `numpy` data structures inplace** using Python's [buffer protocol](https://docs.python.org/3/c-api/buffer.html).

`xtensor` requires a modern C++ compiler supporting C++14. The following C+ compilers are supported:

 - On Windows platforms, Visual C++ 2015 Update 2, or more recent
 - On Unix platforms, gcc 4.9 or a recent version of Clang

<!---
## Installation

`xtensor` is a header-only library. We provide a package for the conda package manager.

```bash
conda install -c conda-forge xtensor
```
--->

## Usage

### Basic Usage

Initialize a 2-D array and computing the sum of one of its rows and a 1-D array.

```cpp
#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

xt::array<double> arr1
  {{1.0, 2.0, 3.0},
   {2.0, 5.0, 7.0},
   {2.0, 5.0, 7.0}};

xt::array<double> arr2
  {5.0, 6.0, 7.0};

std::cout << xt::make_xview(arr, 1) + arr2;
```

Outputs:

```
{7.0, 11.0, 14.0}
```

## Lazy Broadcasting with `xtensor`

We can operate on arrays of different shapes of dimensions in an elementwise fashion. Broadcasting rules of xtensor are similar to those of [numpy](http://www.numpy.org) and [libdynd](http://libdynd.org).

### Broadcasting rules

In an operation involving two arrays of different dimensions, the array with the lesser dimensions is broadcast across the leading dimensions of the other.

For example, if `A` has shape `(2, 3)`, and `B` has shape `(4, 2, 3)`, the result of a broadcasted operation with `A` and `B` has shape `(4, 2, 3)`. 

```
   (2, 3) # A
(4, 2, 3) # B
---------
(4, 2, 3) # Result
```

The same rule holds for scalars, which are handled as 0-D expressions. If `A` is scalar, the equation becomes:

```
       () # A
(4, 2, 3) # B
---------
(4, 2, 3) # Result
```

If matched up dimensions of two input arrays are different, and one of them has size 1, it is broadcast to match the size of the other. Let's say B has the shape (4, 2, 1) in the previous example, so the broadcasting happens as follows:

```
   (2, 3) # A
(4, 2, 1) # B
---------
(4, 2, 3) # Result
```

### Universal functions, Laziness and Vectorization

With `xtensor`, if `x`, `y` and `z` are arrays of *broadcastable shapes*, the return type of an expression like `x + y * sin(z)` is **not an array**. It is an `xexpression` object offering the same interface as an N-dimensional array, which does not hold the result. **Values are only computed upon access or when the expression is assigned to an xarray object**. This allows to operate symbolically on very large arrays and then only compute the result on the area of interest.

We provide utilities to **vectorize any scalar function** (taking multiple scalar argument) into a function that will perform on `xexpression`s, apply the lazy broadcasting rules which we just described. Such functions are called *xfunction*s. They are `xtensor`'s counterpart to numpy's universal functions.

In `xtensor`, arithmetic operations (`+`, `-`, `*`, `/`) and all special functions are *xfunction*s.

### Iterating over `xexpression`s and Broadcasting Iterators

All `xexpression`s offer two sets of functions to retrieve iterator pairs (and their `const` counterpart).

 - `begin()` and `end()` provide instances of `xiterator`s which can be used to iterate over all the elements of the expression. The order in which elements are listed is `row-major` in that the index of last dimension is incremented first.
 - `xbegin(shape)` and `xend(shape)` are similar but take a *broadcasting shape* as an argument. Elements are iterated upon in a row-major way, but certain dimensions are repeated to match the provided shape as per the rules described above. For an expression `e`, `e.xbegin(e.shape())` and `e.begin()` are equivalent.

## Building and Running the Tests

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
