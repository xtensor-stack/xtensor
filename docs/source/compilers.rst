.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Compiler workarounds
====================

This page tracks the workarounds for the various compiler issues that we
encountered in the development. This is mostly of interest for developers
interested in contributing to xtensor.

Visual Studio 2015 and ``std::enable_if``
-----------------------------------------

With Visual Studio, ``std::enable_if`` evaluates its second argument, even if
the condition is false. This is the reason for the presence of the indirection
in the implementation of the ``xfunction_type_t`` meta-function.

Visual Studio 2017 and alias templates with non-class template parameters and multiple aliasing levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alias template with non-class parameters only, and multiple levels of aliasing
are not properly considered as types by Visual Studio 2017. The base
``xcontainer`` template class underlying xtensor container types has such alias
templates defined. We avoid the multiple levels of aliasing in the case of Visual
Studio.

Visual Studio and ``min`` and ``max`` macros
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visual Studio defines ``min`` and ``max`` macros causing calls to e.g.
``std::min`` and ``std::max`` to be interpreted as syntax errors. The
``NOMINMAX`` definition may be used to disable these macros.

In xtensor, to prevent macro replacements of ``min`` and ``max`` functions, we
wrap them with parentheses, so that client code does not need the ``NOMINMAX``
definition.

Visual Studio 2017 (15.7.1) seeing declarations as extra overloads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``xvectorize.hpp``, Visual Studio 15.7.1 sees the forward declaration of ``vectorize(E&&)`` as a separate overload.

Visual Studio 2017 double non-class parameter pack expansion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ``xfixed.hpp`` we add a level of indirection to expand one parameter pack before the other.
Not doing this results in VS2017 complaining about a parameter pack that needs to be expanded in this
context while it actually is.

GCC-4.9 and Clang < 3.8 and constexpr ``std::min`` and ``std::max``
-------------------------------------------------------------------

``std::min`` and ``std::max`` are not constexpr in these compilers. In
``xio.hpp``, we locally define a ``XTENSOR_MIN`` macro used instead of
``std::min``. The macro is undefined right after it is used.

Clang < 3.8 not matching ``initializer_list`` with static arrays
----------------------------------------------------------------

Old versions of Clang don't handle overload resolution with braced initializer
lists correctly: braced initializer lists are not properly matched to static
arrays. This prevent compile-time detection of the length of a braced
initializer list.

A consequence is that we need to use stack-allocated shape types in these cases.
Workarounds for this compiler bug arise in various files of the code base.
Everywhere, the handling of `Clang < 3.8` is wrapped with checks for the
``X_OLD_CLANG`` macro.

GCC < 5.1 and ``std::is_trivially_default_constructible``
---------------------------------------------------------

The versions of the STL shipped with versions of GCC older than 5.1 are missing
a number of type traits, such as ``std::is_trivially_default_constructible``.
However, for some of them, equivalent type traits with different names are
provided, such as ``std::has_trivial_default_constructor``.

In this case, we polyfill the proper standard names using the deprecated
``std::has_trivial_default_constructor``. This must also be done when the
compiler is clang when it makes use of the GCC implementation of the STL,
which is the default behavior on linux. Properly detecting the version of the
GCC STL used by clang cannot be done with the ``__GNUC__``  macro, which is
overridden by clang. Instead, we check for the definition of the macro
``_GLIBCXX_USE_CXX11_ABI`` which is only defined with GCC versions greater than
``5``.

GCC-6 and the signature of ``std::isnan`` and ``std::isinf``
------------------------------------------------------------

We are not directly using ``std::isnan`` or ``std::isinf`` for the
implementation of ``xt::isnan`` and ``xt::isinf``, as a workaround to the
following bug in GCC-6 for the following reason.

- C++11 requires that the ``<cmath>`` header declares ``bool std::isnan(double)`` and ``bool std::isinf(double)``.
- C99 requires that the ``<math.h>`` header declares ``int ::isnan(double)`` and ``int ::isinf(double)``.

These two definitions would clash when importing both headers and using namespace std.

As of version 6, GCC detects whether the obsolete functions are present in the
C header ``<math.h>`` and uses them if they are, avoiding the clash. However,
this means that the function might return int instead of bool as C++11
requires, which is a bug.

GCC-8 and deleted functions
---------------------------

GCC-8 (8.2 specifically) doesn't seem to SFINAE deleted functions correctly. A
strided view on a dynamic_view errors with a message: use of deleted function.
It should pick the *other* implementation by SFINAE on the function
signature, because our ``has_strides<dynamic_view>`` meta-function should return
false. Instantiating the ``has_strides<dynamic_view>`` in the inner_types fixes the issue.
Original issue here: https://github.com/xtensor-stack/xtensor/issues/1273

Apple LLVM version >= 8.0.0
---------------------------

``tuple_cat`` is bugged and propagates the constness of its tuple arguments to the types
inside the tuple. When checking if the resulting tuple contains a given type, the const
qualified type also needs to be checked.
