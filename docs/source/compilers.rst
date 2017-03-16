.. Copyright (c) 2016, Johan Mabille and Sylvain Corlay

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Compiler workarounds
====================

This page tracks the workarounds for the various compiler issues that we encountered in the development. This is mostly of interest for developers interested in contributing to xtensor.

Visual Studio 2015 and ``std::enable_if``
-----------------------------------------

With Visual Studio, ``std::enable_if`` evaluates its second argument, even if the condition is false. This is the reason for the presence of the indirection in the implementation of the ``xfunction_type_t`` meta-function.

GCC-4.9 and Clang < 3.8 and constexpr ``std::min`` and ``std::max``
-------------------------------------------------------------------

``std::min`` and ``std::max`` are not constexpr in these compilers. In ``xio.hpp``, we locally define a ``XTENSOR_MIN`` macro used instead of ``std::min``. The macro is undefined right after it is used.

Clang < 3.8 matching ``initializer_list`` with static arrays
------------------------------------------------------------

Old versions of Clang don't handle overload resolution with braced initializer lists correctly: braced initializer lists are not properly matched to static arrays. This prevent compile-time detection of the length of a braced initializer list.

A consequence is that we need to use stack-allocated shape types in these cases. Workarounds for this compiler bug arise in various files of the code base. Everywhere, the handling of `Clang < 3.8` is wrapped with checks for the ``X_OLD_CLANG`` macro.

GCC-6 and the signature of ``std::isnan`` and ``std::isinf``
------------------------------------------------------------

We are not directly using ``std::isnan`` or ``std::isinf`` for the implementation of ``xt::isnan`` and ``xt::isinf``, as a workaround to the following bug in GCC-6 for the following reason.

- C++11 requires that the ``<cmath>`` header declares ``bool std::isnan(double)`` and ``bool std::isinf(double)``.
- C99 requires that the ``<math.h>`` header declares ``int ::isnan(double)`` and ``int ::isinf(double)``.

These two definitions would clash when importing both headers and using namespace std.

As of version 6, GCC detects whether the obsolete functions are present in the C ``<math.h>`` header and uses them if they are, avoiding the clash. However, this means that the function might return int instead of bool as C++11 requires, which is a bug.
