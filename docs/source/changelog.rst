.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Changelog
=========

0.13.2
------

- Support for complex version of ``isclose``
  `#512 <https://github.com/QuantStack/xtensor/pull/512>`_.
- Fixup static layout in ``xstrided_view``
  `#536 <https://github.com/QuantStack/xtensor/pull/536>`_.
- ``xexpression::operator[]`` now take support any type of sequence
  `#537 <https://github.com/QuantStack/xtensor/pull/537`_.
- Fixing ``xinfo`` issues for Visual Studio.
  `#529 <https://github.com/QuantStack/xtensor/pull/529>`_.
- Fix const-correctness in ``xstrided_view``.
  `#526 <https://github.com/QuantStack/xtensor/pull/526>`_.


0.13.1
------

- More general floating point type
  `#518 <https://github.com/QuantStack/xtensor/pull/518>`_.
- Do not require functor to be passed via rvalue reference
  `#519 <https://github.com/QuantStack/xtensor/pull/519>`_.
- Documentation improved
  `#520 <https://github.com/QuantStack/xtensor/pull/520>`_.
- Fix in xreducer
  `#521 <https://github.com/QuantStack/xtensor/pull/521>`_.

0.13.0
------

Breaking changes
~~~~~~~~~~~~~~~~

- The API for ``xbuffer_adaptor`` has changed. The template parameter is the type of the buffer, not just the value type
  `#482 <https://github.com/QuantStack/xtensor/pull/482>`_.
- Change ``edge_items`` print option to ``edgeitems`` for better numpy consistency
  `#489 <https://github.com/QuantStack/xtensor/pull/489>`_.
- xtensor now depends on ``xtl`` version `~0.3.3`
  `#508 <https://github.com/QuantStack/xtensor/pull/508>`_.

New features
~~~~~~~~~~~~

- Support for parsing the ``npy`` file format
  `#465 <https://github.com/QuantStack/xtensor/pull/465>`_.
- Creation of optional expressions from value and boolean expressions (optional assembly)
  `#496 <https://github.com/QuantStack/xtensor/pull/496>`_.
- Support for the explicit cast of expressions with different value types
  `#491 <https://github.com/QuantStack/xtensor/pull/491>`_.

Other changes
~~~~~~~~~~~~~

- Addition of broadcasting bitwise operators
  `#459 <https://github.com/QuantStack/xtensor/pull/459>`_.
- More efficient optional expression system
  `#467 <https://github.com/QuantStack/xtensor/pull/467>`_.
- Migration of benchmarks to the Google benchmark framework
  `#473 <https://github.com/QuantStack/xtensor/pull/473>`_.
- Container semantic and adaptor semantic merged
  `#475 <https://github.com/QuantStack/xtensor/pull/475>`_.
- Various fixes and improvements of the strided views
  `#480 <https://github.com/QuantStack/xtensor/pull/480>`_.
  `#481 <https://github.com/QuantStack/xtensor/pull/481>`_.
- Assignment now performs basic type conversion
  `#486 <https://github.com/QuantStack/xtensor/pull/486>`_.
- Workaround for a compiler bug in Visual Studio 2017
  `#490 <https://github.com/QuantStack/xtensor/pull/490>`_.
- MSVC 2017 workaround
  `#492 <https://github.com/QuantStack/xtensor/pull/492>`_.
- The ``size()`` method for containers now returns the total number of elements instead of the buffer size, which may differ when the smallest stride is greater than ``1``
  `#502 <https://github.com/QuantStack/xtensor/pull/502>`_.
- The behavior of ``linspace`` with integral types has been made consistent with numpy
  `#510 <https://github.com/QuantStack/xtensor/pull/510>`_.

0.12.1
------

- Fix issue with slicing when using heterogeneous integral types
  `#451 <https://github.com/QuantStack/xtensor/pull/451>`_.

0.12.0
------

Breaking changes
~~~~~~~~~~~~~~~~

- ``xtensor`` now depends on ``xtl`` version `0.2.x`
  `#421 <https://github.com/QuantStack/xtensor/pull/421>`_.

New features
~~~~~~~~~~~~

- ``xtensor`` has an optional dependency on ``xsimd`` for enabling simd acceleration
  `#426 <https://github.com/QuantStack/xtensor/pull/426>`_.

- All expressions have an additional safe access function (``at``)
  `#420 <https://github.com/QuantStack/xtensor/pull/420>`_.

- norm functions
  `#440 <https://github.com/QuantStack/xtensor/pull/440>`_.

- ``closure_pointer`` used in iterators returning temporaries so their ``operator->`` can be
  correctly defined
  `#446 <https://github.com/QuantStack/xtensor/pull/446>`_.

- expressions tags added so ``xtensor`` expression system can be extended
  `#447 <https://github.com/QuantStack/xtensor/pull/447>`_.

Other changes
~~~~~~~~~~~~~

- Preconditions and exceptions
  `#409 <https://github.com/QuantStack/xtensor/pull/409>`_.

- ``isclose`` is now symmetric
  `#411 <https://github.com/QuantStack/xtensor/pull/411>`_.

- concepts added
  `#414 <https://github.com/QuantStack/xtensor/pull/414>`_.

- narrowing cast for mixed arithmetic
  `#432 <https://github.com/QuantStack/xtensor/pull/432>`_.

- ``is_xexpression`` concept fixed
  `#439 <https://github.com/QuantStack/xtensor/pull/439>`_.

- ``void_t`` implementation fixed for compilers affected by C++14 defect CWG 1558
  `#448 <https://github.com/QuantStack/xtensor/pull/448>`_.

0.11.3
------

- Fixed bug in length-1 statically dimensioned tensor construction
  `#431 <https://github.com/QuantStack/xtensor/pull/431>`_.

0.11.2
------

- Fixup compilation issue with latest clang compiler. (missing `constexpr` keyword)
  `#407 <https://github.com/QuantStack/xtensor/pull/407>`_.

0.11.1
------

- Fixes some warnings in julia and python bindings

0.11.0
------

Breaking changes
~~~~~~~~~~~~~~~~

- ``xbegin`` / ``xend``, ``xcbegin`` / ``xcend``, ``xrbegin`` / ``xrend`` and ``xcrbegin`` / ``xcrend`` methods replaced
  with classical ``begin`` / ``end``, ``cbegin`` / ``cend``, ``rbegin`` / ``rend`` and ``crbegin`` / ``crend`` methods.
  Old ``begin`` / ``end`` methods and their variants have been removed.
  `#370 <https://github.com/QuantStack/xtensor/pull/370>`_.

- ``xview`` now uses a const stepper when its underlying expression is const.
  `#385 <https://github.com/QuantStack/xtensor/pull/385>`_.

Other changes
~~~~~~~~~~~~~

- ``xview`` copy semantic and move semantic fixed.
  `#377 <https://github.com/QuantStack/xtensor/pull/377>`_.

- ``xoptional`` can be implicitly constructed from a scalar.
  `#382 <https://github.com/QuantStack/xtensor/pull/382>`_.

- build with Emscripten fixed.
  `#388 <https://github.com/QuantStack/xtensor/pull/388>`_.

- STL version detection improved.
  `#396 <https://github.com/QuantStack/xtensor/pull/396>`_.

- Implicit conversion between signed and unsigned integers fixed.
  `#397 <https://github.com/QuantStack/xtensor/pull/397>`_.

