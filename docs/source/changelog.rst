.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Changelog
=========

0.12.0
------

Breaking changes
~~~~~~~~~~~~~~~~

- ``xtensor`` now depends on ``xtl``.
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

