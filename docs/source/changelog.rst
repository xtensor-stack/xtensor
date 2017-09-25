.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Changelog
=========

0.11.3
------

- Fixed bug in length-1 statically dimensioned tensor construction.

0.11.2
------

- Fixup compilation issue with latest clang compiler. (missing `constexpr` keyword).

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

