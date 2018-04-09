.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Changelog
=========

0.15.7
------

- nan related functions
  `#718 <https://github.com/QuantStack/xtensor/pull/718>`_.
- return types fixed in dynamic view helper
  `#722 <https://github.com/QuantStack/xtensor/pull/722>`_.
- xview on constant expressions
  `#723 <https://github.com/QuantStack/xtensor/pull/723>`_.
- added decays to make const ``value_type`` compile
  `#727 <https://github.com/QuantStack/xtensor/pull/727>`_.
- iterator for constant ``strided_view`` fixed
  `#729 <https://github.com/QuantStack/xtensor/pull/729>`_.
- ``strided_view`` on ``xfunction`` fixed
  `#732 <https://github.com/QuantStack/xtensor/pull/732>`_.
- Fixes in ``xstrided_view``
  `#736 <https://github.com/QuantStack/xtensor/pull/736>`_.
- View semantic (broadcast on assign) fixed
  `#742 <https://github.com/QuantStack/xtensor/pull/742>`_.
- Compilation prevented when using ellipsis with ``xview``
  `#743 <https://github.com/QuantStack/xtensor/pull/743>`_.
- Index of ``xiterator`` set to shape when reaching the end
  `#744 <https://github.com/QuantStack/xtensor/pull/744>`_.
- ``xscalar`` fixed
  `#748 <https://github.com/QuantStack/xtensor/pull/748>`_.
- Updated README and related projects
  `#749 <https://github.com/QuantStack/xtensor/pull/749>`_.
- Perfect forwarding in ``xfunction``  and views
  `#750 <https://github.com/QuantStack/xtensor/pull/750>`_.
- Missing include in ``xassign.hpp``
  `#752 <https://github.com/QuantStack/xtensor/pull/752>`_.
- More related projects in the README
  `#754 <https://github.com/QuantStack/xtensor/pull/754>`_.
- Fixed stride computation for ``xtensorf``
  `#755 <https://github.com/QuantStack/xtensor/pull/755>`_.
- Added tests for backstrides
  `#758 <https://github.com/QuantStack/xtensor/pull/758>`_.
- Clean up ``has_raw_data`` ins strided view
  `#759 <https://github.com/QuantStack/xtensor/pull/759>`_.
- Switch to ``ptrdiff_t`` for slices
  `#760 <https://github.com/QuantStack/xtensor/pull/760>`_.
- Fixed ``xview`` strides computation
  `#762 <https://github.com/QuantStack/xtensor/pull/762>`_.
- Additional methods in slices, required for ``xframe``
  `#764 <https://github.com/QuantStack/xtensor/pull/764>`_.

0.15.6
------

- zeros, ones, full and empty_like functions
  `#686 <https://github.com/QuantStack/xtensor/pull/686>`_.
- squeeze view
  `#687 <https://github.com/QuantStack/xtensor/pull/687>`_.
- bitwise shift left and shift right
  `#688 <https://github.com/QuantStack/xtensor/pull/688>`_.
- ellipsis, unique and trim functions
  `#689 <https://github.com/QuantStack/xtensor/pull/689>`_.
- xview iterator benchmark
  `#696 <https://github.com/QuantStack/xtensor/pull/696>`_.
- optimize stepper increment
  `#697 <https://github.com/QuantStack/xtensor/pull/697>`_.
- minmax reducers
  `#698 <https://github.com/QuantStack/xtensor/pull/698>`_.
- where fix with SIMD
  `#704 <https://github.com/QuantStack/xtensor/pull/704>`_.
- additional doc for scalars and views
  `#705 <https://github.com/QuantStack/xtensor/pull/705>`_.
- mixed arithmetic with SIMD
  `#713 <https://github.com/QuantStack/xtensor/pull/713>`_.
- broadcast fixed
  `#717 <https://github.com/QuantStack/xtensor/pull/717>`_.

0.15.5
------

- assign functions optimized 
  `#650 <https://github.com/QuantStack/xtensor/pull/650>`_.
- transposed view fixed
  `#652 <https://github.com/QuantStack/xtensor/pull/652>`_.
- exceptions refactoring
  `#654 <https://github.com/QuantStack/xtensor/pull/654>`_.
- performances improved
  `#655 <https://github.com/QuantStack/xtensor/pull/655>`_.
- view data accessor fixed
  `#660 <https://github.com/QuantStack/xtensor/pull/660>`_.
- new dynamic view using variant
  `#656 <https://github.com/QuantStack/xtensor/pull/656>`_.
- alignment added to fixed xtensor
  `#659 <https://github.com/QuantStack/xtensor/pull/659>`_.
- code cleanup
  `#664 <https://github.com/QuantStack/xtensor/pull/664>`_.
- xtensorf and new dynamic view documentation
  `#667 <https://github.com/QuantStack/xtensor/pull/667>`_.
- qualify namespace for compute_size
  `#665 <https://github.com/QuantStack/xtensor/pull/665>`_.
- make xio use ``dynamic_view`` instead of ``view``
  `#662 <https://github.com/QuantStack/xtensor/pull/662>`_.
- transposed view on any expression
  `#671 <https://github.com/QuantStack/xtensor/pull/671>`_.
- docs typos and grammar plus formatting
  `#676 <https://github.com/QuantStack/xtensor/pull/676>`_.
- index view test assertion fixed
  `#680 <https://github.com/QuantStack/xtensor/pull/680>`_.
- flatten view
  `#678 <https://github.com/QuantStack/xtensor/pull/678>`_.
- handle the case of pointers to const element in ``xadapt``
  `#679 <https://github.com/QuantStack/xtensor/pull/679>`_.
- use quotes in #include statements for xtl
  `#681 <https://github.com/QuantStack/xtensor/pull/681>`_.
- additional constructors for ``svector``
  `#682 <https://github.com/QuantStack/xtensor/pull/682>`_.
- removed ``test_xsemantics.hpp`` from test CMakeLists
  `#684 <https://github.com/QuantStack/xtensor/pull/684>`_.

0.15.4
------

- fix gcc-7 error w.r.t. the use of ``assert``
  `#648 <https://github.com/QuantStack/xtensor/pull/648>`_.

0.15.3
------

- add missing headers to cmake installation and tests
  `#647 <https://github.com/QuantStack/xtensor/pull/647>`_.


0.15.2
------

- ``xshape`` implementation
  `#572 <https://github.com/QuantStack/xtensor/pull/572>`_.
- xfixed container
  `#586 <https://github.com/QuantStack/xtensor/pull/586>`_.
- protected ``xcontainer::derived_cast``
  `#627 <https://github.com/QuantStack/xtensor/pull/627>`_.
- const reference fix
  `#632 <https://github.com/QuantStack/xtensor/pull/632>`_.
- ``xgenerator`` access operators fixed
  `#643 <https://github.com/QuantStack/xtensor/pull/643>`_.
- contiguous layout optiimzation
  `#645 <https://github.com/QuantStack/xtensor/pull/645>`_.


0.15.1
------

- ``xarray_adaptor`` fixed
  `#618 <https://github.com/QuantStack/xtensor/pull/618>`_.
- ``xtensor_adaptor`` fixed
  `#620 <https://github.com/QuantStack/xtensor/pull/620>`_.
- fix in ``xreducer`` steppers
  `#622 <https://github.com/QuantStack/xtensor/pull/622>`_.
- documentation improved
  `#621 <https://github.com/QuantStack/xtensor/pull/621>`_.
  `#623 <https://github.com/QuantStack/xtensor/pull/623>`_.
  `#625 <https://github.com/QuantStack/xtensor/pull/625>`_.
- warnings removed
  `#624 <https://github.com/QuantStack/xtensor/pull/624>`_.

0.15.0
------

Breaking changes
~~~~~~~~~~~~~~~~

- change ``reshape`` to ``resize``, and add throwing ``reshape``
  `#598 <https://github.com/QuantStack/xtensor/pull/598>`_.
- moved to modern cmake
  `#611 <https://github.com/QuantStack/xtensor/pull/611>`_.

New features
~~~~~~~~~~~~

- unravel function
  `#589 <https://github.com/QuantStack/xtensor/pull/589>`_.
- random access iterators
  `#596 <https://github.com/QuantStack/xtensor/pull/596>`_.


Other changes
~~~~~~~~~~~~~

- upgraded to google/benchmark version 1.3.0
  `#583 <https://github.com/QuantStack/xtensor/pull/583>`_.
- ``XTENSOR_ASSERT`` renamed into ``XTENSOR_TRY``, new ``XTENSOR_ASSERT``
  `#603 <https://github.com/QuantStack/xtensor/pull/603>`_.
- ``adapt`` fixed
  `#604 <https://github.com/QuantStack/xtensor/pull/604>`_.
- VC14 warnings removed
  `#608 <https://github.com/QuantStack/xtensor/pull/608>`_.
- ``xfunctor_iterator`` is now a random access iterator
  `#609 <https://github.com/QuantStack/xtensor/pull/609>`_.
- removed ``old-style-cast`` warnings
  `#610 <https://github.com/QuantStack/xtensor/pull/610>`_.

0.14.1
------

New features
~~~~~~~~~~~~

- sort, argmin and argmax
  `#549 <https://github.com/QuantStack/xtensor/pull/549>`_.
- ``xscalar_expression_tag``
  `#582 <https://github.com/QuantStack/xtensor/pull/582>`_.

Other changes
~~~~~~~~~~~~~

- accumulator improvements
  `#570 <https://github.com/QuantStack/xtensor/pull/570>`_.
- benchmark cmake fixed
  `#571 <https://github.com/QuantStack/xtensor/pull/571>`_.
- allocator_type added to container interface
  `#573 <https://github.com/QuantStack/xtensor/pull/573>`_.
- allow conda-forge as fallback channel
  `#575 <https://github.com/QuantStack/xtensor/pull/575>`_.
- arithmetic mixing optional assemblies and scalars fixed
  `#578 <https://github.com/QuantStack/xtensor/pull/578>`_.
- arithmetic mixing optional assemblies and optionals fixed
  `#579 <https://github.com/QuantStack/xtensor/pull/579>`_.
- ``operator==`` restricted to xtensor and xoptional expressions
  `#580 <https://github.com/QuantStack/xtensor/pull/580>`_.

0.14.0
------

Breaking changes
~~~~~~~~~~~~~~~~

- ``xadapt`` renamed into ``adapt``
  `#563 <https://github.com/QuantStack/xtensor/pull/563>`_.
- Naming consistency
  `#565 <https://github.com/QuantStack/xtensor/pull/565>`_.

New features
~~~~~~~~~~~~

- add ``random::choice``
  `#547 <https://github.com/QuantStack/xtensor/pull/547>`_.
- evaluation strategy and accumulators.
  `#550 <https://github.com/QuantStack/xtensor/pull/550>`_.
- modulus operator
  `#556 <https://github.com/QuantStack/xtensor/pull/556>`_.
- ``adapt``: default overload for 1D arrays
  `#560 <https://github.com/QuantStack/xtensor/pull/560>`_.
- Move semantic on ``adapt``
  `#564 <https://github.com/QuantStack/xtensor/pull/564>`_.

Other changes
~~~~~~~~~~~~~

- optional fixes to avoid ambiguous calls
  `#541 <https://github.com/QuantStack/xtensor/pull/541>`_.
- narrative documentation about ``xt::adapt``
  `#544 <https://github.com/QuantStack/xtensor/pull/544>`_.
- ``xfunction`` refactoring
  `#545 <https://github.com/QuantStack/xtensor/pull/545>`_.
- SIMD acceleration for AVX fixed
  `#557 <https://github.com/QuantStack/xtensor/pull/557>`_.
- allocator fixes
  `#558 <https://github.com/QuantStack/xtensor/pull/558>`_.
  `#559 <https://github.com/QuantStack/xtensor/pull/559>`_.
- return type of ``view::strides()`` fixed
  `#568 <https://github.com/QuantStack/xtensor/pull/568>`_.


0.13.2
------

- Support for complex version of ``isclose``
  `#512 <https://github.com/QuantStack/xtensor/pull/512>`_.
- Fixup static layout in ``xstrided_view``
  `#536 <https://github.com/QuantStack/xtensor/pull/536>`_.
- ``xexpression::operator[]`` now take support any type of sequence
  `#537 <https://github.com/QuantStack/xtensor/pull/537>`_.
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

