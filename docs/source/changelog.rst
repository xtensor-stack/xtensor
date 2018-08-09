.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Changelog
=========

0.17.1
------

- Add std namespace to size_t everywhere, remove std::copysign for MSVC
  `#1053 <https://github.com/QuantStack/xtensor/pull/1053>`_.
- Fix (wrong) bracket warnings for older clang versions (e.g. clang 5 on OS X)
  `#1050 <https://github.com/QuantStack/xtensor/pull/1050>`_.
- Fix strided view on view by using std::addressof
  `#1049 <https://github.com/QuantStack/xtensor/pull/1049>`_.
- Add more adapt functions and shorthands
  `#1043 <https://github.com/QuantStack/xtensor/pull/1043>`_.
- Improve CRTP base class detection
  `#1041 <https://github.com/QuantStack/xtensor/pull/1041>`_.
- Fix rebind container ambiguous template for C++17 / GCC 8 regression
  `#1038 <https://github.com/QuantStack/xtensor/pull/1038>`_.
- Fix functor return value
  `#1035 <https://github.com/QuantStack/xtensor/pull/1035>`_.

0.17.0
------

Breaking changes
~~~~~~~~~~~~~~~~

- Changed strides to ``std::ptrdiff_t``
  `#925 <https://github.com/QuantStack/xtensor/pull/925>`_.
- Renamed ``count_nonzeros`` in ``count_nonzero``
  `#974 <https://github.com/QuantStack/xtensor/pull/974>`_.
- homogenize ``xfixed`` constructors
  `#970 <https://github.com/QuantStack/xtensor/pull/970>`_.
- Improve ``random::choice``
  `#1011 <https://github.com/QuantStack/xtensor/pull/1011>`_.

New features
~~~~~~~~~~~~

- add ``signed char`` to npy deserialization format
  `#1017 <https://github.com/QuantStack/xtensor/pull/1017>`_.
- simd assignment now requires convertible types instead of same type
  `#1000 <https://github.com/QuantStack/xtensor/pull/1000>`_.
- shared expression and automatic xclosure detection
  `#992 <https://github.com/QuantStack/xtensor/pull/992>`_.
- average function
  `#987 <https://github.com/QuantStack/xtensor/pull/987>`_.
- added simd support for complex
  `#985 <https://github.com/QuantStack/xtensor/pull/985>`_.
- argsort function
  `#977 <https://github.com/QuantStack/xtensor/pull/977>`_.
- propagate fixed shape
  `#922 <https://github.com/QuantStack/xtensor/pull/922>`_.
- added xdrop_slice
  `#972 <https://github.com/QuantStack/xtensor/pull/972>`_.
- added doc for ``xmasked_view``
  `#971 <https://github.com/QuantStack/xtensor/pull/971>`_.
- added ``xmasked_view``
  `#969 <https://github.com/QuantStack/xtensor/pull/969>`_.
- added ``dynamic_view``
  `#966 <https://github.com/QuantStack/xtensor/pull/966>`_.
- added ability to use negative indices in keep slice
  `#964 <https://github.com/QuantStack/xtensor/pull/964>`_.
- added an easy way to create lambda expressions, square and cube
  `#961 <https://github.com/QuantStack/xtensor/pull/961>`_.
- noalias on rvalue
  `#965 <https://github.com/QuantStack/xtensor/pull/965>`_.

Other changes
~~~~~~~~~~~~~

- ``xshared_expression`` fixed
  `#1025 <https://github.com/QuantStack/xtensor/pull/1025>`_.
- fix ``make_xshared``
  `#1024 <https://github.com/QuantStack/xtensor/pull/1024>`_.
- add tests to evaluate shared expressions
  `#1019 <https://github.com/QuantStack/xtensor/pull/1019>`_.
- fix ``where`` on ``xview``
  `#1012 <https://github.com/QuantStack/xtensor/pull/1012>`_.
- basic usage replaced with getting started
  `#1004 <https://github.com/QuantStack/xtensor/pull/1004>`_.
- avoided installation failure in absence of ``nlohmann_json``
  `#1001 <https://github.com/QuantStack/xtensor/pull/1001>`_.
- code and documentation clean up
  `#998 <https://github.com/QuantStack/xtensor/pull/998>`_.
- removed g++ "pedantic" compiler warnings
  `#997 <https://github.com/QuantStack/xtensor/pull/997>`_.
- added missing header in basic_usage.rst
  `#996 <https://github.com/QuantStack/xtensor/pull/996>`_.
- warning pass
  `#990 <https://github.com/QuantStack/xtensor/pull/990>`_.
- added missing include in ``xview``
  `#989 <https://github.com/QuantStack/xtensor/pull/989>`_.
- added missing ``<map>`` include
  `#983 <https://github.com/QuantStack/xtensor/pull/983>`_.
- xislice refactoring
  `#962 <https://github.com/QuantStack/xtensor/pull/962>`_.
- added missing operators to noalias
  `#932 <https://github.com/QuantStack/xtensor/pull/932>`_.
- cmake fix for Intel compiler on Windows
  `#951 <https://github.com/QuantStack/xtensor/pull/951>`_.
- fixed xsimd abs deduction
  `#946 <https://github.com/QuantStack/xtensor/pull/946>`_.
- added islice example to view doc
  `#940 <https://github.com/QuantStack/xtensor/pull/940>`_.

0.16.4
------

- removed usage of ``std::transfomr`` in assign
  `#868 <https://github.com/QuantStack/xtensor/pull/868>`_.
- add strided assignment
  `#901 <https://github.com/QuantStack/xtensor/pull/901>`_.
- simd activated for conditional ternary functor
  `#903 <https://github.com/QuantStack/xtensor/pull/903>`_.
- ``xstrided_view`` split
  `#905 <https://github.com/QuantStack/xtensor/pull/905>`_.
- assigning an expression to a view throws if it has more dimensions
  `#910 <https://github.com/QuantStack/xtensor/pull/910>`_.
- faster random
  `#913 <https://github.com/QuantStack/xtensor/pull/913>`_.
- ``xoptional_assembly_base`` storage type
  `#915 <https://github.com/QuantStack/xtensor/pull/915>`_.
- new tests and warning pass
  `#916 <https://github.com/QuantStack/xtensor/pull/916>`_.
- norm immediate reducer
  `#924 <https://github.com/QuantStack/xtensor/pull/924>`_.
- add ``reshape_view``
  `#927 <https://github.com/QuantStack/xtensor/pull/927>`_.
- fix immediate reducers with 0 strides
  `#935 <https://github.com/QuantStack/xtensor/pull/935>`_.

0.16.3
------

- simd on mathematical functions fixed
  `#886 <https://github.com/QuantStack/xtensor/pull/886>`_.
- ``fill`` method added to containers
  `#887 <https://github.com/QuantStack/xtensor/pull/887>`_.
- access with more arguments than dimensions
  `#889 <https://github.com/QuantStack/xtensor/pull/889>`_.
- unchecked method implemented
  `#890 <https://github.com/QuantStack/xtensor/pull/890>`_.
- ``fill`` method implemented in view
  `#893 <https://github.com/QuantStack/xtensor/pull/893>`_.
- documentation fixed and warnings removed
  `#894 <https://github.com/QuantStack/xtensor/pull/894>`_.
- negative slices and new range syntax
  `#895 <https://github.com/QuantStack/xtensor/pull/895>`_.
- ``xview_stepper`` with implicit ``xt::all`` bug fix
  `#899 <https://github.com/QuantStack/xtensor/pull/899>`_.

0.16.2
------

- Add include of ``xview.hpp`` in example
  `#884 <https://github.com/QuantStack/xtensor/pull/884>`_.
- Remove ``FS`` identifier
  `#885 <https://github.com/QuantStack/xtensor/pull/885>`_.

0.16.1
------

- Workaround for Visual Studio Bug
  `#858 <https://github.com/QuantStack/xtensor/pull/858>`_.
- Fixup example notebook
  `#861 <https://github.com/QuantStack/xtensor/pull/861>`_.
- Prevent expansion of min and max macros on Windows
  `#863 <https://github.com/QuantStack/xtensor/pull/863>`_.
- Renamed ``m_data`` to ``m_storage``
  `#864 <https://github.com/QuantStack/xtensor/pull/864>`_.
- Fix regression with respect to random access stepping with views
  `#865 <https://github.com/QuantStack/xtensor/pull/865>`_.
- Remove use of CS, DS and ES qualifiers for Solaris builds
  `#866 <https://github.com/QuantStack/xtensor/pull/866>`_.
- Removal of precision type
  `#870 <https://github.com/QuantStack/xtensor/pull/870>`_.
- Make json tests optional, bump xtl/xsimd versions
  `#871 <https://github.com/QuantStack/xtensor/pull/871>`_.
- Add more benchmarks
  `#876 <https://github.com/QuantStack/xtensor/pull/876>`_.
- Forbid simd fixed
  `#877 <https://github.com/QuantStack/xtensor/pull/877>`_.
- Add more asserts
  `#879 <https://github.com/QuantStack/xtensor/pull/879>`_.
- Add missing ``batch_bool`` typedef
  `#881 <https://github.com/QuantStack/xtensor/pull/881>`_.
- ``simd_return_type`` hack removed
  `#882 <https://github.com/QuantStack/xtensor/pull/882>`_.
- Removed test guard and fixed dimension check in ``xscalar``
  `#883 <https://github.com/QuantStack/xtensor/pull/883>`_.

0.16.0
------

Breaking changes
~~~~~~~~~~~~~~~~

- ``data`` renamed in ``storage``, ``raw_data`` renamed in ``data``
  `#792 <https://github.com/QuantStack/xtensor/pull/792>`_.
- Added layout template parameter to ``xstrided_view``
  `#796 <https://github.com/QuantStack/xtensor/pull/796>`_.
- Remove equality operator from stepper
  `#824 <https://github.com/QuantStack/xtensor/pull/824>`_.
- ``dynamic_view`` renamed in ``strided_view``
  `#832 <https://github.com/QuantStack/xtensor/pull/832>`_.
- ``xtensorf`` renamed in ``xtensor_fixed``
  `#846 <https://github.com/QuantStack/xtensor/pull/846>`_.

New features
~~~~~~~~~~~~

- Added strided view selector
  `#765 <https://github.com/QuantStack/xtensor/pull/765>`_.
- Added ``count_nonzeros``
  `#781 <https://github.com/QuantStack/xtensor/pull/781>`_.
- Added implicit conversion to scalar in ``xview``
  `#788 <https://github.com/QuantStack/xtensor/pull/788>`_.
- Added tracking allocators to ``xutils.hpp``
  `#789 <https://github.com/QuantStack/xtensor/pull/789>`_.
- ``xindexslice`` and ``shuffle`` function
  `#804 <https://github.com/QuantStack/xtensor/pull/804>`_.
- Allow ``xadapt`` with dynamic layout
  `#816 <https://github.com/QuantStack/xtensor/pull/816>`_.
- Added ``xtensorf`` initialization from C array
  `#819 <https://github.com/QuantStack/xtensor/pull/819>`_.
- Added policy to allocation tracking for throw option
  `#820 <https://github.com/QuantStack/xtensor/pull/820>`_.
- Free function ``empty`` for construction from shape
  `#827 <https://github.com/QuantStack/xtensor/pull/827>`_.
- Support for JSON serialization and deserialization of xtensor expressions
  `#830 <https://github.com/QuantStack/xtensor/pull/830>`_.
- Add ``trapz`` function
  `#837 <https://github.com/QuantStack/xtensor/pull/837>`_.
- Add ``diff`` and ``trapz(y, x)`` functions
  `#841 <https://github.com/QuantStack/xtensor/pull/841>`_.

Other changes
~~~~~~~~~~~~~

- Added fast path for specific assigns
  `#767 <https://github.com/QuantStack/xtensor/pull/767>`_.
- Renamed internal macros to prevent collisions
  `#772 <https://github.com/QuantStack/xtensor/pull/772>`_.
- ``dynamic_view`` unwrapping
  `#775 <https://github.com/QuantStack/xtensor/pull/775>`_.
- ``xreducer_stepper`` copy semantic fixed
  `#785 <https://github.com/QuantStack/xtensor/pull/785>`_.
- ``xfunction`` copy constructor fixed
  `#787 <https://github.com/QuantStack/xtensor/pull/787>`_.
- warnings removed
  `#791 <https://github.com/QuantStack/xtensor/pull/791>`_.
- ``xscalar_stepper`` fixed
  `#802 <https://github.com/QuantStack/xtensor/pull/802>`_.
- Fixup ``xadapt`` on const pointers
  `#809 <https://github.com/QuantStack/xtensor/pull/809>`_.
- Fix in owning buffer adaptors
  `#810 <https://github.com/QuantStack/xtensor/pull/810>`_.
- Macros fixup
  `#812 <https://github.com/QuantStack/xtensor/pull/812>`_.
- More fixes in ``xadapt``
  `#813 <https://github.com/QuantStack/xtensor/pull/813>`_.
- Mute unused variable warning
  `#815 <https://github.com/QuantStack/xtensor/pull/815>`_.
- Remove comparison of steppers in assign loop
  `#823 <https://github.com/QuantStack/xtensor/pull/823>`_.
- Fix reverse iterators
  `#825 <https://github.com/QuantStack/xtensor/pull/825>`_.
- gcc-8 fix for template method calls
  `#833 <https://github.com/QuantStack/xtensor/pull/833>`_.
- refactor benchmarks for upcoming release
  `#842 <https://github.com/QuantStack/xtensor/pull/842>`_.
- ``flip`` now returns a view
  `#843 <https://github.com/QuantStack/xtensor/pull/843>`_.
- initial warning pass
  `#850 <https://github.com/QuantStack/xtensor/pull/850>`_.
- Fix warning on diff function
  `#851 <https://github.com/QuantStack/xtensor/pull/851>`_.
- xsimd assignment fixed
  `#852 <https://github.com/QuantStack/xtensor/pull/852>`_.

0.15.9
------

- missing layout method in xfixed
  `#777 <https://github.com/QuantStack/xtensor/pull/777>`_.
- fixed uninitialized backstrides
  `#774 <https://github.com/QuantStack/xtensor/pull/774>`_.
- update xtensor-blas in binder
  `#773 <https://github.com/QuantStack/xtensor/pull/773>`_.

0.15.8
------

- comparison operators for slices
  `#770 <https://github.com/QuantStack/xtensor/pull/770>`_.
- use default-assignable layout for strided views.
  `#769 <https://github.com/QuantStack/xtensor/pull/769>`_.

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

