.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Changelog
=========

0.20.4
------

- Buffer adaptor default constructor
  `#1524 <https://github.com/QuantStack/xtensor/pull/1524>`_

0.20.3
------

- Fix xbuffer adaptor 
  `#1523 <https://github.com/QuantStack/xtensor/pull/1523>`_

0.20.2
------

- Fixed broadcast linear assign
  `#1493 <https://github.com/QuantStack/xtensor/pull/1493>`_
- Fixed ``do_stirdes_match``
  `#1497 <https://github.com/QuantStack/xtensor/pull/1497>`_
- Removed unused capture
  `#1499 <https://github.com/QuantStack/xtensor/pull/1499>`_
- Upgraded to ``xtl`` 0.6.2
  `#1502 <https://github.com/QuantStack/xtensor/pull/1502>`_
- Added missing methods in ``xshared_expression``
  `#1503 <https://github.com/QuantStack/xtensor/pull/1503>`_
- Fixed iterator types of ``xcontainer``
  `#1504 <https://github.com/QuantStack/xtensor/pull/1504>`_
- Typo correction in external-structure.rst
  `#1505 <https://github.com/QuantStack/xtensor/pull/1505>`_
- Added extension base to adaptors
  `#1507 <https://github.com/QuantStack/xtensor/pull/1507>`_
- Fixed shared expression iterator methods
  `#1509 <https://github.com/QuantStack/xtensor/pull/1509>`_
- Strided view fixes
  `#1512 <https://github.com/QuantStack/xtensor/pull/1512>`_
- Improved range documentation
  `#1515 <https://github.com/QuantStack/xtensor/pull/1515>`_
- Fixed ``ravel`` and ``flatten`` implementation
  `#1511 <https://github.com/QuantStack/xtensor/pull/1511>`_
- Fixed ``xfixed_adaptor`` temporary assign
  `#1516 <https://github.com/QuantStack/xtensor/pull/1516>`_
- Changed struct -> class in ``xiterator_adaptor``
  `#1513 <https://github.com/QuantStack/xtensor/pull/1513>`_
- Fxed ``argmax`` for expressions with strides 0
  `#1519 <https://github.com/QuantStack/xtensor/pull/1519>`_
- Add ``has_linear_assign`` to ``sdynamic_view``
  `#1520 <https://github.com/QuantStack/xtensor/pull/1520>`_

0.20.1
------

- Add a test for mimetype rendering and fix forward declaration
  `#1490 <https://github.com/QuantStack/xtensor/pull/1490>`_
- Fix special case of view iteration
  `#1491 <https://github.com/QuantStack/xtensor/pull/1491>`_

0.20.0
------

Breaking changes
~~~~~~~~~~~~~~~~

- Removed ``xmasked_value`` and ``promote_type_t``
  `#1389 <https://github.com/QuantStack/xtensor/pull/1389>`_
- Removed deprecated type ``slice_vector``
  `#1459 <https://github.com/QuantStack/xtensor/pull/1459>`_
- Upgraded to ``xtl`` 0.6.1
  `#1468 <https://github.com/QuantStack/xtensor/pull/1465>`_
- Added ``keep_dims`` option to reducers
  `#1474 <https://github.com/QuantStack/xtensor/pull/1474>`_
- ``do_strides_match`` now accept an addition base stride value
  `#1479 <https://github.com/QuantStack/xtensor/pull/1479>`_

Other changes
~~~~~~~~~~~~~

- Add ``partition``, ``argpartition`` and ``median``
  `#991 <https://github.com/QuantStack/xtensor/pull/991>`_
- Fix tets on avx512
  `#1410 <https://github.com/QuantStack/xtensor/pull/1410>`_
- Implemented ``xcommon_tensor_t`` with tests
  `#1412 <https://github.com/QuantStack/xtensor/pull/1412>`_
- Code reorganization
  `#1416 <https://github.com/QuantStack/xtensor/pull/1416>`_
- ``reshape`` now accepts ``initializer_list`` parameter
  `#1417 <https://github.com/QuantStack/xtensor/pull/1417>`_
- Improved documentation
  `#1419 <https://github.com/QuantStack/xtensor/pull/1419>`_
- Fixed ``noexcept`` specifier
  `#1418 <https://github.com/QuantStack/xtensor/pull/1418>`_
- ``view`` now accepts lvalue slices
  `#1420 <https://github.com/QuantStack/xtensor/pull/1420>`_
- Removed warnings
  `#1422 <https://github.com/QuantStack/xtensor/pull/1422>`_
- Added ``reshape`` member to ``xgenerator`` to make ``arange`` more flexible
  `#1421 <https://github.com/QuantStack/xtensor/pull/1421>`_
- Add ``std::decay_t`` to ``shape_type`` in strided view
  `#1425 <https://github.com/QuantStack/xtensor/pull/1425>`_
- Generic reshape for ``xgenerator``
  `#1426 <https://github.com/QuantStack/xtensor/pull/1426>`_
- Fix out of bounds accessing in ``xview::compute_strides``
  `#1437 <https://github.com/QuantStack/xtensor/pull/1437>`_
- Added quick reference section to documentation
  `#1438 <https://github.com/QuantStack/xtensor/pull/1438>`_
- Improved getting started CMakeLists.txt
  `#1440 <https://github.com/QuantStack/xtensor/pull/1440>`_
- Added periodic indices
  `#1430 <https://github.com/QuantStack/xtensor/pull/1430>`_
- Added build section to narrative documentation
  `#1442 <https://github.com/QuantStack/xtensor/pull/1442>`_
- Fixed ``linspace`` corner case
  `#1443 <https://github.com/QuantStack/xtensor/pull/1443>`_
- Fixed type-o in documentation
  `#1446 <https://github.com/QuantStack/xtensor/pull/1446>`_
- Added ``xt::xpad``
  `#1441 <https://github.com/QuantStack/xtensor/pull/1441>`_
- Added warning in ``resize`` documentation
  `#1447 <https://github.com/QuantStack/xtensor/pull/1447>`_
- Added ``in_bounds`` method
  `#1444 <https://github.com/QuantStack/xtensor/pull/1444>`_
- ``xstrided_view_base`` is now a CRTP base class
  `#1453 <https://github.com/QuantStack/xtensor/pull/1453>`_
- Turned ``xfunctor_applier_base`` into a CRTP base class
  `#1455 <https://github.com/QuantStack/xtensor/pull/1455>`_
- Removed out of bound access in ``data_offset``
  `#1456 <https://github.com/QuantStack/xtensor/pull/1456>`_
- Added ``xaccessible`` base class
  `#1451 <https://github.com/QuantStack/xtensor/pull/1451>`_
- Refactored ``operator[]``
  `#1460 <https://github.com/QuantStack/xtensor/pull/1460>`_
- Splitted ``xaccessible``
  `#1461 <https://github.com/QuantStack/xtensor/pull/1461>`_
- Refactored ``size``
  `#1462 <https://github.com/QuantStack/xtensor/pull/1462>`_
- Implemented ``nanvar`` and ``nanstd`` with tests
  `#1424 <https://github.com/QuantStack/xtensor/pull/1424>`_
- Removed warnings
  `#1463 <https://github.com/QuantStack/xtensor/pull/1463>`_
- Added ``periodic`` and ``in_bounds`` method to ``xoptional_assembly_base``
  `#1464 <https://github.com/QuantStack/xtensor/pull/1464>`_
- Updated documentation according to last changes
  `#1465 <https://github.com/QuantStack/xtensor/pull/1465>`_
- Fixed ``flatten_sort_result_type``
  `#1470 <https://github.com/QuantStack/xtensor/pull/1470>`_
- Fixed ``unique`` with expressions not defining ``temporary_type``
  `#1472 <https://github.com/QuantStack/xtensor/pull/1472>`_
- Fixed ``xstrided_view_base`` constructor
  `#1473 <https://github.com/QuantStack/xtensor/pull/1473>`_
- Avoid signed integer overflow in integer printer
  `#1475 <https://github.com/QuantStack/xtensor/pull/1475>`_
- Fixed ``xview::inner_backstrides_type``
  `#1480 <https://github.com/QuantStack/xtensor/pull/1480>`_
- Fixed compiler warnings
  `#1481 <https://github.com/QuantStack/xtensor/pull/1481>`_
- ``slice_implementation_getter`` now forwards its lice argument
  `#1486 <https://github.com/QuantStack/xtensor/pull/1486>`_
- ``linspace`` can now be reshaped
  `#1488 <https://github.com/QuantStack/xtensor/pull/1488>`_

0.19.4
------

- Add missing include
  `#1391 <https://github.com/QuantStack/xtensor/pull/1391>`_
- Fixes in xfunctor_view
  `#1393 <https://github.com/QuantStack/xtensor/pull/1393>`_
- Add tests for xfunctor_view
  `#1395 <https://github.com/QuantStack/xtensor/pull/1395>`_
- Add `empty` method to fixed_shape
  `#1396 <https://github.com/QuantStack/xtensor/pull/1396>`_
- Add accessors to slice members
  `#1401 <https://github.com/QuantStack/xtensor/pull/1401>`_
- Allow adaptors on shared pointers
  `#1218 <https://github.com/QuantStack/xtensor/pull/1218>`_
- Fix `eye` with negative index
  `#1406 <https://github.com/QuantStack/xtensor/pull/1406>`_
- Add documentation for shared pointer adaptor
  `#1407 <https://github.com/QuantStack/xtensor/pull/1407>`_
- Add `nanmean` function
  `#1408 <https://github.com/QuantStack/xtensor/pull/1408>`_

0.19.3
------

- Fix arange
  `#1361 <https://github.com/QuantStack/xtensor/pull/1361>`_.
- Adaptors for C stack-allocated arrays
  `#1363 <https://github.com/QuantStack/xtensor/pull/1363>`_.
- Add support for optionals in ``conditional_ternary``
  `#1365 <https://github.com/QuantStack/xtensor/pull/1365>`_.
- Add tests for ternary operator on xoptionals
  `#1368 <https://github.com/QuantStack/xtensor/pull/1368>`_.
- Enable ternary operation for a mix of ``xoptional<value>`` and ``value``
  `#1370 <https://github.com/QuantStack/xtensor/pull/1370>`_.
- ``reduce`` now accepts a single reduction function
  `#1371 <https://github.com/QuantStack/xtensor/pull/1371>`_.
- Implemented share method
  `#1372 <https://github.com/QuantStack/xtensor/pull/1372>`_.
- Documentation of shared improved
  `#1373 <https://github.com/QuantStack/xtensor/pull/1373>`_.
- ``make_lambda_xfunction`` more generic
  `#1374 <https://github.com/QuantStack/xtensor/pull/1374>`_.
- minimum/maximum for ``xoptional``
  `#1378 <https://github.com/QuantStack/xtensor/pull/1378>`_.
- Added missing methods in ``uvector`` and ``svector``
  `#1379 <https://github.com/QuantStack/xtensor/pull/1379>`_.
- Clip ``xoptional_assembly``
  `#1380 <https://github.com/QuantStack/xtensor/pull/1380>`_.
- Improve gtest cmake
  `#1382 <https://github.com/QuantStack/xtensor/pull/1382>`_.
- Implement ternary operator for scalars
  `#1385 <https://github.com/QuantStack/xtensor/pull/1385>`_.
- Added missing ``at`` method in ``uvector`` and ``svector``
  `#1386 <https://github.com/QuantStack/xtensor/pull/1386>`_.
- Fixup binder environment
  `#1387 <https://github.com/QuantStack/xtensor/pull/1387>`_.
- Fixed ``resize`` and ``swap`` of ``svector``
  `#1388 <https://github.com/QuantStack/xtensor/pull/1388>`_.

0.19.2
------

- Enable CI for C++17
  `#1324 <https://github.com/QuantStack/xtensor/pull/1324>`_.
- Fix assignment of masked views
  `#1328 <https://github.com/QuantStack/xtensor/pull/1328>`_.
- Set CMAKE_CXX_STANDARD instead of CMAKE_CXX_FLAGS
  `#1330 <https://github.com/QuantStack/xtensor/pull/1330>`_.
- Allow specifying traversal order to argmin and argmax
  `#1331 <https://github.com/QuantStack/xtensor/pull/1331>`_.
- Update section on differences with NumPy
  `#1336 <https://github.com/QuantStack/xtensor/pull/1336>`_.
- Fix accumulators for shapes containing 1
  `#1337 <https://github.com/QuantStack/xtensor/pull/1337>`_.
- Decouple XTENSOR_DEFAULT_LAYOUT and XTENSOR_DEFAULT_TRAVERSAL
  `#1339 <https://github.com/QuantStack/xtensor/pull/1339>`_.
- Prevent embiguity with `xsimd::reduce`
  `#1343 <https://github.com/QuantStack/xtensor/pull/1343>`_.
- Require `xtl` 0.5.3
  `#1346 <https://github.com/QuantStack/xtensor/pull/1346>`_.
- Use concepts instead of SFINAE
  `#1347 <https://github.com/QuantStack/xtensor/pull/1347>`_.
- Document good practice for xtensor-based API design
  `#1348 <https://github.com/QuantStack/xtensor/pull/1348>`_.
- Fix rich display of tensor expressions
  `#1353 <https://github.com/QuantStack/xtensor/pull/1353>`_.
- Fix xview on fixed tensor
  `#1354 <https://github.com/QuantStack/xtensor/pull/1354>`_.
- Fix issue with `keep_slice` in case of `dynamic_view` on `view`
  `#1355 <https://github.com/QuantStack/xtensor/pull/1355>`_.
- Prevent installation of gtest artifacts
  `#1357 <https://github.com/QuantStack/xtensor/pull/1357>`_.

0.19.1
------

- Add string specialization to ``lexical_cast``
  `#1281 <https://github.com/QuantStack/xtensor/pull/1281>`_.
- Added HDF5 reference for ``xtensor-io``
  `#1284 <https://github.com/QuantStack/xtensor/pull/1284>`_.
- Fixed view index remap issue
  `#1288 <https://github.com/QuantStack/xtensor/pull/1288>`_.
- Fixed gcc 8.2 deleted functions
  `#1289 <https://github.com/QuantStack/xtensor/pull/1289>`_.
- Fixed reducer for 0d input
  `#1292 <https://github.com/QuantStack/xtensor/pull/1292>`_.
- Fixed ``check_element_index``
  `#1295 <https://github.com/QuantStack/xtensor/pull/1295>`_.
- Added comparison functions
  `#1297 <https://github.com/QuantStack/xtensor/pull/1297>`_.
- Add some tests to ensure chrono works with xexpressions
  `#1272 <https://github.com/QuantStack/xtensor/pull/1272>`_.
- Refactor ``functor_view``
  `#1276 <https://github.com/QuantStack/xtensor/pull/1276>`_.
- Documentation improved
  `#1302 <https://github.com/QuantStack/xtensor/pull/1302>`_.
- Implementation of shift operators
  `#1304 <https://github.com/QuantStack/xtensor/pull/1304>`_.
- Make functor adaptor stepper work for proxy specializations 
  `#1305 <https://github.com/QuantStack/xtensor/pull/1305>`_.
- Replaced ``auto&`` with ``auto&&`` in ``assign_to``
  `#1306 <https://github.com/QuantStack/xtensor/pull/1306>`_.
- Fix namespace in ``xview_utils.hpp``
  `#1308 <https://github.com/QuantStack/xtensor/pull/1308>`_.
- Introducing ``flatten_indices`` and ``unravel_indices``
  `#1300 <https://github.com/QuantStack/xtensor/pull/1300>`_.
- Default layout parameter for ``ravel``
  `#1311 <https://github.com/QuantStack/xtensor/pull/1311>`_.
- Fixed ``xvie_stepper``
  `#1317 <https://github.com/QuantStack/xtensor/pull/1317>`_.
- Fixed assignment of view on view 
  `#1314 <https://github.com/QuantStack/xtensor/pull/1314>`_.
- Documented indices
  `#1318 <https://github.com/QuantStack/xtensor/pull/1318>`_.
- Fixed shift operators return type
  `#1319 <https://github.com/QuantStack/xtensor/pull/1319>`_.

0.19.0
------

Breaking changes
~~~~~~~~~~~~~~~~

- Upgraded to ``xtl 0.5``
  `#1275 <https://github.com/QuantStack/xtensor/pull/1275>`_.

Other changes
~~~~~~~~~~~~~

- Removed type-o in docs, minor code style consistency update
  `#1255 <https://github.com/QuantStack/xtensor/pull/1255>`_.
- Removed most of the warnings
  `#1261 <https://github.com/QuantStack/xtensor/pull/1261>`_.
- Optional bitwise fixed
  `#1263 <https://github.com/QuantStack/xtensor/pull/1263>`_.
- Prevent macro expansion in ``std::max``
  `#1265 <https://github.com/QuantStack/xtensor/pull/1265>`_.
- Update numpy.rst
  `#1267 <https://github.com/QuantStack/xtensor/pull/1267>`_.
- Update getting_started.rst
  `#1268 <https://github.com/QuantStack/xtensor/pull/1268>`_.
- keep and drop ``step_size`` fixed
  `#1270 <https://github.com/QuantStack/xtensor/pull/1270>`_.
- Fixed typo in ``xadapt``
  `#1277 <https://github.com/QuantStack/xtensor/pull/1277>`_.
- Fixed typo
  `#1278 <https://github.com/QuantStack/xtensor/pull/1278>`_.

0.18.3
------

- Exporting optional dependencies
  `#1253 <https://github.com/QuantStack/xtensor/pull/1253>`_.
- 0-D HTML rendering
  `#1252 <https://github.com/QuantStack/xtensor/pull/1252>`_.
- Include nlohmann_json in xio for mime bundle repr
  `#1251 <https://github.com/QuantStack/xtensor/pull/1251>`_.
- Fixup xview scalar assignment
  `#1250 <https://github.com/QuantStack/xtensor/pull/1250>`_.
- Implemented `from_indices`
  `#1240 <https://github.com/QuantStack/xtensor/pull/1240>`_.
- xtensor_forward.hpp cleanup
  `#1243 <https://github.com/QuantStack/xtensor/pull/1243>`_.
- default layout-type for `unravel_from_strides` and `unravel_index`
  `#1239 <https://github.com/QuantStack/xtensor/pull/1239>`_.
- xfunction iterator fix
  `#1241 <https://github.com/QuantStack/xtensor/pull/1241>`_.
- xstepper fixes
  `#1237 <https://github.com/QuantStack/xtensor/pull/1237>`_.
- print_options io manipulators
  `#1231 <https://github.com/QuantStack/xtensor/pull/1231>`_.
- Add syntactic sugar for reducer on single axis
  `#1228 <https://github.com/QuantStack/xtensor/pull/1228>`_.
- Added view vs. adapt benchmark
  `#1229 <https://github.com/QuantStack/xtensor/pull/1229>`_.
- added precisions to the installation instructions
  `#1226 <https://github.com/QuantStack/xtensor/pull/1226>`_.
- removed data interface from dynamic view
  `#1225 <https://github.com/QuantStack/xtensor/pull/1225>`_.
- add xio docs
  `#1223 <https://github.com/QuantStack/xtensor/pull/1223>`_.
- Fixup xview assignment
  `#1216 <https://github.com/QuantStack/xtensor/pull/1216>`_.
- documentation updated to be consistent with last changes
  `#1214 <https://github.com/QuantStack/xtensor/pull/1214>`_.
- prevents macro expansion of std::max
  `#1213 <https://github.com/QuantStack/xtensor/pull/1213>`_.
- Fix minor typos
  `#1212 <https://github.com/QuantStack/xtensor/pull/1212>`_.
- Added missing assign operator in xstrided_view 
  `#1210 <https://github.com/QuantStack/xtensor/pull/1210>`_.
- argmax on axis with single element fixed 
  `#1209 <https://github.com/QuantStack/xtensor/pull/1209>`_.

0.18.2
------

- expression tag system fixed
  `#1207 <https://github.com/QuantStack/xtensor/pull/1207>`_.
- optional extension for generator
  `#1206 <https://github.com/QuantStack/xtensor/pull/1206>`_.
- optional extension for ``xview``
  `#1205 <https://github.com/QuantStack/xtensor/pull/1205>`_.
- optional extension for ``xstrided_view``
  `#1204 <https://github.com/QuantStack/xtensor/pull/1204>`_.
- optional extension for reducer
  `#1203 <https://github.com/QuantStack/xtensor/pull/1203>`_.
- optional extension for ``xindex_view``
  `#1202 <https://github.com/QuantStack/xtensor/pull/1202>`_.
- optional extension for ``xfunctor_view``
  `#1201 <https://github.com/QuantStack/xtensor/pull/1201>`_.
- optional extension for broadcast
  `#1198 <https://github.com/QuantStack/xtensor/pull/1198>`_.
- extension API and code cleanup
  `#1197 <https://github.com/QuantStack/xtensor/pull/1197>`_.
- ``xscalar`` optional refactoring
  `#1196 <https://github.com/QuantStack/xtensor/pull/1196>`_.
- Extension mechanism
  `#1192 <https://github.com/QuantStack/xtensor/pull/1192>`_.
- Many small fixes
  `#1191 <https://github.com/QuantStack/xtensor/pull/1191>`_.
- Slight refactoring in ``step_size`` logic
  `#1188 <https://github.com/QuantStack/xtensor/pull/1188>`_.
- Fixup call of const overload in assembly storage
  `#1187 <https://github.com/QuantStack/xtensor/pull/1187>`_.

0.18.1
------

- Fixup xio forward declaration
  `#1185 <https://github.com/QuantStack/xtensor/pull/1185>`_.

0.18.0
------

Breaking changes
~~~~~~~~~~~~~~~~

- Assign and trivial_broadcast refactoring
  `#1150 <https://github.com/QuantStack/xtensor/pull/1150>`_.
- Moved array manipulation functions (``transpose``, ``ravel``, ``flatten``, ``trim_zeros``, ``squeeze``, ``expand_dims``, ``split``, ``atleast_Nd``, ``atleast_1d``, ``atleast_2d``, ``atleast_3d``, ``flip``) from ``xstrided_view.hpp`` to ``xmanipulation.hpp``
  `#1153 <https://github.com/QuantStack/xtensor/pull/1153>`_.
- iterator API improved
  `#1155 <https://github.com/QuantStack/xtensor/pull/1155>`_.
- Fixed ``where`` and ``nonzero`` function behavior to mimic the behavior from NumPy
  `#1157 <https://github.com/QuantStack/xtensor/pull/1157>`_.
- xsimd and functor refactoring
  `#1173 <https://github.com/QuantStack/xtensor/pull/1173>`_.

New features
~~~~~~~~~~~~

- Implement ``rot90``
  `#1153 <https://github.com/QuantStack/xtensor/pull/1153>`_.
- Implement ``argwhere`` and ``flatnonzero``
  `#1157 <https://github.com/QuantStack/xtensor/pull/1157>`_.
- Implemented ``xexpression_holder``
  `#1164 <https://github.com/QuantStack/xtensor/pull/1164>`_.

Other changes
~~~~~~~~~~~~~

- Warnings removed
  `#1159 <https://github.com/QuantStack/xtensor/pull/1159>`_.
- Added missing include 
  `#1162 <https://github.com/QuantStack/xtensor/pull/1162>`_.
- Removed unused type alias in ``xmath/average``
  `#1163 <https://github.com/QuantStack/xtensor/pull/1163>`_.
- Slices improved
  `#1168 <https://github.com/QuantStack/xtensor/pull/1168>`_.
- Fixed ``xdrop_slice``
  `#1181 <https://github.com/QuantStack/xtensor/pull/1181>`_.

0.17.4
------

- perfect forwarding in ``xoptional_function`` constructor
  `#1101 <https://github.com/QuantStack/xtensor/pull/1101>`_.
- fix issue with ``base_simd``
  `#1103 <https://github.com/QuantStack/xtensor/pull/1103>`_.
- ``XTENSOR_ASSERT`` fixed on Windows
  `#1104 <https://github.com/QuantStack/xtensor/pull/1104>`_.
- Implement ``xmasked_value``
  `#1032 <https://github.com/QuantStack/xtensor/pull/1032>`_.
- Added ``setdiff1d`` using stl interface
  `#1109 <https://github.com/QuantStack/xtensor/pull/1109>`_.
- Added test case for ``setdiff1d``
  `#1110 <https://github.com/QuantStack/xtensor/pull/1110>`_.
- Added missing reference to ``diff`` in ``From numpy to xtensor`` section
  `#1116 <https://github.com/QuantStack/xtensor/pull/1116>`_.
- Add ``amax`` and ``amin`` to the documentation
  `#1121 <https://github.com/QuantStack/xtensor/pull/1121>`_.
- ``histogram`` and ``histogram_bin_edges`` implementation
  `#1108 <https://github.com/QuantStack/xtensor/pull/1108>`_.
- Added numpy comparison for interp
  `#1111 <https://github.com/QuantStack/xtensor/pull/1111>`_.
- Allow multiple return type reducer functions
  `#1113 <https://github.com/QuantStack/xtensor/pull/1113>`_.
- Fixes ``average`` bug + adds Numpy based tests
  `#1118 <https://github.com/QuantStack/xtensor/pull/1118>`_.
- Static ``xfunction`` cache for fixed sizes
  `#1105 <https://github.com/QuantStack/xtensor/pull/1105>`_.
- Add negative reshaping axis
  `#1120 <https://github.com/QuantStack/xtensor/pull/1120>`_.
- Updated ``xmasked_view`` using ``xmasked_value``
  `#1074 <https://github.com/QuantStack/xtensor/pull/1074>`_.
- Clean documentation for views
  `#1131 <https://github.com/QuantStack/xtensor/pull/1131>`_.
- Build with ``xsimd`` on Windows fixed
  `#1127 <https://github.com/QuantStack/xtensor/pull/1127>`_.
- Implement ``mime_bundle_repr`` for ``xmasked_view``
  `#1132 <https://github.com/QuantStack/xtensor/pull/1132>`_.
- Modify shuffle to use identical algorithms for any number of dimensions
  `#1135 <https://github.com/QuantStack/xtensor/pull/1135>`_.
- Warnings removal on windows
  `#1139 <https://github.com/QuantStack/xtensor/pull/1135>`_.
- Add permutation function to random
  `#1141 <https://github.com/QuantStack/xtensor/pull/1141>`_.
- ``xfunction_iterator`` permutation
  `#933 <https://github.com/QuantStack/xtensor/pull/933>`_.
- Add ``bincount`` to ``xhistogram``
  `#1140 <https://github.com/QuantStack/xtensor/pull/1140>`_.
- Add contiguous iterable base class and remove layout param from storage iterator
  `#1057 <https://github.com/QuantStack/xtensor/pull/1057>`_.
- Add ``storage_iterator`` to view and strided view
  `#1045 <https://github.com/QuantStack/xtensor/pull/1045>`_.
- Removes ``data_element`` from ``xoptional``
  `#1137 <https://github.com/QuantStack/xtensor/pull/1137>`_.
- ``xtensor`` default constructor and scalar assign fixed
  `#1148 <https://github.com/QuantStack/xtensor/pull/1148>`_.
- Add ``resize / reshape`` to ``xfixed_container``
  `#1147 <https://github.com/QuantStack/xtensor/pull/1147>`_.
- Iterable refactoring
  `#1149 <https://github.com/QuantStack/xtensor/pull/1149>`_.
- ``inner_strides_type`` imported in ``xstrided_view``
  `#1151 <https://github.com/QuantStack/xtensor/pull/1151>`_.

0.17.3
------

- ``xslice`` fix
  `#1099 <https://github.com/QuantStack/xtensor/pull/1099>`_.
- added missing ``static_layout`` in ``xmasked_view``
  `#1100 <https://github.com/QuantStack/xtensor/pull/1100>`_.

0.17.2
------

- Add experimental TBB support for parallelized multicore assign
  `#948 <https://github.com/QuantStack/xtensor/pull/948>`_.
- Add inline statement to all functions in xnpy
  `#1097 <https://github.com/QuantStack/xtensor/pull/1097>`_.
- Fix strided assign for certain assignments
  `#1095 <https://github.com/QuantStack/xtensor/pull/1095>`_.
- CMake, remove gtest warnings
  `#1085 <https://github.com/QuantStack/xtensor/pull/1085>`_.
- Add conversion operators to slices
  `#1093 <https://github.com/QuantStack/xtensor/pull/1093>`_.
- Add optimization to unchecked accessors when contiguous layout is known
  `#1060 <https://github.com/QuantStack/xtensor/pull/1060>`_.
- Speedup assign by computing ``any`` layout on vectors
  `#1063 <https://github.com/QuantStack/xtensor/pull/1063>`_.
- Skip resizing for fixed shapes
  `#1072 <https://github.com/QuantStack/xtensor/pull/1072>`_.
- Add xsimd apply to xcomplex functors (conj, norm, arg)
  `#1086 <https://github.com/QuantStack/xtensor/pull/1086>`_.
- Propagate contiguous layout through views
  `#1039 <https://github.com/QuantStack/xtensor/pull/1039>`_.
- Fix C++17 ambiguity for GCC 7
  `#1081 <https://github.com/QuantStack/xtensor/pull/1081>`_.
- Correct shape type in argmin, fix svector growth
  `#1079 <https://github.com/QuantStack/xtensor/pull/1079>`_.
- Add ``interp`` function to xmath
  `#1071 <https://github.com/QuantStack/xtensor/pull/1071>`_.
- Fix valgrind warnings + memory leak in xadapt
  `#1078 <https://github.com/QuantStack/xtensor/pull/1078>`_.
- Remove more clang warnings & errors on OS X
  `#1077 <https://github.com/QuantStack/xtensor/pull/1077>`_.
- Add move constructor from xtensor <-> xarray
  `#1051 <https://github.com/QuantStack/xtensor/pull/1051>`_.
- Add global support for negative axes in reducers/accumulators
  allow multiple axes in average
  `#1010 <https://github.com/QuantStack/xtensor/pull/1010>`_.
- Fix reference usage in xio
  `#1076 <https://github.com/QuantStack/xtensor/pull/1076>`_.
- Remove occurences of std::size_t and double
  `#1073 <https://github.com/QuantStack/xtensor/pull/1073>`_.
- Add missing parantheses around min/max for MSVC
  `#1061 <https://github.com/QuantStack/xtensor/pull/1061>`_.

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
