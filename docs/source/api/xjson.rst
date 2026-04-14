.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xjson: serialize to/from JSON
=============================

Defined in ``xtensor/io/xjson.hpp``

Available overload families
---------------------------

- ``xt::to_json(nlohmann::basic_json<M>&, const E&)``
- ``xt::from_json(const nlohmann::basic_json<M>&, E&)``

``xt::from_json`` is provided for both container and view semantics.
