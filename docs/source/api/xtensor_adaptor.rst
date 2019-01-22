.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xtensor_adaptor
===============

Defined in ``xtensor/xtensor.hpp``

.. doxygenclass:: xt::xtensor_adaptor
   :project: xtensor
   :members:

adapt (xtensor_adaptor)
========================

Defined in ``xtensor/xadapt.hpp``

.. doxygenfunction:: xt::adapt(C&&, layout_type)
   :project: xtensor

.. doxygenfunction:: xt::adapt(C&&, const SC&, layout_type)
   :project: xtensor

.. doxygenfunction:: xt::adapt(C&&, SC&&, SS&&)
   :project: xtensor

.. doxygenfunction:: xt::adapt(P&&, typename A::size_type, O, layout_type, const A&)
   :project: xtensor

.. doxygenfunction:: xt::adapt(P&&, typename A::size_type, O, const SC&, layout_type, const A&)
   :project: xtensor

.. doxygenfunction:: xt::adapt(P&&, typename A::size_type, O, SC&&, SS&&, const A&)
   :project: xtensor

.. doxygenfunction:: xt::adapt(T (&)[N], const SC&, layout_type)
   :project: xtensor

.. doxygenfunction:: xt::adapt(T (&)[N], SC&&, SS&&)
   :project: xtensor
