.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xtensor_adaptor
===============

Defined in ``xtensor/xtensor.hpp``

.. doxygenclass:: xt::xtensor_adaptor
   :project: xtensor
   :members:

xadapt (xtensor_adaptor)
========================

Defined in ``xtensor/xadapt.hpp``

.. doxygenfunction:: xt::xadapt(C&&, const std::array<typename std::decay_t<C>::size_type, N>&, layout_type)
   :project: xtensor

.. doxygenfunction:: xt::xadapt(C&&, const std::array<typename std::decay_t<C>::size_type, N>&, const std::array<typename std::decay_t<C>::size_type, N>&)
   :project: xtensor

.. doxygenfunction:: xt::xadapt(P&&, typename A::size_type, O, const std::array<typename A::size_type, N>&, layout_type, const A&)
   :project: xtensor

.. doxygenfunction:: xt::xadapt(P&&, typename A::size_type, O, const std::array<typename A::size_type, N>&, const std::array<typename A::size_type, N>&, const A&)
   :project: xtensor
