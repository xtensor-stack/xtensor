.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xarray_adaptor
==============

Defined in ``xtensor/xarray.hpp``

.. doxygenclass:: xt::xarray_adaptor
   :members:

adapt
=====

Defined in ``xtensor/xadapt.hpp``

.. doxygenfunction:: xt::adapt(C&&, const SC&, layout_type)

.. doxygenfunction:: xt::adapt(C&&, SC&&, SS&&)

.. doxygenfunction:: xt::adapt(P&&, typename A::size_type, O, const SC&, layout_type, const A&)

.. doxygenfunction:: xt::adapt(P&&, typename A::size_type, O, SC&&, SS&&, const A&)

.. doxygenfunction:: xt::adapt(T (&)[N], const SC&, layout_type)

.. doxygenfunction:: xt::adapt(T (&)[N], SC&&, SS&&)

.. doxygenfunction:: xt::adapt(C&& pointer, const fixed_shape<X...>&);

.. doxygenfunction:: xt::adapt(C&&, layout_type)

.. doxygenfunction:: xt::adapt(P&&, typename A::size_type, O, layout_type, const A&)

.. doxygenfunction:: xt::adapt_smart_ptr(P&&, const SC&, layout_type)

.. doxygenfunction:: xt::adapt_smart_ptr(P&&, const SC&, D&&, layout_type)

.. doxygenfunction:: xt::adapt_smart_ptr(P&&, const I (&)[N], layout_type)

.. doxygenfunction:: xt::adapt_smart_ptr(P&&, const I (&)[N], D&&, layout_type)
