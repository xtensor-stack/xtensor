.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xstrided_view
=============

Defined in ``xtensor/xstrided_view.hpp``

.. doxygenclass:: xt::xstrided_view
   :members:

.. doxygentypedef:: xt::xstrided_slice_vector

.. doxygenfunction:: xt::strided_view(E&&, S&&, X&&, std::size_t, layout_type)

.. doxygenfunction:: xt::strided_view(E&&, const xstrided_slice_vector&)

.. doxygenfunction:: xt::reshape_view(E&&, S&&, layout_type)
