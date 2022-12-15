
.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xaxis_slice_iterator
====================

Defined in ``xtensor/xaxis_slice_iterator.hpp``

.. doxygenclass:: xt::xaxis_slice_iterator
   :members:

.. doxygenfunction:: operator==(const xaxis_slice_iterator<CT>&, const xaxis_slice_iterator<CT>&)


.. doxygenfunction:: operator!=(const xaxis_slice_iterator<CT>&, const xaxis_slice_iterator<CT>&)

.. doxygenfunction:: axis_slice_begin(E&&)

.. doxygenfunction:: axis_slice_begin(E&&, typename std::decay_t<E>::size_type)

.. doxygenfunction:: axis_slice_end(E&&)

.. doxygenfunction:: axis_slice_end(E&&, typename std::decay_t<E>::size_type)
