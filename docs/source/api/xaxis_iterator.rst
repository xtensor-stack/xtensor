.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xaxis_iterator
==============

Defined in ``xtensor/xaxis_iterator.hpp``

.. doxygenclass:: xt::xaxis_iterator
   :members:

.. doxygenfunction:: operator==(const xaxis_iterator<CT>&, const xaxis_iterator<CT>&)


.. doxygenfunction:: operator!=(const xaxis_iterator<CT>&, const xaxis_iterator<CT>&)

.. doxygenfunction:: axis_begin(E&&)

.. doxygenfunction:: axis_begin(E&&, typename std::decay_t<E>::size_type)

.. doxygenfunction:: axis_end(E&&)

.. doxygenfunction:: axis_end(E&&, typename std::decay_t<E>::size_type)
