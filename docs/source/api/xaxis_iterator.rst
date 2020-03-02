.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xaxis_iterator
==============

Defined in ``xtensor/xaxis_iterator.hpp``

.. doxygenclass:: xt::xaxis_iterator
   :project: xtensor
   :members:

.. doxygenfunction:: operator==(const xaxis_iterator<CT>&, const xaxis_iterator<CT>&)
   :project: xtensor


.. doxygenfunction:: operator!=(const xaxis_iterator<CT>&, const xaxis_iterator<CT>&)
   :project: xtensor

.. doxygenfunction:: axis_begin(E&&)
   :project: xtensor

.. doxygenfunction:: axis_begin(E&&, typename std::decay_t<E>::size_type)
   :project: xtensor

.. doxygenfunction:: axis_end(E&&)
   :project: xtensor

.. doxygenfunction:: axis_end(E&&, typename std::decay_t<E>::size_type)
   :project: xtensor
