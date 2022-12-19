.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Containers and views
====================

Containers are in-memory expressions that share a common implementation of most of the methods of the xexpression API.
The final container classes (:cpp:type:`xt::xarray`, :cpp:type:`xt::xtensor`) mainly implement constructors and value semantic, most of the
xexpression API is actually implemented in ``xstrided_container`` and ``xcontainer``.

.. toctree::

   xcontainer
   xaccessible
   xiterable
   xarray
   xarray_adaptor
   xchunked_array
   xtensor
   xtensor_adaptor
   xfixed
   xoptional_assembly_base
   xoptional_assembly
   xoptional_assembly_adaptor
   xmasked_view
   xview
   xstrided_view
   xbroadcast
   xindex_view
   xfunctor_view
   xrepeat
