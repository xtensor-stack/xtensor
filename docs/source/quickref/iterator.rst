.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Iterators
=========

Default iteration
-----------------

.. code::

    #include <iostream>
    #include <iterator>
    #include "xarray.hpp"

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}};
    std::copy(a.begin(), a.end(), std::ostream_iterator<int>(std::cout, ", "));
    // Prints 1, 2, 3, 4, 5, 6,

Specified traversal order
-------------------------

.. code::

    #include <iostream>
    #include <iterator>
    #include "xarray.hpp"

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}};
    std::copy(a.begin<layout_type::row_major>(),
              a.end<layout_type::row_major>(),
              std::ostream_iterator<int>(std::cout, ", "));
    // Prints 1, 2, 3, 4, 5, 6,

    std::copy(a.begin<layout_type::column_major>(),
              a.end<layout_type>::column_major>(),
              std::ostream_iterator<int>(std::cout, ", "));
    // Prints 1, 4, 2, 5, 3, 6,

Broacasting iteration
---------------------

.. code::

    #include <iostream>
    #include <iterator>
    #include "xarray.hpp"

    xt::xarray<int> a = {{1, 2, 3}, {4, 5, 6}};
    using shape_type = xt::dynamic_shape<std::size_t>;
    shape_type s = {2, 2, 3};
    
    std::copy(a.begin(s), a.end(s), std::ostream_iterator<int>(std::cout, ", "));
    // Prints 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 

    std::copy(a.begin<layout_type::row_major>(s),
              a.end<layout_type::row_major>(s),
              std::ostream_iterator<int>(std::cout, ", "));
    // Prints 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 

    std::copy(a.begin<layout_type::column_major>(s),
              a.end<layout_type>::column_major>(s),
              std::ostream_iterator<int>(std::cout, ", "));
    // Prints 1, 4, 2, 5, 3, 6, 1, 4, 2, 5, 3, 6,

1-D slice iteration
-------------------

Iterating over axis 0:

.. code::

    #include "xarray.hpp"
    #include "xaxis_slice_iterator.hpp"
    #include "xio.hpp"

    xarray<int> a = {{{1, 2, 3, 4},
                      {5, 6, 7, 8},
                      {9, 10, 11, 12}},
                     {{13, 14, 15, 16},
                      {17, 18, 19, 20},
                      {21, 22, 23, 24}}};

    auto iter = axis_slice_begin(a, 0);
    auto end = axis_slice_end(a, 0);
    while(iter != end)
    {
        std::cout << *iter++ << std::endl;
    }
    // Prints:
    // { 1, 13 }
    // { 2, 14 }
    // { 3, 15 }
    // { 4, 16 }
    // { 5, 17 }
    // { 6, 18 }
    // { 7, 19 }
    // { 8, 20 }
    // { 9, 21 }
    // { 10, 22 }
    // { 11, 23 }
    // { 12, 24 }

Iterating over axis 1:

.. code::

    #include "xarray.hpp"
    #include "xaxis_slice_iterator.hpp"
    #include "xio.hpp"

    xarray<int> a = {{{1, 2, 3, 4},
                      {5, 6, 7, 8},
                      {9, 10, 11, 12}},
                     {{13, 14, 15, 16},
                      {17, 18, 19, 20},
                      {21, 22, 23, 24}}};

    auto iter = axis_slice_begin(a, 1u);
    auto end = axis_slice_end(a, 1u);
    while(iter != end)
    {
        std::cout << *iter++ << std::endl;
    }
    // Prints:
    // { 1, 5, 9 } 
    // { 2, 6, 10 }
    // { 3, 7, 11 }
    // { 4, 8, 12 }
    // { 13, 17, 21 }
    // { 14, 18, 22 }
    // { 15, 19, 23 }
    // { 16, 20, 24 }

Iterating over axis 2:

.. code::

    #include "xarray.hpp"
    #include "xaxis_slice_iterator.hpp"
    #include "xio.hpp"

    xarray<int> a = {{{1, 2, 3, 4},
                      {5, 6, 7, 8},
                      {9, 10, 11, 12}},
                     {{13, 14, 15, 16},
                      {17, 18, 19, 20},
                      {21, 22, 23, 24}}};

    auto iter = axis_slice_begin(a, 2u);
    auto end = axis_slice_end(a, 2u);
    while(iter != end)
    {
        std::cout << *iter++ << std::endl;
    }
    // Prints:
    // { 1, 2, 3, 4 }
    // { 5, 6, 7, 8 }
    // { 9, 10, 11, 12 }
    // { 13, 14, 15, 16 }
    // { 17, 18, 19, 20 }
    // { 21, 22, 23, 24 }

(N-1)-dimensional iteration
---------------------------

Iterating over axis 0:

.. code::

    #include "xarray.hpp"
    #include "xaxis_iterator.hpp"
    #include "xio.hpp"

    xarray<int> a = {{{1, 2, 3, 4},
                      {5, 6, 7, 8},
                      {9, 10, 11, 12}},
                     {{13, 14, 15, 16},
                      {17, 18, 19, 20},
                      {21, 22, 23, 24}}};

    auto iter = axis_begin(a, 0);
    auto end = axis_end(a, 0);
    while(iter != end)
    {
        std::cout << *iter++ << std::endl;
    }
    // Prints:
    // {{ 1,  2,  3,  4 },
    //  { 5,  6,  7,  9 },
    //  { 9, 10, 11, 12 }}
    // {{ 13, 14, 15, 16 },
    //  { 17, 18, 19, 20 },
    //  { 21, 22, 23, 24 }}

Iterating over axis 1:

.. code::

    #include "xarray.hpp"
    #include "xaxis_iterator.hpp"
    #include "xio.hpp"

    xarray<int> a = {{{1, 2, 3, 4},
                      {5, 6, 7, 8},
                      {9, 10, 11, 12}},
                     {{13, 14, 15, 16},
                      {17, 18, 19, 20},
                      {21, 22, 23, 24}}};

    auto iter = axis_begin(a, 1u);
    auto end = axis_end(a, 1u);
    while(iter != end)
    {
        std::cout << *iter++ << std::endl;
    }
    // Prints:
    // {{  1,  2,  3,  4 },
    //  { 13, 14, 15, 16 }}
    // {{  5,  6,  7,  8 },
    //  { 17, 18, 19, 20 }}
    // {{  9, 10, 11, 12 },
    //  { 21, 22, 23, 24 }}

Iterating over axis 2:

.. code::

    #include "xarray.hpp"
    #include "xaxis_iterator.hpp"
    #include "xio.hpp"

    xarray<int> a = {{{1, 2, 3, 4},
                      {5, 6, 7, 8},
                      {9, 10, 11, 12}},
                     {{13, 14, 15, 16},
                      {17, 18, 19, 20},
                      {21, 22, 23, 24}}};

    auto iter = axis_begin(a, 2u);
    auto end = axis_end(a, 2u);
    while(iter != end)
    {
        std::cout << *iter++ << std::endl;
    }
    // Prints:
    // {{  1,  5,  9 }
    //  { 13, 17, 21 }}
    // {{  2,  6, 10 },
    //  { 14, 18, 22 }}
    // {{  3,  7, 11 },
    //  { 15, 19, 23 }}
    // {{  4,  8, 12 },
    //  { 16, 20, 24 }}
