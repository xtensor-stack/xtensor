/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_FORWARD_HPP
#define XTENSOR_FORWARD_HPP

#include <vector>
#include <memory>

namespace xt
{
    template <class C>
    struct xcontainer_inner_types;

    template <class T, class EA = std::allocator<T>, class SA = std::allocator<typename std::vector<T, EA>::size_type>>
    class xarray;

    template <class T, std::size_t N, class A = std::allocator<T>>
    class xtensor;

    template <class CT, class... S>
    class xview;
}

#endif
