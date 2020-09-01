/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZMATH_HPP
#define XTENSOR_ZMATH_HPP

#include "xmath.hpp"
#include "zarray_impl.hpp"

namespace xt
{
    namespace detail
    {
        // For further improvement: move shape computation
        // at the beginning of a zarray assignment so it is computed
        // only once
        template <class E1, class E2>
        inline void zassign_data(xexpression<E1>& e1, const xexpression<E2>& e2)
        {
            e1.derived_cast() = e2.derived_cast();
        }
    }

    template <class T1, class T2, class R>
    inline void zadd(const ztyped_array<T1>& z1,
                     const ztyped_array<T2>& z2,
                     ztyped_array<R>& zres)
    {
        detail::zassign_data(zres.get_array(), z1.get_array() + z2.get_array());
    }

    template <class T1, class R>
    inline void zexp(const ztyped_array<T1>& z1,
                     ztyped_array<R>& zres)
    {
        detail::zassign_data(zres.get_array(), xt::exp(z1.get_array()));
    }
}

#endif
