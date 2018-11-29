/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_OFFSET_VIEW_HPP
#define XTENSOR_OFFSET_VIEW_HPP

#include <xtl/xcomplex.hpp>

#include "xtensor/xfunctor_view.hpp"

namespace xt
{
    namespace detail
    {
        template <class M, std::size_t I>
        struct offset_forwarder
        {
            using value_type = M;
            using reference = M&;
            using const_reference = const M&;
            using pointer = M*;
            using const_pointer = const M*;

            using proxy = xtl::xproxy_wrapper<M>;

            template <class value_type, class requested_type>
            using simd_return_type = xsimd::simd_return_type<value_type, requested_type>;

            template <class T>
            decltype(auto) operator()(T&& t) const
            {
                return xtl::forward_offset<M, I>(t);
            }

            // template <class align, class requested_type, std::size_t N, class E>
            // auto proxy_simd_load(const E& expr, std::size_t n) const
            // {
            //     using simd_value_type = xsimd::simd_type<value_type>;
            //     auto v1 = xsimd::load_aligned((double*) expr.data() + n);
            //     auto v2 = xsimd::load_aligned((double*) expr.data() + n + N);

            //     if (std::is_same<M, double>::value && I == sizeof(double))
            //     {
            //         return simd_value_type(_mm256_permute4x64_pd(_mm256_unpackhi_pd(v1, v2), _MM_SHUFFLE(3, 1, 2, 0)));
            //     }
            //     else if (std::is_same<M, double>::value && I == sizeof(double))
            //     {
            //         return simd_value_type(_mm256_permute4x64_pd(_mm256_unpacklo_pd(v1, v2), _MM_SHUFFLE(3, 1, 2, 0)));
            //     }
            //     // return expr.template load_simd<align, double, N>(n);
            // }

            // template <class align, class simd, class E>
            // auto proxy_simd_store(E& expr, std::size_t n, const simd& batch) const
            // {
            //     using simd_value_type = typename E::simd_value_type;
            //     // return expr.template store_simd<align>(n, xsimd::select(batch, simd_value_type(0), simd_value_type(1)));
            //     return 0;
            // }

        };
    }

    template <class CT, class M, std::size_t I>
    using xoffset_view = xfunctor_view<detail::offset_forwarder<M, I>, CT>;
}

#endif
