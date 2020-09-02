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

    template <class XF>
    struct get_zmapped_functor;
    
    template <class XF>
    using get_zmapped_functor_t = typename get_zmapped_functor<XF>::type;

/*#define DEFINE_ZFUNCTOR_MAPPING(XF, ZF) \
    template <>                         \
    struct get_zmapped_functor<XF>      \
    { using type = ZF; }
*/
    struct zadd
    {
        template <class T1, class T2, class R>
        static void run(const ztyped_array<T1>& z1,
                        const ztyped_array<T2>& z2,
                        ztyped_array<R>& zres)
        {
            detail::zassign_data(zres.get_array(), z1.get_array() + z2.get_array());
        }

        template <class T1, class T2>
        static size_t index(const ztyped_array<T1>&, const ztyped_array<T2>&)
        {
            using result_type = ztyped_array<decltype(std::declval<T1>() + std::declval<T2>())>;
            return result_type::get_class_static_index();
        }
    };

    template <>
    struct get_zmapped_functor<detail::plus>
    {
        using type = zadd;
    };

    //DEFINE_ZFUNCTOR_MAPPING((detail::plus), zadd);

    struct zexp
    {
        template <class T, class R>
        static void run(const ztyped_array<T>& z,
                        ztyped_array<R>& zres)
        {
            detail::zassign_data(zres.get_array(), xt::exp(z.get_array()));
        }

        template <class T>
        static size_t index(const ztyped_array<T>&)
        {
            using value_type = decltype(std::declval<math::exp_fun>()(std::declval<T>()));
            return ztyped_array<value_type>::get_class_static_index();
        }
    };

    template <>
    struct get_zmapped_functor<math::exp_fun>
    {
        using type = zexp;
    };
    //DEFINE_ZFUNCTOR_MAPPING((math::exp_fun), zexp);
}

#endif
