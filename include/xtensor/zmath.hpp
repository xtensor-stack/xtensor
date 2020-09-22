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

#define XTENSOR_ZMAPPED_FUNCTOR(ZFUN, XFUN)                                        \
    template <>                                                                    \
    struct get_zmapped_functor<XFUN>                                               \
    { using type = ZFUN; }

#define XTENSOR_UNARY_ZOPERATOR(ZNAME, XOP, XFUN)                                  \
    struct ZNAME                                                                   \
    {                                                                              \
        template <class T, class  R>                                               \
        static void run(const ztyped_array<T>& z, ztyped_array<R>& zres)           \
        {                                                                          \
            detail::zassign_data(zres.get_array(), XOP z.get_array());             \
        }                                                                          \
        template <class T>                                                         \
        static size_t index(const ztyped_array<T>&)                                \
        {                                                                          \
            using result_type = ztyped_array<decltype(XOP std::declval<T>())>;     \
            return result_type::get_class_static_index();                          \
        }                                                                          \
    };                                                                             \
    XTENSOR_ZMAPPED_FUNCTOR(ZNAME, XFUN)

#define XTENSOR_BINARY_ZOPERATOR(ZNAME, XOP, XFUN)                                 \
    struct ZNAME                                                                   \
    {                                                                              \
        template <class T1, class T2, class R>                                     \
        static void run(const ztyped_array<T1>& z1,                                \
                        const ztyped_array<T2>& z2,                                \
                        ztyped_array<R>& zres)                                     \
        {                                                                          \
            detail::zassign_data(zres.get_array(),                                 \
                                 z1.get_array() XOP z2.get_array());               \
        }                                                                          \
        template <class T1, class T2>                                              \
        static size_t index(const ztyped_array<T1>&, const ztyped_array<T2>&)      \
        {                                                                          \
            using result_type =                                                    \
                ztyped_array<decltype(std::declval<T1>() XOP std::declval<T2>())>; \
            return result_type::get_class_static_index();                          \
        }                                                                          \
    };                                                                             \
    XTENSOR_ZMAPPED_FUNCTOR(ZNAME, XFUN)

#define XTENSOR_UNARY_ZFUNCTOR(ZNAME, XEXP, XFUN)                                  \
    struct ZNAME                                                                   \
    {                                                                              \
        template <class T, class R>                                                \
        static void run(const ztyped_array<T>& z,                                  \
                        ztyped_array<R>& zres)                                     \
        {                                                                          \
            detail::zassign_data(zres.get_array(), XEXP(z.get_array()));           \
        }                                                                          \
        template <class T>                                                         \
        static size_t index(const ztyped_array<T>&)                                \
        {                                                                          \
            using value_type = decltype(std::declval<XFUN>()(std::declval<T>()));  \
            return ztyped_array<value_type>::get_class_static_index();             \
        }                                                                          \
    };                                                                             \
    XTENSOR_ZMAPPED_FUNCTOR(ZNAME, XFUN)

#define XTENSOR_BINARY_ZFUNCTOR(ZNAME, XEXP, XFUN)                                 \
    struct ZNAME                                                                   \
    {                                                                              \
        template <class T1, class T2, class R>                                     \
        static void run(const ztyped_array<T1>& z1,                                \
                        const ztyped_array<T2>& z2,                                \
                        ztyped_array<R>& zres)                                     \
        {                                                                          \
            detail::zassign_data(zres.get_array(),                                 \
                                 XEXP(z1.get_array(), z2.get_array()));            \
        }                                                                          \
        template <class T1, class T2>                                              \
        static size_t index(const ztyped_array<T1>&, const ztyped_array<T2>&)      \
        {                                                                          \
            using value_type = decltype(                                           \
                std::declval<XFUN>()(std::declval<T1>(), std::declval<T2>()));     \
            return ztyped_array<value_type>::get_class_static_index();             \
        }                                                                          \
    };                                                                             \
    XTENSOR_ZMAPPED_FUNCTOR(ZNAME, XFUN)

    XTENSOR_UNARY_ZOPERATOR(zidentity, +, detail::identity);
    XTENSOR_UNARY_ZOPERATOR(znegate, -, detail::negate);
    XTENSOR_BINARY_ZOPERATOR(zplus, +, detail::plus);
    XTENSOR_BINARY_ZOPERATOR(zminus, -, detail::minus);
    XTENSOR_BINARY_ZOPERATOR(zmultiuplies, *, detail::multiplies);
    XTENSOR_BINARY_ZOPERATOR(zdivides, /, detail::divides);
    XTENSOR_BINARY_ZOPERATOR(zmodulus, %, detail::modulus);
    XTENSOR_BINARY_ZOPERATOR(zlogical_or, ||, detail::logical_or);
    XTENSOR_BINARY_ZOPERATOR(zlogical_and, &&, detail::logical_and);
    XTENSOR_UNARY_ZOPERATOR(zlogical_not, !, detail::logical_not);
    XTENSOR_BINARY_ZOPERATOR(zbitwise_or, |, detail::bitwise_or);
    XTENSOR_BINARY_ZOPERATOR(zbitwise_and, &, detail::bitwise_and);
    XTENSOR_BINARY_ZOPERATOR(zbitwise_xor, ^, detail::bitwise_xor);
    XTENSOR_UNARY_ZOPERATOR(zbitwise_not, ~, detail::bitwise_not);
    XTENSOR_BINARY_ZOPERATOR(zleft_shift, <<, detail::left_shift);
    XTENSOR_BINARY_ZOPERATOR(zright_shift, >>, detail::right_shift);
    XTENSOR_BINARY_ZOPERATOR(zless, <, detail::less);
    XTENSOR_BINARY_ZOPERATOR(zless_equal, <=, detail::less_equal);
    XTENSOR_BINARY_ZOPERATOR(zgreater, >, detail::greater);
    XTENSOR_BINARY_ZOPERATOR(zgreater_equal, >=, detail::greater_equal);
    XTENSOR_BINARY_ZOPERATOR(zequal_to, ==, detail::equal_to);
    XTENSOR_BINARY_ZOPERATOR(znot_equal_to, !=, detail::not_equal_to);


    XTENSOR_UNARY_ZFUNCTOR(zfabs, xt::fabs, math::fabs_fun);
    XTENSOR_BINARY_ZFUNCTOR(zfmod, xt::fmod, math::fmod_fun);
    XTENSOR_BINARY_ZFUNCTOR(zremainder, xt::remainder, math::remainder_fun);
    //XTENSOR_TERNARY_ZFUNCTOR(fma);
    XTENSOR_BINARY_ZFUNCTOR(zfmax, xt::fmax, math::fmax_fun);
    XTENSOR_BINARY_ZFUNCTOR(zfmin, xt::fmin, math::fmin_fun);
    XTENSOR_BINARY_ZFUNCTOR(zfdim, xt::fdim, math::fdim_fun);
    XTENSOR_UNARY_ZFUNCTOR(zexp, xt::exp, math::exp_fun);
    XTENSOR_UNARY_ZFUNCTOR(zexp2, xt::exp2, math::exp2_fun);
    XTENSOR_UNARY_ZFUNCTOR(zexpm1, xt::expm1, math::expm1_fun);
    XTENSOR_UNARY_ZFUNCTOR(zlog, xt::log, math::log_fun);
    XTENSOR_UNARY_ZFUNCTOR(zlog10, xt::log10, math::log10_fun);
    XTENSOR_UNARY_ZFUNCTOR(zlog2, xt::log2, math::log2_fun);
    XTENSOR_UNARY_ZFUNCTOR(zlog1p, xt::log1p, math::log1p_fun);
    XTENSOR_BINARY_ZFUNCTOR(zpow, xt::pow, math::pow_fun);
    XTENSOR_UNARY_ZFUNCTOR(zsqrt, xt::sqrt, math::sqrt_fun);
    XTENSOR_UNARY_ZFUNCTOR(zcbrt, xt::cbrt, math::cbrt_fun);
    XTENSOR_BINARY_ZFUNCTOR(zhypot, xt::hypot, math::hypot_fun);
    XTENSOR_UNARY_ZFUNCTOR(zsin, xt::sin, math::sin_fun);
    XTENSOR_UNARY_ZFUNCTOR(zcos, xt::cos, math::cos_fun);
    XTENSOR_UNARY_ZFUNCTOR(ztan, xt::tan, math::tan_fun);
    XTENSOR_UNARY_ZFUNCTOR(zasin, xt::asin, math::asin_fun);
    XTENSOR_UNARY_ZFUNCTOR(zacos, xt::acos, math::acos_fun);
    XTENSOR_UNARY_ZFUNCTOR(zatan, xt::atan, math::atan_fun);
    XTENSOR_BINARY_ZFUNCTOR(zatan2, xt::atan2, math::atan2_fun);
    XTENSOR_UNARY_ZFUNCTOR(zsinh, xt::sinh, math::sinh_fun);
    XTENSOR_UNARY_ZFUNCTOR(zcosh, xt::cosh, math::cosh_fun);
    XTENSOR_UNARY_ZFUNCTOR(ztanh, xt::tanh, math::tanh_fun);
    XTENSOR_UNARY_ZFUNCTOR(zasinh, xt::asinh, math::asinh_fun);
    XTENSOR_UNARY_ZFUNCTOR(zacosh, xt::acosh, math::acosh_fun);
    XTENSOR_UNARY_ZFUNCTOR(zatanh, xt::atanh, math::atanh_fun);
    XTENSOR_UNARY_ZFUNCTOR(zerf, xt::erf, math::erf_fun);
    XTENSOR_UNARY_ZFUNCTOR(zerfc, xt::erfc, math::erfc_fun);
    XTENSOR_UNARY_ZFUNCTOR(ztgamma, xt::tgamma, math::tgamma_fun);
    XTENSOR_UNARY_ZFUNCTOR(zlgamma, xt::lgamma, math::lgamma_fun);
    /*XTENSOR_UNARY_ZFUNCTOR(zceil, xt::ceil, math::ceil_fun);
    XTENSOR_UNARY_ZFUNCTOR(zfloor, xt::floor, math::floor_fun);
    XTENSOR_UNARY_ZFUNCTOR(ztrunc, xt::trunc, math::trunc_fun);
    XTENSOR_UNARY_ZFUNCTOR(zround, xt::round, math::round_fun);
    XTENSOR_UNARY_ZFUNCTOR(znearbyint, xt::nearbyint, math::nearbyint_fun);
    XTENSOR_UNARY_ZFUNCTOR(zrint, xt::rint, math::rint_fun);
    XTENSOR_UNARY_ZFUNCTOR(zisfinite, xt::isfinite, math::isfinite_fun);
    XTENSOR_UNARY_ZFUNCTOR(zisinf, xt::isinf, math::isinf_fun);
    XTENSOR_UNARY_ZFUNCTOR(zisnan, xt::isnan, math::isnan_fun);*/

#undef XTENSOR_BINARY_ZFUNCTOR
#undef XTENSOR_UNARY_ZFUNCTOR
#undef XTENSOR_BINARY_ZOPERATOR
#undef XTENSOR_UNARY_ZOPERATOR
#undef XTENSOR_ZMAPPED_FUNCTOR

}

#endif
