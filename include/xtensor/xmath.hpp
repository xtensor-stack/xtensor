/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

/**
 * @brief standard mathematical functions for xexpressions
 */

#ifndef XTENSOR_MATH_HPP
#define XTENSOR_MATH_HPP

#include <cmath>
#include <array>
#include <complex>
#include <type_traits>

#include <xtl/xcomplex.hpp>

#include "xaccumulator.hpp"
#include "xoperation.hpp"
#include "xreducer.hpp"
#include "xslice.hpp"
#include "xstrided_view.hpp"
#include "xeval.hpp"

namespace xt
{
    template <class T = double>
    struct numeric_constants
    {
        static constexpr T PI = 3.141592653589793238463;
        static constexpr T PI_2 = 1.57079632679489661923;
        static constexpr T PI_4 = 0.785398163397448309616;
        static constexpr T D_1_PI = 0.318309886183790671538;
        static constexpr T D_2_PI = 0.636619772367581343076;
        static constexpr T D_2_SQRTPI = 1.12837916709551257390;
        static constexpr T SQRT2 = 1.41421356237309504880;
        static constexpr T SQRT1_2 = 0.707106781186547524401;
        static constexpr T E = 2.71828182845904523536;
        static constexpr T LOG2E = 1.44269504088896340736;
        static constexpr T LOG10E = 0.434294481903251827651;
        static constexpr T LN2 = 0.693147180559945309417;
    };

    /***********
     * Helpers *
     ***********/

#define XTENSOR_UNSIGNED_ABS_FUNC(T)                                              \
constexpr inline T abs(const T& x)                                                \
{                                                                                 \
    return x;                                                                     \
}                                                                                 \

#define XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, T)                 \
constexpr inline bool FUNC_NAME(const T& /*x*/) noexcept                          \
{                                                                                 \
    return RETURN_VAL;                                                            \
}                                                                                 \

#define XTENSOR_INT_SPECIALIZATION(FUNC_NAME, RETURN_VAL)                         \
XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, char);                     \
XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, short);                    \
XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, int);                      \
XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, long);                     \
XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, long long);                \
XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned char);            \
XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned short);           \
XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned int);             \
XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned long);            \
XTENSOR_INT_SPECIALIZATION_IMPL(FUNC_NAME, RETURN_VAL, unsigned long long);       \


#define XTENSOR_UNARY_MATH_FUNCTOR_IMPL(NAME, R)                                  \
    template <class T>                                                            \
    struct NAME##_fun                                                             \
    {                                                                             \
        static auto exec(const T& arg)                                            \
        {                                                                         \
            using math::NAME;                                                     \
            return NAME(arg);                                                     \
        }                                                                         \
        template <class U, class RT>                                              \
        using frt = xt::detail::functor_return_type<U, RT>;                       \
        using return_type = frt<T, R>;                                            \
        using argument_type = T;                                                  \
        using result_type = decltype(exec(std::declval<T>()));                    \
        using simd_value_type = xsimd::simd_type<T>;                              \
        using simd_result_type = typename return_type::simd_type;                 \
        constexpr result_type operator()(const T& arg) const                      \
        {                                                                         \
            using math::NAME;                                                     \
            return NAME(arg);                                                     \
        }                                                                         \
        template <class B>                                                        \
        constexpr typename frt<get_value_type_t<B>, R>::simd_type                 \
        simd_apply(const B& arg) const                                            \
        {                                                                         \
            using math::NAME;                                                     \
            return NAME(arg);                                                     \
        }                                                                         \
        template <class U>                                                        \
        struct rebind                                                             \
        {                                                                         \
            using type = NAME##_fun<U>;                                           \
        };                                                                        \
    }

#define XTENSOR_UNARY_MATH_FUNCTOR(NAME) XTENSOR_UNARY_MATH_FUNCTOR_IMPL(NAME, T)
#define XTENSOR_UNARY_BOOL_FUNCTOR(NAME) XTENSOR_UNARY_MATH_FUNCTOR_IMPL(NAME, bool)

#define XTENSOR_UNARY_MATH_FUNCTOR_COMPLEX_REDUCING(NAME)                         \
    template <class T>                                                            \
    struct NAME##_fun                                                             \
    {                                                                             \
        static auto exec(const T& arg)                                            \
        {                                                                         \
            using math::NAME;                                                     \
            return NAME(arg);                                                     \
        }                                                                         \
        using argument_type = T;                                                  \
        using result_type = decltype(exec(std::declval<T>()));                    \
        using simd_value_type = argument_type;                                    \
        using simd_result_type = result_type;                                     \
        constexpr result_type operator()(const T& arg) const                      \
        {                                                                         \
            using math::NAME;                                                     \
            return NAME(arg);                                                     \
        }                                                                         \
        template <class B>                                                        \
        constexpr simd_result_type simd_apply(const B& arg) const                 \
        {                                                                         \
            using math::NAME;                                                     \
            return NAME(arg);                                                     \
        }                                                                         \
        template <class U>                                                        \
        struct rebind                                                             \
        {                                                                         \
            using type = NAME##_fun<U>;                                           \
        };                                                                        \
    }

#define XTENSOR_BINARY_MATH_FUNCTOR_IMPL(NAME, R)                                 \
    template <class T>                                                            \
    struct NAME##_fun                                                             \
    {                                                                             \
        static auto exec(const T& arg1, const T& arg2)                            \
        {                                                                         \
            using math::NAME;                                                     \
            return NAME(arg1, arg2);                                              \
        }                                                                         \
        template <class U, class RT>                                              \
        using frt = xt::detail::functor_return_type<U, RT>;                       \
        using return_type = xt::detail::functor_return_type<T, R>;                \
        using first_argument_type = T;                                            \
        using second_argument_type = T;                                           \
        using result_type = decltype(exec(std::declval<T>(), std::declval<T>())); \
        using simd_value_type = xsimd::simd_type<T>;                              \
        using simd_result_type = typename return_type::simd_type;                 \
        constexpr result_type operator()(const T& arg1, const T& arg2) const      \
        {                                                                         \
            using math::NAME;                                                     \
            return NAME(arg1, arg2);                                              \
        }                                                                         \
        template <class B>                                                        \
        constexpr typename frt<get_value_type_t<B>, R>::simd_type                 \
        simd_apply(const B& arg1, const B& arg2) const                            \
        {                                                                         \
            using math::NAME;                                                     \
            return NAME(arg1, arg2);                                              \
        }                                                                         \
        template <class U>                                                        \
        struct rebind                                                             \
        {                                                                         \
            using type = NAME##_fun<U>;                                           \
        };                                                                        \
    }

#define XTENSOR_BINARY_MATH_FUNCTOR(NAME) XTENSOR_BINARY_MATH_FUNCTOR_IMPL(NAME, T)
#define XTENSOR_BINARY_BOOL_FUNCTOR(NAME) XTENSOR_BINARY_MATH_FUNCTOR_IMPL(NAME, bool)

#define XTENSOR_TERNARY_MATH_FUNCTOR_IMPL(NAME, R)                                \
    template <class T>                                                            \
    struct NAME##_fun                                                             \
    {                                                                             \
        static auto exec(const T& arg1, const T& arg2, const T& arg3)             \
        {                                                                         \
            using math::NAME;                                                     \
            return NAME(arg1, arg2, arg3);                                        \
        }                                                                         \
        template <class U, class RT>                                              \
        using frt = xt::detail::functor_return_type<U, RT>;                       \
        using return_type = xt::detail::functor_return_type<T, R>;                \
        using first_argument_type = T;                                            \
        using second_argument_type = T;                                           \
        using third_argument_type = T;                                            \
        using result_type = decltype(exec(std::declval<T>(), std::declval<T>(),   \
                    std::declval<T>()));                                          \
        using simd_value_type = xsimd::simd_type<T>;                              \
        using simd_result_type = typename return_type::simd_type;                 \
        constexpr result_type operator()(const T& arg1,                           \
                                         const T& arg2,                           \
                                         const T& arg3) const                     \
        {                                                                         \
            using math::NAME;                                                     \
            return NAME(arg1, arg2, arg3);                                        \
        }                                                                         \
        template <class B>                                                        \
        constexpr typename frt<get_value_type_t<B>, R>::simd_type                 \
        simd_apply(const B& arg1, const B& arg2, const B& arg3) const             \
        {                                                                         \
            using math::NAME;                                                     \
            return NAME(arg1, arg2, arg3);                                        \
        }                                                                         \
        template <class U>                                                        \
        struct rebind                                                             \
        {                                                                         \
            using type = NAME##_fun<U>;                                           \
        };                                                                        \
    }

#define XTENSOR_TERNARY_MATH_FUNCTOR(NAME) XTENSOR_TERNARY_MATH_FUNCTOR_IMPL(NAME, T)
#define XTENSOR_TERNARY_BOOL_FUNCTOR(NAME) XTENSOR_TERNARY_MATH_FUNCTOR_IMPL(NAME, bool)

    namespace math
    {
        using std::abs;
        using std::fabs;

        using std::cos;
        using std::sin;
        using std::tan;
        using std::acos;
        using std::asin;
        using std::atan;

        using std::cosh;
        using std::sinh;
        using std::tanh;
        using std::acosh;
        using std::asinh;
        using std::atanh;

        using std::sqrt;
        using std::cbrt;

        using std::exp;
        using std::exp2;
        using std::expm1;
        using std::log;
        using std::log2;
        using std::log10;
        using std::log1p;
        using std::logb;
        using std::ilogb;

        using std::floor;
        using std::ceil;
        using std::trunc;
        using std::round;
        using std::lround;
        using std::llround;
        using std::rint;
        using std::nearbyint;
        using std::remainder;

        using std::erf;
        using std::erfc;
        using std::erfc;
        using std::tgamma;
        using std::lgamma;

        using std::conj;
        using std::real;
        using std::imag;
        using std::arg;

        using std::atan2;
        using std::copysign;
        using std::fdim;
        using std::fmax;
        using std::fmin;
        using std::fmod;
        using std::hypot;
        using std::pow;

        using std::fma;

        using std::isnan;
        using std::isinf;
        using std::isfinite;
        using std::fpclassify;

        // Overload isinf, isnan and isfinite for complex datatypes,
        // following the Python specification:
        template <class T>
        inline bool isinf(const std::complex<T>& c)
        {
            return std::isinf(std::real(c)) || std::isinf(std::imag(c));
        }

        template <class T>
        inline bool isnan(const std::complex<T>& c)
        {
            return std::isnan(std::real(c)) || std::isnan(std::imag(c));
        }

        template <class T>
        inline bool isfinite(const std::complex<T>& c)
        {
            return !isinf(c) && !isnan(c);
        }

        // The following specializations are needed to avoid 'ambiguous overload' errors,
        // whereas 'unsigned char' and 'unsigned short' are automatically converted to 'int'.
        // we're still adding those functions to silence warnings
        XTENSOR_UNSIGNED_ABS_FUNC(unsigned char);
        XTENSOR_UNSIGNED_ABS_FUNC(unsigned short);
        XTENSOR_UNSIGNED_ABS_FUNC(unsigned int);
        XTENSOR_UNSIGNED_ABS_FUNC(unsigned long);
        XTENSOR_UNSIGNED_ABS_FUNC(unsigned long long);

#ifdef _WIN32
        XTENSOR_INT_SPECIALIZATION(isinf, false);
        XTENSOR_INT_SPECIALIZATION(isnan, false);
        XTENSOR_INT_SPECIALIZATION(isfinite, true);
#endif

        XTENSOR_UNARY_MATH_FUNCTOR_COMPLEX_REDUCING(abs);

        XTENSOR_UNARY_MATH_FUNCTOR(fabs);
        XTENSOR_BINARY_MATH_FUNCTOR(fmod);
        XTENSOR_BINARY_MATH_FUNCTOR(remainder);
        XTENSOR_TERNARY_MATH_FUNCTOR(fma);
        XTENSOR_BINARY_MATH_FUNCTOR(fmax);
        XTENSOR_BINARY_MATH_FUNCTOR(fmin);
        XTENSOR_BINARY_MATH_FUNCTOR(fdim);
        XTENSOR_UNARY_MATH_FUNCTOR(exp);
        XTENSOR_UNARY_MATH_FUNCTOR(exp2);
        XTENSOR_UNARY_MATH_FUNCTOR(expm1);
        XTENSOR_UNARY_MATH_FUNCTOR(log);
        XTENSOR_UNARY_MATH_FUNCTOR(log10);
        XTENSOR_UNARY_MATH_FUNCTOR(log2);
        XTENSOR_UNARY_MATH_FUNCTOR(log1p);
        XTENSOR_BINARY_MATH_FUNCTOR(pow);
        XTENSOR_UNARY_MATH_FUNCTOR(sqrt);
        XTENSOR_UNARY_MATH_FUNCTOR(cbrt);
        XTENSOR_BINARY_MATH_FUNCTOR(hypot);
        XTENSOR_UNARY_MATH_FUNCTOR(sin);
        XTENSOR_UNARY_MATH_FUNCTOR(cos);
        XTENSOR_UNARY_MATH_FUNCTOR(tan);
        XTENSOR_UNARY_MATH_FUNCTOR(asin);
        XTENSOR_UNARY_MATH_FUNCTOR(acos);
        XTENSOR_UNARY_MATH_FUNCTOR(atan);
        XTENSOR_BINARY_MATH_FUNCTOR(atan2);
        XTENSOR_UNARY_MATH_FUNCTOR(sinh);
        XTENSOR_UNARY_MATH_FUNCTOR(cosh);
        XTENSOR_UNARY_MATH_FUNCTOR(tanh);
        XTENSOR_UNARY_MATH_FUNCTOR(asinh);
        XTENSOR_UNARY_MATH_FUNCTOR(acosh);
        XTENSOR_UNARY_MATH_FUNCTOR(atanh);
        XTENSOR_UNARY_MATH_FUNCTOR(erf);
        XTENSOR_UNARY_MATH_FUNCTOR(erfc);
        XTENSOR_UNARY_MATH_FUNCTOR(tgamma);
        XTENSOR_UNARY_MATH_FUNCTOR(lgamma);
        XTENSOR_UNARY_MATH_FUNCTOR(ceil);
        XTENSOR_UNARY_MATH_FUNCTOR(floor);
        XTENSOR_UNARY_MATH_FUNCTOR(trunc);
        XTENSOR_UNARY_MATH_FUNCTOR(round);
        XTENSOR_UNARY_MATH_FUNCTOR(nearbyint);
        XTENSOR_UNARY_MATH_FUNCTOR(rint);
        XTENSOR_UNARY_BOOL_FUNCTOR(isfinite);
        XTENSOR_UNARY_BOOL_FUNCTOR(isinf);
        XTENSOR_UNARY_BOOL_FUNCTOR(isnan);
    }

#undef XTENSOR_UNARY_MATH_FUNCTOR
#undef XTENSOR_UNARY_BOOL_FUNCTOR
#undef XTENSOR_UNARY_MATH_FUNCTOR_IMPL
#undef XTENSOR_BINARY_MATH_FUNCTOR
#undef XTENSOR_BINARY_BOOL_FUNCTOR
#undef XTENSOR_BINARY_MATH_FUNCTOR_IMPL
#undef XTENSOR_TERNARY_MATH_FUNCTOR
#undef XTENSOR_TERNARY_BOOL_FUNCTOR
#undef XTENSOR_TERNARY_MATH_FUNCTOR_IMPL
#undef XTENSOR_UNARY_MATH_FUNCTOR_COMPLEX_REDUCING
#undef XTENSOR_UNSIGNED_ABS_FUNC

#define XTENSOR_REDUCER_FUNCTION(NAME, FUNCTOR, RESULT_TYPE)                                                      \
    template <class E, class X, class EVS = DEFAULT_STRATEGY_REDUCERS,                                            \
              class = std::enable_if_t<!std::is_base_of<evaluation_strategy::base, std::decay_t<X>>::value, int>> \
    inline auto NAME(E&& e, X&& axes, EVS es = EVS())                                                             \
    {                                                                                                             \
        using result_type = RESULT_TYPE;                                                                          \
        using functor_type = FUNCTOR<result_type>;                                                                \
        return reduce(make_xreducer_functor(functor_type()), std::forward<E>(e),                                  \
                      std::forward<X>(axes), es);                                                                 \
    }                                                                                                             \
                                                                                                                  \
    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS,                                                     \
              class = std::enable_if_t<std::is_base_of<evaluation_strategy::base, EVS>::value, int>>              \
    inline auto NAME(E&& e, EVS es = EVS())                                                                       \
    {                                                                                                             \
        using result_type = RESULT_TYPE;                                                                          \
        using functor_type = FUNCTOR<result_type>;                                                                \
        return reduce(make_xreducer_functor(functor_type()), std::forward<E>(e), es);                             \
    }

#define XTENSOR_OLD_CLANG_REDUCER(NAME, FUNCTOR, RESULT_TYPE)                                                     \
    template <class E, class I, class EVS = DEFAULT_STRATEGY_REDUCERS>                                            \
        inline auto NAME(E&& e, std::initializer_list<I> axes, EVS es = EVS())                                    \
        {                                                                                                         \
            using result_type = RESULT_TYPE;                                                                      \
            using functor_type = FUNCTOR<result_type>;                                                            \
            return reduce(make_xreducer_functor(functor_type()), std::forward<E>(e), axes, es);                   \
        }                                                                                                         \

#define XTENSOR_MODERN_CLANG_REDUCER(NAME, FUNCTOR, RESULT_TYPE)                                                  \
    template <class E, class I, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>                             \
    inline auto NAME(E&& e, const I (&axes)[N], EVS es = EVS())                                                   \
    {                                                                                                             \
        using result_type = RESULT_TYPE;                                                                          \
        using functor_type = FUNCTOR<result_type>;                                                                \
        return reduce(make_xreducer_functor(functor_type()), std::forward<E>(e), axes, es);                       \
    }

    /*******************
     * basic functions *
     *******************/

    /**
     * @defgroup basic_functions Basic functions
     */

    /**
     * @ingroup basic_functions
     * @brief Absolute value function.
     *
     * Returns an \ref xfunction for the element-wise absolute value
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto abs(E&& e) noexcept
        -> detail::xfunction_type_t<math::abs_fun, E>
    {
        return detail::make_xfunction<math::abs_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup basic_functions
     * @brief Absolute value function.
     *
     * Returns an \ref xfunction for the element-wise absolute value
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto fabs(E&& e) noexcept
        -> detail::xfunction_type_t<math::fabs_fun, E>
    {
        return detail::make_xfunction<math::fabs_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup basic_functions
     * @brief Remainder of the floating point division operation.
     *
     * Returns an \ref xfunction for the element-wise remainder of
     * the floating point division operation <em>e1 / e2</em>.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fmod(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::fmod_fun, E1, E2>
    {
        return detail::make_xfunction<math::fmod_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Signed remainder of the division operation.
     *
     * Returns an \ref xfunction for the element-wise signed remainder
     * of the floating point division operation <em>e1 / e2</em>.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto remainder(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::remainder_fun, E1, E2>
    {
        return detail::make_xfunction<math::remainder_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Fused multiply-add operation.
     *
     * Returns an \ref xfunction for <em>e1 * e2 + e3</em> as if
     * to infinite precision and rounded only once to fit the result type.
     * @param e1 an \ref xfunction or a scalar
     * @param e2 an \ref xfunction or a scalar
     * @param e3 an \ref xfunction or a scalar
     * @return an \ref xfunction
     * @note e1, e2 and e3 can't be scalars every three.
     */
    template <class E1, class E2, class E3>
    inline auto fma(E1&& e1, E2&& e2, E3&& e3) noexcept
        -> detail::xfunction_type_t<math::fma_fun, E1, E2, E3>
    {
        return detail::make_xfunction<math::fma_fun>(std::forward<E1>(e1), std::forward<E2>(e2), std::forward<E3>(e3));
    }

    /**
     * @ingroup basic_functions
     * @brief Maximum function.
     *
     * Returns an \ref xfunction for the element-wise maximum
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fmax(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::fmax_fun, E1, E2>
    {
        return detail::make_xfunction<math::fmax_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Minimum function.
     *
     * Returns an \ref xfunction for the element-wise minimum
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fmin(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::fmin_fun, E1, E2>
    {
        return detail::make_xfunction<math::fmin_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Positive difference function.
     *
     * Returns an \ref xfunction for the element-wise positive
     * difference of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fdim(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::fdim_fun, E1, E2>
    {
        return detail::make_xfunction<math::fdim_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    namespace math
    {
        template <class T>
        struct minimum
        {
            using result_type = T;
            using simd_value_type = xsimd::simd_type<T>;

            constexpr result_type operator()(const T& t1, const T& t2) const noexcept
            {
                return (t1 < t2) ? t1 : t2;
            }

            constexpr simd_value_type simd_apply(const simd_value_type& t1, const simd_value_type& t2) const noexcept
            {
                return xsimd::select(t1 < t2, t1, t2);
            }
        };

        template <class T>
        struct maximum
        {
            using result_type = T;
            using simd_value_type = xsimd::simd_type<T>;

            constexpr result_type operator()(const T& t1, const T& t2) const noexcept
            {
                return (t1 > t2) ? t1 : t2;
            }

            constexpr simd_value_type simd_apply(const simd_value_type& t1, const simd_value_type& t2) const noexcept
            {
                return xsimd::select(t1 > t2, t1, t2);
            }
        };

        template <class T>
        struct clamp_fun
        {
            using first_argument_type = T;
            using second_argument_type = T;
            using third_argument_type = T;
            using result_type = T;
            using simd_value_type = xsimd::simd_type<T>;

            constexpr T operator()(const T& v, const T& lo, const T& hi) const
            {
                return v < lo ? lo : hi < v ? hi : v;
            }

            constexpr simd_value_type simd_apply(const simd_value_type& v,
                                                 const simd_value_type& lo,
                                                 const simd_value_type& hi) const
            {
                return xsimd::select(v < lo, lo, xsimd::select(hi < v, hi, v));
            }
        };
    }

    /**
     * @ingroup basic_functions
     * @brief Elementwise maximum
     *
     * Returns an \ref xfunction for the element-wise
     * maximum between e1 and e2.
     * @param e1 an \ref xexpression
     * @param e2 an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto maximum(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::maximum, E1, E2>
    {
        return detail::make_xfunction<math::maximum>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Elementwise minimum
     *
     * Returns an \ref xfunction for the element-wise
     * minimum between e1 and e2.
     * @param e1 an \ref xexpression
     * @param e2 an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto minimum(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::minimum, E1, E2>
    {
        return detail::make_xfunction<math::minimum>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup basic_functions
     * @brief Maximum element along given axis.
     *
     * Returns an \ref xreducer for the maximum of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the maximum is found (optional)
     * @param es evaluation strategy of the reducer
     * @return an \ref xreducer
     */
    XTENSOR_REDUCER_FUNCTION(amax, math::maximum, typename std::decay_t<E>::value_type);
#ifdef X_OLD_CLANG
    XTENSOR_OLD_CLANG_REDUCER(amax, math::maximum, typename std::decay_t<E>::value_type);
#else
    XTENSOR_MODERN_CLANG_REDUCER(amax, math::maximum, typename std::decay_t<E>::value_type);
#endif

    /**
     * @ingroup basic_functions
     * @brief Minimum element along given axis.
     *
     * Returns an \ref xreducer for the minimum of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the minimum is found (optional)
     * @param es evaluation strategy of the reducer
     * @return an \ref xreducer
     */
    XTENSOR_REDUCER_FUNCTION(amin, math::minimum, typename std::decay_t<E>::value_type);
#ifdef X_OLD_CLANG
    XTENSOR_OLD_CLANG_REDUCER(amin, math::minimum, typename std::decay_t<E>::value_type);
#else
    XTENSOR_MODERN_CLANG_REDUCER(amin, math::minimum, typename std::decay_t<E>::value_type);
#endif

    /**
     * @ingroup basic_functions
     * @brief Clip values between hi and lo
     *
     * Returns an \ref xfunction for the element-wise clipped
     * values between lo and hi
     * @param e1 an \ref xexpression or a scalar
     * @param lo a scalar
     * @param hi a scalar
     *
     * @return a \ref xfunction
     */
    template <class E1, class E2, class E3>
    inline auto clip(E1&& e1, E2&& lo, E3&& hi) noexcept
        -> detail::xfunction_type_t<math::clamp_fun, E1, E2, E3>
    {
        return detail::make_xfunction<math::clamp_fun>(std::forward<E1>(e1), std::forward<E2>(lo), std::forward<E3>(hi));
    }

    namespace math
    {
        namespace detail
        {
            template <typename T>
            constexpr std::enable_if_t<std::is_signed<T>::value, T>
            sign_impl(T x)
            {
                return std::isnan(x) ? std::numeric_limits<T>::quiet_NaN() : x == 0 ? T(copysign(T(0), x)) : T(copysign(T(1), x));
            }

            template <typename T>
            inline std::enable_if_t<xtl::is_complex<T>::value, T>
            sign_impl(T x)
            {
                typename T::value_type e = (x.real() != T(0)) ? x.real() : x.imag();
                return T(sign_impl(e), 0);
            }

            template <typename T>
            constexpr std::enable_if_t<std::is_unsigned<T>::value, T>
            sign_impl(T x)
            {
                return T(x > T(0));
            }
        }

        template <class T>
        struct sign_fun
        {
            using argument_type = T;
            using result_type = T;

            constexpr T operator()(const T& x) const
            {
                return detail::sign_impl(x);
            }
        };
    }

    /**
     * @ingroup basic_functions
     * @brief Returns an element-wise indication of the sign of a number
     *
     * If the number is positive, returns +1. If negative, -1. If the number
     * is zero, returns 0.
     *
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sign(E&& e) noexcept
        -> detail::xfunction_type_t<math::sign_fun, E>
    {
        return detail::make_xfunction<math::sign_fun>(std::forward<E>(e));
    }

    /*************************
     * exponential functions *
     *************************/

    /**
     * @defgroup exp_functions Exponential functions
     */

    /**
     * @ingroup exp_functions
     * @brief Natural exponential function.
     *
     * Returns an \ref xfunction for the element-wise natural
     * exponential of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto exp(E&& e) noexcept
        -> detail::xfunction_type_t<math::exp_fun, E>
    {
        return detail::make_xfunction<math::exp_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Base 2 exponential function.
     *
     * Returns an \ref xfunction for the element-wise base 2
     * exponential of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto exp2(E&& e) noexcept
        -> detail::xfunction_type_t<math::exp2_fun, E>
    {
        return detail::make_xfunction<math::exp2_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Natural exponential minus one function.
     *
     * Returns an \ref xfunction for the element-wise natural
     * exponential of \em e, minus 1.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto expm1(E&& e) noexcept
        -> detail::xfunction_type_t<math::expm1_fun, E>
    {
        return detail::make_xfunction<math::expm1_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Natural logarithm function.
     *
     * Returns an \ref xfunction for the element-wise natural
     * logarithm of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log(E&& e) noexcept
        -> detail::xfunction_type_t<math::log_fun, E>
    {
        return detail::make_xfunction<math::log_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Base 10 logarithm function.
     *
     * Returns an \ref xfunction for the element-wise base 10
     * logarithm of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log10(E&& e) noexcept
        -> detail::xfunction_type_t<math::log10_fun, E>
    {
        return detail::make_xfunction<math::log10_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Base 2 logarithm function.
     *
     * Returns an \ref xfunction for the element-wise base 2
     * logarithm of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log2(E&& e) noexcept
        -> detail::xfunction_type_t<math::log2_fun, E>
    {
        return detail::make_xfunction<math::log2_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup exp_functions
     * @brief Natural logarithm of one plus function.
     *
     * Returns an \ref xfunction for the element-wise natural
     * logarithm of \em e, plus 1.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log1p(E&& e) noexcept
        -> detail::xfunction_type_t<math::log1p_fun, E>
    {
        return detail::make_xfunction<math::log1p_fun>(std::forward<E>(e));
    }

    /*******************
     * power functions *
     *******************/

    /**
     * @defgroup pow_functions Power functions
     */

    /**
     * @ingroup pow_functions
     * @brief Power function.
     *
     * Returns an \ref xfunction for the element-wise value of
     * of \em e1 raised to the power \em e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto pow(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::pow_fun, E1, E2>
    {
        return detail::make_xfunction<math::pow_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup pow_functions
     * @brief Square root function.
     *
     * Returns an \ref xfunction for the element-wise square
     * root of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sqrt(E&& e) noexcept
        -> detail::xfunction_type_t<math::sqrt_fun, E>
    {
        return detail::make_xfunction<math::sqrt_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup pow_functions
     * @brief Cubic root function.
     *
     * Returns an \ref xfunction for the element-wise cubic
     * root of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto cbrt(E&& e) noexcept
        -> detail::xfunction_type_t<math::cbrt_fun, E>
    {
        return detail::make_xfunction<math::cbrt_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup pow_functions
     * @brief Hypotenuse function.
     *
     * Returns an \ref xfunction for the element-wise square
     * root of the sum of the square of \em e1 and \em e2, avoiding
     * overflow and underflow at intermediate stages of computation.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto hypot(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::hypot_fun, E1, E2>
    {
        return detail::make_xfunction<math::hypot_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /***************************
     * trigonometric functions *
     ***************************/

    /**
     * @defgroup trigo_functions Trigonometric function
     */

    /**
     * @ingroup trigo_functions
     * @brief Sine function.
     *
     * Returns an \ref xfunction for the element-wise sine
     * of \em e (measured in radians).
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sin(E&& e) noexcept
        -> detail::xfunction_type_t<math::sin_fun, E>
    {
        return detail::make_xfunction<math::sin_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Cosine function.
     *
     * Returns an \ref xfunction for the element-wise cosine
     * of \em e (measured in radians).
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto cos(E&& e) noexcept
        -> detail::xfunction_type_t<math::cos_fun, E>
    {
        return detail::make_xfunction<math::cos_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Tangent function.
     *
     * Returns an \ref xfunction for the element-wise tangent
     * of \em e (measured in radians).
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto tan(E&& e) noexcept
        -> detail::xfunction_type_t<math::tan_fun, E>
    {
        return detail::make_xfunction<math::tan_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Arcsine function.
     *
     * Returns an \ref xfunction for the element-wise arcsine
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto asin(E&& e) noexcept
        -> detail::xfunction_type_t<math::asin_fun, E>
    {
        return detail::make_xfunction<math::asin_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Arccosine function.
     *
     * Returns an \ref xfunction for the element-wise arccosine
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto acos(E&& e) noexcept
        -> detail::xfunction_type_t<math::acos_fun, E>
    {
        return detail::make_xfunction<math::acos_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Arctangent function.
     *
     * Returns an \ref xfunction for the element-wise arctangent
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto atan(E&& e) noexcept
        -> detail::xfunction_type_t<math::atan_fun, E>
    {
        return detail::make_xfunction<math::atan_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup trigo_functions
     * @brief Artangent function, using signs to determine quadrants.
     *
     * Returns an \ref xfunction for the element-wise arctangent
     * of <em>e1 / e2</em>, using the signs of arguments to determine the
     * correct quadrant.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto atan2(E1&& e1, E2&& e2) noexcept
        -> detail::xfunction_type_t<math::atan2_fun, E1, E2>
    {
        return detail::make_xfunction<math::atan2_fun>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /************************
     * hyperbolic functions *
     ************************/

    /**
     * @defgroup hyper_functions Hyperbolic functions
     */

    /**
     * @ingroup hyper_functions
     * @brief Hyperbolic sine function.
     *
     * Returns an \ref xfunction for the element-wise hyperbolic
     * sine of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sinh(E&& e) noexcept
        -> detail::xfunction_type_t<math::sinh_fun, E>
    {
        return detail::make_xfunction<math::sinh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Hyperbolic cosine function.
     *
     * Returns an \ref xfunction for the element-wise hyperbolic
     * cosine of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto cosh(E&& e) noexcept
        -> detail::xfunction_type_t<math::cosh_fun, E>
    {
        return detail::make_xfunction<math::cosh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Hyperbolic tangent function.
     *
     * Returns an \ref xfunction for the element-wise hyperbolic
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto tanh(E&& e) noexcept
        -> detail::xfunction_type_t<math::tanh_fun, E>
    {
        return detail::make_xfunction<math::tanh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Inverse hyperbolic sine function.
     *
     * Returns an \ref xfunction for the element-wise inverse hyperbolic
     * sine of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto asinh(E&& e) noexcept
        -> detail::xfunction_type_t<math::asinh_fun, E>
    {
        return detail::make_xfunction<math::asinh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Inverse hyperbolic cosine function.
     *
     * Returns an \ref xfunction for the element-wise inverse hyperbolic
     * cosine of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto acosh(E&& e) noexcept
        -> detail::xfunction_type_t<math::acosh_fun, E>
    {
        return detail::make_xfunction<math::acosh_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup hyper_functions
     * @brief Inverse hyperbolic tangent function.
     *
     * Returns an \ref xfunction for the element-wise inverse hyperbolic
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto atanh(E&& e) noexcept
        -> detail::xfunction_type_t<math::atanh_fun, E>
    {
        return detail::make_xfunction<math::atanh_fun>(std::forward<E>(e));
    }

    /*****************************
     * error and gamma functions *
     *****************************/

    /**
     * @defgroup err_functions Error and gamma functions
     */

    /**
     * @ingroup err_functions
     * @brief Error function.
     *
     * Returns an \ref xfunction for the element-wise error function
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto erf(E&& e) noexcept
        -> detail::xfunction_type_t<math::erf_fun, E>
    {
        return detail::make_xfunction<math::erf_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup err_functions
     * @brief Complementary error function.
     *
     * Returns an \ref xfunction for the element-wise complementary
     * error function of \em e, whithout loss of precision for large argument.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto erfc(E&& e) noexcept
        -> detail::xfunction_type_t<math::erfc_fun, E>
    {
        return detail::make_xfunction<math::erfc_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup err_functions
     * @brief Gamma function.
     *
     * Returns an \ref xfunction for the element-wise gamma function
     * of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto tgamma(E&& e) noexcept
        -> detail::xfunction_type_t<math::tgamma_fun, E>
    {
        return detail::make_xfunction<math::tgamma_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup err_functions
     * @brief Natural logarithm of the gamma function.
     *
     * Returns an \ref xfunction for the element-wise logarithm of
     * the asbolute value fo the gamma function of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto lgamma(E&& e) noexcept
        -> detail::xfunction_type_t<math::lgamma_fun, E>
    {
        return detail::make_xfunction<math::lgamma_fun>(std::forward<E>(e));
    }

    /*********************************************
     * nearest integer floating point operations *
     *********************************************/

    /**
     * @defgroup nearint_functions Nearest integer floating point operations
     */

    /**
     * @ingroup nearint_functions
     * @brief ceil function.
     *
     * Returns an \ref xfunction for the element-wise smallest integer value
     * not less than \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto ceil(E&& e) noexcept
        -> detail::xfunction_type_t<math::ceil_fun, E>
    {
        return detail::make_xfunction<math::ceil_fun>(std::forward<E>(e));
    }

    /**
    * @ingroup nearint_functions
    * @brief floor function.
    *
    * Returns an \ref xfunction for the element-wise smallest integer value
    * not greater than \em e.
    * @param e an \ref xexpression
    * @return an \ref xfunction
    */
    template <class E>
    inline auto floor(E&& e) noexcept
        -> detail::xfunction_type_t<math::floor_fun, E>
    {
        return detail::make_xfunction<math::floor_fun>(std::forward<E>(e));
    }

    /**
    * @ingroup nearint_functions
    * @brief trunc function.
    *
    * Returns an \ref xfunction for the element-wise nearest integer not greater
    * in magnitude than \em e.
    * @param e an \ref xexpression
    * @return an \ref xfunction
    */
    template <class E>
    inline auto trunc(E&& e) noexcept
        -> detail::xfunction_type_t<math::trunc_fun, E>
    {
        return detail::make_xfunction<math::trunc_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup nearint_functions
     * @brief round function.
     *
     * Returns an \ref xfunction for the element-wise nearest integer value
     * to \em e, rounding halfway cases away from zero, regardless of the
     * current rounding mode.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto round(E&& e) noexcept
        -> detail::xfunction_type_t<math::round_fun, E>
    {
        return detail::make_xfunction<math::round_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup nearint_functions
     * @brief nearbyint function.
     *
     * Returns an \ref xfunction for the element-wise rounding of \em e to integer
     * values in floating point format, using the current rounding mode. nearbyint
     * never raises FE_INEXACT error.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto nearbyint(E&& e) noexcept
        -> detail::xfunction_type_t<math::nearbyint_fun, E>
    {
        return detail::make_xfunction<math::nearbyint_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup nearint_functions
     * @brief rint function.
     *
     * Returns an \ref xfunction for the element-wise rounding of \em e to integer
     * values in floating point format, using the current rounding mode. Contrary
     * to nearbyint, rint may raise FE_INEXACT error.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto rint(E&& e) noexcept
        -> detail::xfunction_type_t<math::rint_fun, E>
    {
        return detail::make_xfunction<math::rint_fun>(std::forward<E>(e));
    }

    /****************************
     * classification functions *
     ****************************/

    /**
     * @defgroup classif_functions Classification functions
     */

    /**
     * @ingroup classif_functions
     * @brief finite value check
     *
     * Returns an \ref xfunction for the element-wise finite value check
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto isfinite(E&& e) noexcept
        -> detail::xfunction_type_t<math::isfinite_fun, E>
    {
        return detail::make_xfunction<math::isfinite_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup classif_functions
     * @brief infinity check
     *
     * Returns an \ref xfunction for the element-wise infinity check
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto isinf(E&& e) noexcept
        -> detail::xfunction_type_t<math::isinf_fun, E>
    {
        return detail::make_xfunction<math::isinf_fun>(std::forward<E>(e));
    }

    /**
     * @ingroup classif_functions
     * @brief NaN check
     *
     * Returns an \ref xfunction for the element-wise NaN check
     * tangent of \em e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto isnan(E&& e) noexcept
        -> detail::xfunction_type_t<math::isnan_fun, E>
    {
        return detail::make_xfunction<math::isnan_fun>(std::forward<E>(e));
    }

    namespace detail
    {
        template <class FUNCTOR, class T, std::size_t... Is>
        inline auto get_functor(T&& args, std::index_sequence<Is...>)
        {
            return FUNCTOR(std::get<Is>(args)...);
        }

        template <template <class...> class F, class... A, class... E>
        inline auto make_xfunction(std::tuple<A...>&& f_args, E&&... e) noexcept
        {
            using functor_type = F<common_value_type_t<std::decay_t<E>...>>;
            using expression_tag = xexpression_tag_t<E...>;
            using type = select_xfunction_expression_t<expression_tag,
                                                       functor_type,
                                                       const_xclosure_t<E>...>;
            auto functor = get_functor<functor_type>(
                std::forward<std::tuple<A...>>(f_args),
                std::make_index_sequence<sizeof...(A)>{}
            );
            return type(std::move(functor), std::forward<E>(e)...);
        }

        template <class T>
        struct isclose
        {
            using result_type = bool;
            isclose(double rtol, double atol, bool equal_nan)
                : m_rtol(rtol), m_atol(atol), m_equal_nan(equal_nan)
            {
            }

            bool operator()(const T& a, const T& b) const
            {
                using internal_type = promote_type_t<T, double>;
                if (math::isnan(a) && math::isnan(b))
                {
                    return m_equal_nan;
                }
                if (math::isinf(a) && math::isinf(b))
                {
                    // check for both infinity signs equal
                    return a == b;
                }
                auto d = math::abs(internal_type(a) - internal_type(b));
                return d <= m_atol || d <= m_rtol * double((std::max)(math::abs(a), math::abs(b)));
            }

            template <class U>
            struct rebind
            {
                using type = isclose<U>;
            };

        private:
            double m_rtol;
            double m_atol;
            bool m_equal_nan;
        };
    }

    /**
     * @ingroup classif_functions
     * @brief Element-wise closeness detection
     *
     * Returns an \ref xfunction that evaluates to
     * true if the elements in ``e1`` and ``e2`` are close to each other
     * according to parameters ``atol`` and ``rtol``.
     * The equation is: ``std::abs(a - b) <= (m_atol + m_rtol * std::abs(b))``.
     * @param e1 input array to compare
     * @param e2 input array to compare
     * @param rtol the relative tolerance parameter (default 1e-05)
     * @param atol the absolute tolerance parameter (default 1e-08)
     * @param equal_nan if true, isclose returns true if both elements of e1 and e2 are NaN
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto isclose(E1&& e1, E2&& e2, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false) noexcept
    {
        return detail::make_xfunction<detail::isclose>(std::make_tuple(rtol, atol, equal_nan),
                                                       std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup classif_functions
     * @brief Check if all elements in \em e1 are close to the
     * corresponding elements in \em e2.
     *
     * Returns true if all elements in ``e1`` and ``e2`` are close to each other
     * according to parameters ``atol`` and ``rtol``.
     * @param e1 input array to compare
     * @param e2 input arrays to compare
     * @param rtol the relative tolerance parameter (default 1e-05)
     * @param atol the absolute tolerance parameter (default 1e-08)
     * @return a boolean
     */
    template <class E1, class E2>
    inline auto allclose(E1&& e1, E2&& e2, double rtol = 1e-05, double atol = 1e-08) noexcept
    {
        return xt::all(isclose(std::forward<E1>(e1), std::forward<E2>(e2), rtol, atol));
    }

    /**********************
     * Reducing functions *
     **********************/


    /**
     * @defgroup  red_functions reducing functions
     */

    /**
     * @ingroup red_functions
     * @brief Sum of elements over given axes.
     *
     * Returns an \ref xreducer for the sum of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the sum is performed (optional)
     * @param es evaluation strategy of the reducer
     * @return an \ref xreducer
     */
    XTENSOR_REDUCER_FUNCTION(sum, std::plus, big_promote_type_t<typename std::decay_t<E>::value_type>);
#ifdef X_OLD_CLANG
    XTENSOR_OLD_CLANG_REDUCER(sum, std::plus, big_promote_type_t<typename std::decay_t<E>::value_type>);
#else
    XTENSOR_MODERN_CLANG_REDUCER(sum, std::plus, big_promote_type_t<typename std::decay_t<E>::value_type>);
#endif

    /**
     * @ingroup red_functions
     * @brief Product of elements over given axes.
     *
     * Returns an \ref xreducer for the product of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the product is computed (optional)
     * @param es evaluation strategy of the reducer
     * @return an \ref xreducer
     */
    XTENSOR_REDUCER_FUNCTION(prod, std::multiplies, big_promote_type_t<typename std::decay_t<E>::value_type>);
#ifdef X_OLD_CLANG
    XTENSOR_OLD_CLANG_REDUCER(prod, std::multiplies, big_promote_type_t<typename std::decay_t<E>::value_type>);
#else
    XTENSOR_MODERN_CLANG_REDUCER(prod, std::multiplies, big_promote_type_t<typename std::decay_t<E>::value_type>);
#endif

    /**
     * @ingroup red_functions
     * @brief Mean of elements over given axes.
     *
     * Returns an \ref xreducer for the mean of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the mean is computed (optional)
     * @return an \ref xexpression
     */
    template <class E, class X>
    inline auto mean(E&& e, X&& axes)
    {
        auto size = e.size();
        auto s = sum(std::forward<E>(e), std::forward<X>(axes));
        return std::move(s) / static_cast<double>(size / s.size());
    }

    template <class E>
    inline auto mean(E&& e)
    {
        auto size = e.size();
        return sum(std::forward<E>(e)) / static_cast<double>(size);
    }

#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto mean(E&& e, std::initializer_list<I> axes)
    {
        auto size = e.size();
        auto s = sum(std::forward<E>(e), axes);
        return std::move(s) / static_cast<double>(size / s.size());
    }
#else
    template <class E, class I, std::size_t N>
    inline auto mean(E&& e, const I (&axes)[N])
    {
        auto size = e.size();
        auto s = sum(std::forward<E>(e), axes);
        return std::move(s) / static_cast<double>(size / s.size());
    }
#endif

    /**
     * @ingroup red_functions
     * @brief Minimum and maximum among the elements of an array or expression.
     *
     * Returns an \ref xreducer for the minimum and maximum of an expression's elements.
     * @param e an \ref xexpression
     * @param es evaluation strategy to use (lazy (default), or immediate)
     * @return an \ref xexpression of type ``std::array<value_type, 2>``, whose first
     *         and second element represent the minimum and maximum respectively
     */
    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS,
              XTENSOR_REQUIRE<std::is_base_of<evaluation_strategy::base, EVS>::value>>
    inline auto minmax(E&& e, EVS es = EVS())
    {
        using std::min;
        using std::max;
        using value_type = typename std::decay_t<E>::value_type;
        using result_type = std::array<value_type, 2>;

        auto reduce_func = [](result_type r, value_type const& v) {
            r[0] = (min)(r[0], v);
            r[1] = (max)(r[1], v);
            return r;
        };
        auto init_func = [](value_type const& v) {
            return result_type{v, v};
        };
        auto merge_func = [](result_type r, result_type const& s) {
            r[0] = (min)(r[0], s[0]);
            r[1] = (max)(r[1], s[1]);
            return r;
        };
        return reduce(make_xreducer_functor(std::move(reduce_func),
                                            std::move(init_func),
                                            std::move(merge_func)),
                      std::forward<E>(e), arange(e.dimension()), es);
    }

    /**
     * @defgroup acc_functions accumulating functions
     */

    /**
     * @ingroup acc_functions
     * @brief Cumulative sum.
     *
     * Returns the accumulated sum for the elements over given
     * \em axis (or flattened).
     * @param e an \ref xexpression
     * @param axis the axes along which the cumulative sum is computed (optional)
     * @return an \ref xarray<T>
     */
    template <class E>
    inline auto cumsum(E&& e, std::size_t axis)
    {
        using result_type = big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(std::plus<result_type>()), std::forward<E>(e), axis);
    }

    template <class E>
    inline auto cumsum(E&& e)
    {
        using result_type = big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(std::plus<result_type>()), std::forward<E>(e));
    }

    /**
     * @ingroup acc_functions
     * @brief Cumulative product.
     *
     * Returns the accumulated product for the elements over given
     * \em axis (or flattened).
     * @param e an \ref xexpression
     * @param axis the axes along which the cumulative product is computed (optional)
     * @return an \ref xarray<T>
     */
    template <class E>
    inline auto cumprod(E&& e, std::size_t axis)
    {
        using result_type = big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(std::multiplies<result_type>()), std::forward<E>(e), axis);
    }

    template <class E>
    inline auto cumprod(E&& e)
    {
        using result_type = big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(std::multiplies<result_type>()), std::forward<E>(e));
    }

    /*****************
     * nan functions *
     *****************/

    namespace detail
    {
        template <class T>
        struct nan_to_num_functor
        {
            using value_type = T;
            using result_type = value_type;

            inline result_type operator()(const value_type a) const
            {
                if (math::isnan(a))
                {
                    return 0;
                }
                if (math::isinf(a))
                {
                    if (a < 0)
                    {
                        return std::numeric_limits<result_type>::lowest();
                    }
                    else
                    {
                        return (std::numeric_limits<result_type>::max)();
                    }
                }
                return a;
            }
        };

        template <class T>
        struct nan_plus
        {
            using value_type = T;
            using result_type = value_type;

            constexpr result_type operator()(const value_type lhs, const value_type rhs) const
            {
                return !math::isnan(rhs) ? lhs + rhs : lhs;
            }
        };

        template <class T>
        struct nan_multiplies
        {
            using value_type = T;
            using result_type = value_type;

            constexpr result_type operator()(const value_type lhs, const value_type rhs) const
            {
                return !math::isnan(rhs) ? lhs * rhs : lhs;
            }
        };

        template <class T, int V>
        struct nan_init
        {
            using value_type = T;
            using result_type = T;
            constexpr result_type operator()(const value_type lhs) const
            {
                return math::isnan(lhs) ? result_type(V) : lhs;
            }
        };
    }

    /**
     * @defgroup  nan_functions nan functions
     */

    /**
     * @ingroup nan_functions
     * @brief Convert nan or +/- inf to numbers
     *
     * This functions converts nan to 0, and +inf to the highest, -inf to the lowest
     * floating point value of the same type.
     *
     * @param e input \ref xexpression
     * @return an \ref xexpression
     */
    template <class E>
    inline auto nan_to_num(E&& e)
    {
        return detail::make_xfunction<detail::nan_to_num_functor>(std::forward<E>(e));
    }

#define XTENSOR_NAN_REDUCER_FUNCTION(NAME, FUNCTOR, RESULT_TYPE, NAN)                                             \
    template <class E, class X, class EVS = DEFAULT_STRATEGY_REDUCERS,                                            \
              class = std::enable_if_t<!std::is_base_of<evaluation_strategy::base, std::decay_t<X>>::value, int>> \
    inline auto NAME(E&& e, X&& axes, EVS es = EVS())                                                             \
    {                                                                                                             \
        using result_type = RESULT_TYPE;                                                                          \
        using functor_type = FUNCTOR<result_type>;                                                                \
        using init_functor_type = detail::nan_init<result_type, NAN>;                                             \
        return reduce(make_xreducer_functor(functor_type(), init_functor_type()), std::forward<E>(e),             \
                      std::forward<X>(axes), es);                                                                 \
    }                                                                                                             \
                                                                                                                  \
    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS,                                                     \
              class = std::enable_if_t<std::is_base_of<evaluation_strategy::base, EVS>::value, int>>              \
    inline auto NAME(E&& e, EVS es = EVS())                                                                       \
    {                                                                                                             \
        using result_type = RESULT_TYPE;                                                                          \
        using functor_type = FUNCTOR<result_type>;                                                                \
        using init_functor_type = detail::nan_init<result_type, NAN>;                                             \
        return reduce(make_xreducer_functor(functor_type(), init_functor_type()), std::forward<E>(e), es);        \
    }

#define OLD_CLANG_NAN_REDUCER(NAME, FUNCTOR, RESULT_TYPE, NAN)                                                       \
    template <class E, class I, class EVS = DEFAULT_STRATEGY_REDUCERS>                                               \
        inline auto NAME(E&& e, std::initializer_list<I> axes, EVS es = EVS())                                       \
        {                                                                                                            \
            using result_type = RESULT_TYPE;                                                                         \
            using functor_type = FUNCTOR<result_type>;                                                               \
            using init_functor_type = detail::nan_init<result_type, NAN>;                                            \
            return reduce(make_xreducer_functor(functor_type(), init_functor_type()), std::forward<E>(e), axes, es); \
        }

#define MODERN_CLANG_NAN_REDUCER(NAME, FUNCTOR, RESULT_TYPE, NAN)                                                 \
    template <class E, class I, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>                             \
    inline auto NAME(E&& e, const I (&axes)[N], EVS es = EVS())                                                   \
    {                                                                                                             \
        using result_type = RESULT_TYPE;                                                                          \
        using functor_type = FUNCTOR<result_type>;                                                                \
        using init_functor_type = detail::nan_init<result_type, NAN>;                                             \
        return reduce(make_xreducer_functor(functor_type(), init_functor_type()), std::forward<E>(e), axes, es);  \
    }

    /**
     * @ingroup nan_functions
     * @brief Sum of elements over given axes, replacing nan with 0.
     *
     * Returns an \ref xreducer for the sum of elements over given
     * \em axes, replacing nan with 0.
     * @param e an \ref xexpression
     * @param axes the axes along which the sum is performed (optional)
     * @param es evaluation strategy of the reducer (optional)
     * @return an \ref xreducer
     */
    XTENSOR_NAN_REDUCER_FUNCTION(nansum, detail::nan_plus, typename std::decay_t<E>::value_type, 0);
#ifdef X_OLD_CLANG
    OLD_CLANG_NAN_REDUCER(nansum, detail::nan_plus, typename std::decay_t<E>::value_type, 0);
#else
    MODERN_CLANG_NAN_REDUCER(nansum, detail::nan_plus, typename std::decay_t<E>::value_type, 0);
#endif

    /**
     * @ingroup nan_functions
     * @brief Product of elements over given axes, replacing nan with 1.
     *
     * Returns an \ref xreducer for the sum of elements over given
     * \em axes, replacing nan with 1.
     * @param e an \ref xexpression
     * @param axes the axes along which the sum is performed (optional)
     * @param es evaluation strategy of the reducer (optional)
     * @return an \ref xreducer
     */
    XTENSOR_NAN_REDUCER_FUNCTION(nanprod, detail::nan_multiplies, typename std::decay_t<E>::value_type, 1);
#ifdef X_OLD_CLANG
    OLD_CLANG_NAN_REDUCER(nanprod, detail::nan_multiplies, typename std::decay_t<E>::value_type, 1);
#else
    MODERN_CLANG_NAN_REDUCER(nanprod, detail::nan_multiplies, typename std::decay_t<E>::value_type, 1);
#endif

#undef XTENSOR_NAN_REDUCER_FUNCTION
#undef OLD_CLANG_NAN_REDUCER
#undef MODERN_CLANG_NAN_REDUCER

#define COUNT_NON_ZEROS_CONTENT                                                 \
    using result_type = std::size_t;                                            \
    using value_type = typename std::decay_t<E>::value_type;                    \
    auto init_fct = [](value_type const& lhs) -> result_type                    \
    {                                                                           \
        return (lhs != value_type(0)) ? result_type(1) : result_type(0);        \
    };                                                                          \
    auto reduce_fct = [](const result_type& lhs, const value_type& rhs)         \
         -> result_type                                                         \
    {                                                                           \
        return (rhs != value_type(0)) ? lhs + result_type(1) : lhs;             \
    };                                                                          \
    auto merge_func = std::plus<result_type>();                                 \

    template <class E, class EVS = DEFAULT_STRATEGY_REDUCERS,
              class = std::enable_if_t<std::is_base_of<evaluation_strategy::base, EVS>::value, int>>
    inline auto count_nonzeros(E&& e, EVS es = EVS())
    {
        COUNT_NON_ZEROS_CONTENT;
        return reduce(make_xreducer_functor(std::move(reduce_fct), std::move(init_fct), std::move(merge_func)),
                      std::forward<E>(e), es);
    }

    template <class E, class X, class EVS = DEFAULT_STRATEGY_REDUCERS,
              class = std::enable_if_t<!std::is_base_of<evaluation_strategy::base, X>::value, int>>
    inline auto count_nonzeros(E&& e, X&& axes, EVS es = EVS())
    {
        COUNT_NON_ZEROS_CONTENT;
        return reduce(make_xreducer_functor(std::move(reduce_fct), std::move(init_fct), std::move(merge_func)),
                      std::forward<E>(e), std::forward<X>(axes), es);
    }

#ifdef X_OLD_CLANG
    template <class E, class I, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto count_nonzeros(E&& e, std::initializer_list<I> axes, EVS es = EVS())
    {
        COUNT_NON_ZEROS_CONTENT;
        return reduce(make_xreducer_functor(std::move(reduce_fct), std::move(init_fct), std::move(merge_func)),
                      std::forward<E>(e), axes, es);
    }
#else
    template <class E, class I, std::size_t N, class EVS = DEFAULT_STRATEGY_REDUCERS>
    inline auto count_nonzeros(E&& e, const I (&axes)[N], EVS es = EVS())
    {
        COUNT_NON_ZEROS_CONTENT;
        return reduce(make_xreducer_functor(std::move(reduce_fct), std::move(init_fct), std::move(merge_func)),
                      std::forward<E>(e), axes, es);
    }
#endif

#undef COUNT_NON_ZEROS_CONTENT

    /**
     * @ingroup nan_functions
     * @brief Cumulative sum, replacing nan with 0.
     *
     * Returns an xaccumulator for the sum of elements over given
     * \em axis, replacing nan with 0.
     * @param e an \ref xexpression
     * @param axis the axis along which the elements are accumulated (optional)
     * @return an xaccumulator
     */
    template <class E>
    inline auto nancumsum(E&& e, std::size_t axis)
    {
        using result_type = big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(detail::nan_plus<result_type>(), detail::nan_init<result_type, 0>()), std::forward<E>(e), axis);
    }

    template <class E>
    inline auto nancumsum(E&& e)
    {
        using result_type = big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(detail::nan_plus<result_type>(), detail::nan_init<result_type, 0>()), std::forward<E>(e));
    }

    /**
     * @ingroup nan_functions
     * @brief Cumulative product, replacing nan with 1.
     *
     * Returns an xaccumulator for the product of elements over given
     * \em axis, replacing nan with 1.
     * @param e an \ref xexpression
     * @param axis the axis along which the elements are accumulated (optional)
     * @return an xaccumulator
     */
    template <class E>
    inline auto nancumprod(E&& e, std::size_t axis)
    {
        using result_type = big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(detail::nan_multiplies<result_type>(), detail::nan_init<result_type, 1>()), std::forward<E>(e), axis);
    }

    template <class E>
    inline auto nancumprod(E&& e)
    {
        using result_type = big_promote_type_t<typename std::decay_t<E>::value_type>;
        return accumulate(make_xaccumulator_functor(detail::nan_multiplies<result_type>(), detail::nan_init<result_type, 1>()), std::forward<E>(e));
    }

    namespace detail
    {
        template <class T>
        struct diff_impl
        {
            template <class Arg>
            inline void operator()(Arg& ad, const std::size_t& n,
                                   xstrided_slice_vector& slice1, xstrided_slice_vector& slice2,
                                   const std::size_t& saxis)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    slice2[saxis] = range(xnone(), ad.shape()[saxis] - 1);
                    ad = strided_view(ad, slice1) - strided_view(ad, slice2);
                }
            };
        };

        template <>
        struct diff_impl<bool>
        {
            template <class Arg>
            inline void operator()(Arg& ad, const std::size_t& n,
                                   xstrided_slice_vector& slice1, xstrided_slice_vector& slice2,
                                   const std::size_t& saxis)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    slice2[saxis] = range(xnone(), ad.shape()[saxis] - 1);
                    ad = not_equal(strided_view(ad, slice1), strided_view(ad, slice2));
                }
            };
        };
    }

    /**
     * @ingroup red_functions
     * @brief Calculate the n-th discrete difference along the given axis.
     *
     * Calculate the n-th discrete difference along the given axis. This function is not lazy (might change in the future).
     * @param a an \ref xexpression
     * @param n The number of times values are differenced. If zero, the input is returned as-is. (optional)
     * @param axis The axis along which the difference is taken, default is the last axis.
     * @return an xarray
     */
    template <class T>
    auto diff(const xexpression<T>& a, std::size_t n = 1, std::ptrdiff_t axis = -1)
    {
        auto ad = a.derived_cast();
        std::size_t saxis = static_cast<std::size_t>(axis);

        if (n == 0)
        {
            return eval(ad);
        }

        if (axis == -1)
        {
            saxis = ad.dimension() - 1;
        }

        xstrided_slice_vector slice1(ad.dimension(), all());
        xstrided_slice_vector slice2(ad.dimension(), all());
        slice1[saxis] = range(1, xnone());

        detail::diff_impl<typename T::value_type> impl;
        impl(ad, n, slice1, slice2, saxis);

        return eval(ad);
    }

    /**
     * @ingroup red_functions
     * @brief Integrate along the given axis using the composite trapezoidal rule.
     *
     * Returns definite integral as approximated by trapezoidal rule. This function is not lazy (might change in the future).
     * @param y an \ref xexpression
     * @param dx the spacing between sample points (optional)
     * @param axis the axis along which to integrate.
     * @return an xarray
     */
    template <class T>
    auto trapz(const xexpression<T>& y, double dx = 1.0, std::ptrdiff_t axis = -1)
    {
        auto& yd = y.derived_cast();
        std::size_t saxis = static_cast<std::size_t>(axis);

        if (axis == -1)
        {
          saxis = yd.dimension() - 1;
        }

        xstrided_slice_vector slice1(yd.dimension(), all());
        xstrided_slice_vector slice2(yd.dimension(), all());
        slice1[saxis] = range(1, xnone());
        slice2[saxis] = range(xnone(), yd.shape()[saxis] - 1);

        auto trap = dx * (strided_view(yd, slice1) + strided_view(yd, slice2)) * 0.5;

        return eval(sum(trap, {saxis}));
    }

    /**
     * @ingroup red_functions
     * @brief Integrate along the given axis using the composite trapezoidal rule.
     *
     * Returns definite integral as approximated by trapezoidal rule. This function is not lazy (might change in the future).
     * @param y an \ref xexpression
     * @param x an \ref xexpression representing the sample points corresponding to the y values.
     * @param axis the axis along which to integrate.
     * @return an xarray
     */
    template <class T, class E>
    auto trapz(const xexpression<T>& y, const xexpression<E>& x, std::ptrdiff_t axis = -1)
    {
        auto& yd = y.derived_cast();
        auto& xd = x.derived_cast();
        decltype(diff(x)) dx;

        std::size_t saxis = static_cast<std::size_t>(axis);

        if (axis == -1)
        {
            saxis = yd.dimension() - 1;
        }

        if (xd.dimension() == 1)
        {
            dx = diff(x);
            typename std::decay_t<decltype(yd)>::shape_type shape;
            resize_container(shape, yd.dimension());
            std::fill(shape.begin(), shape.end(), 1);
            shape[saxis] = dx.shape()[0];
            dx.reshape(shape);
        }
        else
        {
            dx = diff(x, 1, axis);
        }

        xstrided_slice_vector slice1(yd.dimension(), all());
        xstrided_slice_vector slice2(yd.dimension(), all());
        slice1[saxis] = range(1, xnone());
        slice2[saxis] = range(xnone(), yd.shape()[saxis] - 1);

        auto trap = dx * (strided_view(yd, slice1) + strided_view(yd, slice2)) * 0.5;

        return eval(sum(trap, {saxis}));
    }
}

#endif
