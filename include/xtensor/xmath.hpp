/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

/**
 * @brief standard mathematical functions for xexpressions
 */

#ifndef XMATH_HPP
#define XMATH_HPP

#include <cmath>
#include <type_traits>
#include <complex>

#include "xfunction.hpp"
#include "xreducer.hpp"

namespace xt
{

    /***********
     * Helpers *
     ***********/

    namespace detail
    {
        template <class R, class... Args, class... E>
        inline auto make_xfunction(R (*f) (Args...), E&&... e) noexcept
        {
            using type = xfunction<R (*) (Args...), R, const_xclosure_t<E>...>;
            return type(f, std::forward<E>(e)...);
        }

        template <class... E>
        using mf_type = common_value_type_t<std::decay_t<E>...> (*) (xvalue_type_t<std::decay_t<E>>...);

        template <class... E>
        using get_xfunction_free_type = std::enable_if_t<has_xexpression<std::decay_t<E>...>::value,
                                                         xfunction<mf_type<E...>,
                                                                   common_value_type_t<std::decay_t<E>...>,
                                                                   const_xclosure_t<E>...>>;
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::abs, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::fabs, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fmod, std::forward<E1>(e1), std::forward<E2>(e2));
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
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::remainder, std::forward<E1>(e1), std::forward<E2>(e2));
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
        -> detail::get_xfunction_free_type<E1, E2, E3>
    {
        using functor_type = detail::mf_type<E1, E2, E3>;
        return detail::make_xfunction((functor_type)std::fma, std::forward<E1>(e1), std::forward<E2>(e2), std::forward<E3>(e3));
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
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fmax, std::forward<E1>(e1), std::forward<E2>(e2));
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
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fmin, std::forward<E1>(e1), std::forward<E2>(e2));
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
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fdim, std::forward<E1>(e1), std::forward<E2>(e2));
    }
    
    namespace detail
    {
        // this function will be part of std with C++17
        template <class T>
        constexpr T clamp(T v, T lo, T hi)
        {
            return v < lo ? lo : hi < v ? hi : v;
        }
    }

    /**
     * @ingroup basic_functions
     * @brief Clip values between hi and lo
     * 
     * Returns an \ref xfunction for the element-wise clipped 
     * values between hi- and lo
     * @param e1 an \ref xexpression or a scalar
     * @param hi a scalar
     * @param lo a scalar
     *
     * @return a \ref xfunction
     */
    template <class E1, class E2, class E3>
    inline auto clip(E1&& e1, E2&& hi, E3&& lo) noexcept
        -> detail::get_xfunction_free_type<E1, E2, E3>
    {
        using functor_type = detail::mf_type<E1, E2, E3>;
        return detail::make_xfunction((functor_type)detail::clamp, std::forward<E1>(e1), std::forward<E2>(hi), std::forward<E3>(lo));
    }

    namespace detail
    {
        template <typename T>
        inline constexpr std::enable_if_t<std::is_signed<T>::value, T>
        sign_impl(T x)
        {
            return std::isnan(x) ? std::numeric_limits<T>::quiet_NaN() : x == 0 ? (T) copysign(T(0), x) : (T) copysign(T(1), x);
        }

        template <typename T>
        inline std::enable_if_t<detail::is_complex<T>::value, T>
        sign_impl(T x)
        {
            typename T::value_type e = x.real() ? x.real() : x.imag();
            return T(sign_impl(e), 0);
        }

        template <typename T>
        inline constexpr std::enable_if_t<std::is_unsigned<T>::value, T>
        sign_impl(T x)
        {
            return T(x > T(0));
        }
    }

    /**
     * @ingroup basic_function
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)detail::sign_impl, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::exp, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::exp2, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::expm1, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::log, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::log10, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::log2, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::log1p, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::pow, std::forward<E1>(e1), std::forward<E2>(e2));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::sqrt, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::cbrt, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::hypot, std::forward<E1>(e1), std::forward<E2>(e2));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::sin, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::cos, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::tan, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::asin, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::acos, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::atan, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::atan2, std::forward<E1>(e1), std::forward<E2>(e2));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::sinh, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::cosh, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::tanh, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::asinh, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::acosh, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::atanh, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::erf, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::erfc, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::tgamma, std::forward<E>(e));
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
        -> detail::get_xfunction_free_type<E>
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::lgamma, std::forward<E>(e));
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
     * @param axes the axes along which the sum is performed
     * @return an \ref xreducer
     */
    template <class E, class X>
    inline auto sum(E&& e, const X& axes) noexcept
    {
        using functor_type = std::plus<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }

#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto sum(E&& e, std::initializer_list<I> axes) noexcept
    {
        using functor_type = std::plus<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }
#else
    template <class E, class I, std::size_t N>
    inline auto sum(E&& e, const I(&axes)[N]) noexcept
    {
        using functor_type = std::plus<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }
#endif

    /**
     * @ingroup red_functions
     * @brief Product of elements over given axes.
     *
     * Returns an \ref xreducer for the product of elements over given
     * \em axes.
     * @param e an \ref xexpression
     * @param axes the axes along which the product is performed
     * @return an \ref xreducer
     */
    template <class E, class X>
    inline auto prod(E&& e, const X& axes) noexcept
    {
        using functor_type = std::multiplies<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }

#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto prod(E&& e, std::initializer_list<I> axes) noexcept
    {
        using functor_type = std::multiplies<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }
#else
    template <class E, class I, std::size_t N>
    inline auto prod(E&& e, const I(&axes)[N]) noexcept
    {
        using functor_type = std::multiplies<typename std::decay_t<E>::value_type>;
        return reduce(functor_type(), std::forward<E>(e), axes);
    }
#endif
}

#endif

