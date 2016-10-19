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
#include "xfunction.hpp"

namespace xt
{

    /***********
     * Helpers *
     ***********/

    namespace detail
    {
        template <class R, class... Args, class... E>
        inline auto make_xfunction(R (*f) (Args...), const E&... e) noexcept
        {
            using type = xfunction<R (*) (Args...), R, get_xexpression_type<E>...>;
            return type(f, get_xexpression(e)...);
        }

        template <class... E>
        using mf_type = common_value_type<E...> (*) (get_value_type<E>...);

        template <class... Args>
        using get_xfunction_free_type = std::enable_if_t<has_xexpression<Args...>::value,
                                                         xfunction<mf_type<Args...>,
                                                                   common_value_type<Args...>,
                                                                   get_xexpression_type<Args>...>>;
    }

    /*******************
     * basic functions *
     *******************/

    /**
     * @defgroup basic_functions Basic functions
     */

    /**
     * @ingroup basic_functions
     * @brief Absolute value function
     * 
     * Returns an \ref xfunction for the element-wise absolute value
     * of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto abs(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::abs, e.derived_cast());
    }

    /**
     * @ingroup basic_functions
     * @brief Absolute value function
     * 
     * Returns an \ref xfunction for the element-wise absolute value
     * of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto fabs(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::fabs, e.derived_cast());
    }

    /**
     * @ingroup basic_functions
     * @brief Remainder of the floating point division operation
     * 
     * Returns an \ref xfunction for the element-wise remainder of
     * the floating point division operation <em>e1 / e2</em>.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fmod(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fmod, e1, e2);
    }

    /**
     * @ingroup basic_functions
     * @brief Signed remainder of the division operation
     * 
     * Returns an \ref xfunction for the element-wise signed remainder
     * of the floating point division operation <em>e1 / e2</em>.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto remainder(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::remainder, e1, e2);
    }

    /**
     * @ingroup basic_functions
     * @brief Fused multiply-add operation
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
    inline auto fma(const E1& e1, const E2& e2, const E3& e3) noexcept
        -> detail::get_xfunction_free_type<E1, E2, E3>
    {
        using functor_type = detail::mf_type<E1, E2, E3>;
        return detail::make_xfunction((functor_type)std::fma, e1, e2, e3);
    }

    /**
     * @ingroup basic_functions
     * @brief Maximum function
     *
     * Returns an \ref xfunction for the element-wise maximum
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fmax(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fmax, e1, e2);
    }

    /**
     * @ingroup basic_functions
     * @brief Minimum function
     *
     * Returns an \ref xfunction for the element-wise minimum
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fmin(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fmin, e1, e2);
    }

    /**
     * @ingroup basic_functions
     * @brief Positive difference function
     *
     * Returns an \ref xfunction for the element-wise positive
     * difference of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto fdim(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fdim, e1, e2);
    }

    /*************************
     * exponential functions *
     *************************/

    /**
     * @defgroup exp_functions Exponential functions
     */

    /**
     * @ingroup exp_functions
     * @brief Natural exponential function
     *
     * Returns an \ref xfunction for the element-wise natural
     * exponential of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto exp(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::exp, e.derived_cast());
    }

    /**
     * @ingroup exp_functions
     * @brief Base 2 exponential function
     *
     * Returns an \ref xfunction for the element-wise base 2
     * exponential of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto exp2(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::exp2, e.derived_cast());
    }

    /**
     * @ingroup exp_functions
     * @brief Natural exponential minus one function
     *
     * Returns an \ref xfunction for the element-wise natural
     * exponential of \em e, minus 1
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto expm1(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::expm1, e.derived_cast());
    }

    /**
     * @ingroup exp_functions
     * @brief Natural logarithm function
     *
     * Returns an \ref xfunction for the element-wise natural
     * logarithm of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::log, e.derived_cast());
    }

    /**
     * @ingroup exp_functions
     * @brief Base 10 logarithm function
     *
     * Returns an \ref xfunction for the element-wise base 10
     * logarithm of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log10(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::log10, e.derived_cast());
    }

    /**
     * @ingroup exp_functions
     * @brief Base 2 logarithm function
     *
     * Returns an \ref xfunction for the element-wise base 2
     * logarithm of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log2(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::log2, e.derived_cast());
    }

    /**
     * @ingroup exp_functions
     * @brief Natural logarithm of one plus function
     *
     * Returns an \ref xfunction for the element-wise natural
     * logarithm of \em e, plus 1
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto log1p(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::log1p, e.derived_cast());
    }

    /*******************
     * power functions *
     *******************/

    /**
     * @defgroup pow_functions Power functions
     */

    /**
     * @ingroup pow_functions
     * @brief Power function
     *
     * Returns an \ref xfunction for the element-wise value of
     * of \em e1 raised to the power \em e2
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto pow(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::pow, e1, e2);
    }

    /**
     * @ingroup pow_functions
     * @brief Square root function
     *
     * Returns an \ref xfunction for the element-wise square 
     * root of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sqrt(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::sqrt, e.derived_cast());
    }

    /**
     * @ingroup pow_functions
     * @brief Cubic root function
     *
     * Returns an \ref xfunction for the element-wise cubic
     * root of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto cbrt(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::cbrt, e.derived_cast());
    }

    /**
     * @ingroup pow_functions
     * @brief Hypotenuse function
     *
     * Returns an \ref xfunction for the element-wise square
     * root of the sum of the square of \em e1 and \em e2, avoiding
     * overflow and underflow at intermediate stages of computation
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto hypot(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::hypot, e1, e2);
    }

    /***************************
     * trigonometric functions *
     ***************************/

    /**
     * @defgroup trigo_functions Trigonometric function
     */

    /**
     * @ingroup trigo_functions
     * @brief Sine function
     *
     * Returns an \ref xfunction for the element-wise sine
     * of \em e (measured in radians)
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sin(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::sin, e.derived_cast());
    }

    /**
     * @ingroup trigo_functions
     * @brief Cosine function
     *
     * Returns an \ref xfunction for the element-wise cosine
     * of \em e (measured in radians)
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto cos(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::cos, e.derived_cast());
    }

    /**
     * @ingroup trigo_functions
     * @brief Tangent function
     *
     * Returns an \ref xfunction for the element-wise tangent
     * of \em e (measured in radians)
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto tan(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::tan, e.derived_cast());
    }

    /**
     * @ingroup trigo_functions
     * @brief Arcsine function
     *
     * Returns an \ref xfunction for the element-wise arcsine
     * of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto asin(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::asin, e.derived_cast());
    }

    /**
     * @ingroup trigo_functions
     * @brief Arccosine function
     *
     * Returns an \ref xfunction for the element-wise arccosine
     * of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto acos(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::acos, e.derived_cast());
    }

    /**
     * @ingroup trigo_functions
     * @brief Arctangent function
     *
     * Returns an \ref xfunction for the element-wise arctangent
     * of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto atan(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::atan, e.derived_cast());
    }

    /**
     * @ingroup trigo_functions
     * @brief Artangent function, using signs to determine quadrants
     *
     * Returns an \ref xfunction for the element-wise arctangent
     * of <em>e1 / e2</em>, using the signs of arguments to determine the
     * correct quadrant
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     * @note e1 and e2 can't be both scalars.
     */
    template <class E1, class E2>
    inline auto atan2(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::atan2, e1, e2);
    }

    /************************
     * hyperbolic functions *
     ************************/

    /**
     * @defgroup hyper_functions Hyperbolic functions
     */

    /**
     * @ingroup hyper_functions
     * @brief Hyperbolic sine function
     *
     * Returns an \ref xfunction for the element-wise hyperbolic
     * sine of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto sinh(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::sinh, e.derived_cast());
    }

    /**
     * @ingroup hyper_functions
     * @brief Hyperbolic cosine function
     *
     * Returns an \ref xfunction for the element-wise hyperbolic
     * cosine of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto cosh(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::cosh, e.derived_cast());
    }

    /**
     * @ingroup hyper_functions
     * @brief Hyperbolic tangent function
     *
     * Returns an \ref xfunction for the element-wise hyperbolic
     * tangent of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto tanh(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::tanh, e.derived_cast());
    }

    /**
     * @ingroup hyper_functions
     * @brief Inverse hyperbolic sine function
     *
     * Returns an \ref xfunction for the element-wise inverse hyperbolic
     * sine of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto asinh(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::asinh, e.derived_cast());
    }

    /**
     * @ingroup hyper_functions
     * @brief Inverse hyperbolic cosine function
     *
     * Returns an \ref xfunction for the element-wise inverse hyperbolic
     * cosine of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto acosh(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::acosh, e.derived_cast());
    }

    /**
     * @ingroup hyper_functions
     * @brief Inverse hyperbolic tangent function
     *
     * Returns an \ref xfunction for the element-wise inverse hyperbolic
     * tangent of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto atanh(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::atanh, e.derived_cast());
    }

    /*****************************
     * error and gamma functions *
     *****************************/

    /**
     * @defgroup err_functions Error and gamma functions
     */

    /**
     * @ingroup err_functions
     * @brief Error function
     *
     * Returns an \ref xfunction for the element-wise error function
     * of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto erf(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::erf, e.derived_cast());
    }

    /**
     * @ingroup err_functions
     * @brief Complementary error function
     *
     * Returns an \ref xfunction for the element-wise complementary
     * error function of \em e, whithout loss of precision for large argument
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto erfc(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::erfc, e.derived_cast());
    }

    /**
     * @ingroup err_functions
     * @brief Gamma function
     *
     * Returns an \ref xfunction for the element-wise gamma function
     * of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto tgamma(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::tgamma, e.derived_cast());
    }

    /**
     * @ingroup err_functions
     * @brief Natural logarithm of the gamma function
     *
     * Returns an \ref xfunction for the element-wise logarithm of
     * the asbolute value fo the gamma function of \em e
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto lgamma(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::lgamma, e.derived_cast());
    }

}

#endif

