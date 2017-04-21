/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XCOMPLEX_HPP
#define XCOMPLEX_HPP

#include <type_traits>
#include <utility>

#include "xtensor/xbuilder.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xoffsetview.hpp"

namespace xt
{

    /******************************
     * real and imag declarations *
     ******************************/

    template <class E>
    decltype(auto) real(E&& e) noexcept;

    template <class E>
    decltype(auto) imag(E&& e) noexcept;

    /********************************
     * real and imag implementation *
     ********************************/

    namespace detail
    {
        template <bool iscomplex = true>
        struct complex_helper
        {
            template <class E>
            static inline auto real(E&& e) noexcept
            {
                using real_type = typename std::decay_t<E>::value_type::value_type;
                return xoffsetview<xclosure_t<E>, real_type, 0>(std::forward<E>(e));
            }

            template <class E>
            static inline auto imag(E&& e) noexcept
            {
                using real_type = typename std::decay_t<E>::value_type::value_type;
                return xoffsetview<xclosure_t<E>, real_type, sizeof(real_type)>(std::forward<E>(e));
            }
        };

        template <>
        struct complex_helper<false>
        {
            template <class E>
            static inline decltype(auto) real(E&& e) noexcept
            {
                return e;
            }

            template <class E>
            static inline auto imag(E&& e) noexcept
            {
                return zeros<typename std::decay_t<E>::value_type>(e.shape());
            }
        };

        template <bool isexpression = true>
        struct complex_expression_helper
        {
            template <class E>
            static inline auto real(E&& e) noexcept
            {
                return detail::complex_helper<is_complex<typename std::decay_t<E>::value_type>::value>::real(e);
            }

            template <class E>
            static inline auto imag(E&& e) noexcept
            {
                return detail::complex_helper<is_complex<typename std::decay_t<E>::value_type>::value>::imag(e);
            }
        };

        template <>
        struct complex_expression_helper<false>
        {
            template <class E>
            static inline decltype(auto) real(E&& e) noexcept
            {
                return forward_real(std::forward<E>(e));
            }

            template <class E>
            static inline decltype(auto) imag(E&& e) noexcept
            {
                return forward_imag(std::forward<E>(e));
            }
        };
    }

    /**
     * @brief Returns an \ref xexpression representing the real part of the given expression.
     *
     * @tparam e the \ref xexpression
     *
     * The returned expression either hold a const reference to \p e or a copy
     * depending on whether \p e is an lvalue or an rvalue.
     */
    template <class E>
    inline decltype(auto) real(E&& e) noexcept
    {
        return detail::complex_expression_helper<is_xexpression<std::decay_t<E>>::value>::real(std::forward<E>(e));
    }

    /**
     * @brief Returns an \ref xexpression representing the imaginary part of the given expression.
     *
     * @tparam e the \ref xexpression
     *
     * The returned expression either hold a const reference to \p e or a copy
     * depending on whether \p e is an lvalue or an rvalue.
     */
    template <class E>
    inline decltype(auto) imag(E&& e) noexcept
    {
        return detail::complex_expression_helper<is_xexpression<std::decay_t<E>>::value>::imag(std::forward<E>(e));
    }

#define UNARY_COMPLEX_FUNCTOR(NAME)\
    template <class T>\
    struct NAME##_fun {\
        using argument_type = T;\
        using result_type = decltype(std::NAME(std::declval<T>()));\
        constexpr result_type operator()(const T& t) const {\
            using std::NAME;\
            return NAME(t);\
        }\
    }

    namespace math
    {
        const double PI  = 3.141592653589793238463;

        UNARY_COMPLEX_FUNCTOR(conj);
        UNARY_COMPLEX_FUNCTOR(norm);
        UNARY_COMPLEX_FUNCTOR(arg);
    }

    /**
     * @brief Returns an \ref xfunction evaluating to the complex conjugate of the given expression.
     *
     * @param e the \ref xexpression
     */
    template <class E>
    inline auto conj(E&& e) noexcept
    {
        using value_type = typename std::decay_t<E>::value_type;
        using functor = math::conj_fun<value_type>;
        using result_type = typename functor::result_type;
        using type = xfunction<functor, result_type, const_xclosure_t<E>>;
        return type(functor(), std::forward<E>(e));
    }

    /**
     * @brief Calculates the phase angle (in radians) elementwise for the complex numbers in e.
     * @param e the \ref xexpression
     */
    template <class E>
    inline auto arg(E&& e) noexcept
    {
        using value_type = typename std::decay_t<E>::value_type;
        using functor = math::arg_fun<value_type>;
        using result_type = typename functor::result_type;
        using type = xfunction<functor, result_type, const_xclosure_t<E>>;
        return type(functor(), std::forward<E>(e));
    }

    /**
     * @brief Calculates the phase angle elementwise for the complex numbers in e.
     * Note that this function might be slightly less perfomant than \ref arg.
     * @param e the \ref xexpression
     * @param deg calculate angle in degrees instead of radians
     */
    template <class E>
    inline auto angle(E&& e, bool deg = false) noexcept
    {
        double multiplier = 1.0;
        if (deg)
        {
            multiplier = 180. / math::PI;
        }
        return arg(std::forward<E>(e)) * std::move(multiplier);
    }

    /**
     * @brief Calculates the squared magnitude elementwise for the complex numbers in e.
     *
     * @param e the \ref xexpression
     */
    template <class E>
    inline auto norm(E&& e) noexcept
    {
        using value_type = typename std::decay_t<E>::value_type;
        using functor = math::norm_fun<value_type>;
        using result_type = typename functor::result_type;
        using type = xfunction<functor, result_type, const_xclosure_t<E>>;
        return type(functor(), std::forward<E>(e));
    }
}
#endif
