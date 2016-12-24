/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XOPERATION_HPP
#define XOPERATION_HPP

#include <functional>
#include <type_traits>

#include "xfunction.hpp"
#include "xscalar.hpp"

namespace xt
{

    /***********
     * helpers *
     ***********/

    template <class T>
    struct identity
    {
        using result_type = T;

        constexpr T operator()(const T& t) const noexcept
        {
            return +t;
        }
    };

    template <class T>
    struct conditional_ternary
    {
        using result_type = T;

        constexpr result_type operator()(const T& t1, const T& t2, const T& t3) const noexcept
        {
            return t1 ? t2 : t3;
        }
    };

    namespace detail
    {
        template <template <class...> class F, class... E>
        inline auto make_xfunction(const E&... e) noexcept
        {
            using functor_type = F<common_value_type<E...>>;
            using result_type = typename functor_type::result_type;
            using type = xfunction<functor_type, result_type, get_xexpression_type<E>...>;
            return type(functor_type(), get_xexpression(e)...);
        }

        template <template <class...> class F, class... E>
        using get_xfunction_type = std::enable_if_t<has_xexpression<E...>::value,
                                                    xfunction<F<common_value_type<E...>>,
                                                                common_value_type<E...>,
                                                                get_xexpression_type<E>...>>;
    }

    /*************
     * operators *
     *************/

    template <class E>
    inline auto operator+(const xexpression<E>& e) noexcept
    {
        return detail::make_xfunction<identity>(e.derived_cast());
    }

    template <class E>
    inline auto operator-(const xexpression<E>& e) noexcept
    {
        return detail::make_xfunction<std::negate>(e.derived_cast());
    }

    template <class E1, class E2>
    inline auto operator+(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_type<std::plus, E1, E2>
    {
        return detail::make_xfunction<std::plus>(e1, e2);
    }

    template <class E1, class E2>
    inline auto operator-(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_type<std::minus, E1, E2>
    {
        return detail::make_xfunction<std::minus>(e1, e2);
    }

    template <class E1, class E2>
    inline auto operator*(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_type<std::multiplies, E1, E2>
    {
        return detail::make_xfunction<std::multiplies>(e1, e2);
    }

    template <class E1, class E2>
    inline auto operator/(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_type<std::divides, E1, E2>
    {
        return detail::make_xfunction<std::divides>(e1, e2);
    }

    template <class E1, class E2>
    inline auto operator||(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_type<std::logical_or, E1, E2>
    {
        return detail::make_xfunction<std::logical_or>(e1, e2);
    }

    template <class E1, class E2>
    inline auto operator&&(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_type<std::logical_and, E1, E2>
    {
        return detail::make_xfunction<std::logical_and>(e1, e2);
    }

    template <class E>
    inline auto operator!(const E& e) noexcept
        -> detail::get_xfunction_type<std::logical_not, E>
    {
        return detail::make_xfunction<std::logical_not>(e);
    }

    template <class E1, class E2, class E3>
    inline auto where(const E1& e1, const E2& e2, const E3& e3) noexcept
        -> detail::get_xfunction_type<conditional_ternary, E1, E2, E3>
    {
         return detail::make_xfunction<conditional_ternary>(e1, e2, e3);
    }

}

#endif

