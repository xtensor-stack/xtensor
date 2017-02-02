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
#include <algorithm>
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
        inline auto make_xfunction(E&&... e) noexcept
        {
            using functor_type = F<common_value_type<typename std::decay<E>::type...>>;
            using result_type = typename functor_type::result_type;
            using type = xfunction<functor_type, result_type, xclosure<E>...>;
            return type(functor_type(), std::forward<E>(e)...);
        }

        template <template <class...> class F, class... E>
        using get_xfunction_type = std::enable_if_t<has_xexpression<typename std::decay<E>::type...>::value,
                                                    xfunction<F<common_value_type<typename std::decay<E>::type...>>,
                                                              typename F<common_value_type<typename std::decay<E>::type...>>::result_type,
                                                              xclosure<E>...>>;
    }

    /*************
     * operators *
     *************/

    template <class E>
    inline auto operator+(E&& e) noexcept
        -> detail::get_xfunction_type<identity, E>
    {
        return detail::make_xfunction<identity>(std::forward<E>(e));
    }

    template <class E>
    inline auto operator-(E&& e) noexcept
        -> detail::get_xfunction_type<std::negate, E>
    {
        return detail::make_xfunction<std::negate>(std::forward<E>(e));
    }

    template <class E1, class E2>
    inline auto operator+(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::plus, E1, E2>
    {
        return detail::make_xfunction<std::plus>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E1, class E2>
    inline auto operator-(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::minus, E1, E2>
    {
        return detail::make_xfunction<std::minus>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E1, class E2>
    inline auto operator*(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::multiplies, E1, E2>
    {
        return detail::make_xfunction<std::multiplies>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E1, class E2>
    inline auto operator/(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::divides, E1, E2>
    {
        return detail::make_xfunction<std::divides>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E1, class E2>
    inline auto operator||(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::logical_or, E1, E2>
    {
        return detail::make_xfunction<std::logical_or>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E1, class E2>
    inline auto operator&&(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::logical_and, E1, E2>
    {
        return detail::make_xfunction<std::logical_and>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E>
    inline auto operator!(E&& e) noexcept
        -> detail::get_xfunction_type<std::logical_not, E>
    {
        return detail::make_xfunction<std::logical_not>(std::forward<E>(e));
    }

    template <class E1, class E2>
    inline auto operator<(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::less, E1, E2>
    {
        return detail::make_xfunction<std::less>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E1, class E2>
    inline auto operator<=(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::less_equal, E1, E2>
    {
        return detail::make_xfunction<std::less_equal>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E1, class E2>
    inline auto operator>(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::greater, E1, E2>
    {
        return detail::make_xfunction<std::greater>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E1, class E2>
    inline auto operator>=(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::greater_equal, E1, E2>
    {
        return detail::make_xfunction<std::greater_equal>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E1, class E2>
    inline bool operator==(const xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        const E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();
        bool res = de1.shape() == de2.shape();
        auto iter1 = de1.begin();
        auto iter2 = de2.begin();
        auto iter_end = de1.end();
        while (res && iter1 != iter_end)
        {
            res = (*iter1++ == *iter2++);
        }
        return res;
    }

    template <class E1, class E2>
    inline bool operator!=(const xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        return !(e1 == e2);
    }

    template <class E1, class E2>
    inline auto equal(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::equal_to, E1, E2>
    {
        return detail::make_xfunction<std::equal_to>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E1, class E2>
    inline auto not_equal(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::not_equal_to, E1, E2>
    {
        return detail::make_xfunction<std::not_equal_to>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    template <class E1, class E2, class E3>
    inline auto where(E1&& e1, E2&& e2, E3&& e3) noexcept
        -> detail::get_xfunction_type<conditional_ternary, E1, E2, E3>
    {
         return detail::make_xfunction<conditional_ternary>(std::forward<E1>(e1), std::forward<E2>(e2), std::forward<E3>(e3));
    }

    template <class E>
    inline bool any(E&& e)
    {
        return std::any_of(e.storage_begin(), e.storage_end(), 
                           [](const typename std::decay<E>::type::value_type& el) { return el; });
    }

    template <class E>
    inline bool all(E&& e)
    {
        return std::all_of(e.storage_begin(), e.storage_end(),
                           [](const typename std::decay<E>::type::value_type& el) { return el; });
    }
}

#endif

