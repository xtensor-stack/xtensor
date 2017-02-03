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

    /**
     * @defgroup arithmetic_operators Arithmetic operators
     */

    /**
     * @ingroup arithmetic_operators
     * @brief Identity
     *
     * Returns an \ref xfunction for the element-wise identity
     * of \a e.
     * @param e an \ref xexpression
     * @return an \ref xfunction
     */
    template <class E>
    inline auto operator+(E&& e) noexcept
        -> detail::get_xfunction_type<identity, E>
    {
        return detail::make_xfunction<identity>(std::forward<E>(e));
    }

    /**
    * @ingroup arithmetic_operators
    * @brief Opposite
    *
    * Returns an \ref xfunction for the element-wise opposite
    * of \a e.
    * @param e an \ref xexpression
    * @return an \ref xfunction
    */
    template <class E>
    inline auto operator-(E&& e) noexcept
        -> detail::get_xfunction_type<std::negate, E>
    {
        return detail::make_xfunction<std::negate>(std::forward<E>(e));
    }

    /**
    * @ingroup arithmetic_operators
    * @brief Addition
    *
    * Returns an \ref xfunction for the element-wise addition
    * of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator+(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::plus, E1, E2>
    {
        return detail::make_xfunction<std::plus>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup arithmetic_operators
    * @brief Substraction
    *
    * Returns an \ref xfunction for the element-wise substraction
    * of \a e2 to \a e1.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator-(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::minus, E1, E2>
    {
        return detail::make_xfunction<std::minus>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup arithmetic_operators
    * @brief Multiplication
    *
    * Returns an \ref xfunction for the element-wise multiplication
    * of \a e1 by \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator*(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::multiplies, E1, E2>
    {
        return detail::make_xfunction<std::multiplies>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup arithmetic_operators
    * @brief Division
    *
    * Returns an \ref xfunction for the element-wise division
    * of \a e1 by \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator/(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::divides, E1, E2>
    {
        return detail::make_xfunction<std::divides>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @defgroup logical_operators Logical operators
     */

     /**
     * @ingroup logical_operators
     * @brief Or
     *
     * Returns an \ref xfunction for the element-wise or
     * of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator||(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::logical_or, E1, E2>
    {
        return detail::make_xfunction<std::logical_or>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup logical_operators
    * @brief And
    *
    * Returns an \ref xfunction for the element-wise and
    * of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator&&(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::logical_and, E1, E2>
    {
        return detail::make_xfunction<std::logical_and>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup logical_operators
    * @brief Not
    *
    * Returns an \ref xfunction for the element-wise not
    * of \a e.
    * @param e an \ref xexpression
    * @return an \ref xfunction
    */
    template <class E>
    inline auto operator!(E&& e) noexcept
        -> detail::get_xfunction_type<std::logical_not, E>
    {
        return detail::make_xfunction<std::logical_not>(std::forward<E>(e));
    }

    /**
     * @defgroup comparison_operators Comparison operators
     */

    /**
     * @ingroup comparison_operators
     * @brief Lesser than
     *
     * Returns an \ref xfunction for the element-wise
     * lesser than comparison of \a e1 and \a e2.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return an \ref xfunction
     */
    template <class E1, class E2>
    inline auto operator<(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::less, E1, E2>
    {
        return detail::make_xfunction<std::less>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup comparison_operators
    * @brief Lesser or equal
    *
    * Returns an \ref xfunction for the element-wise
    * lesser or equal comparison of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator<=(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::less_equal, E1, E2>
    {
        return detail::make_xfunction<std::less_equal>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup comparison_operators
    * @brief Greater than
    *
    * Returns an \ref xfunction for the element-wise
    * greater than comparison of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator>(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::greater, E1, E2>
    {
        return detail::make_xfunction<std::greater>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup comparison_operators
    * @brief Greater or equal
    *
    * Returns an \ref xfunction for the element-wise
    * greater or equal comparison of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto operator>=(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::greater_equal, E1, E2>
    {
        return detail::make_xfunction<std::greater_equal>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
     * @ingroup comparison_operators
     * @brief Equality
     *
     * Returns true if \a e1 and \a e2 have the same shape
     * and hold the same values. Unlike other comparison
     * operators, this does not return an \ref xfunction.
     * @param e1 an \ref xexpression or a scalar
     * @param e2 an \ref xexpression or a scalar
     * @return a boolean
     */
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

    /**
    * @ingroup comparison_operators
    * @brief Inequality
    *
    * Returns true if \a e1 and \a e2 have different shapes
    * or hold the different values. Unlike other comparison
    * operators, this does not return an \ref xfunction.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return a boolean
    */
    template <class E1, class E2>
    inline bool operator!=(const xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        return !(e1 == e2);
    }

    /**
    * @ingroup comparison_operators
    * @brief Element-wise equality
    *
    * Returns an \ref xfunction for the element-wise
    * equality of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto equal(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::equal_to, E1, E2>
    {
        return detail::make_xfunction<std::equal_to>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup comparison_operators
    * @brief Element-wise inequality
    *
    * Returns an \ref xfunction for the element-wise
    * inequality of \a e1 and \a e2.
    * @param e1 an \ref xexpression or a scalar
    * @param e2 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2>
    inline auto not_equal(E1&& e1, E2&& e2) noexcept
        -> detail::get_xfunction_type<std::not_equal_to, E1, E2>
    {
        return detail::make_xfunction<std::not_equal_to>(std::forward<E1>(e1), std::forward<E2>(e2));
    }

    /**
    * @ingroup logical_operators
    * @brief Ternary selection
    *
    * Returns an \ref xfunction for the element-wise
    * ternary selection (i.e. operator ? :) of \a e1,
    * \a e2 and \a e3.
    * @param e1 a boolean \ref xexpression
    * @param e2 an \ref xexpression or a scalar
    * @param e3 an \ref xexpression or a scalar
    * @return an \ref xfunction
    */
    template <class E1, class E2, class E3>
    inline auto where(E1&& e1, E2&& e2, E3&& e3) noexcept
        -> detail::get_xfunction_type<conditional_ternary, E1, E2, E3>
    {
         return detail::make_xfunction<conditional_ternary>(std::forward<E1>(e1), std::forward<E2>(e2), std::forward<E3>(e3));
    }

    /**
    * @ingroup logical_operators
    * @brief Any
    *
    * Returns true if any of the values of \a e is truthy,
    * false otherwise.
    * @param e an \ref xexpression
    * @return a boolean
    */
    template <class E>
    inline bool any(E&& e)
    {
        return std::any_of(e.storage_begin(), e.storage_end(), 
                           [](const typename std::decay<E>::type::value_type& el) { return el; });
    }

    /**
    * @ingroup logical_operators
    * @brief Any
    *
    * Returns true if all of the values of \a e are truthy,
    * false otherwise.
    * @param e an \ref xexpression
    * @return a boolean
    */
    template <class E>
    inline bool all(E&& e)
    {
        return std::all_of(e.storage_begin(), e.storage_end(),
                           [](const typename std::decay<E>::type::value_type& el) { return el; });
    }
}

#endif

