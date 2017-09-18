/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XOPTIONAL_HPP
#define XOPTIONAL_HPP

#include <type_traits>
#include <utility>

#include "xtl/xoptional.hpp"
#include "xtl/xoptional_sequence.hpp"

#include "xfunctorview.hpp"

namespace xt
{

    using xtl::xoptional;
    using xtl::missing;
    using xtl::disable_xoptional;
    using xtl::enable_xoptional;

    namespace detail
    {
        template <class CT, class CB>
        struct functor_return_type<xoptional<CT, CB>, bool>
        {
            using type = ::xtl::xoptional<bool>;
            using simd_type = ::xtl::xoptional<bool>;
        };
    }

    namespace math
    {
        template <class T, class B>
        struct sign_fun<xoptional<T, B>>
        {
            using argument_type = xoptional<T, B>;
            using result_type = xoptional<std::decay_t<T>>;

            constexpr result_type operator()(const xoptional<T, B>& x) const
            {
                return x.has_value() ? xoptional<T>(detail::sign_impl(x.value()))
                                     : missing<std::decay_t<T>>();
            }
        };
    }

    template <class T, class B>
    inline auto sign(const xoptional<T, B>& e)
    {
        return e.has_value() ? math::detail::sign_impl(e.value()) : missing<std::decay_t<T>>();
    }

    /*******************************************************
     * value() and has_value() xfunctorview implementation *
     *******************************************************/

    namespace detail
    {
        template <class E>
        struct value_forwarder
        {
            // internal types
            using xexpression_type = std::decay_t<E>;
            using optional_type = typename xexpression_type::value_type;
            using optional_reference = typename xexpression_type::reference;
            using optional_const_reference = typename xexpression_type::const_reference;

            // types
            using value_type = typename optional_type::value_type;
            using reference = typename optional_reference::value_closure;
            using const_reference = typename optional_const_reference::value_closure;
            using pointer = value_type*;
            using const_pointer = const value_type*;

            template <class T>
            decltype(auto) operator()(T&& t) const
            {
                return std::forward<T>(t).value();
            }
        };

        template <class E>
        struct flag_forwarder
        {
            // internal types
            using xexpression_type = std::decay_t<E>;
            using optional_type = typename xexpression_type::value_type;
            using optional_reference = typename xexpression_type::reference;
            using optional_const_reference = typename xexpression_type::const_reference;

            // types
            using value_type = typename optional_type::flag_type;
            using reference = typename optional_reference::flag_closure;
            using const_reference = typename optional_const_reference::flag_closure;
            using pointer = value_type*;
            using const_pointer = const value_type*;

            template <class T>
            decltype(auto) operator()(T&& t) const
            {
                return std::forward<T>(t).has_value();
            }
        };
    }

    template <class E>
    auto value(E&& e)
        -> disable_xoptional<typename std::decay_t<E>::value_type, E>
    {
        return std::forward<E>(e);
    }

    template <class E>
    auto has_value(E&& e)
        -> disable_xoptional<typename std::decay_t<E>::value_type, decltype(ones<bool>(std::forward<E>(e).shape()))>
    {
        return ones<bool>(std::forward<E>(e).shape());
    }

    template <class E>
    auto value(E&& e)
        -> enable_xoptional<typename std::decay_t<E>::value_type, xfunctorview<detail::value_forwarder<E>, xclosure_t<E>>>
    {
        using type = xfunctorview<detail::value_forwarder<E>, xclosure_t<E>>;
        return type(std::forward<E>(e));
    }

    template <class E>
    auto has_value(E&& e)
        -> enable_xoptional<typename std::decay_t<E>::value_type, xfunctorview<detail::flag_forwarder<E>, xclosure_t<E>>>
    {
        using type = xfunctorview<detail::flag_forwarder<E>, xclosure_t<E>>;
        return type(std::forward<E>(e));
    }
}

#endif
