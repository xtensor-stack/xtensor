/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XEVAL_HPP
#define XEVAL_HPP

#include "xarray.hpp"
#include "xtensor.hpp"

namespace xt
{
    namespace detail
    {
        template <class T>
        using is_container = std::is_base_of<xcontainer<std::remove_const_t<T>>, T>;

        template <class T, class E = void>
        struct has_container_base_impl
        {
            static constexpr bool value = false;
        };

        template <class T>
        struct has_container_base_impl<T, std::enable_if_t<is_container<std::decay_t<T>>::value>>
        {
            static constexpr bool value = true;
        };

        template<class CT, class... S>
        struct has_container_base_impl<xview<CT, S...>, std::enable_if_t<is_container<std::decay_t<CT>>::value>>
        {
            static constexpr bool value = true;
        };

        template <class T>
        using has_container_base = has_container_base_impl<T>;
    }

    /**
     * Force evaluation of xexpression.
     * @return xarray or xtensor depending on shape type
     * 
     * \code{.cpp}
     * xarray<double> a = {1,2,3,4};
     * auto&& b = xt::eval(a); // b is a reference to a, no copy!
     * auto&& c = xt::eval(a + b); // c is xarray<double>, not an xexpression
     * \endcode
     */
    template <class T>
    inline auto eval(T&& t)
        -> std::enable_if_t<detail::has_container_base<T>::value, T&&>
    {
        return t;
    }

    /// @cond DOXYGEN_INCLUDE_SFINAE
    template <class T, class I = std::decay_t<T>>
    inline auto eval(T&& t)
        -> std::enable_if_t<!detail::has_container_base<T>::value && detail::is_array<typename I::shape_type>::value, xtensor<typename I::value_type, std::tuple_size<typename I::shape_type>::value>>
    {
        return xtensor<typename I::value_type, std::tuple_size<typename I::shape_type>::value>(std::forward<T>(t));
    }

    template <class T, class I = std::decay_t<T>>
    inline auto eval(T&& t)
        -> std::enable_if_t<!detail::has_container_base<T>::value && !detail::is_array<typename I::shape_type>::value, xt::xarray<typename I::value_type>>
    {
        return xarray<typename I::value_type>(std::forward<T>(t));
    }
    /// @endcond
}

#endif