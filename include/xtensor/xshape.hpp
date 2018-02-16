/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_XSHAPE_HPP
#define XTENSOR_XSHAPE_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>

#include "xstorage.hpp"
#include "xexception.hpp"

namespace xt
{
    template <class T>
    using dynamic_shape = svector<T, 4>;

    template <class T, std::size_t N>
    using static_shape = std::array<T, N>;

    template <std::size_t... X>
    class fixed_shape;

    using xindex = dynamic_shape<std::size_t>;

    /*************************************
     * promote_shape and promote_strides *
     *************************************/

    namespace detail
    {
        template <class T1, class T2>
        constexpr std::common_type_t<T1, T2> imax(const T1& a, const T2& b)
        {
            return a > b ? a : b;
        }

        // Variadic meta-function returning the maximal size of std::arrays.
        template <class... T>
        struct max_array_size;

        template <>
        struct max_array_size<>
        {
            static constexpr std::size_t value = 0;
        };

        template <class T, class... Ts>
        struct max_array_size<T, Ts...> : std::integral_constant<std::size_t, imax(std::tuple_size<T>::value, max_array_size<Ts...>::value)>
        {
        };

        // Simple is_array and only_array meta-functions
        template <class S>
        struct is_array
        {
            static constexpr bool value = false;
        };

        template <class T, std::size_t N>
        struct is_array<std::array<T, N>>
        {
            static constexpr bool value = true;
        };

        template <class... S>
        using only_array = xtl::conjunction<is_array<S>...>;

        template <class... S>
        struct pack_first
        {
            using type = void;
        };

        template <class S1, class... S>
        struct pack_first<S1, S...>
        {
            using type = S1;
        };

        template <class... S>
        using pack_first_t = typename pack_first<S...>::type;

        template <class... S>
        using all_same = xtl::conjunction<std::is_same<pack_first_t<S...>, S>...>;

        // The promote_index meta-function returns std::vector<promoted_value_type> in the
        // general case and an array of the promoted value type and maximal size if all
        // arguments are of type std::array
        // also, if all shape types are the same, it returns this kind of type. This allows 
        // for fixed shape types to be promoted to (e.g. fixed_shape<3, 5, 2>)
        template <bool A, bool B, class... S>
        struct promote_index_impl;

        template <class... S>
        struct promote_index_impl<false, false, S...>
        {
            using type = xt::dynamic_shape<typename std::common_type<typename S::value_type...>::type>;
        };

        template <class... S>
        struct promote_index_impl<false, true, S...>
        {
            using type = std::array<typename std::common_type<typename S::value_type...>::type, max_array_size<S...>::value>;
        };


        template <class... S>
        struct promote_index_impl<true, true, S...>
        {
            using type = pack_first_t<S...>;
        };

        template <class... S>
        struct promote_index_impl<true, false, S...>
        {
            using type = pack_first_t<S...>;
        };

        template <class... S>
        struct promote_index
        {
            using type = std::array<std::size_t, 0>;
        };

        template <class S1, class... S>
        struct promote_index<S1, S...>
        {
            using type = typename promote_index_impl<all_same<S1, S...>::value, only_array<S1, S...>::value, S1, S...>::type; 
        };
    }

    template <class... S>
    using promote_shape_t = typename detail::promote_index<S...>::type;

    template <class... S>
    using promote_strides_t = typename detail::promote_index<S...>::type;

    /***********************************
     * equal dimensions implementation *
     ***********************************/

    namespace detail
    {
        template <std::size_t I, std::size_t... J>
        struct all_equal_int
        {
            constexpr static bool value = sizeof...(J) == 0 || xtl::conjunction<std::integral_constant<bool, (I == J)>...>::value;
        };

        template <class V>
        struct container_static_size
        {
            constexpr static std::size_t value = 0;
        };

        template <class T, std::size_t N>
        struct container_static_size<std::array<T, N>>
        {
            constexpr static std::size_t value = N;
        };

        template <class... Args>
        struct equal_dimensions
        {
            constexpr static bool value = all_equal_int<container_static_size<Args>::value...>::value && xtl::conjunction<is_array<Args>...>::value;
        };

        template<>
        struct equal_dimensions<>
        {
            constexpr static bool value = true;
        };
    }
}

#endif
