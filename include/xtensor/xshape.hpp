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

#include "xexception.hpp"
#include "xstorage.hpp"

namespace xt
{
    template <class T>
    using dynamic_shape = svector<T, 4>;

    template <class T, std::size_t N>
    using static_shape = std::array<T, N>;

    template <std::size_t... X>
    class fixed_shape;

    using xindex = dynamic_shape<std::size_t>;
}

namespace xtl
{
    namespace detail
    {
        template <std::size_t... I>
        struct sequence_builder<xt::fixed_shape<I...>>
        {
            using sequence_type = xt::fixed_shape<I...>;
            using value_type = typename sequence_type::value_type;

            inline static sequence_type make(std::size_t /*size*/)
            {
                return sequence_type{};
            }

            inline static sequence_type make(std::size_t /*size*/, value_type /*v*/)
            {
                return sequence_type{};
            }
        };
    }
}

namespace xt
{

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

        // Broadcasting for fixed shapes
        template <std::size_t IDX, std::size_t... X>
        struct at
        {
            constexpr static std::size_t arr[sizeof...(X)] = {X...};
            constexpr static std::size_t value = (IDX < sizeof...(X)) ? arr[IDX] : 0;
        };

        template <class S1, class S2>
        struct broadcast_fixed_shape;

        template <class IX, class A, class B>
        struct broadcast_fixed_shape_impl;

        template <std::size_t IX, class A, class B>
        struct broadcast_fixed_shape_cmp_impl;

        template <std::size_t IX, std::size_t... I, std::size_t... J>
        struct broadcast_fixed_shape_cmp_impl<IX, fixed_shape<I...>, fixed_shape<J...>>
        {
            static constexpr std::size_t index_diff = sizeof...(I) - sizeof...(J);
            static constexpr std::size_t value = detail::at<IX, I...>::value == 1 && (IX - index_diff) < sizeof...(J) ? 
                                                    detail::at<IX - index_diff, J...>::value :
                                                    detail::at<IX, I...>::value;
        };

        template <std::size_t... IX, std::size_t... I, std::size_t... J>
        struct broadcast_fixed_shape_impl<std::index_sequence<IX...>, fixed_shape<I...>, fixed_shape<J...>>
        {
            using type = xt::fixed_shape<broadcast_fixed_shape_cmp_impl<IX, fixed_shape<I...>, fixed_shape<J...>>::value...>;
        };

        template <std::size_t... I, std::size_t... J>
        struct broadcast_fixed_shape<fixed_shape<I...>, fixed_shape<J...>>
        {
            using type = std::conditional_t<(sizeof...(I) > sizeof...(J)),
                    typename broadcast_fixed_shape_impl<decltype(std::make_index_sequence<sizeof...(I)>()), fixed_shape<I...>, fixed_shape<J...>>::type,
                    typename broadcast_fixed_shape_impl<decltype(std::make_index_sequence<sizeof...(J)>()), fixed_shape<J...>, fixed_shape<I...>>::type
                >;
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

        template <class S>
        struct is_fixed
        {
            static constexpr bool value = false;
        };

        template <std::size_t... N>
        struct is_fixed<fixed_shape<N...>>
        {
            static constexpr bool value = true;
        };

        template <class... S>
        using only_array = xtl::conjunction<xtl::disjunction<is_array<S>, is_fixed<S>>...>;

        template <class... S>
        using only_fixed = xtl::conjunction<is_fixed<S>...>;

        // The promote_index meta-function returns std::vector<promoted_value_type> in the
        // general case and an array of the promoted value type and maximal size if all
        // arguments are of type std::array

        template <class... S>
        struct promote_array
        {
            using type = std::array<typename std::common_type<typename S::value_type...>::type, max_array_size<S...>::value>;
        };

        template <>
        struct promote_array<>
        {
            using type = std::array<std::size_t, 0>;
        };

        template <class... S>
        struct broadcast_fixed;

        template <class S1, class S2>
        struct broadcast_fixed<S1, S2>
        {
            using type = typename broadcast_fixed_shape<S1, S2>::type;
        };

        template <class S1, class... S>
        struct broadcast_fixed<S1, S...>
        {
            using type = typename broadcast_fixed_shape<S1, typename broadcast_fixed<S...>::type>::type;
        };

        template <bool all_index, bool all_array, class... S>
        struct select_promote_index;

        template <class... S>
        struct select_promote_index<true, true, S...>
        {
            using type = typename broadcast_fixed<S...>::type;
        };

        template <class... S>
        struct select_promote_index<false, true, S...>
        {
            using type = typename promote_array<S...>::type;
        };

        template <class... S>
        struct select_promote_index<false, false, S...>
        {
            using type = dynamic_shape<typename std::common_type<typename S::value_type...>::type>;
        };

        template <class... S>
        struct promote_index
        {
            using type = typename select_promote_index<only_fixed<S...>::value,
                                                       only_array<S...>::value,
                                                       S...>::type;
        };

        template <class T>
        struct index_from_shape_impl
        {
            using type = T;
        };

        template <std::size_t... N>
        struct index_from_shape_impl<fixed_shape<N...>>
        {
            using type = std::array<std::size_t, sizeof...(N)>;
        };
    }

    template <class... S>
    using promote_shape_t = typename detail::promote_index<S...>::type;

    template <class... S>
    using promote_strides_t = typename detail::promote_index<S...>::type;

    template <class S>
    using index_from_shape_t = typename detail::index_from_shape_impl<S>::type;
}

#endif
