/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XUTILS_HPP
#define XUTILS_HPP

#include <cstddef>
#include <array>
#include <utility>
#include <tuple>
#include <type_traits>
#include <initializer_list>
#include <algorithm>
#include "xtensor_config.hpp"

namespace xt
{

    template <class T>
    struct remove_class;

    template <class F, class... T>
    void for_each(F&& f, std::tuple<T...>& t) noexcept(noexcept(std::declval<F>()));

    template <class F, class R, class... T>
    R accumulate(F&& f, R init, const std::tuple<T...>& t) noexcept(noexcept(std::declval<F>()));

    template <class... T>
    struct or_;

    template <class... T>
    struct and_;

    template <std::size_t I, class... Args>
    constexpr decltype(auto) argument(Args&&... args) noexcept;

    template <class R, class F, class... S>
    R apply(std::size_t index, F&& func, const std::tuple<S...>& s) noexcept(noexcept(std::declval<F>()));

    template <class U>
    struct initializer_dimension;

    /*******************************
     * remove_class implementation *
     *******************************/

    template <class T>
    struct remove_class
    {
    };

    template <class C, class R, class... Args>
    struct remove_class<R (C::*) (Args...)>
    {
        typedef R type(Args...);
    };

    template <class C, class R, class... Args>
    struct remove_class<R (C::*) (Args...) const>
    {
        typedef R type(Args...);
    };

    template <class T>
    using remove_class_t = typename remove_class<T>::type;

    /***************************
     * for_each implementation *
     ***************************/

    namespace detail
    {
        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I == sizeof...(T), void>::type
        for_each_impl(F&& /*f*/, std::tuple<T...>& /*t*/) noexcept(noexcept(std::declval<F>()))
        {
        }

        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I < sizeof...(T), void>::type
        for_each_impl(F&& f, std::tuple<T...>& t) noexcept(noexcept(std::declval<F>()))
        {
            f(std::get<I>(t));
            for_each_impl<I + 1, F, T...>(std::forward<F>(f), t);
        }
    }

    template <class F, class... T>
    inline void for_each(F&& f, std::tuple<T...>& t) noexcept(noexcept(std::declval<F>()))
    {
        detail::for_each_impl<0, F, T...>(std::forward<F>(f), t);
    }

    /*****************************
     * accumulate implementation *
     *****************************/

    namespace detail
    {
        template <std::size_t I, class F, class R, class... T>
        inline std::enable_if_t<I == sizeof...(T), R>
        accumulate_impl(F&& /*f*/, R init, const std::tuple<T...>& /*t*/) noexcept(noexcept(std::declval<F>()))
        {
            return init;
        }

        template <std::size_t I, class F, class R, class... T>
        inline std::enable_if_t<I < sizeof...(T), R>
        accumulate_impl(F&& f, R init, const std::tuple<T...>& t) noexcept(noexcept(std::declval<F>()))
        {
            R res = f(init, std::get<I>(t));
            return accumulate_impl<I + 1, F, R, T...>(std::forward<F>(f), res, t);
        }
    }

    template <class F, class R, class... T>
    inline R accumulate(F&& f, R init, const std::tuple<T...>& t) noexcept(noexcept(std::declval<F>()))
    {
        return detail::accumulate_impl<0, F, R, T...>(f, init, t);
    }

    /**********************
     * or_ implementation *
     **********************/

    template <>
    struct or_<> : std::integral_constant<bool, false>
    {
    };

    template <class T, class... Ts>
    struct or_<T, Ts...>
        : std::integral_constant<bool, T::value || or_<Ts...>::value>
    {
    };

    /**********************
     * and_ implementation *
     **********************/

    template <>
    struct and_<> : std::integral_constant<bool, true>
    {
    };

    template <class T, class... Ts>
    struct and_<T, Ts...>
        : std::integral_constant<bool, T::value && and_<Ts...>::value>
    {
    };

    /***************************
     * argument implementation *
     ***************************/

    namespace detail
    {
        template <std::size_t I>
        struct getter
        {
            template <class Arg, class... Args>
            static constexpr decltype(auto) get(Arg&& /*arg*/, Args&&... args) noexcept
            {
                return getter<I - 1>::get(std::forward<Args>(args)...);
            }
        };

        template <>
        struct getter<0>
        {
            template <class Arg, class... Args>
            static constexpr Arg&& get(Arg&& arg, Args&&... /*args*/) noexcept
            {
                return std::forward<Arg>(arg);
            }
        };
    }

    template <std::size_t I, class... Args>
    constexpr decltype(auto) argument(Args&&... args) noexcept
    {
        static_assert(I < sizeof...(Args), "I should be lesser than sizeof...(Args)");
        return detail::getter<I>::get(std::forward<Args>(args)...);
    }

    /************************
     * apply implementation *
     ************************/

    namespace detail
    {
        template <class R, class F, std::size_t I, class... S>
        R apply_one(F&& func, const std::tuple<S...>& s) noexcept(noexcept(std::declval<F>()))
        {
            return func(std::get<I>(s));
        }

        template <class R, class F, std::size_t... I, class... S>
        R apply(std::size_t index, F&& func, std::index_sequence<I...> /*seq*/, const std::tuple<S...>& s) noexcept(noexcept(std::declval<F>()))
        {
            using FT = std::add_pointer_t<R(F&&, const std::tuple<S...>&)>;
            static const std::array<FT, sizeof...(I)> ar = { &apply_one<R, F, I, S...>... };
            return ar[index](std::forward<F>(func), s);
        }
    }

    template <class R, class F, class... S>
    inline R apply(std::size_t index, F&& func, const std::tuple<S...>& s) noexcept(noexcept(std::declval<F>()))
    {
        return detail::apply<R>(index, std::forward<F>(func), std::make_index_sequence<sizeof...(S)>(), s);
    }

    /******************************
     * nested_copy implementation *
     ******************************/

    template <class T, class S>
    inline void nested_copy(T&& iter, const S& s)
    {
        *iter++ = s;
    }

    template <class T, class S>
    inline void nested_copy(T&& iter, std::initializer_list<S> s)
    {
        for (auto it = s.begin(); it != s.end(); ++it)
        {
            nested_copy(std::forward<T>(iter), *it);
        }
    }

    /****************************************
     * initializer_dimension implementation *
     ****************************************/

    namespace detail
    {
        template <class U>
        struct initializer_depth_impl
        {
            static constexpr std::size_t value = 0;
        };

        template <class T>
        struct initializer_depth_impl<std::initializer_list<T>>
        {
            static constexpr std::size_t value = 1 + initializer_depth_impl<T>::value;
        };
    }

    template <class U>
    struct initializer_dimension
    {
        static constexpr std::size_t value = detail::initializer_depth_impl<U>::value;
    };

    /************************************
     * initializer_shape implementation *
     ************************************/

    namespace detail
    {
        template <std::size_t I>
        struct initializer_shape_impl
        {
            template <class T>
            static constexpr std::size_t value(T t)
            {
                return t.size() == 0 ? 0 : initializer_shape_impl<I - 1>::value(*t.begin());
            }
        };

        template <>
        struct initializer_shape_impl<0>
        {
            template <class T>
            static constexpr std::size_t value(T t)
            {
                return t.size();
            }
        };

        template <class R, class U, std::size_t... I>
        constexpr R initializer_shape(U t, std::index_sequence<I...>)
        {
             return { initializer_shape_impl<I>::value(t)... };
        }
    }

    template <class R, class T>
    constexpr R shape(T t)
    {
        return detail::initializer_shape<R, decltype(t)>(t, std::make_index_sequence<initializer_dimension<decltype(t)>::value>());
    }

    /******************************
     * check_shape implementation *
     ******************************/

    namespace detail
    {
        template <class T, class S>
        struct predshape
        {
            constexpr predshape(S first, S last): m_first(first), m_last(last)
            {}

            constexpr bool operator()(const T&) const
            {
                return m_first == m_last;
            }

            S m_first;
            S m_last;
        };

        template <class T, class S>
        struct predshape<std::initializer_list<T>, S>
        {
            constexpr predshape(S first, S last): m_first(first), m_last(last)
            {}

            constexpr bool operator()(std::initializer_list<T> t) const
            {
                return *m_first == t.size() && std::all_of(t.begin(), t.end(), predshape<T, S>(m_first + 1, m_last));
            }

            S m_first;
            S m_last;
        };
    }

    template <class T, class S>
    constexpr bool check_shape(T t, S first, S last)
    {
        return detail::predshape<decltype(t), S>(first, last)(t);
    }

    /***********************************
     * resize_container implementation *
     ***********************************/

    template <class C>
    inline bool resize_container(C& c, typename C::size_type size)
    {
        c.resize(size);
        return true;
    }

    template <class T, std::size_t N>
    inline bool resize_container(std::array<T, N>& /*a*/, typename std::array<T, N>::size_type size)
    {
        return size == N;
    }

    /********************************
     * make_sequence implementation *
     ********************************/

    namespace detail
    {
        template <class S>
        struct sequence_builder
        {
            using value_type = typename S::value_type;
            using size_type = typename S::size_type;

            inline static S make(size_type size, value_type v)
            {
                return S(size, v);
            }
        };

        template <class T, std::size_t N>
        struct sequence_builder<std::array<T, N>>
        {
            using sequence_type = std::array<T, N>;
            using value_type = typename sequence_type::value_type;
            using size_type = typename sequence_type::size_type;

            inline static sequence_type make(size_type /*size*/, value_type v)
            {
                sequence_type s;
                s.fill(v);
                return s;
            }
        };
    }

    template <class S>
    inline S make_sequence(typename S::size_type size, typename S::value_type v)
    {
        return detail::sequence_builder<S>::make(size, v);
    }

    /*************************************
     * promote_shape and promote_strides *
     *************************************/

    namespace detail
    {
        template<class T1, class T2>
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
        using only_array = and_<is_array<S>...>;

        // The promote_index meta-function returns std::vector<promoted_value_type> in the
        // general case and an array of the promoted value type and maximal size if all
        // arguments are of type std::array

        template <bool A, class... S>
        struct promote_index_impl;

        template <class... S>
        struct promote_index_impl<false, S...>
        {
            using type = std::vector<typename std::common_type<typename S::value_type...>::type>;
        };

        template <class... S>
        struct promote_index_impl<true, S...>
        {
            using type = std::array<typename std::common_type<typename S::value_type...>::type, max_array_size<S...>::value>;
        };

        template <>
        struct promote_index_impl<true>
        {
            using type = std::array<std::size_t, 0>;
        };

        template <class... S>
        struct promote_index
        {
            using type = typename promote_index_impl<only_array<S...>::value, S...>::type;
        };
    }

    template <class... S>
    using promote_shape_t = typename detail::promote_index<S...>::type;

    template <class... S>
    using promote_strides_t = typename detail::promote_index<S...>::type;
}

#endif

