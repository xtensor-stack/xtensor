/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_UTILS_HPP
#define XTENSOR_UTILS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "xtl/xtype_traits.hpp"

#include "xtensor_config.hpp"

namespace xt
{
    /****************
     * declarations *
     ****************/

    template <class T>
    struct remove_class;

    template <class F, class... T>
    void for_each(F&& f, std::tuple<T...>& t) noexcept(noexcept(std::declval<F>()));

    template <class F, class R, class... T>
    R accumulate(F&& f, R init, const std::tuple<T...>& t) noexcept(noexcept(std::declval<F>()));

    template <std::size_t I, class... Args>
    constexpr decltype(auto) argument(Args&&... args) noexcept;

    template <class R, class F, class... S>
    R apply(std::size_t index, F&& func, const std::tuple<S...>& s) noexcept(noexcept(std::declval<F>()));

    template <class T, class S>
    void nested_copy(T&& iter, const S& s);

    template <class T, class S>
    void nested_copy(T&& iter, std::initializer_list<S> s);

    template <class U>
    struct initializer_dimension;

    template <class R, class T>
    constexpr R shape(T t);

    template <class T, class S>
    constexpr bool check_shape(T t, S first, S last);

    template <class C>
    bool resize_container(C& c, typename C::size_type size);

    template <class T, std::size_t N>
    bool resize_container(std::array<T, N>& a, typename std::array<T, N>::size_type size);

    // gcc 4.9 is affected by C++14 defect CGW 1558
    // see http://open-std.org/JTC1/SC22/WG21/docs/cwg_defects.html#1558
    template <class... T>
    struct make_void
    {
        using type = void;
    };

    template <class... T>
    using void_t = typename make_void<T...>::type;

    /*******************************
     * remove_class implementation *
     *******************************/

    template <class T>
    struct remove_class
    {
    };

    template <class C, class R, class... Args>
    struct remove_class<R (C::*)(Args...)>
    {
        typedef R type(Args...);
    };

    template <class C, class R, class... Args>
    struct remove_class<R (C::*)(Args...) const>
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
            return static_cast<R>(func(std::get<I>(s)));
        }

        template <class R, class F, std::size_t... I, class... S>
        R apply(std::size_t index, F&& func, std::index_sequence<I...> /*seq*/, const std::tuple<S...>& s) noexcept(noexcept(std::declval<F>()))
        {
            using FT = std::add_pointer_t<R(F&&, const std::tuple<S...>&)>;
            static const std::array<FT, sizeof...(I)> ar = {{&apply_one<R, F, I, S...>...}};
            return ar[index](std::forward<F>(func), s);
        }
    }

    template <class R, class F, class... S>
    inline R apply(std::size_t index, F&& func, const std::tuple<S...>& s) noexcept(noexcept(std::declval<F>()))
    {
        return detail::apply<R>(index, std::forward<F>(func), std::make_index_sequence<sizeof...(S)>(), s);
    }

    /***************************
     * nested_initializer_list *
     ***************************/

    template <class T, std::size_t I>
    struct nested_initializer_list
    {
        using type = std::initializer_list<typename nested_initializer_list<T, I - 1>::type>;
    };

    template <class T>
    struct nested_initializer_list<T, 0>
    {
        using type = T;
    };

    template <class T, std::size_t I>
    using nested_initializer_list_t = typename nested_initializer_list<T, I>::type;

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
            using size_type = typename R::value_type;
            return {size_type(initializer_shape_impl<I>::value(t))...};
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
            constexpr predshape(S first, S last)
                : m_first(first), m_last(last)
            {
            }

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
            constexpr predshape(S first, S last)
                : m_first(first), m_last(last)
            {
            }

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

    /***************************
     * apply_cv implementation *
     ***************************/

    namespace detail
    {
        template <class T, class U, bool = std::is_const<std::remove_reference_t<T>>::value,
                  bool = std::is_volatile<std::remove_reference_t<T>>::value>
        struct apply_cv_impl
        {
            using type = U;
        };

        template <class T, class U>
        struct apply_cv_impl<T, U, true, false>
        {
            using type = const U;
        };

        template <class T, class U>
        struct apply_cv_impl<T, U, false, true>
        {
            using type = volatile U;
        };

        template <class T, class U>
        struct apply_cv_impl<T, U, true, true>
        {
            using type = const volatile U;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, false, false>
        {
            using type = U&;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, true, false>
        {
            using type = const U&;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, false, true>
        {
            using type = volatile U&;
        };

        template <class T, class U>
        struct apply_cv_impl<T&, U, true, true>
        {
            using type = const volatile U&;
        };
    }

    template <class T, class U>
    struct apply_cv
    {
        using type = typename detail::apply_cv_impl<T, U>::type;
    };

    template <class T, class U>
    using apply_cv_t = typename apply_cv<T, U>::type;


    /**************************
    * to_array implementation *
    ***************************/

    namespace detail
    {
        template <class T, std::size_t N, std::size_t... I>
        constexpr std::array<std::remove_cv_t<T>, N> to_array_impl(T (&a)[N], std::index_sequence<I...>)
        {
            return {{a[I]...}};
        }
    }

    template <class T, std::size_t N>
    constexpr std::array<std::remove_cv_t<T>, N> to_array(T (&a)[N])
    {
        return detail::to_array_impl(a, std::make_index_sequence<N>{});
    }

    /********************************
     * sequence_size implementation *
     ********************************/

    // equivalent to std::size(c) in c++17
    template <class C>
    constexpr auto sequence_size(const C& c) -> decltype(c.size())
    {
        return c.size();
    }

    // equivalent to std::size(a) in c++17
    template <class T, std::size_t N>
    constexpr std::size_t sequence_size(const T (&)[N])
    {
        return N;
    }

    /*****************************************
     * has_raw_data_interface implementation *
     *****************************************/

    template <class T>
    class has_raw_data_interface
    {
        template <class C>
        static std::true_type test(decltype(std::declval<C>().raw_data_offset()));

        template <class C>
        static std::false_type test(...);

    public:

        constexpr static bool value = decltype(test<T>(std::size_t(0)))::value == true;
    };

    /******************
     * enable_if_type *
     ******************/

    template <class T>
    struct enable_if_type
    {
        using type = void;
    };

    /*****************************
     * is_complete implemenation *
     *****************************/

    namespace detail
    {
        template <class T>
        class is_complete_impl
        {
            template <class U>
            static auto test(U*) -> std::integral_constant<bool, sizeof(U) == sizeof(U)>;

            static auto test(...) -> std::false_type;

        public:

            using type = decltype(test((T*)0));
        };
    }

    template <class T>
    struct is_complete : detail::is_complete_impl<T>::type
    {
    };

    /*************
     * static_if *
     *************/

    struct identity_functor
    {
        template <class T>
        T&& operator()(T&& x) const
        {
            return std::forward<T>(x);
        }
    };

    template <class TF, class FF>
    auto static_if(std::true_type, const TF& tf, const FF&)
    {
        return tf(identity_functor());
    }

    template <class TF, class FF>
    auto static_if(std::false_type, const TF&, const FF& ff)
    {
        return ff(identity_functor());
    }

    template <bool cond, class TF, class FF>
    auto static_if(const TF& tf, const FF& ff)
    {
        return static_if(std::integral_constant<bool, cond>(), tf, ff);
    }

    /********************************************
     * xtrivial_default_construct implemenation *
     ********************************************/

    #if !defined(__GNUG__) || defined(_LIBCPP_VERSION) || defined(_GLIBCXX_USE_CXX11_ABI)

    template <class T>
    using xtrivially_default_constructible = std::is_trivially_default_constructible<T>;

    #else

    template <class T>
    using xtrivially_default_constructible = std::has_trivial_default_constructor<T>;

    #endif

    /*************************
     * conditional type cast *
     *************************/

    template <bool condition, class T>
    struct conditional_cast_functor;

    template <class T>
    struct conditional_cast_functor<false, T>
    : public identity_functor
    {
    };

    template <class T>
    struct conditional_cast_functor<true, T>
    {
        template <class U>
        inline auto operator()(U&& u) const
        {
            return static_cast<T>(std::forward<U>(u));
        }
    };

    /**
     * @brief Perform a type cast when a condition is true.
     * If <tt>condition</tt> is true, return <tt>static_cast<T>(u)</tt>,
     * otherwise return <tt>u</tt> unchanged. This is useful when an unconditional
     * static_cast would force undesired type conversions in some situations where
     * an error or warning would be desired. The condition determines when the
     * explicit cast is ok.
     */
    template <bool condition, class T, class U>
    inline auto conditional_cast(U&& u)
    {
        return conditional_cast_functor<condition, T>()(std::forward<U>(u));
    };

    /************************************
     * arithmetic type promotion traits *
     ************************************/

    /**
     * @brief Traits class for the result type of mixed arithmetic expressions.
     * For example, <tt>promote_type<unsigned char, unsigned char>::type</tt> tells
     * the user that <tt>unsigned char + unsigned char => int</tt>.
     */
    template <class... T>
    struct promote_type;

    template <class T>
    struct promote_type<T>
    {
        using type = typename promote_type<T, T>::type;
    };

    template <class T0, class T1>
    struct promote_type<T0, T1>
    {
        using type = decltype(*(std::decay_t<T0>*)0 + *(std::decay_t<T1>*)0);
    };

    template <class T0, class... REST>
    struct promote_type<T0, REST...>
    {
        using type = decltype(*(std::decay_t<T0>*)0 + *(typename promote_type<REST...>::type*)0);
    };

    template <>
    struct promote_type<bool>
    {
        using type = bool;
    };

    template <class T>
    struct promote_type<bool, T>
    {
        using type = T;
    };

    template <class... REST>
    struct promote_type<bool, REST...>
    {
        using type = typename promote_type<bool, typename promote_type<REST...>::type>::type;
    };

    /** 
     * @brief Abbreviation of 'typename promote_type<T>::type'.
     */
    template <class... T>
    using promote_type_t = typename promote_type<T...>::type;

    /**
     * @brief Traits class to find the biggest type of the same kind.
     *
     * For example, <tt>big_promote_type<unsigned char>::type</tt> is <tt>unsigned long long</tt>.
     * The default implementation only supports built-in types and <tt>std::complex</tt>. All
     * other types remain unchanged unless <tt>big_promote_type</tt> gets specialized for them.
     */
    template <class T>
    struct big_promote_type
    {
    private:
        
        using V = std::decay_t<T>;
        static constexpr bool is_arithmetic = std::is_arithmetic<V>::value;
        static constexpr bool is_signed = std::is_signed<V>::value;
        static constexpr bool is_integral = std::is_integral<V>::value;
        static constexpr bool is_long_double = std::is_same<V, long double>::value;

    public:
        using type = std::conditional_t<is_arithmetic,
                        std::conditional_t<is_integral,
                            std::conditional_t<is_signed, long long, unsigned long long>,
                            std::conditional_t<is_long_double, long double, double>
                        >,
                        V
                     >;
    };

    template <class T>
    struct big_promote_type<std::complex<T>>
    {
        using type = std::complex<typename big_promote_type<T>::type>;
    };

    /**
     * @brief Abbreviation of 'typename big_promote_type<T>::type'.
     */
    template <class T>
    using big_promote_type_t = typename big_promote_type<T>::type;

    namespace traits_detail
    {
        using std::sqrt;

        template <class T>
        using real_promote_type_t = decltype(sqrt(*(std::decay_t<T>*)0));
    }

    /**
     * @brief Result type of algebraic expressions.
     *
     * For example, <tt>real_promote_type<int>::type</tt> tells the
     * user that <tt>sqrt(int) => double</tt>.
     */
    template <class T>
    struct real_promote_type
    {
        using type = traits_detail::real_promote_type_t<T>;
    };

    /**
     * @brief Abbreviation of 'typename real_promote_type<T>::type'.
     */
    template <class T>
    using real_promote_type_t = typename real_promote_type<T>::type;

    /**
     * @brief Traits class to replace 'bool' with 'uint8_t' and keep everything else.
     *
     * This is useful for scientific computing, where a boolean mask array is
     * usually implemented as an array of bytes.
     */
    template <class T>
    struct bool_promote_type
    {
        using type = typename std::conditional<std::is_same<T, bool>::value, uint8_t, T>::type;
    };

    /**
     * @brief Abbreviation for typename bool_promote_type<T>::type
     */
    template <class T>
    using bool_promote_type_t = typename bool_promote_type<T>::type;

    /********************************************
     * type inference for norm and squared norm *
     ********************************************/

    template <class T>
    struct norm_type;

    template <class T>
    struct squared_norm_type;

    namespace traits_detail
    {

        template <class T, bool scalar = std::is_arithmetic<T>::value>
        struct norm_of_scalar_impl;

        template <class T>
        struct norm_of_scalar_impl<T, false>
        {
            static const bool value = false;
            using norm_type = void*;
            using squared_norm_type = void*;
        };

        template <class T>
        struct norm_of_scalar_impl<T, true>
        {
            static const bool value = true;
            using norm_type = promote_type_t<T>;
            using squared_norm_type = promote_type_t<T>;
        };

        template <class T, bool integral = std::is_integral<T>::value,
                  bool floating = std::is_floating_point<T>::value>
        struct norm_of_array_elements_impl;

        template <>
        struct norm_of_array_elements_impl<void*, false, false>
        {
            using norm_type = void*;
            using squared_norm_type = void*;
        };

        template <class T>
        struct norm_of_array_elements_impl<T, false, false>
        {
            using norm_type = typename norm_type<T>::type;
            using squared_norm_type = typename squared_norm_type<T>::type;
        };

        template <class T>
        struct norm_of_array_elements_impl<T, true, false>
        {
            static_assert(!std::is_same<T, char>::value,
                          "'char' is not a numeric type, use 'signed char' or 'unsigned char'.");

            using norm_type = double;
            using squared_norm_type = uint64_t;
        };

        template <class T>
        struct norm_of_array_elements_impl<T, false, true>
        {
            using norm_type = double;
            using squared_norm_type = double;
        };

        template <>
        struct norm_of_array_elements_impl<long double, false, true>
        {
            using norm_type = long double;
            using squared_norm_type = long double;
        };

        template <class ARRAY>
        struct norm_of_vector_impl
        {
            static void* test(...);

            template <class U>
            static typename U::value_type test(U*, typename U::value_type* = 0);

            using T = decltype(test((ARRAY*)0));

            static const bool value = !std::is_same<T, void*>::value;

            using norm_type = typename norm_of_array_elements_impl<T>::norm_type;
            using squared_norm_type = typename norm_of_array_elements_impl<T>::squared_norm_type;
        };

        template <class U>
        struct norm_type_base
        {
            using T = std::decay_t<U>;

            static_assert(!std::is_same<T, char>::value,
                          "'char' is not a numeric type, use 'signed char' or 'unsigned char'.");

            using norm_of_scalar = norm_of_scalar_impl<T>;
            using norm_of_vector = norm_of_vector_impl<T>;

            static const bool value = norm_of_scalar::value || norm_of_vector::value;

            static_assert(value, "norm_type<T> are undefined for type U.");
        };
    }  // namespace traits_detail

    /**
     * @brief Traits class for the result type of the <tt>norm_l2()</tt> function.
     *
     * Member 'type' defines the result of <tt>norm_l2(t)</tt>, where <tt>t</tt>
     * is of type @tparam T. It implements the following rules designed to
     * minimize the potential for overflow:
     *   - @tparam T is an arithmetic type: 'type' is the result type of <tt>abs(t)</tt>.
     *   - @tparam T is a container of 'long double' elements: 'type' is <tt>long double</tt>.
     *   - @tparam T is a container of another arithmetic type: 'type' is <tt>double</tt>.
     *   - @tparam T is a container of some other type: 'type' is the element's norm type,
     *
     * Containers are recognized by having an embedded typedef 'value_type'.
     * To change the behavior for a case not covered here, specialize the
     * <tt>traits_detail::norm_type_base</tt> template.
     */
    template <class T>
    struct norm_type
        : public traits_detail::norm_type_base<T>
    {
        using base_type = traits_detail::norm_type_base<T>;

        using type =
            typename std::conditional<base_type::norm_of_vector::value,
                                      typename base_type::norm_of_vector::norm_type,
                                      typename base_type::norm_of_scalar::norm_type>::type;
    };

    /**
     * Abbreviation of 'typename norm_type<T>::type'.
     */
    template <class T>
    using norm_type_t = typename norm_type<T>::type;

    /**
     * @brief Traits class for the result type of the <tt>norm_sq()</tt> function.
     *
     * Member 'type' defines the result of <tt>norm_sq(t)</tt>, where <tt>t</tt>
     * is of type @tparam T. It implements the following rules designed to
     * minimize the potential for overflow:
     *   - @tparam T is an arithmetic type: 'type' is the result type of <tt>t*t</tt>.
     *   - @tparam T is a container of 'long double' elements: 'type' is <tt>long double</tt>.
     *   - @tparam T is a container of another floating-point type: 'type' is <tt>double</tt>.
     *   - @tparam T is a container of integer elements: 'type' is <tt>uint64_t</tt>.
     *   - @tparam T is a container of some other type: 'type' is the element's squared norm type,
     *
     *  Containers are recognized by having an embedded typedef 'value_type'.
     *  To change the behavior for a case not covered here, specialize the
     *  <tt>traits_detail::norm_type_base</tt> template.
     */
    template <class T>
    struct squared_norm_type
        : public traits_detail::norm_type_base<T>
    {
        using base_type = traits_detail::norm_type_base<T>;

        using type =
            typename std::conditional<base_type::norm_of_vector::value,
                                      typename base_type::norm_of_vector::squared_norm_type,
                                      typename base_type::norm_of_scalar::squared_norm_type>::type;
    };

    /**
     * Abbreviation of 'typename squared_norm_type<T>::type'.
     */
    template <class T>
    using squared_norm_type_t = typename squared_norm_type<T>::type;
}

#endif
