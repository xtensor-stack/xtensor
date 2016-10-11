#ifndef XUTILS_HPP
#define XUTILS_HPP

#include <cstddef>
#include <utility>
#include <tuple>
#include <type_traits>
#include <initializer_list>

namespace qs
{

    template <class T>
    struct remove_class;

    template <class F, class... T>
    void for_each(F&& f, std::tuple<T...>& t);

    template <class F, class R, class... T>
    R accumulate(F&& f, R init, const std::tuple<T...>& t);

    template <class... T>
    struct or_;

    template <std::size_t I, class... Args>
    decltype(auto) argument(Args&&... args) noexcept;

    template<class R, class F, class... S>
    R apply(std::size_t index, F&& func, S&&... s);

    template<class R, class F, class... S>
    R apply(std::size_t index, F&& func, std::tuple<S...>& s);

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

    /**************************
     * for_each implementation
     **************************/

    namespace detail
    {
        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I == sizeof...(T), void>::type
        for_each_impl(F&& f, std::tuple<T...>& t)
        {
        }

        template <std::size_t I, class F, class... T>
        inline typename std::enable_if<I < sizeof...(T), void>::type
        for_each_impl(F&& f, std::tuple<T...>& t)
        {
            f(std::get<I>(t));
            for_each_impl<I + 1, F, T...>(std::forward<F>(f), t);
        }
    }

    template <class F, class... T>
    inline void for_each(F&& f, std::tuple<T...>& t)
    {
        detail::for_each_impl<0, F, T...>(std::forward<F>(f), t);
    }

    /****************************
     * accumulate implementation
     ****************************/

    namespace detail
    {
        template <std::size_t I, class F, class R, class... T>
        inline std::enable_if_t<I == sizeof...(T), R>
        accumulate_impl(F&& f, R init, const std::tuple<T...>& t)
        {
            return init;
        }

        template <std::size_t I, class F, class R, class... T>
        inline std::enable_if_t<I < sizeof...(T), R>
        accumulate_impl(F&& f, R init, const std::tuple<T...>& t)
        {
            R res = f(init, std::get<I>(t));
            return accumulate_impl<I + 1, F, R, T...>(std::forward<F>(f), res, t);
        }
    }

    template <class F, class R, class... T>
    inline R accumulate(F&& f, R init, const std::tuple<T...>& t)
    {
        return detail::accumulate_impl<0, F, R, T...>(f, init, t);
    }

    /**********************
     * or_ implementation
     **********************/

    template <class T>
    struct or_<T> : std::integral_constant<bool, T::value>
    {
    };

    template <class T, class... Ts>
    struct or_<T, Ts...>
        : std::integral_constant<bool, T::value || or_<Ts...>::value>
    {
    };

    /**************************
     * argument implementation
     **************************/
 
    namespace detail
    {
        template <std::size_t I>
        struct getter
        {
            template <class Arg, class... Args>
            static inline decltype(auto) get(Arg&& arg, Args&&... args) noexcept
            {
                return getter<I - 1>::get(std::forward<Args>(args)...);
            }
        };

        template <>
        struct getter<0>
        {
            template <class Arg, class... Args>
            static inline Arg&& get(Arg&& arg, Args&&... args) noexcept
            {
                return std::forward<Arg>(arg);
            }
        };
    }

    template <std::size_t I, class... Args>
    inline decltype(auto) argument(Args&&... args) noexcept
    {
        static_assert(I < sizeof...(Args), "I should be lesser than sizeof...(Args)");
        return detail::getter<I>::get(std::forward<Args>(args)...);
    }
    
    /************************
     * apply implementation
     ************************/

    namespace detail
    {
        template<class R, class F, std::size_t I, class... S>
        R apply_one(F&& func, S&&... s)
        {
            return func(argument<I>(s...));
        }

        template<class R, class F, std::size_t... I, class... S>
        R apply(std::size_t index, F&& func, std::index_sequence<I...>, S&&... s)
        {
            using FT = R(F, S&&...);
            static constexpr FT* arr[] = { &apply_one<R, F, I, S...>... };
            return arr[index](std::forward<F>(func), std::forward<S>(s)...);
        }

        template <class R, class F, std::size_t... I, class... S>
        R apply(std::size_t index, F&& func, std::index_sequence<I...> seq, std::tuple<S...>& s)
        {
            return apply<R>(index, std::forward<F>(func), seq, std::get<I>(s)...);
        }
    }

    template<class R, class F, class... S>
    inline R apply(std::size_t index, F&& func, S&&... s)
    {
        return detail::apply<R>(index, std::forward<F>(func), std::make_index_sequence<sizeof...(S)>(), std::forward<S>(s)...);
    }

    template<class R, class F, class... S>
    inline R apply(std::size_t index, F&& func, std::tuple<S...>& s)
    {
        return detail::apply<R>(index, std::forward<F>(func), std::make_index_sequence<sizeof...(S)>(), s);
    }

    /***************************************
     * initializer_dimension implementation
     ***************************************/

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

    /***********************************
     * initializer_shape implementation
     ***********************************/

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
    constexpr R initializer_shape(T t)
    {
        return detail::initializer_shape<R, decltype(t)>(t, std::make_index_sequence<initializer_dimension<decltype(t)>::value>());
    } 

    template <class R, class T>
    constexpr R initializer_shape(std::initializer_list<T> t)
    {
        return detail::initializer_shape<R, decltype(t)>(t, std::make_index_sequence<initializer_dimension<decltype(t)>::value>());
    }
    
    template <class R, class T>
    constexpr R initializer_shape(std::initializer_list<std::initializer_list<T>> t)
    {
        return detail::initializer_shape<R, decltype(t)>(t, std::make_index_sequence<initializer_dimension<decltype(t)>::value>());
    }

    /*****************************
     * nested_copy implementation 
     *****************************/

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

    template <class T, class S>
    inline void nested_copy(T&& iter, std::initializer_list<std::initializer_list<S>> s)
    {
        for (auto it = s.begin(); it != s.end(); ++it)
        {
            nested_copy(std::forward<T>(iter), *it);
        }
    }

}

#endif

