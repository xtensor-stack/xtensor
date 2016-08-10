#ifndef NDUTILS_HPP
#define NDUTILS_HPP

#include <utility>
#include <tuple>

namespace qs
{

    template <size_t... Ints>
    struct index_sequence;

    /*
    template <size_t N>
    using make_index_sequence;
    */

    template <class F, class... T>
    void for_each(F&& f, std::tuple<T...>& t);


    template <class F, class... Args>
    void for_each_arg(F&&, Args&&...);


    /********************
     * index_sequence
     ********************/

    template <size_t... Ints>
    struct index_sequence
    {
        using type = index_sequence<Ints...>;

        static constexpr size_t size() noexcept
        {
            return sizeof...(Ints);
        }
    };

    namespace detail
    {
        template <class S1, class S2>
        struct merge_and_renumber;

        template <size_t... I1, size_t... I2>
        struct merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>
            : index_sequence<I1..., (sizeof...(I1) + I2)...>
        {};

        template <size_t N>
        struct make_index_sequence_impl
            : merge_and_renumber<typename make_index_sequence_impl<N/2>::type,
                                 typename make_index_sequence_impl<N-N/2>::type>
        {};

        template <>
        struct make_index_sequence_impl<0> : index_sequence<> {};

        template <>
        struct make_index_sequence_impl<1> : index_sequence<0> {};
    }

    template <size_t N>
    using make_index_sequence = typename detail::make_index_sequence_impl<N>::type;


    /***********************
     * for_each on tuple
     ***********************/

    namespace detail
    {
        template <class F, class... T, size_t I>
        inline typename std::enable_if<I == sizeof...(T), void>::type
        for_each_impl(F&&, std::tuple<T...>& t)
        {
        }

        template <class F, class... T, size_t I>
        inline typename std::enable_if<I < sizeof...(T), void>::type
        for_each_impl(F&& f, std::tuple<T...>& t)
        {
            f(std::get<I>(t));
            for_each_impl<T..., F, I+1>(std::forward<F>(f), t);
        }
    }

    template <class F, class... T>
    inline void for_each(F&& f, std::tuple<T...>& t)
    {
        detail::for_each_impl<T..., F, 0>(std::forward<F>(f), f);
    }


    /********************************
     * for_each on parameter pack
     ********************************/

    namespace detail
    {
        template <size_t I>
        struct invoker_base
        {
            template <class F, class T>
            invoker_base(F&& f, T&& t)
            {
                f(std::forward<T>(t));
            }
        };

        template <size_t... Ints>
        struct invoker : invoker_base<Ints>...
        {
            template <class F, class... Args>
            invoker(F&& f, Args&&... args)
                : invoker_base<Ints>(std::forward<F>(f), std::forward<Args>(args))...
            {
            }
        };

        template <class F, size_t... Ints, class... Args>
        inline void for_each_arg_impl(F&& f, index_sequence<Ints...>, Args&&... args)
        {
            invoker<Ints...> invoker(std::forward<F>(f), std::forward<Args>(args)...);
        }
    }

    template <class F, class... Args>
    inline void for_each_arg(F&& f, Args&&... args)
    {
        detail::for_each_arg_impl(std::forward<F>(f), make_index_sequence<sizeof...(Args)>(), std::forward<Args>(args)...);
    }

}

#endif

