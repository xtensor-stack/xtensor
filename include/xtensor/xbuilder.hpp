/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

/**
 * @brief standard mathematical functions for xexpressions
 */

#ifndef XBUILDER_HPP
#define XBUILDER_HPP

#include <utility>

#include "xfunction.hpp"
#include "xbroadcast.hpp"
#include "xgenerator.hpp"

#ifdef X_OLD_CLANG
    #include <initializer_list>
    #include <vector>
#else
    #include <array>
#endif

namespace xt
{

    /********
     * ones *
     ********/

    /**
     * @brief Returns an \ref xexpression containing ones of the specified shape.
     *
     * @tparam shape the shape of the returned expression.
     */
    template <class T, class S>
    inline auto ones(S shape) noexcept
    {
        return broadcast(T(1), std::forward<S>(shape));
    }

#ifdef X_OLD_CLANG
    template <class T, class I>
    inline auto ones(std::initializer_list<I> shape) noexcept
    {
        return broadcast(T(1), shape);
    }
#else
    template <class T, class I, std::size_t L>
    inline auto ones(const I(&shape)[L]) noexcept
    {
        return broadcast(T(1), shape);
    }
#endif

    /*********
     * zeros *
     *********/

    /**
     * @brief Returns an \ref xexpression containing zeros of the specified shape.
     *
     * @tparam shape the shape of the returned expression.
     */
    template <class T, class S>
    inline auto zeros(S shape) noexcept
    {
        return broadcast(T(0), std::forward<S>(shape));
    }
    
#ifdef X_OLD_CLANG
    template <class T, class I>
    inline auto zeros(std::initializer_list<I> shape) noexcept
    {
        return broadcast(T(0), shape);
    }
#else
    template <class T, class I, std::size_t L>
    inline auto zeros(const I(&shape)[L]) noexcept
    {
        return broadcast(T(0), shape);
    }
#endif

    namespace detail
    {

        template <class Functor, class I>
        inline auto make_xgenerator(Functor&& f, std::initializer_list<I> shape) noexcept
        {
            using type = xgenerator<Functor, typename Functor::value_type>;
            return type(std::forward<Functor>(f), std::vector<std::size_t>(shape));
        }
        
        template <class Functor, class S>
        inline auto make_xgenerator(Functor&& f, const S& shape) noexcept
        {
            using type = xgenerator<Functor, typename Functor::value_type>;
            return type(std::forward<Functor>(f), shape);
        }

        template <class T>
        struct arange_impl
        {
            using value_type = T;

            arange_impl(T start, T stop, T step) :
                m_start(start), m_stop(stop), m_step(step)
            {
            }

            template <class... Args>
            inline T operator()(Args... args) const
            {
                return access_impl(args...);
            }

            inline T operator[](const xindex& idx) const
            {
                return m_start + m_step * idx[0];
            }

            template <class It>
            inline T element(It first, It /*last*/) const
            {
                return m_start + m_step * (*first);
            }

        private:
            value_type m_start;
            value_type m_stop;
            value_type m_step;

            template <class T1, class... Args>
            inline T access_impl(T1 t, Args... /*args*/) const
            {
                return m_start + m_step * t;
            }

            inline T access_impl() const
            {
                return m_start;
            }
        };
    }

    template <class T>
    inline auto arange(T start, T stop, T step = 1) noexcept
    {
        std::size_t shape = (stop - start) / step;
        return detail::make_xgenerator(detail::arange_impl<T>(start, stop, step), {shape});
    }

    template <class T>
    inline auto arange(T stop) noexcept
    {
        return arange<T>(T(0), stop, T(1));
    }

    template <class T>
    inline auto linspace(T start, T stop, std::size_t num_samples = 50, bool endpoint = true) noexcept
    {
        T step = (stop - start) / T(num_samples - (endpoint ? 1 : 0));
        return detail::make_xgenerator(detail::arange_impl<T>(start, stop, step), {num_samples});
    }

    template <class T>
    inline auto logspace(T start, T stop, std::size_t num_samples, T base = 10, bool endpoint = true) noexcept
    {
        return pow(base, linspace(start, stop, num_samples, endpoint));
    }

}
#endif