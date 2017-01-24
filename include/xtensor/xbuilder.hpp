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
#include <functional>
#include <cmath>

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
                return T(m_start + m_step * idx[0]);
            }

            template <class It>
            inline T element(It first, It /*last*/) const
            {
                return T(m_start + m_step * (*first));
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

        template <class T, template <class> class K, class F = K<T>>
        struct fn_impl
        {
            using value_type = T;
            using size_type = std::size_t;

            inline T operator()() const
            {
                // special case when called without args (happens when printing)
                return T();
            }

            template <class... Args>
            inline T operator()(Args... args) const
            {
                size_type idx [sizeof...(Args)] = {static_cast<size_type>(args)...};
                return access_impl(std::begin(idx), std::end(idx));
            }

            inline T operator[](const xindex& idx) const
            {
                return access_impl(idx.begin(), idx.end());
            }

            template <class It>
            inline T element(It first, It last) const
            {
                return access_impl(first, last);
            }

        private:
            F m_ft;
            template <class It>
            inline T access_impl(const It& begin, const It& end) const
            {
                return m_ft(begin, end);
            }
        };

        template <class T>
        struct eye_fn
        {
            template <class It>
            inline T operator()(const It& /*begin*/, const It& end) const
            {
                // workaround windows compile error by using temporary 
                // iterators and operator-=
                auto end_1 = end;
                auto end_2 = end;
                end_1 -= 1;
                end_2 -= 2;
                return *(end_1) == *(end_2) ? T(1) : T(0);
            }
        };
    }

    template <class T = bool>
    inline auto eye(const std::vector<size_t>& shape)
    {
        return detail::make_xgenerator(detail::fn_impl<T, detail::eye_fn>(), shape);
    }

    template <class T = bool>
    inline auto eye(std::size_t n)
    {
        return eye<T>({n, n});
    }

    /**
     * @function arange(T start, T stop, T step = 1)
     * @brief generate numbers evenly spaced within given half-open interval [start, stop).
     *
     * @param start start of the interval
     * @param stop stop of the interval
     * @param step stepsize
     *
     * @tparam T value_type of xexpression
     *
     * @return xgenerator that generates the values on access
     */
    template <class T>
    inline auto arange(T start, T stop, T step = 1) noexcept
    {
        std::size_t shape = static_cast<std::size_t>(std::ceil((stop - start) / step));
        return detail::make_xgenerator(detail::arange_impl<T>(start, stop, step), {shape});
    }

    /**
     * @function arange(T stop)
     * @brief generate numbers evenly spaced within given half-open interval [0, stop)
     *        with a step size of 1.
     *
     * @param stop stop of the interval
     *
     * @tparam T value_type of xexpression
     *
     * @return xgenerator that generates the values on access
     */
    template <class T>
    inline auto arange(T stop) noexcept
    {
        return arange<T>(T(0), stop, T(1));
    }

    /**
     * @function linspace
     * @brief generate @a num_samples evenly spaced numbers over given interval
     *
     * @param start start of interval
     * @param stop stop of interval
     * @param num_samples number of samples (defaults to 50)
     * @param endpoint if true, include endpoint (defaults to true)
     *
     * @tparam T value_type of xexpression
     *
     * @return xgenerator that generates the values on access
     */
    template <class T>
    inline auto linspace(T start, T stop, std::size_t num_samples = 50, bool endpoint = true) noexcept
    {
        T step = (stop - start) / T(num_samples - (endpoint ? 1 : 0));
        return detail::make_xgenerator(detail::arange_impl<T>(start, stop, step), {num_samples});
    }

    /**
     * @function logspace
     * @brief generate @num_samples numbers evenly spaced on a log scale over given interval
     *
     * @param start start of interval (pow(base, start) is the first value).
     * @param stop stop of interval (pow(base, stop) is the final value, except if endpoint = false)
     * @param num_samples number of samples (defaults to 50)
     * @param base the base of the log space.
     * @param endpoint if true, include endpoint (defaults to true)
     *
     * @tparam T value_type of xexpression
     *
     * @return xgenerator that generates the values on access
     */
    template <class T>
    inline auto logspace(T start, T stop, std::size_t num_samples, T base = 10, bool endpoint = true) noexcept
    {
        return pow(base, linspace(start, stop, num_samples, endpoint));
    }

}
#endif
