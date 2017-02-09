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
#include "xexpression.hpp"
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
     * Returns an \ref xexpression containing ones of the specified shape.
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
     * Returns an \ref xexpression containing zeros of the specified shape.
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
                return T(m_start + m_step * T(idx[0]));
            }

            template <class It>
            inline T element(It first, It /*last*/) const
            {
                return T(m_start + m_step * T(*first));
            }

        private:
            value_type m_start;
            value_type m_stop;
            value_type m_step;

            template <class T1, class... Args>
            inline T access_impl(T1 t, Args... /*args*/) const
            {
                return m_start + m_step * T(t);
            }

            inline T access_impl() const
            {
                return m_start;
            }
        };

        template <class F>
        struct fn_impl
        {
            using value_type = typename F::value_type;
            using size_type = std::size_t;

            fn_impl(F&& f) : m_ft(f)
            {
            }

            inline value_type operator()() const
            {
                // special case when called without args (happens when printing)
                return value_type();
            }

            template <class... Args>
            inline value_type operator()(Args... args) const
            {
                size_type idx [sizeof...(Args)] = {static_cast<size_type>(args)...};
                return access_impl(std::begin(idx), std::end(idx));
            }

            inline value_type operator[](const xindex& idx) const
            {
                return access_impl(idx.begin(), idx.end());
            }

            template <class It>
            inline value_type element(It first, It last) const
            {
                return access_impl(first, last);
            }

        private:
            F m_ft;
            template <class It>
            inline value_type access_impl(const It& begin, const It& end) const
            {
                return m_ft(begin, end);
            }
        };

        template <class T>
        struct eye_fn
        {
            using value_type = T;

            eye_fn(int k) : m_k(k)
            {
            }

            template <class It>
            inline T operator()(const It& /*begin*/, const It& end) const
            {
                // workaround windows compile error by using temporary 
                // iterators and operator-=
                auto end_1 = end;
                auto end_2 = end;
                end_1 -= 1;
                end_2 -= 2;
                return *(end_1) == *(end_2) + m_k ? T(1) : T(0);
            }

        private:
            int m_k;
        };
    }

    /**
     * Generates an array with ones on the diagonal.
     * @param shape shape of the resulting expression
     * @param k index of the diagonal. 0 (default) refers to the main diagonal,
     *          a positive value refers to an upper diagonal, and a negative
     *          value to a lower diagonal.
     * @tparam T value_type of xexpression
     * @return xgenerator that generates the values on access
     */
    template <class T = bool>
    inline auto eye(const std::vector<std::size_t>& shape, int k = 0)
    {
        return detail::make_xgenerator(detail::fn_impl<detail::eye_fn<T>>(detail::eye_fn<T>(k)), shape);
    }

    /**
     * Generates a (n x n) array with ones on the diagonal.
     * @param n length of the diagonal.
     * @param k index of the diagonal. 0 (default) refers to the main diagonal,
     *          a positive value refers to an upper diagonal, and a negative
     *          value to a lower diagonal.
     * @tparam T value_type of xexpression
     * @return xgenerator that generates the values on access
     */
    template <class T = bool>
    inline auto eye(std::size_t n, int k = 0)
    {
        return eye<T>({n, n}, k);
    }

    /**
     * Generates numbers evenly spaced within given half-open interval [start, stop).
     * @param start start of the interval
     * @param stop stop of the interval
     * @param step stepsize
     * @tparam T value_type of xexpression
     * @return xgenerator that generates the values on access
     */
    template <class T>
    inline auto arange(T start, T stop, T step = 1) noexcept
    {
        std::size_t shape = static_cast<std::size_t>(std::ceil((stop - start) / step));
        return detail::make_xgenerator(detail::arange_impl<T>(start, stop, step), {shape});
    }

    /**
     * Generate numbers evenly spaced within given half-open interval [0, stop)
     * with a step size of 1.
     * @param stop stop of the interval
     * @tparam T value_type of xexpression
     * @return xgenerator that generates the values on access
     */
    template <class T>
    inline auto arange(T stop) noexcept
    {
        return arange<T>(T(0), stop, T(1));
    }

    /**
     * Generates @a num_samples evenly spaced numbers over given interval
     * @param start start of interval
     * @param stop stop of interval
     * @param num_samples number of samples (defaults to 50)
     * @param endpoint if true, include endpoint (defaults to true)
     * @tparam T value_type of xexpression
     * @return xgenerator that generates the values on access
     */
    template <class T>
    inline auto linspace(T start, T stop, std::size_t num_samples = 50, bool endpoint = true) noexcept
    {
        T step = (stop - start) / T(num_samples - (endpoint ? 1 : 0));
        return detail::make_xgenerator(detail::arange_impl<T>(start, stop, step), {num_samples});
    }

    /**
     * Generates @a num_samples numbers evenly spaced on a log scale over given interval
     * @param start start of interval (pow(base, start) is the first value).
     * @param stop stop of interval (pow(base, stop) is the final value, except if endpoint = false)
     * @param num_samples number of samples (defaults to 50)
     * @param base the base of the log space.
     * @param endpoint if true, include endpoint (defaults to true)
     * @tparam T value_type of xexpression
     * @return xgenerator that generates the values on access
     */
    template <class T>
    inline auto logspace(T start, T stop, std::size_t num_samples, T base = 10, bool endpoint = true) noexcept
    {
        return pow(base, linspace(start, stop, num_samples, endpoint));
    }

    namespace detail
    {
        template <class CA, class CB>
        struct concatenate_impl
        {
            using size_type = std::size_t;
            using value_type = std::common_type_t<typename std::decay_t<CA>::value_type,
                                                  typename std::decay_t<CB>::value_type>;

            concatenate_impl(const CA& a, const CB& b, std::size_t axis) :
                m_a(a), m_b(b), m_axis(axis)
            {
            }

            template <class... Args>
            value_type operator()(Args... args) const
            {
                return access_impl(xindex({{static_cast<size_type>(args)...}}));
            }

            value_type operator[](const xindex& idx) const
            {
                return access_impl(idx);
            }

            template <class It>
            value_type element(It first, It last) const
            {
                return access_impl(xindex(first, last));
            }

        private:
            inline value_type access_impl(xindex idx) const
            {
                if (idx[m_axis] >= m_a.shape()[m_axis])
                {
                    idx[m_axis] -= m_a.shape()[m_axis];
                    return m_b[idx];
                }
                else
                {
                    return m_a[idx];
                }
            }

            const CA m_a;
            const CB m_b;
            const size_type m_axis;
        };
    }

    /**
     * @function concatentate
     * @brief Concatenate xexpressions along \em axis.
     *
     * @param a xexpression to concatenate
     * @param b xexpression to concatenate
     * @returns xgenerator evaluating to concatenated elements
     */
    template <class T, class U>
    inline auto concatenate(T&& a, U&& b, std::size_t axis = 0)
    {
        std::vector<std::size_t> new_shape(a.shape());
        new_shape[axis] += b.shape()[axis];
        using concat_t = detail::concatenate_impl<detail::const_closure_t<T>, detail::const_closure_t<U>>;
        return detail::make_xgenerator(concat_t(std::forward<T>(a), std::forward<U>(b), axis), new_shape);
    }

    /**
     * @function hstack
     * @brief Stack xexpressions horizontally (column-wise).
     *        \em a and \em b have to have the same dimensions.
     *
     * @param a first xexpression to stack
     * @param b second xexpression to stack
     * @returns xgenerator evaluating to stacked elements
     */
    template <class T, class U>
    inline auto hstack(T&& a, U&& b) {
        if (a.dimension() != b.dimension())
        {
            throw std::invalid_argument("All inputs to hstack must have same number of dimensions");
        }
        std::size_t axis = 1;
        if (a.dimension() == 1) 
        {
            axis = 0;
        }
        return concatenate(std::forward<T>(a), std::forward<U>(b), axis);
    }

    /**
     * @function vstack
     * @brief Stack xexpressions vertically (row-wise). 
     *        \em a and \em b have to be at least two-dimensional.
     *
     * @param a first xexpression to stack
     * @param b second xexpression to stack
     * @returns xgenerator evaluating to stacked elements
     */
    template <class T, class U>
    inline auto vstack(T&& a, U&& b) {
        if (a.dimension() < 2 || b.dimension() < 2)
        {
            throw std::invalid_argument("All inputs to vstack must have at least dimension 2.");
        }

        std::size_t axis = 0;
        // handle special case for 1D case
        return concatenate(std::forward<T>(a), std::forward<U>(b), axis);
    }
}
#endif
