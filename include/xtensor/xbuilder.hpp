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

        template <class T>
        struct diagonal_fn
        {
            using xtype = T;
            using value_type = typename T::value_type;

            diagonal_fn(const T& arr) : m_arr(arr)
            {
            }

            template <class It>
            inline value_type operator()(const It& begin, const It& end) const
            {
                return m_arr(*begin, *begin);
            }

        private:
            const xtype& m_arr;
        };

        template <class T>
        struct diag_fn
        {
            using xtype = T;
            using value_type = typename T::value_type;

            diag_fn(const T& arr) : m_arr(arr)
            {
            }

            template <class It>
            inline value_type operator()(const It& begin, const It& end) const
            {
                auto other = begin;
                other += 1;
                return *begin == *other ? m_arr(*begin) : value_type(0);
            }

        private:
            const T& m_arr;
        };

        template <class T>
        struct flipud_fn
        {
            using xtype = T;
            using value_type = typename T::value_type;

            flipud_fn(const T& arr) : m_arr(arr), shape_first(m_arr.shape().front() - 1)
            {
            }

            template <class It>
            inline value_type operator()(const It& begin, const It& end) const
            {
                xindex idx(begin, end);
                idx.front() = shape_first - idx.front();
                return m_arr.element(idx.begin(), idx.end());
            }

        private:
            const T& m_arr;
            const std::size_t shape_first;
        };

        template <class T>
        struct fliplr_fn
        {
            using xtype = T;
            using value_type = typename T::value_type;

            fliplr_fn(const T& arr) : m_arr(arr), shape_last(m_arr.shape().back() - 1)
            {
            }

            template <class It>
            inline value_type operator()(const It& begin, const It& end) const
            {
                xindex idx(begin, end);
                idx.back() = shape_last - idx.back();
                return m_arr.element(idx.begin(), idx.end());
            }

        private:
            const T& m_arr;
            const std::size_t shape_last;
        };

        template <class T, class C>
        struct tril_fn
        {
            using xtype = T;
            using value_type = typename T::value_type;
            using signed_idx_type = long int;

            tril_fn(const T& arr, int k, const C& comp) : m_arr(arr), m_k(k), m_comp(comp)
            {
            }

            template <class It>
            inline value_type operator()(const It& begin, const It& end) const
            {
                // have to cast to signed int otherwise -1 can lead to overflow
                auto begin_next = begin;
                begin_next += 1;
                return m_comp(signed_idx_type(*begin) + m_k, signed_idx_type(*begin_next)) ? m_arr.element(begin, end) : value_type(0);
            }

        private:
            const xtype& m_arr;
            const signed_idx_type m_k;
            const C m_comp;
        };

    }

    /**
     * @function eye(const std::vector<std::size_t>& shape, int k = 0)
     * @brief generate array with ones on the diagonal
     *
     * @param shape shape of the resulting expression
     * @param k index of the diagonal. 0 (default) refers to the main diagonal,
     *          a positive value refers to an upper diagonal, and a negative
     *          value to a lower diagonal.
     *
     * @tparam T value_type of xexpression
     *
     * @return xgenerator that generates the values on access
     */
    template <class T = bool>
    inline auto eye(const std::vector<std::size_t>& shape, int k = 0)
    {
        return detail::make_xgenerator(detail::fn_impl<detail::eye_fn<T>>(detail::eye_fn<T>(k)), shape);
    }

    /**
     * @function eye(std::size_t n, int k = 0)
     * @brief like eye with a shape of n x n
     */
    template <class T = bool>
    inline auto eye(std::size_t n, int k = 0)
    {
        return eye<T>({n, n}, k);
    }

    /**
     * @function identity(std::size_t n, int k = 0)
     * @brief return identity matrix with 1 on the diagonal. Same as \ref eye(n).
     */
    template <class T = bool>
    inline auto identity(std::size_t n)
    {
        return eye<T>({n, n});
    }

    /**
     * @function diagonal(const xexpression<T>& arr)
     * @brief Returns the elements on the diagonal of arr
     *
     * @param arr the input array
     *
     * @return xexpression with values of the diagonal
     */
    template <class T>
    inline auto diagonal(const xexpression<T>& arr)
    {
        const T& arr_dc = arr.derived_cast();
        return detail::make_xgenerator(detail::fn_impl<detail::diagonal_fn<T>>(detail::diagonal_fn<T>(arr_dc)), {arr_dc.shape()[0]});
    }

    /**
     * @function diag(const xexpression<T>& arr)
     * @brief xexpression with values of arr on the diagonal, zeroes otherwise
     *
     * @param arr the 1D input array of length n
     *
     * @return xexpression function with shape n x n and arr on the diagonal
     */
    template <class T>
    inline auto diag(const xexpression<T>& arr)
    {
        const T& arr_dc = arr.derived_cast();
        return detail::make_xgenerator(detail::fn_impl<detail::diag_fn<T>>(detail::diag_fn<T>(arr_dc)), 
                                       {arr_dc.shape()[0], arr_dc.shape()[0]});
    }

    /**
     * @function fliplr(const xexpression<T>& arr)
     * @brief Flip xexpression in the left/right direction. Essentially flips the last axis.
     *
     * @param arr the input array
     *
     * @return xexpression with values flipped in left/right direction
     */
    template <class T>
    inline auto fliplr(const xexpression<T>& arr)
    {
        const T& arr_dc = arr.derived_cast();
        return detail::make_xgenerator(detail::fn_impl<detail::fliplr_fn<T>>(detail::fliplr_fn<T>(arr_dc)), 
                                       arr_dc.shape());
    }

    /**
     * @function flipud(const xexpression<T>& arr)
     * @brief Flip xexpression in the up/down direction. Essentially flips the last axis.
     *
     * @param arr the input array
     *
     * @return xexpression with values flipped in up/down direction
     */
    template <class T>
    inline auto flipud(const xexpression<T>& arr)
    {
        const T& arr_dc = arr.derived_cast();
        return detail::make_xgenerator(detail::fn_impl<detail::flipud_fn<T>>(detail::flipud_fn<T>(arr_dc)), 
                                       arr_dc.shape());
    }

    /**
     * @function tril(const xexpression<T>& arr, int k = 0)
     * @brief Extract lower triangular matrix from xexpression. The parameter k selects the
     *        offset of the diagonal.
     *
     * @param arr the input array
     * @param k the diagonal above which to zero elements. 0 (default) selects the main diagonal, 
     *          k < 0 is below the main diagonal, k > 0 above.
     *
     * @return xexpression containing lower triangle from arr, 0 otherwise
     */
    template <class T>
    inline auto tril(const xexpression<T>& arr, int k = 0)
    {
        const T& arr_dc = arr.derived_cast();
        return detail::make_xgenerator(detail::fn_impl<detail::tril_fn<T, std::greater_equal<long int>>>(
                                       detail::tril_fn<T, std::greater_equal<long int>>(arr_dc, k, std::greater_equal<long int>())), 
                                       arr_dc.shape());
    }

    /**
     * @function triu(const xexpression<T>& arr, int k = 0)
     * @brief Extract upper triangular matrix from xexpression. The parameter k selects the
     *        offset of the diagonal.
     *
     * @param arr the input array
     * @param k the diagonal below which to zero elements. 0 (default) selects the main diagonal, 
     *          k < 0 is below the main diagonal, k > 0 above.
     *
     * @return xexpression containing lower triangle from arr, 0 otherwise
     */
    template <class T>
    inline auto triu(const xexpression<T>& arr, int k = 0)
    {
        const T& arr_dc = arr.derived_cast();
        return detail::make_xgenerator(detail::fn_impl<detail::tril_fn<T, std::less_equal<long int>>>(
                    detail::tril_fn<T, std::less_equal<long int>>(arr_dc, k, std::less_equal<long int>())), 
                                       arr_dc.shape());
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
