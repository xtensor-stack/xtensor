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

#include <initializer_list>
#include <utility>
#include <vector>

#include "xbroadcast.hpp"
#include "xindex_function.hpp"
#include "xmath.hpp"

namespace xt
{

    /********
     * ones *
     ********/

    /**
     * @function ones
     * @brief Returns an \ref xexpression containing ones of the specified shape.
     *
     * @tparam shape the shape of the returned expression.
     */
    template <class T, class S>
    inline auto ones(S shape) noexcept
    {
        return broadcast(T(1), std::forward<S>(shape));
    }

    template <class T, class I>
    inline auto ones(std::initializer_list<I> shape) noexcept
    {
        // TODO: In the case of an initializer_list, use an array instead of a vector.
        return ones<T>(std::vector<I>(shape));
    }

    /*********
     * zeros *
     *********/

    /**
     * @function ones
     * @brief Returns an \ref xexpression containing zeros of the specified shape.
     *
     * @tparam shape the shape of the returned expression.
     */
    template <class T, class S>
    inline auto zeros(S shape) noexcept
    {
        return broadcast(T(0), std::forward<S>(shape));
    }
    
    template <class T, class I>
    inline auto zeros(std::initializer_list<I> shape) noexcept
    {
        // TODO: In the case of an initializer_list, use an array instead of a vector.
        return zeros<T>(std::vector<I>(shape));
    }

    namespace detail {
        template <class Functor>
        inline auto make_xindex_function(Functor&& f, std::initializer_list<std::size_t> shape) noexcept
        {
            using type = xindex_function<Functor, typename Functor::value_type>;
            return type(f, std::forward<std::vector<std::size_t>>(std::vector<std::size_t>(shape)));
        }
        
        template <class Functor, class S>
        inline auto make_xindex_function(Functor&& f, const S& shape) noexcept
        {
            using type = xindex_function<Functor, typename Functor::value_type>;
            return type(f, std::forward<S>(shape));
        }

        template <class T>
        struct arange_impl {
            T m_start, m_stop, m_step;
            using value_type = T;
            arange_impl(T start, T stop, T step) : 
                m_start(start), m_stop(stop), m_step(step) {
            }

            template <class... Args>
            T operator()(Args... args) const {
                std::vector<T> args_arr({static_cast<T>(args)...});
                return m_start + m_step * args_arr[0];
            }

            T operator[](const xindex& idx) const {
                return m_start + m_step * idx[0];
            }
        };

        template <class T>
        struct repeat_impl {
            using value_type = typename T::value_type;
            using size_type = typename T::size_type;
            using return_type = T;
            
            const T& m_V;
            const size_type m_axis;

            repeat_impl(const T& V, const size_type axis) :
                m_V(V), m_axis(axis) {
            }

            template <class... Args>
            value_type operator()(Args... args) const {
                std::vector<size_type> args_arr({static_cast<size_type>(args)...});
                return m_V(args_arr[m_axis]);
            }

            value_type operator[](const xindex& idx) const {
                return m_V[{idx[m_axis]}];
            }
        };

        template <class T, class U, class v_t = typename T::value_type>
        struct concat_impl {
            const T& m_A;
            const U& m_B;
            using size_type = std::size_t;
            using value_type = v_t;
            std::size_t m_axis;
            concat_impl(const T& a, const U& b, const std::size_t axis) :
                m_A(a), m_B(b), m_axis(axis) {
            }

            template <class... Args>
            value_type operator()(Args... args) const {
                std::vector<size_type> args_arr({static_cast<size_type>(args)...});
                if (args_arr[m_axis] >= m_A.shape()[m_axis]) 
                {
                    args_arr[m_axis] -= m_A.shape()[m_axis];
                    return m_B[args_arr];
                }
                else
                {
                    return m_A[args_arr];
                }
            }

            value_type operator[](const xindex& idx) const {
                if (idx[m_axis] > m_A.shape()[m_axis]) 
                {
                    xindex temp(idx);
                    temp[m_axis] -= m_A.shape()[m_axis];
                    return m_B[temp];
                }
                else
                {
                    return m_A[idx];
                }
            }
        };
    }

    template <class T, class U>
    inline auto concat(const T& a, const U& b, std::size_t axis = 0) {
        std::vector<std::size_t> new_shape(a.shape());
        new_shape[axis] += b.shape()[axis];
        return detail::make_xindex_function(detail::concat_impl<T, U>(a, b, axis), new_shape);
    }

    template <class T, class U>
    inline auto hstack(const T& a, const U& b) {
        return concat(a, b, 1);
    }

    template <class T, class U>
    inline auto vstack(const T& a, const U& b) {
        return concat(a, b, 0);
    }

    template <class T>
    inline auto arange(const T stop) noexcept
    {
        return arange(0, stop, 1);
    }

    template <class T>
    inline auto arange(const T start, const T stop) noexcept
    {
        return arange(start, stop, 1);
    }

    template <class T>
    inline auto arange(const T start, const T stop, const T step) noexcept
    {
        std::size_t shape = (stop - start) / step;
        return detail::make_xindex_function(detail::arange_impl<T>(start, stop, step), {shape});
    }

    template <class T>
    inline auto linspace(const T start, const T stop) noexcept
    {
        return linspace(start, stop, 50, true);
    }

    template <class T>
    inline auto linspace(const T start, const T stop, const std::size_t num_samples, bool endpoint=true) noexcept
    {
        T step = (stop - start) / T(num_samples - (endpoint ? 1 : 0));
        return detail::make_xindex_function(detail::arange_impl<T>(start, stop, step), {num_samples});
    }

    template <class T>
    inline auto logspace(T start, T stop, std::size_t num_samples, T base = 10, bool endpoint=true) noexcept
    {
        return pow(base, linspace(start, stop, num_samples, endpoint));
    }

    template <class T, class U>
    inline auto meshgrid(const T& X, const U& Y) noexcept
    {
        return std::make_tuple(
            detail::make_xindex_function(detail::repeat_impl<T>(X, 1), {Y.shape()[0], X.shape()[0]}),
            detail::make_xindex_function(detail::repeat_impl<U>(Y, 0), {Y.shape()[0], X.shape()[0]})
        );
    }
}

#endif