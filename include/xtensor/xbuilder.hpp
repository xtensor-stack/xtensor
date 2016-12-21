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

    namespace detail
    {

        template <class Functor>
        inline auto make_xindex_function(Functor&& f, std::initializer_list<std::size_t> shape) noexcept
        {
            using type = xindex_function<Functor, typename Functor::value_type>;
            return type(f, std::vector<std::size_t>(shape));
        }
        
        template <class Functor, class S>
        inline auto make_xindex_function(Functor&& f, const S& shape) noexcept
        {
            using type = xindex_function<Functor, typename Functor::value_type>;
            return type(f, shape);
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
                access_impl(args...);
            }

            inline T operator[](const xindex& idx) const
            {
                return m_start + m_step * idx[0];
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

        template <class T>
        struct repeat_impl
        {
            using value_type = typename T::value_type;
            using size_type = typename T::size_type;

            repeat_impl(const T& source, size_type axis) :
                m_source(source), m_axis(axis)
            {
            }

            template <class... Args>
            value_type operator()(Args... args) const
            {
                // to catch a -Wuninitialized
                if (sizeof...(Args))
                {
                    std::array<size_type, sizeof...(Args)> args_arr({static_cast<size_type>(args)...});
                    return m_source(args_arr[m_axis]);
                }
                return m_source();
            }

            value_type operator[](const xindex& idx) const
            {
                return m_source[{idx[m_axis]}];
            }

        private:
            const T& m_source;
            const size_type m_axis;
        };

        template <class T, class U, class V = typename T::value_type>
        struct concatenate_impl
        {
            using size_type = std::size_t;
            using value_type = V;

            concatenate_impl(const T& a, const U& b, const std::size_t axis) :
                m_a(a), m_b(b), m_axis(axis)
            {
            }

            template <class... Args>
            value_type operator()(Args... args) const
            {
                std::vector<size_type> args_arr = {static_cast<size_type>(args)...};
                if (args_arr[m_axis] >= m_a.shape()[m_axis]) 
                {
                    args_arr[m_axis] -= m_a.shape()[m_axis];
                    return m_b[args_arr];
                }
                else
                {
                    return m_a[args_arr];
                }
            }

            value_type operator[](const xindex& idx) const
            {
                if (idx[m_axis] > m_a.shape()[m_axis]) 
                {
                    xindex temp(idx);
                    temp[m_axis] -= m_a.shape()[m_axis];
                    return m_b[temp];
                }
                else
                {
                    return m_a[idx];
                }
            }

        private:
            const T& m_a;
            const U& m_b;
            size_type m_axis;
        };

        template <class T, class U, class V = typename T::value_type>
        struct stack_impl
        {
            using size_type = std::size_t;
            using shape_type = std::vector<size_type>;
            using value_type = V;

            stack_impl(const T& a, const U& b, std::size_t axis) :
                m_a(a), m_b(b), m_axis(axis), m_a_shape(a.shape())
            {
                if (axis == 0 && m_a.dimension() == 1) {
                    m_a_shape.insert(m_a_shape.begin(), 1);
                }
                if (axis == 1 && m_a.dimension() == 1) {
                    m_axis = 0;
                }
            }

            template <class... Args>
            value_type operator()(Args... args) const
            {
                std::vector<size_type> args_arr = {static_cast<size_type>(args)...};
                if (args_arr[m_axis] >= m_a_shape[m_axis]) 
                {
                    args_arr[m_axis] -= m_a_shape[m_axis];
                    return m_b[args_arr];
                }
                else
                {
                    return m_a[args_arr];
                }
            }

            value_type operator[](const xindex& idx) const
            {
                if (idx[m_axis] > m_a_shape[m_axis]) 
                {
                    xindex temp(idx);
                    temp[m_axis] -= m_a_shape[m_axis];
                    return m_b[temp];
                }
                else
                {
                    return m_a[idx];
                }
            }

        private:
            const T& m_a;
            const U& m_b;
            size_type m_axis;
            shape_type m_a_shape;
        };
    }

    /**
     * @function concatentate
     * @brief Stack xexpressions horizontally.
     *
     * @param a xexpression to stack
     * @param b xexpression to stack
     * @returns xindex_function evaluating to stacked elements
     */
    template <class T, class U>
    inline auto concatenate(const T& a, const U& b, std::size_t axis = 0)
    {
        std::vector<std::size_t> new_shape(a.shape());
        new_shape[axis] += b.shape()[axis];
        return detail::make_xindex_function(detail::concatenate_impl<T, U>(a, b, axis), new_shape);
    }

    /**
     * @function hstack
     * @brief Stack xexpressions horizontally (column wise).
     *
     * @param a first xexpression to stack
     * @param b second xexpression to stack
     * @returns xindex_function evaluating to stacked elements
     */
    template <class T, class U>
    inline auto hstack(const T& a, const U& b) {
        std::vector<std::size_t> new_shape(a.shape());
        std::size_t axis = 1;
        // handle special case for 1D case
        if (a.dimension() == 1)
        {
            new_shape[0] += b.shape()[0];
        }
        else
        {
            new_shape[1] += b.shape()[1];
        }
        return detail::make_xindex_function(detail::stack_impl<T, U>(a, b, axis), new_shape);
    }

    /**
     * @function vstack
     * @brief Stack xexpressions vertically (column wise), adding an axis in the 1D case.
     *
     * @param a first xexpression to stack
     * @param b second xexpression to stack
     * @returns xindex_function evaluating to stacked elements
     */
    template <class T, class U>
    inline auto vstack(const T& a, const U& b) {
        std::vector<std::size_t> new_shape(a.shape());
        std::size_t axis = 0;
        // handle special case for 1D case
        if (a.dimension() == 1 || b.dimension() == 1)
        {
            if (a.dimension() == 1 && b.dimension() == 1)
            {
                new_shape.insert(new_shape.begin(), 2);
            }
            else
            {
                double shs = (1 + (a.dimension() == 1 ? b.shape()[0] : a.shape()[0]) );
                if (a.dimension() == 1)
                {
                    new_shape.insert(new_shape.begin(), 1 + b.shape()[0]);
                }
                else // b.dimension() == 1
                {
                    new_shape[0] += 1;
                }
            }
        }
        else
        {
            new_shape[0] += b.shape()[0];
        }
        return detail::make_xindex_function(detail::stack_impl<T, U>(a, b, axis), new_shape);
    }

    template <class T>
    inline auto arange(T start, T stop, T step = 1) noexcept
    {
        std::size_t shape = (stop - start) / step;
        return detail::make_xindex_function(detail::arange_impl<T>(start, stop, step), {shape});
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
        return detail::make_xindex_function(detail::arange_impl<T>(start, stop, step), {num_samples});
    }

    template <class T>
    inline auto logspace(T start, T stop, std::size_t num_samples, T base = 10, bool endpoint = true) noexcept
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