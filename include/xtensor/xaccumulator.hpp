/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ACCUMULATOR_HPP
#define XTENSOR_ACCUMULATOR_HPP

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <cstddef>

#include "xtensor_forward.hpp"
#include "xstrides.hpp"
#include "xexpression.hpp"
#include <iostream>

namespace xt
{

#define DEFAULT_STRATEGY_ACCUMULATORS evaluation_strategy::immediate

    /**************
     * accumulate *
     **************/

    namespace detail
    {
        template <class F, class E, class ES>
        xarray<typename std::decay_t<E>::value_type> accumulator_impl(F&&, E&&, std::size_t, ES)
        {
            static_assert(!std::is_same<evaluation_strategy::lazy, ES>::value, "Lazy accumulators not yet implemented.");
        }

        template <class F, class E, class ES>
        xarray<typename std::decay_t<E>::value_type> accumulator_impl(F&&, E&&, ES)
        {
            static_assert(!std::is_same<evaluation_strategy::lazy, ES>::value, "Lazy accumulators not yet implemented.");
        }

        template <class F, class E>
        inline auto accumulator_impl(F&& f, E&& e, std::size_t axis, evaluation_strategy::immediate)
        {
            using T = typename F::result_type;

            if (axis >= e.dimension())
            {
                throw std::runtime_error("Axis larger than expression dimension in accumulator.");
            }

            // Investigate if doing a trick with transpose is better here: by transposing the
            // xarray such that the accumulation axis is last, it should be cache friendly
            xarray<T, layout_type::row_major> result = e;  // assign + make a copy, we need it anyways

            std::size_t outer_stride = result.strides().back();
            std::size_t inner_stride = result.strides()[axis];

            std::size_t outer_loop_size = std::accumulate(result.shape().cbegin(),
                                                          result.shape().cbegin() + std::ptrdiff_t(axis),
                                                          std::size_t(1), std::multiplies<std::size_t>());
            std::size_t inner_loop_size = std::accumulate(result.shape().cbegin() + std::ptrdiff_t(axis),
                                                          result.shape().cend(),
                                                          std::size_t(1), std::multiplies<std::size_t>());
            inner_loop_size -= inner_stride;

            std::size_t pos = 0;

            for (std::size_t i = 0; i < outer_loop_size; ++i)
            {
                for (std::size_t j = 0; j < inner_loop_size; ++j)
                {
                    result.data()[pos + inner_stride] = f(result.data()[pos],
                                                          result.data()[pos + inner_stride]);
                    pos += outer_stride;
                }
                pos += inner_stride;
            }
            return result;
        }

        template <class F, class E>
        inline auto accumulator_impl(F&& f, E&& e, evaluation_strategy::immediate)
        {
            using T = typename F::result_type;
            std::size_t sz = e.size();

            // if layout == row_major, avoid a copy
            if (e.layout() == layout_type::row_major)
            {
                xarray<T, layout_type::row_major> result = xarray<T, layout_type::row_major>::from_shape({sz});
                result.data()[0] = e.data()[0];
                for (std::size_t i = 0; i < sz - 1; ++i)
                {
                    result.data()[i + 1] = f(result.data()[i], e.data()[i + 1]);
                }
                return result;
            }
            else
            {
                xarray<T, layout_type::row_major> result = e;
                result.reshape({sz});

                for (std::size_t i = 0; i < sz - 1; ++i)
                {
                    result.data()[i + 1] = f(result.data()[i], result.data()[i + 1]);
                }
                return result;
            }
        }
    }

    /**
     * Accumulate and flatten array
     * **NOTE** This function is not lazy!
     *
     * @param f functor to use for accumulation
     * @param e xexpression to be accumulated
     * @param evaluation_strategy evaluation strategy of the accumulation
     *
     * @return returns xarray<T> filled with accumulated values
     */
    template <class F, class E, class ES = DEFAULT_STRATEGY_ACCUMULATORS,
              typename std::enable_if_t<!std::is_integral<ES>::value, int> = 0>
    inline auto accumulate(F&& f, E&& e, ES evaluation_strategy = ES())
    {
        // Note we need to check is_integral above in order to prohibit ES = int, and not taking the std::size_t
        // overload below!
        return detail::accumulator_impl(std::forward<F>(f), std::forward<E>(e), evaluation_strategy);
    }

    /**
     * Accumulate over axis
     * **NOTE** This function is not lazy!
     *
     * @param f Functor to use for accumulation
     * @param e xexpression to accumulate
     * @param axis Axis to perform accumulation over
     * @param evaluation_strategy evaluation strategy of the accumulation
     *
     * @return returns xarray<T> filled with accumulated values
     */
    template <class F, class E, class ES = DEFAULT_STRATEGY_ACCUMULATORS>
    inline auto accumulate(F&& f, E&& e, std::size_t axis, ES evaluation_strategy = ES())
    {
        return detail::accumulator_impl(std::forward<F>(f), std::forward<E>(e), axis, evaluation_strategy);
    }

}

#endif