/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XACCUMULATOR_HPP
#define XACCUMULATOR_HPP

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <tuple>

#ifdef X_OLD_CLANG
#include <vector>
#endif

#include "xtl/xsequence.hpp"

#include "xbuilder.hpp"
#include "xreducer.hpp"
#include "xexpression.hpp"
#include "xgenerator.hpp"
#include "xiterable.hpp"
#include "xutils.hpp"
#include <iostream>

namespace xt
{
    /**************
     * accumulate *
     **************/

    template <class F, class E>
    xarray<typename std::decay_t<E>::value_type> accumulate_greedy(F&& f, E&& e)
    {
        using T = typename std::decay_t<E>::value_type;
        std::size_t sz = compute_size(e.shape());
        auto result = xarray<T>::from_shape({sz});
        result[0] = e[0];
        for (std::size_t i = 0; i < sz - 1; ++i)
        {
            result[i + 1] = f(result[i], e[i + 1]);
        }
        return result;
    }

    template <class F, class E>
    xarray<typename std::decay_t<E>::value_type> accumulate_greedy(F&& f, E&& e, std::size_t axis)
    {
        using T = typename std::decay_t<E>::value_type;
        std::size_t sz = compute_size(e.shape());

        xarray<T> result = e;  // assign + make a copy, we need it anyways

        std::size_t outer_stride = result.strides().back();
        std::size_t inner_stride = result.strides()[axis];

        std::size_t inner_loop_dim = 1, outer_loop_dim = 1;

        std::size_t i = 0;
        for (; i < axis; ++i)
        {
            outer_loop_dim *= result.shape()[i];
        }

        for (; i < result.dimension(); ++i)
        {
            inner_loop_dim *= result.shape()[i];
        }

        inner_loop_dim -= inner_stride;

        std::size_t pos = 0;
        for (std::size_t i = 0; i < outer_loop_dim; ++i)
        {
            for (std::size_t j = 0; j < inner_loop_dim; ++j)
            {
                result[pos + inner_stride] = f(result[pos], result[pos + inner_stride]);
                pos += outer_stride;
            }
            pos += inner_stride;
        }

        return result;
    }


    /**
     * Accumulate and flatten array
     * **NOTE** This function is not lazy!
     *
     * @param f Functor to use for accumulation
     * @param e xexpression to be accumulated
     *
     * @return returns xarray<T> filled with accumulated values
     */
    template <class F, class E>
    auto accumulate(F&& f, E&& e)
    {
        return accumulate_greedy(std::forward<F>(f), std::forward<E>(e));
    }

    /**
     * Accumulate over axis
     * **NOTE** This function is not lazy!
     *
     * @param f Functor to use for accumulation
     * @param e xexpression to accumulate
     * @param axis Axis to perform accumulation over
     *
     * @return returns xarray<T> filled with accumulated values
     */
    template <class F, class E>
    auto accumulate(F&& f, E&& e, std::size_t axis)
    {
        return accumulate_greedy(std::forward<F>(f), std::forward<E>(e), axis);
    }

}

#endif
