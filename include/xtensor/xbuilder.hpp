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
#include <functional>
#include <random>

#include "xfunction.hpp"
#include "xarray.hpp"
#include "xbroadcast.hpp"

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

    namespace random
    {
        // default random number engine
        static std::mt19937 mt;

        template <class T, class RE = decltype(mt)>
        inline auto rand(std::vector<std::size_t> shape, T lower = 0, T upper = 1, RE engine = mt) noexcept
        {
            std::uniform_real_distribution<T> dist(lower, upper);
            return xt::xarray<T>(shape, std::bind(dist, engine));
        }

        template <class T, class RE = decltype(mt)>
        inline auto randint(std::vector<std::size_t> shape, T lower, T upper, RE engine = mt) noexcept
        {
            std::uniform_int_distribution<T> dist(lower, upper);
            return xt::xarray<T>(shape, std::bind(dist, engine));
        }
    }

}
#endif

