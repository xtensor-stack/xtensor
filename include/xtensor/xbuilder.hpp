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

}
#endif

