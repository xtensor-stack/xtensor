/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_EVAL_HPP
#define XTENSOR_EVAL_HPP

#include "xexpression_traits.hpp"
#include "xtensor_forward.hpp"
#include "xshape.hpp"

namespace xt
{

    namespace detail
    {
        template <class T>
        using is_container = std::is_base_of<xcontainer<std::remove_const_t<T>>, T>;
    }

    /**
     * Force evaluation of xexpression.
     * @return xarray or xtensor depending on shape type
     *
     * \code{.cpp}
     * xarray<double> a = {1,2,3,4};
     * auto&& b = xt::eval(a); // b is a reference to a, no copy!
     * auto&& c = xt::eval(a + b); // c is xarray<double>, not an xexpression
     * \endcode
     */
    template <class T>
    inline auto eval(T&& t)
        -> std::enable_if_t<detail::is_container<std::decay_t<T>>::value, T&&>
    {
        return std::forward<T>(t);
    }

    /// @cond DOXYGEN_INCLUDE_SFINAE
    template <class T>
    inline auto eval(T&& t)
        -> std::enable_if_t<!detail::is_container<std::decay_t<T>>::value, temporary_type_t<T>>
    {
        return std::forward<T>(t);
    }
}

#endif
