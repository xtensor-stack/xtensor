/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XCOMPLEX_HPP
#define XCOMPLEX_HPP

#include <utility>
#include <type_traits>

#include "xtensor/xoffsetview.hpp"
#include "xtensor/xbuilder.hpp"

namespace xt
{

    /******************************
     * real and imag declarations *
     ******************************/


    /********************************
     * real and imag implementation *
     ********************************/

    /**
     * @brief Returns an \ref xexpression representing the real part of the given expression.
     *
     * @tparam e the \ref xexpression
     *
     * The returned expression either hold a const reference to \p e or a copy
     * depending on whether \p e is an lvalue or an rvalue.
     */
    template <class E>
    std::enable_if_t<is_complex<typename std::decay_t<E>::value_type>::value, 
                     xoffsetview<xclosure_t<E>,
                                 typename std::decay_t<E>::value_type::value_type, 
                                 0>>
    real(E&& e) noexcept
    {
        using real_type = typename std::decay_t<E>::value_type::value_type;
        return xoffsetview<xclosure_t<E>, real_type, 0>(std::forward<E>(e));
    }

    template <class E>
    std::enable_if_t<!is_complex<typename std::decay_t<E>::value_type>::value,
                     detail::forward_type_t<E, E>>
    real(E&& e) noexcept
    {
        return e;
    }

    /**
     * @brief Returns an \ref xexpression representing the imaginary part of the given expression.
     *
     * @tparam e the \ref xexpression
     *
     * The returned expression either hold a const reference to \p e or a copy
     * depending on whether \p e is an lvalue or an rvalue.
     */
    template <class E>
    std::enable_if_t<is_complex<typename std::decay_t<E>::value_type>::value,
                     xoffsetview<xclosure_t<E>,
                                 typename std::decay_t<E>::value_type::value_type, 
                                 sizeof(typename std::decay_t<E>::value_type::value_type)>>
    imag(E&& e) noexcept
    {
        using real_type = typename std::decay_t<E>::value_type::value_type;
        return xoffsetview<xclosure_t<E>, real_type, sizeof(real_type)>(std::forward<E>(e));
    }

    template <class E>
    std::enable_if_t<!is_complex<typename std::decay_t<E>::value_type>::value,
                     xbroadcast<xscalar<const typename std::decay_t<E>::value_type>, 
                                typename std::decay_t<E>::shape_type>>
    imag(E&& e) noexcept
    {
        return zeros<typename std::decay_t<E>::value_type>(e.shape());
    }

}
#endif

