/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XLAYOUT_HPP
#define XLAYOUT_HPP

namespace xt
{
    /*! Layout enum for xcontainer based xexpressions */
    enum class layout
    {
        dynamic = 0x00, /*! dynamic layout: you can reshape to row major, column major, or use custom strides */
        any = 0xFF, /*! layout compatible with all others */
        row_major = 0x01, /*! row major layout */
        column_major = 0x02 /*! column major layout */
    };

    /**
     * Implementation of the following logical table:
     *
     *   | d | a | r | c |
     * --+---+---+---+---+
     * d | d | d | d | d |
     * a | d | a | r | c |
     * r | d | r | r | d |
     * c | d | c | d | c |
     *
     * d = dynamic, a = any, r = row_major, c = column_major.
     * Using bitmasks to avoid nested if-else statements.
     */
    template <class... Args>
    constexpr layout compute_layout(Args... args) noexcept;

    /******************
     * Implementation *
     ******************/

    namespace detail
    {
        constexpr layout compute_layout_impl() noexcept
        {
            return xt::layout::any;
        }

        constexpr layout compute_layout_impl(layout l) noexcept
        {
            return l;
        }

        constexpr layout compute_layout_impl(layout lhs, layout rhs) noexcept
        {
            using type = std::underlying_type_t<layout>;
            return layout(static_cast<type>(lhs) & static_cast<type>(rhs));
        }

        template <class... Args>
        constexpr layout compute_layout_impl(layout lhs, Args... args) noexcept
        {
            return compute_layout_impl(lhs, compute_layout_impl(args...));
        }
    }

    template <class... Args>
    constexpr layout compute_layout(Args... args) noexcept
    {
        return detail::compute_layout_impl(args...);
    }


}

#endif
