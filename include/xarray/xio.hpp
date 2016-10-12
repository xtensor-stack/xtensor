/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XIO_HPP
#define XIO_HPP

#include <cstddef>
#include <iostream>

#include "xexpression.hpp"
#include "xview.hpp"

namespace xt
{

    template <class E>
    inline std::ostream& operator<<(std::ostream& out, const xexpression<E>& e);

    /**************************************
     * xexpression ostream implementation *
     **************************************/

    namespace detail
    {
        template <std::size_t I>
        struct xout
        {
            template <class E>
            static std::ostream& output(std::ostream& out, const E& e)
            {
                if (e.dimension() == 0)
                {
                    out << e();
                }
                else
                {
                    typename E::size_type i = 0;
                    out << '{';
                    for (;i != e.shape()[0] - 1; ++i)
                        xout<I-1>::output(out, make_xview(e, i)) << ',' << ' ';
                    xout<I-1>::output(out, make_xview(e, i)) << '}';
                }
                return out;
            }
        };

        template <>
        struct xout<0>
        {
            template <class E>
            static std::ostream& output(std::ostream& out, const E&)
            {
                return out << "...";
            }
        };
    }

    template <class E>
    inline std::ostream& operator<<(std::ostream& out, const xexpression<E>& e)
    {
        return detail::xout<5>::output(out, e.derived_cast());
    }
}

#endif
