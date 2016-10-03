#ifndef XIO_HPP
#define XIO_HPP

#include <iostream>
#include "xexpression.hpp"
#include "xview.hpp"

namespace qs
{

    /*********************************
     * Representing xexpressions
     *********************************/

    namespace detail
    {
        template <size_t I>
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
                    size_t i = 0;
                    out << '{';
                    for (;i != e.shape()[0] - 1; i++)
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
        return detail::xout<3>::output(out, e.derived_cast());
    }

}

#endif
