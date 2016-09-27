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
        template <class E>
        std::ostream& output(std::ostream& out, const E& e)
        {
            if (e.dimension() == 0)
            {
                out << e();
            }
            else
            {
                size_t i = 0;
                out << '{';
                for (;i != e.shape()[0] - 1; ++i)
                    out << make_xview(e, i) << ',' << ' ';
                return out << make_xview(e, i) << '}';
            }
            return out;
        }
    }

    template <class E>
    inline std::ostream& operator<<(std::ostream& out, const xexpression<E>& e)
    {
        return detail::output(out, e.derived_cast());
    }

}

#endif

