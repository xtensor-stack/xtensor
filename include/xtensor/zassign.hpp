/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZASSIGN_HPP
#define XTENSOR_ZASSIGN_HPP

#include "xassign.hpp"
#include "zarray_impl.hpp"

namespace xt
{
    template <>
    class xexpression_assigner<zarray_expression_tag>
    {
    public:

        template <class E1, class E2>
        static void assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2)
        {
            std::unique_ptr<zarray_impl> res_impl = e2.derived_cast().allocate_result();
            e2.derived_cast().assign_to(*res_impl);
            e1.derived_cast() = std::move(res_impl);
        }
    };

}

#endif

