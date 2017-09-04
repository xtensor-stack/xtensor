/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef TEST_XSEMANTIC_HPP
#define TEST_XSEMANTIC_HPP

#include <functional>

#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

#include "xtensor/testing/test_common.hpp"

namespace xt
{
    using std::size_t;
    using xarray_dynamic = xarray<int, layout_type::dynamic>;
    using xtensor_dynamic = xtensor<int, 3, layout_type::dynamic>;

    template <class F, class C>
    struct operation_tester
    {
        using container_type = C;
        using shape_type = typename C::shape_type;

        container_type a;
        container_type ra;
        container_type ca;
        container_type cta;
        container_type ua;

        container_type res_rr;
        container_type res_rc;
        container_type res_rct;
        container_type res_ru;

        operation_tester();
    };

    template <class F, class C>
    inline operation_tester<F, C>::operation_tester()
    {
        F f;
        row_major_result<shape_type> rmr;
        a.reshape(rmr.shape(), rmr.strides());
        assign_array(a, rmr.m_assigner);
        ra.reshape(rmr.shape(), rmr.strides());
        assign_array(ra, rmr.m_assigner);

        column_major_result<shape_type> cmr;
        ca.reshape(cmr.shape(), cmr.strides());
        assign_array(ca, cmr.m_assigner);

        central_major_result<shape_type> ctmr;
        cta.reshape(ctmr.shape(), ctmr.strides());
        assign_array(cta, ctmr.m_assigner);

        unit_shape_result<shape_type> usr;
        ua.reshape(usr.shape(), usr.strides());
        assign_array(ua, usr.m_assigner);

        res_rr.reshape(rmr.shape(), rmr.strides());
        res_rc.reshape(rmr.shape(), rmr.strides());
        res_rct.reshape(rmr.shape(), rmr.strides());
        res_ru.reshape(rmr.shape(), rmr.strides());

        for (size_t i = 0; i < rmr.shape()[0]; ++i)
        {
            for (size_t j = 0; j < rmr.shape()[1]; ++j)
            {
                for (size_t k = 0; k < rmr.shape()[2]; ++k)
                {
                    res_rr(i, j, k) = f(a(i, j, k), ra(i, j, k));
                    res_rc(i, j, k) = f(a(i, j, k), ca(i, j, k));
                    res_rct(i, j, k) = f(a(i, j, k), cta(i, j, k));
                    res_ru(i, j, k) = f(a(i, j, k), ua(i, j, k));
                }
            }
        }
    }

    template <class F, class C>
    struct scalar_operation_tester
    {
        using container_type = C;
        using shape_type = typename C::shape_type;

        int b;
        container_type ra;
        container_type ca;
        container_type cta;
        container_type ua;

        container_type res_r;
        container_type res_c;
        container_type res_ct;
        container_type res_u;

        scalar_operation_tester();
    };

    template <class F, class C>
    inline scalar_operation_tester<F, C>::scalar_operation_tester()
    {
        F f;
        b = 2;
        row_major_result<shape_type> rmr;
        ra.reshape(rmr.shape(), rmr.strides());
        assign_array(ra, rmr.m_assigner);

        column_major_result<shape_type> cmr;
        ca.reshape(cmr.shape(), cmr.strides());
        assign_array(ca, cmr.m_assigner);

        central_major_result<shape_type> ctmr;
        cta.reshape(ctmr.shape(), ctmr.strides());
        assign_array(cta, ctmr.m_assigner);

        unit_shape_result<shape_type> usr;
        ua.reshape(usr.shape(), usr.strides());
        assign_array(ua, usr.m_assigner);

        res_r.reshape(rmr.shape(), rmr.strides());
        res_c.reshape(cmr.shape(), cmr.strides());
        res_ct.reshape(ctmr.shape(), ctmr.strides());
        res_u.reshape(usr.shape(), usr.strides());

        for (size_t i = 0; i < rmr.shape()[0]; ++i)
        {
            for (size_t j = 0; j < rmr.shape()[1]; ++j)
            {
                for (size_t k = 0; k < rmr.shape()[2]; ++k)
                {
                    res_r(i, j, k) = f(ra(i, j, k), b);
                    res_c(i, j, k) = f(ca(i, j, k), b);
                    res_ct(i, j, k) = f(cta(i, j, k), b);
                    res_u(i, j, k) = f(ua(i, j, k), b);
                }
            }
        }
    }
}

#endif
