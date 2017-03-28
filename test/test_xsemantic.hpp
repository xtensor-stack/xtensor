/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef TEST_XSEMANTIC_HPP
#define TEST_XSEMANTIC_HPP

#include <functional>
#include "xtensor/xarray.hpp"
#include "test_common.hpp"

namespace xt
{
    using std::size_t;

    template <class F>
    struct operation_tester
    {
        xarray<int> a;
        xarray<int> ra;
        xarray<int> ca;
        xarray<int> cta;
        xarray<int> ua;

        xarray<int> res_rr;
        xarray<int> res_rc;
        xarray<int> res_rct;
        xarray<int> res_ru;

        operation_tester();
    };

    template <class F>
    inline operation_tester<F>::operation_tester()
    {
        F f;
        row_major_result<> rmr;
        a.reshape(rmr.shape(), rmr.strides());
        assign_array(a, rmr.m_assigner);
        ra.reshape(rmr.shape(), rmr.strides());
        assign_array(ra, rmr.m_assigner);

        column_major_result<> cmr;
        ca.reshape(cmr.shape(), cmr.strides());
        assign_array(ca, cmr.m_assigner);

        central_major_result<> ctmr;
        cta.reshape(ctmr.shape(), ctmr.strides());
        assign_array(cta, ctmr.m_assigner);

        unit_shape_result<> usr;
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

    template <class F>
    struct scalar_operation_tester
    {
        int b;
        xarray<int> ra;
        xarray<int> ca;
        xarray<int> cta;
        xarray<int> ua;

        xarray<int> res_r;
        xarray<int> res_c;
        xarray<int> res_ct;
        xarray<int> res_u;

        scalar_operation_tester();
    };

    template <class F>
    inline scalar_operation_tester<F>::scalar_operation_tester()
    {
        F f;
        b = 2;
        row_major_result<> rmr;
        ra.reshape(rmr.shape(), rmr.strides());
        assign_array(ra, rmr.m_assigner);

        column_major_result<> cmr;
        ca.reshape(cmr.shape(), cmr.strides());
        assign_array(ca, cmr.m_assigner);

        central_major_result<> ctmr;
        cta.reshape(ctmr.shape(), ctmr.strides());
        assign_array(cta, ctmr.m_assigner);

        unit_shape_result<> usr;
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
                    res_c(i, j, k) = f( ca(i, j, k), b);
                    res_ct(i, j, k) = f(cta(i, j, k), b);
                    res_u(i, j, k) = f(ua(i, j, k), b);
                }
            }
        }
    }
}

#endif

