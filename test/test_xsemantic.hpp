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
#include "test_common.hpp"

namespace xt
{
    using std::size_t;
    using xarray_dynamic = xarray<int, layout_type::dynamic>;
    using xtensor_dynamic = xtensor<int, 3, layout_type::dynamic>;

    template <class F, class C>
    struct operation_tester
    {
        using storage_type = C;
        using shape_type = typename C::shape_type;

        storage_type a;
        storage_type ra;
        storage_type ca;
        storage_type cta;
        storage_type ua;

        storage_type res_rr;
        storage_type res_rc;
        storage_type res_rct;
        storage_type res_ru;

        operation_tester();
    };

    template <class F, class C>
    inline operation_tester<F, C>::operation_tester()
    {
        F f;
        row_major_result<shape_type> rmr;
        a.resize(rmr.shape(), rmr.strides());
        assign_array(a, rmr.m_assigner);
        ra.resize(rmr.shape(), rmr.strides());
        assign_array(ra, rmr.m_assigner);

        column_major_result<shape_type> cmr;
        ca.resize(cmr.shape(), cmr.strides());
        assign_array(ca, cmr.m_assigner);

        central_major_result<shape_type> ctmr;
        cta.resize(ctmr.shape(), ctmr.strides());
        assign_array(cta, ctmr.m_assigner);

        unit_shape_result<shape_type> usr;
        ua.resize(usr.shape(), usr.strides());
        assign_array(ua, usr.m_assigner);

        res_rr.resize(rmr.shape(), rmr.strides());
        res_rc.resize(rmr.shape(), rmr.strides());
        res_rct.resize(rmr.shape(), rmr.strides());
        res_ru.resize(rmr.shape(), rmr.strides());

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
        using storage_type = C;
        using shape_type = typename C::shape_type;

        int b;
        storage_type ra;
        storage_type ca;
        storage_type cta;
        storage_type ua;

        storage_type res_r;
        storage_type res_c;
        storage_type res_ct;
        storage_type res_u;

        scalar_operation_tester();
    };

    template <class F, class C>
    inline scalar_operation_tester<F, C>::scalar_operation_tester()
    {
        F f;
        b = 2;
        row_major_result<shape_type> rmr;
        ra.resize(rmr.shape(), rmr.strides());
        assign_array(ra, rmr.m_assigner);

        column_major_result<shape_type> cmr;
        ca.resize(cmr.shape(), cmr.strides());
        assign_array(ca, cmr.m_assigner);

        central_major_result<shape_type> ctmr;
        cta.resize(ctmr.shape(), ctmr.strides());
        assign_array(cta, ctmr.m_assigner);

        unit_shape_result<shape_type> usr;
        ua.resize(usr.shape(), usr.strides());
        assign_array(ua, usr.m_assigner);

        res_r.resize(rmr.shape(), rmr.strides());
        res_c.resize(cmr.shape(), cmr.strides());
        res_ct.resize(ctmr.shape(), ctmr.strides());
        res_u.resize(usr.shape(), usr.strides());

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
