/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xview.hpp"
#include "test_xsemantic.hpp"

namespace xt
{

    template <class F>
    struct view_op_tester : operation_tester<F>
    {
        xarray<int> vres_rr;
        xarray<int> vres_rc;
        xarray<int> vres_rct;
        xarray<int> vres_ru;

        size_t x_slice;
        xrange<size_t> y_slice;
        xrange<size_t> z_slice;

        view_op_tester();
    };

    template <class F>
    view_op_tester<F>::view_op_tester()
        : operation_tester<F>(), x_slice(0),
        y_slice(0, 2), z_slice(1, 4)
    {
        xshape<size_t> shape = this->a.shape();
        vres_rr = this->a;
        vres_rc = this->a;
        vres_rct = this->a;
        vres_ru = this->a;

        size_t imax = y_slice.size();
        size_t jmax = z_slice.size();
        for(size_t i = 0; i < imax; ++i)
        {
            for(size_t j = 0; j < jmax; ++j)
            {
                size_t si = y_slice(i);
                size_t sj = z_slice(j);
                vres_rr(x_slice, si, sj) = this->res_rr(x_slice, y_slice(i), z_slice(j));
                vres_rc(x_slice, i, j) = this->res_rc(x_slice, y_slice(i), z_slice(j));
                vres_rct(x_slice, i, j) = this->res_rct(x_slice, y_slice(i), z_slice(j));
                vres_ru(x_slice, i, j) = this->res_ru(x_slice, y_slice(i), z_slice(j));
            }
        }
    }

    TEST(xview_semantic, a_plus_b)
    {
        view_op_tester<std::plus<>> t;

        {
            SCOPED_TRACE("row_major + row_major");
            xarray<int> b = t.a;
            auto viewb = make_xview(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewa = make_xview(t.a, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = make_xview(t.ra, t.x_slice, t.y_slice, t.z_slice);

            viewb = viewa + viewra;

            //EXPECT_EQ(t.vres_rr, b);
        }
    }
}

