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
#include "xtensor/xnoalias.hpp"
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
        std::vector<size_t> shape = this->a.shape();
        vres_rr = this->a;
        vres_rc = this->a;
        vres_rct = this->a;
        vres_ru = this->a;

        size_t imax = y_slice.size();
        size_t jmax = z_slice.size();
        for (size_t i = 0; i < imax; ++i)
        {
            for (size_t j = 0; j < jmax; ++j)
            {
                size_t si = y_slice(i);
                size_t sj = z_slice(j);
                vres_rr(x_slice, si, sj) = this->res_rr(x_slice, si, sj);
                vres_rc(x_slice, si, sj) = this->res_rc(x_slice, si, sj);
                vres_rct(x_slice, si, sj) = this->res_rct(x_slice, si, sj);
                vres_ru(x_slice, si, sj) = this->res_ru(x_slice, si, sj);
            }
        }
    }

    TEST(xview_semantic, a_plus_b)
    {
        view_op_tester<std::plus<>> t;
        auto viewa = view(t.a, t.x_slice, t.y_slice, t.z_slice);

        {
            SCOPED_TRACE("row_major + row_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa + viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major + column_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa + viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major + central_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa + viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major + unit_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa + viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }
    
    TEST(xview_semantic, a_minus_b)
    {
        view_op_tester<std::minus<>> t;
        auto viewa = view(t.a, t.x_slice, t.y_slice, t.z_slice);

        {
            SCOPED_TRACE("row_major - row_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa - viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major - column_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa - viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major - central_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa - viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major - unit_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa - viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TEST(xview_semantic, a_times_b)
    {
        view_op_tester<std::multiplies<>> t;
        auto viewa = view(t.a, t.x_slice, t.y_slice, t.z_slice);

        {
            SCOPED_TRACE("row_major * row_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa * viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major * column_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa * viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major * central_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa * viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major * unit_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa * viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TEST(xview_semantic, a_divdide_by_b)
    {
        view_op_tester<std::divides<>> t;
        auto viewa = view(t.a, t.x_slice, t.y_slice, t.z_slice);

        {
            SCOPED_TRACE("row_major / row_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa / viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major / column_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa / viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major / central_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa / viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major / unit_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb = viewa / viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TEST(xview_semantic, a_plus_equal_b)
    {
        view_op_tester<std::plus<>> t;
        auto viewa = view(t.a, t.x_slice, t.y_slice, t.z_slice);

        {
            SCOPED_TRACE("row_major += row_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb += viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major += column_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb += viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major += central_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb += viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major += unit_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb += viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TEST(xview_semantic, a_minus_equal_b)
    {
        view_op_tester<std::minus<>> t;
        auto viewa = view(t.a, t.x_slice, t.y_slice, t.z_slice);

        {
            SCOPED_TRACE("row_major -= row_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb -= viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major -= column_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb -= viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major -= central_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb -= viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major -= unit_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb -= viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TEST(xview_semantic, a_times_equal_b)
    {
        view_op_tester<std::multiplies<>> t;
        auto viewa = view(t.a, t.x_slice, t.y_slice, t.z_slice);

        {
            SCOPED_TRACE("row_major *= row_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb *= viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major *= column_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb *= viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major *= central_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb *= viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major *= unit_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb *= viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }
    
    TEST(xview_semantic, a_divide_by_equal_b)
    {
        view_op_tester<std::divides<>> t;
        auto viewa = view(t.a, t.x_slice, t.y_slice, t.z_slice);

        {
            SCOPED_TRACE("row_major /= row_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewra = view(t.ra, t.x_slice, t.y_slice, t.z_slice);
            viewb /= viewra;
            EXPECT_EQ(t.vres_rr, b);
        }

        {
            SCOPED_TRACE("row_major /= column_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewca = view(t.ca, t.x_slice, t.y_slice, t.z_slice);
            viewb /= viewca;
            EXPECT_EQ(t.vres_rc, b);
        }

        {
            SCOPED_TRACE("row_major /= central_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewcta = view(t.cta, t.x_slice, t.y_slice, t.z_slice);
            viewb /= viewcta;
            EXPECT_EQ(t.vres_rct, b);
        }

        {
            SCOPED_TRACE("row_major /= unit_major");
            xarray<int> b = t.a;
            auto viewb = view(b, t.x_slice, t.y_slice, t.z_slice);
            auto viewua = view(t.ua, t.x_slice, t.y_slice, t.z_slice);
            viewb /= viewua;
            EXPECT_EQ(t.vres_ru, b);
        }
    }

    TEST(xview_semantic, broadcast_equal)
    {
        xarray<int> a = { {1,  2,  3,  4},
                          {5,  6,  7,  8},
                          {9, 10, 11, 12} };
        xarray<int> b = a;
        auto viewa = view(a, all(), range(1, 4));
        auto viewb = view(b, all(), range(1, 4));
        xarray<int> c = {1, 2, 3};
        viewa = c;
        noalias(viewb) = c;
        xarray<int> res = { {1, 1, 2, 3},
                            {5, 1, 2, 3},
                            {9, 1, 2, 3} };

        EXPECT_EQ(res, a);
        EXPECT_EQ(res, b);
    }

    TEST(xview_semantic, scalar_equal)
    {
        xarray<int> a = { { 1,  2,  3,  4 },
                          { 5,  6,  7,  8 },
                          { 9, 10, 11, 12 } };
        auto viewa = view(a, all(), range(1, 4));
        int b = 1;
        viewa = b;
        xarray<int> res = { { 1, 1, 1, 1 },
                            { 5, 1, 1, 1 },
                            { 9, 1, 1, 1 } };

        EXPECT_EQ(res, a);
    }
}

