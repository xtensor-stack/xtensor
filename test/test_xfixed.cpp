/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xfixed.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

// On VS2015, when compiling in x86 mode, alignas(T) leads to C2718
// when used for a function parameter, even indirectly. This means that
// we cannot pass parameters whose class is declared with alignas specifier
// or any type wrapping or inheriting from such a type.
// The xtensorf class internally uses aligned_array which is declared as
// alignas(something_different_from_0), hence the workaround.
#if _MSC_VER < 1910 && !_WIN64
#define VS_X86_WORKAROUND 1
#endif

#if _MSC_VER < 1910 || (_MSC_VER >= 1910 && !defined(DISABLE_VS2017))

namespace xt
{
    using xtensorf3x4 = xtensorf<double, xt::xshape<3, 4>>;
    using xtensorf4 = xtensorf<double, xt::xshape<4>>;

    TEST(xtensorf, basic)
    {
        xtensorf3x4 a({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
        xtensorf3x4 b({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});

        xtensorf3x4 res1 = a + b;
        xtensorf3x4 res2 = 2.0 * a;

#ifndef VS_X86_WORKAROUND
        EXPECT_EQ(res1, res2);
#else
        bool res = std::equal(res1.cbegin(), res1.cend(), res2.cbegin());
        EXPECT_TRUE(res);
#endif
    }

    TEST(xtensorf, broadcast)
    {
        xtensorf3x4 a({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
        xtensorf4 b({4, 5, 6, 7});

        xtensorf3x4 res = a * b;
        xtensorf3x4 resb = b * a;

        xarray<double> ax = a;
        xarray<double> bx = b;
        xarray<double> arx = a * b;
        xarray<double> brx = b * a;

#ifndef VS_X86_WORKAROUND
        EXPECT_EQ(res, arx);
        EXPECT_EQ(resb, brx);
#else
        bool bresa = std::equal(res.cbegin(), res.cend(), arx.cbegin());
        EXPECT_TRUE(bresa);
        bool bresb = std::equal(resb.cbegin(), resb.cend(), brx.cbegin());
        EXPECT_TRUE(bresb);
#endif

#ifdef XTENSOR_ENABLE_ASSERT
        EXPECT_THROW(a.resize({2, 2}), std::runtime_error);
#endif
        // reshaping fixed container
        EXPECT_THROW(a.reshape({1, 9}), std::runtime_error);
        EXPECT_NO_THROW(a.reshape({3, 4}));
        EXPECT_NO_THROW(a.reshape({3, 4}, DEFAULT_LAYOUT));
        EXPECT_THROW(a.reshape({3, 4}, layout_type::any), std::runtime_error);
    }

    TEST(xtensorf, strides)
    {
        xtensorf<double, xshape<3, 7, 2, 5, 3>, layout_type::row_major> arm;
        xtensor<double, 5, layout_type::row_major> brm = xtensor<double, 5, layout_type::row_major>::from_shape({3, 7, 2, 5, 3});

        EXPECT_TRUE(std::equal(arm.strides().begin(), arm.strides().end(), brm.strides().begin()));
        EXPECT_EQ(arm.strides().size(), brm.strides().size());
        EXPECT_TRUE(std::equal(arm.backstrides().begin(), arm.backstrides().end(), brm.backstrides().begin()));
        EXPECT_EQ(arm.backstrides().size(), brm.backstrides().size());
        EXPECT_EQ(arm.size(), 3 * 7 * 2 * 5 * 3);

        xtensorf<double, xshape<3, 7, 2, 5, 3>, layout_type::column_major> acm;
        xtensor<double, 5, layout_type::column_major> bcm = xtensor<double, 5, layout_type::column_major>::from_shape({3, 7, 2, 5, 3});

        EXPECT_TRUE(std::equal(acm.strides().begin(), acm.strides().end(), bcm.strides().begin()));
        EXPECT_EQ(acm.strides().size(), bcm.strides().size());
        EXPECT_TRUE(std::equal(acm.backstrides().begin(), acm.backstrides().end(), bcm.backstrides().begin()));
        EXPECT_EQ(acm.backstrides().size(), bcm.backstrides().size());
        EXPECT_EQ(acm.size(), 3 * 7 * 2 * 5 * 3);

        auto s = get_strides<layout_type::row_major>(xshape<3, 4, 5>());
        EXPECT_EQ(s[0], 20);
        EXPECT_EQ(s[1], 5);
        EXPECT_EQ(s[2], 1);

        auto sc = get_strides<layout_type::column_major>(xshape<3, 4, 5>());
        EXPECT_EQ(sc[0], 1);
        EXPECT_EQ(sc[1], 3);
        EXPECT_EQ(sc[2], 12);
    }

    TEST(xtensorf, adapt)
    {
        std::vector<double> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::vector<double> b = a;

        xfixed_adaptor<std::vector<double>&, xt::xshape<3, 4>> ad(a);
        auto bd = adapt(b, std::array<std::size_t, 2>{3, 4});

        EXPECT_EQ(ad.layout(), DEFAULT_LAYOUT);

        EXPECT_EQ(ad(1, 1), bd(1, 1));
        auto expr = ad + bd;
        EXPECT_EQ(expr(1, 1), bd(1, 1) * 2);
        ad = bd * 2;
        EXPECT_EQ(bd(1, 1) * 2, ad(1, 1));
        EXPECT_EQ(a[0], 2);
    }

    TEST(xtensorf, layout)
    {
        xtensorf<double, xshape<2, 2>, layout_type::row_major> a;
        EXPECT_EQ(a.layout(), layout_type::row_major);
        xtensorf<double, xshape<2, 2>, layout_type::column_major> b;
        EXPECT_EQ(b.layout(), layout_type::column_major);
    }
}

#endif