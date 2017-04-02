/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xtensor_config.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xview.hpp"

namespace xt
{

    TEST(xeval, has_container_base)
    {
        xarray<double> a = {1,2,3,4};
        xtensor<double, 2> b = {{1,2,3}, {4,5,6}};
        auto c = linspace(0, 10);
        auto va = view(a, range(0, 2));
        auto vb = view(b, range(0, 2));
        auto vc = view(c, range(0, 2));

        EXPECT_TRUE(detail::has_container_base<decltype(a)>::value);
        EXPECT_TRUE(detail::has_container_base<decltype(b)>::value);
        EXPECT_TRUE(detail::has_container_base<decltype(va)>::value);
        EXPECT_TRUE(detail::has_container_base<decltype(vb)>::value);
        EXPECT_FALSE(detail::has_container_base<decltype(c)>::value);
        EXPECT_FALSE(detail::has_container_base<decltype(vc)>::value);
        EXPECT_FALSE(detail::has_container_base<decltype(a * b)>::value);
    }

    TEST(xeval, array_tensor)
    {
        xarray<double> a = {1,2,3,4};

        auto&& b = eval(a);

        EXPECT_EQ(a.data().data(), b.data().data());
        EXPECT_EQ(&a, &b);
        bool type_eq = std::is_same<decltype(b), xarray<double>&>::value;
        EXPECT_TRUE(type_eq);

        xtensor<double, 2> t({3, 3});

        auto&& i = eval(t);

        EXPECT_EQ(t.data().data(), i.data().data());
        EXPECT_EQ(&t, &i);
        bool type_eq_2 = std::is_same<decltype(i), xtensor<double, 2>&>::value;
        EXPECT_TRUE(type_eq_2);
    }

    TEST(xeval, funcs)
    {
        xarray<double> a = {1,2,3,4};

        auto f = a * a - 2;
        auto&& b = eval(f);

        bool type_eq = std::is_same<decltype(b), xarray<double>&&>::value;
        EXPECT_TRUE(type_eq);

        xtensor<int, 2> k({3, 3});
        auto m = k * k - 4;
        auto&& n = eval(m);
        bool type_eq_3 = std::is_same<decltype(n), xtensor<int, 2>&&>::value;
        EXPECT_TRUE(type_eq_3);

#ifndef X_OLD_CLANG
        auto&& i = eval(linspace(0, 100));
        bool type_eq_2 = std::is_same<decltype(i), xtensor<int, 1>&&>::value;
        EXPECT_TRUE(type_eq_2);
#endif
    }
}

