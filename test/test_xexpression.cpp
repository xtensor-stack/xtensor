/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xmath.hpp"

namespace xt
{
    template <class E>
    double fun(xexpression_dimensioned<E, 2>& a)
    {
        auto& da = a.derived_cast();
        return (da * 2.0)(0, 0);
    }

    template <class E>
    double fun_shaped(xexpression_shaped<E, sshape<2>>& a)
    {
        auto& da = a.derived_cast();
        return (da * 2.0)(0, 0);
    }

    template <class E>
    double fun(xexpression_dimensioned<E, -1>& a)
    {
        auto& da = a.derived_cast();
        return (da * -1.0)(0, 0);
    }

    template <class E>
    double fun_shaped(xexpression_shaped<E, dshape>& a)
    {
        auto& da = a.derived_cast();
        return (da * -1.0)(0, 0);
    }

    TEST(xexpression, shared_basic)
    {
        xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        xtensor<double, 2> b = {{1,2,3,4}, {5,6,7,8}};

        EXPECT_EQ(fun(a), -1.0);
        EXPECT_EQ(fun(b), 2.0);
        EXPECT_EQ(fun_shaped(a), -1.0);
        EXPECT_EQ(fun_shaped(b), 2.0);
    }
}  // namespace xt
