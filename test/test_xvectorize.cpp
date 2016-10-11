#include "gtest/gtest.h"

#include "xarray/xarray.hpp"
#include "xarray/xvectorize.hpp"

namespace qs
{
    double f1(double d1, double d2)
    {
        return d1 + d2;
    }

    TEST(xvectorize, function)
    {
        auto vecf1 = vectorize(f1);
        xshape<std::size_t> shape = { 3, 2 };
        xarray<double> a(shape, 1.5);
        xarray<double> b(shape, 2.3);
        xarray<double> c = vecf1(a, b);
        EXPECT_EQ(a(0,0) + b(0, 0), c(0, 0));
    }

    TEST(xvectorize, lambda)
    {
        auto lambda = [](double d1, double d2) { return d1 + d2; };
        auto vec_lambda = vectorize(lambda);
        xshape<std::size_t> shape = { 3, 2 };
        xarray<double> a(shape, 1.5);
        xarray<double> b(shape, 2.3);
        xarray<double> c = vec_lambda(a, b);
        EXPECT_EQ(a(0,0) + b(0, 0), c(0, 0));
    }

    struct func
    {
        inline double operator()(double d1, double d2) const
        {
            return d1 + d2;
        }
    };

    TEST(xvectorize, functor)
    {
        func f;
        auto vecfunc = vectorize(f);
        xshape<std::size_t> shape = { 3, 2 };
        xarray<double> a(shape, 1.5);
        xarray<double> b(shape, 2.3);
        xarray<double> c = vecfunc(a, b);
        EXPECT_EQ(a(0,0) + b(0, 0), c(0, 0));
    }

}

