/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

namespace xt
{
    using std::size_t;

    /**********************
     * Basic operations
     **********************/


    TEST(xmath, abs)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        EXPECT_EQ(abs(a)(0, 0), std::abs(a(0, 0)));
    }

    TEST(xmath, fabs)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        EXPECT_EQ(fabs(a)(0, 0), std::fabs(a(0, 0)));
    }

    TEST(xmath, fmod)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(fmod(a, b)(0, 0), std::fmod(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        EXPECT_EQ(fmod(a, sb)(0, 0), std::fmod(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(fmod(sa, b)(0, 0), std::fmod(sa, b(0, 0)));
    }

    TEST(xmath, remainder)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(remainder(a, b)(0, 0), std::remainder(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        EXPECT_EQ(remainder(a, sb)(0, 0), std::remainder(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(remainder(sa, b)(0, 0), std::remainder(sa, b(0, 0)));
    }

    TEST(xmath, fma)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        xarray<double> c(shape, 2.6);
        EXPECT_EQ(fma(a, b, c)(0, 0), std::fma(a(0, 0), b(0, 0), c(0, 0)));
        
        double sb = 1.2;
        EXPECT_EQ(fma(a, sb, c)(0, 0), std::fma(a(0, 0), sb, c(0, 0)));

        double sa = 4.6;
        EXPECT_EQ(fma(sa, b, c)(0, 0), std::fma(sa, b(0, 0), c(0, 0)));

        EXPECT_EQ(fma(sa, sb, c)(0, 0), std::fma(sa, sb, c(0, 0)));
    }

    TEST(xmath, fmax)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(fmax(a, b)(0, 0), std::fmax(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        EXPECT_EQ(fmax(a, sb)(0, 0), std::fmax(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(fmax(sa, b)(0, 0), std::fmax(sa, b(0, 0)));
    }

    TEST(xmath, fmin)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(fmin(a, b)(0, 0), std::fmin(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        EXPECT_EQ(fmin(a, sb)(0, 0), std::fmin(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(fmin(sa, b)(0, 0), std::fmin(sa, b(0, 0)));
    }

    TEST(xmath, fdim)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(fdim(a, b)(0, 0), std::fdim(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        EXPECT_EQ(fdim(a, sb)(0, 0), std::fdim(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(fdim(sa, b)(0, 0), std::fdim(sa, b(0, 0)));
    }


    /***************************
     * Exponential functions
     ***************************/

    TEST(xmath, exp)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(exp(a)(0, 0), std::exp(a(0, 0)));
    }

    TEST(xmath, exp2)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(exp2(a)(0, 0), std::exp2(a(0, 0)));
    }

    TEST(xmath, expm1)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(expm1(a)(0, 0), std::expm1(a(0, 0)));
    }

    TEST(xmath, log)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(log(a)(0, 0), std::log(a(0, 0)));
    }

    TEST(xmath, log2)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(log2(a)(0, 0), std::log2(a(0, 0)));
    }

    TEST(xmath, log1p)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(log1p(a)(0, 0), std::log1p(a(0, 0)));
    }


    /*********************
     * Power functions
     *********************/

    TEST(xmath, pow)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(pow(a, b)(0, 0), std::pow(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        EXPECT_EQ(pow(a, sb)(0, 0), std::pow(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(pow(sa, b)(0, 0), std::pow(sa, b(0, 0)));
    }

    TEST(xmath, sqrt)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(sqrt(a)(0, 0), std::sqrt(a(0, 0)));
    }

    TEST(xmath, cbrt)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(cbrt(a)(0, 0), std::cbrt(a(0, 0)));
    }

    TEST(xmath, hypot)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(hypot(a, b)(0, 0), std::hypot(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        EXPECT_EQ(hypot(a, sb)(0, 0), std::hypot(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(hypot(sa, b)(0, 0), std::hypot(sa, b(0, 0)));
    }


    /*****************************
     * Trigonometric functions
     *****************************/

    TEST(xmath, sin)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(sin(a)(0, 0), std::sin(a(0, 0)));
    }

    TEST(xmath, cos)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(cos(a)(0, 0), std::cos(a(0, 0)));
    }

    TEST(xmath, tan)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(tan(a)(0, 0), std::tan(a(0, 0)));
    }

    TEST(xmath, asin)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(asin(a)(0, 0), std::asin(a(0, 0)));
    }

    TEST(xmath, acos)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(acos(a)(0, 0), std::acos(a(0, 0)));
    }

    TEST(xmath, atan)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(atan(a)(0, 0), std::atan(a(0, 0)));
    }

    TEST(xmath, atan2)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(atan2(a, b)(0, 0), std::atan2(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        EXPECT_EQ(atan2(a, sb)(0, 0), std::atan2(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(atan2(sa, b)(0, 0), std::atan2(sa, b(0, 0)));
    }


    /**************************
     * Hyperbolic functions
     **************************/

    TEST(xmath, sinh)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(sinh(a)(0, 0), std::sinh(a(0, 0)));
    }

    TEST(xmath, cosh)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(cosh(a)(0, 0), std::cosh(a(0, 0)));
    }

    TEST(xmath, tanh)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(tanh(a)(0, 0), std::tanh(a(0, 0)));
    }

    TEST(xmath, asinh)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(asinh(a)(0, 0), std::asinh(a(0, 0)));
    }

    TEST(xmath, acosh)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 1.7);
        EXPECT_EQ(acosh(a)(0, 0), std::acosh(a(0, 0)));
    }

    TEST(xmath, atanh)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(atanh(a)(0, 0), std::atanh(a(0, 0)));
    }


    /*******************************
     * Error and gamma functions
     *******************************/

    TEST(xmath, erf)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(erf(a)(0, 0), std::erf(a(0, 0)));
    }

    TEST(xmath, erfc)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(erfc(a)(0, 0), std::erfc(a(0, 0)));
    }

    TEST(xmath, tgamma)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(tgamma(a)(0, 0), std::tgamma(a(0, 0)));
    }

    TEST(xmath, lgamma)
    {
        xshape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(lgamma(a)(0, 0), std::lgamma(a(0, 0)));
    }

}

