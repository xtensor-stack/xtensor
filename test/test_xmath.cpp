#include "gtest/gtest.h"
#include "xarray/xarray.hpp"

namespace qs
{

    /**********************
     * Basic operations
     **********************/


    TEST(xmath, abs)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        ASSERT_TRUE(abs(a)(0, 0) == std::abs(a(0, 0)));
    }

    TEST(xmath, fabs)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        ASSERT_TRUE(fabs(a)(0, 0) == std::fabs(a(0, 0)));
    }

    TEST(xmath, fmod)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        ASSERT_TRUE(fmod(a, b)(0, 0) == std::fmod(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        ASSERT_TRUE(fmod(a, sb)(0, 0) == std::fmod(a(0, 0), sb));

        double sa = 4.6;
        ASSERT_TRUE(fmod(sa, b)(0, 0) == std::fmod(sa, b(0, 0)));
    }

    TEST(xmath, remainder)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        ASSERT_TRUE(remainder(a, b)(0, 0) == std::remainder(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        ASSERT_TRUE(remainder(a, sb)(0, 0) == std::remainder(a(0, 0), sb));

        double sa = 4.6;
        ASSERT_TRUE(remainder(sa, b)(0, 0) == std::remainder(sa, b(0, 0)));
    }

    TEST(xmath, fma)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        xarray<double> c(shape, 2.6);
        ASSERT_TRUE(fma(a, b, c)(0, 0) == std::fma(a(0, 0), b(0, 0), c(0, 0)));
        
        double sb = 1.2;
        ASSERT_TRUE(fma(a, sb, c)(0, 0) == std::fma(a(0, 0), sb, c(0, 0)));

        double sa = 4.6;
        ASSERT_TRUE(fma(sa, b, c)(0, 0) == std::fma(sa, b(0, 0), c(0, 0)));

        ASSERT_TRUE(fma(sa, sb, c)(0, 0) == std::fma(sa, sb, c(0, 0)));
    }

    TEST(xmath, fmax)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        ASSERT_TRUE(fmax(a, b)(0, 0) == std::fmax(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        ASSERT_TRUE(fmax(a, sb)(0, 0) == std::fmax(a(0, 0), sb));

        double sa = 4.6;
        ASSERT_TRUE(fmax(sa, b)(0, 0) == std::fmax(sa, b(0, 0)));
    }

    TEST(xmath, fmin)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        ASSERT_TRUE(fmin(a, b)(0, 0) == std::fmin(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        ASSERT_TRUE(fmin(a, sb)(0, 0) == std::fmin(a(0, 0), sb));

        double sa = 4.6;
        ASSERT_TRUE(fmin(sa, b)(0, 0) == std::fmin(sa, b(0, 0)));
    }

    TEST(xmath, fdim)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        ASSERT_TRUE(fdim(a, b)(0, 0) == std::fdim(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        ASSERT_TRUE(fdim(a, sb)(0, 0) == std::fdim(a(0, 0), sb));

        double sa = 4.6;
        ASSERT_TRUE(fdim(sa, b)(0, 0) == std::fdim(sa, b(0, 0)));
    }


    /***************************
     * Exponential functions
     ***************************/

    TEST(xmath, exp)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(exp(a)(0, 0) == std::exp(a(0, 0)));
    }

    TEST(xmath, exp2)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(exp2(a)(0, 0) == std::exp2(a(0, 0)));
    }

    TEST(xmath, expm1)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(expm1(a)(0, 0) == std::expm1(a(0, 0)));
    }

    TEST(xmath, log)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(log(a)(0, 0) == std::log(a(0, 0)));
    }

    TEST(xmath, log2)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(log2(a)(0, 0) == std::log2(a(0, 0)));
    }

    TEST(xmath, log1p)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(log1p(a)(0, 0) == std::log1p(a(0, 0)));
    }


    /*********************
     * Power functions
     *********************/

    TEST(xmath, pow)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        ASSERT_TRUE(pow(a, b)(0, 0) == std::pow(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        ASSERT_TRUE(pow(a, sb)(0, 0) == std::pow(a(0, 0), sb));

        double sa = 4.6;
        ASSERT_TRUE(pow(sa, b)(0, 0) == std::pow(sa, b(0, 0)));
    }

    TEST(xmath, sqrt)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(sqrt(a)(0, 0) == std::sqrt(a(0, 0)));
    }

    TEST(xmath, cbrt)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(cbrt(a)(0, 0) == std::cbrt(a(0, 0)));
    }

    TEST(xmath, hypot)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        ASSERT_TRUE(hypot(a, b)(0, 0) == std::hypot(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        ASSERT_TRUE(hypot(a, sb)(0, 0) == std::hypot(a(0, 0), sb));

        double sa = 4.6;
        ASSERT_TRUE(hypot(sa, b)(0, 0) == std::hypot(sa, b(0, 0)));
    }


    /*****************************
     * Trigonometric functions
     *****************************/

    TEST(xmath, sin)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(sin(a)(0, 0) == std::sin(a(0, 0)));
    }

    TEST(xmath, cos)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(cos(a)(0, 0) == std::cos(a(0, 0)));
    }

    TEST(xmath, tan)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(tan(a)(0, 0) == std::tan(a(0, 0)));
    }

    TEST(xmath, asin)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        ASSERT_TRUE(asin(a)(0, 0) == std::asin(a(0, 0)));
    }

    TEST(xmath, acos)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        ASSERT_TRUE(acos(a)(0, 0) == std::acos(a(0, 0)));
    }

    TEST(xmath, atan)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(atan(a)(0, 0) == std::atan(a(0, 0)));
    }

    TEST(xmath, atan2)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        ASSERT_TRUE(atan2(a, b)(0, 0) == std::atan2(a(0, 0), b(0, 0)));
        
        double sb = 1.2;
        ASSERT_TRUE(atan2(a, sb)(0, 0) == std::atan2(a(0, 0), sb));

        double sa = 4.6;
        ASSERT_TRUE(atan2(sa, b)(0, 0) == std::atan2(sa, b(0, 0)));
    }


    /**************************
     * Hyperbolic functions
     **************************/

    TEST(xmath, sinh)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(sinh(a)(0, 0) == std::sinh(a(0, 0)));
    }

    TEST(xmath, cosh)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(cosh(a)(0, 0) == std::cosh(a(0, 0)));
    }

    TEST(xmath, tanh)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 3.7);
        ASSERT_TRUE(tanh(a)(0, 0) == std::tanh(a(0, 0)));
    }

    TEST(xmath, asinh)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        ASSERT_TRUE(asinh(a)(0, 0) == std::asinh(a(0, 0)));
    }

    TEST(xmath, acosh)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 1.7);
        ASSERT_TRUE(acosh(a)(0, 0) == std::acosh(a(0, 0)));
    }

    TEST(xmath, atanh)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        ASSERT_TRUE(atanh(a)(0, 0) == std::atanh(a(0, 0)));
    }


    /*******************************
     * Error and gamma functions
     *******************************/

    TEST(xmath, erf)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        ASSERT_TRUE(erf(a)(0, 0) == std::erf(a(0, 0)));
    }

    TEST(xmath, erfc)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        ASSERT_TRUE(erfc(a)(0, 0) == std::erfc(a(0, 0)));
    }

    TEST(xmath, tgamma)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        ASSERT_TRUE(tgamma(a)(0, 0) == std::tgamma(a(0, 0)));
    }

    TEST(xmath, lgamma)
    {
        array_shape<size_t> shape = {3, 2};
        xarray<double> a(shape, 0.7);
        ASSERT_TRUE(lgamma(a)(0, 0) == std::lgamma(a(0, 0)));
    }

}

