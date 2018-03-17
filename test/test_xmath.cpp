/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <complex>
#include <limits>

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

namespace xt
{
    using std::size_t;
    using shape_type = dynamic_shape<size_t>;

    /*******************
     * type conversion *
     *******************/

#define CHECK_RESULT_TYPE(EXPRESSION, EXPECTED_TYPE)                                  \
    {                                                                                \
        using result_type = typename std::decay_t<decltype(EXPRESSION)>::value_type; \
        EXPECT_TRUE((std::is_same<result_type, EXPECTED_TYPE>::value));              \
    }
#define ARRAY_TYPE(VALUE_TYPE)  \
    std::array<VALUE_TYPE, 2>

    TEST(xmath, result_type)
    {
        shape_type shape = {3, 2};
        xarray<unsigned char> auchar(shape);
        xarray<short> ashort(shape);
        xarray<int> aint(shape);
        xarray<unsigned int> auint(shape);
        xarray<unsigned long long> aulong(shape);
        xarray<float> afloat(shape);
        xarray<double> adouble(shape);
        xarray<std::complex<float>> afcomplex(shape);
        xarray<std::complex<double>> adcomplex(shape);

        /*****************
         * unsigned char *
         *****************/
        CHECK_RESULT_TYPE(auchar + auchar, int);
        CHECK_RESULT_TYPE(2 * auchar, int);
        CHECK_RESULT_TYPE(2.0 * auchar, double);
        CHECK_RESULT_TYPE(sqrt(auchar), double);
        CHECK_RESULT_TYPE(abs(auchar), int);
        CHECK_RESULT_TYPE(sum(auchar), unsigned long long);
        CHECK_RESULT_TYPE(mean(auchar), double);
        CHECK_RESULT_TYPE(minmax(auchar), ARRAY_TYPE(unsigned char));

        /*********
         * short *
         *********/
        CHECK_RESULT_TYPE(ashort + ashort, int);
        CHECK_RESULT_TYPE(2 * ashort, int);
        CHECK_RESULT_TYPE(2.0 * ashort, double);
        CHECK_RESULT_TYPE(sqrt(ashort), double);
        CHECK_RESULT_TYPE(abs(ashort), int);
        CHECK_RESULT_TYPE(sum(ashort), long long);
        CHECK_RESULT_TYPE(mean(ashort), double);
        CHECK_RESULT_TYPE(minmax(ashort), ARRAY_TYPE(short));

        /*******
         * int *
         *******/
        CHECK_RESULT_TYPE(aint + aint, int);
        CHECK_RESULT_TYPE(2 * aint, int);
        CHECK_RESULT_TYPE(2.0 * aint, double);
        CHECK_RESULT_TYPE(sqrt(aint), double);
        CHECK_RESULT_TYPE(abs(aint), int);
        CHECK_RESULT_TYPE(sum(aint), long long);
        CHECK_RESULT_TYPE(mean(aint), double);
        CHECK_RESULT_TYPE(minmax(aint), ARRAY_TYPE(int));

        /****************
         * unsigned int *
         ****************/
        CHECK_RESULT_TYPE(auint + auint, unsigned int);
        CHECK_RESULT_TYPE(2 * auint, unsigned int);
        CHECK_RESULT_TYPE(2.0 * auint, double);
        CHECK_RESULT_TYPE(sqrt(auint), double);
        CHECK_RESULT_TYPE(abs(auint), unsigned int);
        CHECK_RESULT_TYPE(sum(auint), unsigned long long);
        CHECK_RESULT_TYPE(mean(auint), double);
        CHECK_RESULT_TYPE(minmax(auint), ARRAY_TYPE(unsigned int));

        /**********************
         * unsigned long long *
         **********************/
        CHECK_RESULT_TYPE(aulong + aulong, unsigned long long);
        CHECK_RESULT_TYPE(2 * aulong, unsigned long long);
        CHECK_RESULT_TYPE(2.0 * aulong, double);
        CHECK_RESULT_TYPE(sqrt(aulong), double);
        CHECK_RESULT_TYPE(abs(aulong), unsigned long long);
        CHECK_RESULT_TYPE(sum(aulong), unsigned long long);
        CHECK_RESULT_TYPE(mean(aulong), double);
        CHECK_RESULT_TYPE(minmax(aulong), ARRAY_TYPE(unsigned long long));

        /*********
         * float *
         *********/
        CHECK_RESULT_TYPE(afloat + afloat, float);
        CHECK_RESULT_TYPE(2.0f * afloat, float);
        CHECK_RESULT_TYPE(2.0 * afloat, double);
        CHECK_RESULT_TYPE(sqrt(afloat), float);
        CHECK_RESULT_TYPE(abs(afloat), float);
        CHECK_RESULT_TYPE(sum(afloat), double);
        CHECK_RESULT_TYPE(mean(afloat), double);
        CHECK_RESULT_TYPE(minmax(afloat), ARRAY_TYPE(float));

        /**********
         * double *
         **********/
        CHECK_RESULT_TYPE(adouble + adouble, double);
        CHECK_RESULT_TYPE(2.0 * adouble, double);
        CHECK_RESULT_TYPE(sqrt(adouble), double);
        CHECK_RESULT_TYPE(abs(adouble), double);
        CHECK_RESULT_TYPE(sum(adouble), double);
        CHECK_RESULT_TYPE(mean(adouble), double);
        CHECK_RESULT_TYPE(minmax(adouble), ARRAY_TYPE(double));

        /***********************
         * std::complex<float> *
         ***********************/
        CHECK_RESULT_TYPE(afcomplex + afcomplex, std::complex<float>);
        CHECK_RESULT_TYPE(std::complex<float>(2.0) * afcomplex, std::complex<float>);
        CHECK_RESULT_TYPE(2.0f * afcomplex, std::complex<float>);
        CHECK_RESULT_TYPE(sqrt(afcomplex), std::complex<float>);
        CHECK_RESULT_TYPE(abs(afcomplex), float);
        CHECK_RESULT_TYPE(sum(afcomplex), std::complex<double>);
        CHECK_RESULT_TYPE(mean(afcomplex), std::complex<double>);

        /************************
         * std::complex<double> *
         ************************/
        CHECK_RESULT_TYPE(adcomplex + adcomplex, std::complex<double>);
        CHECK_RESULT_TYPE(std::complex<double>(2.0) * adcomplex, std::complex<double>);
        CHECK_RESULT_TYPE(2.0 * adcomplex, std::complex<double>);
        CHECK_RESULT_TYPE(sqrt(adcomplex), std::complex<double>);
        CHECK_RESULT_TYPE(abs(adcomplex), double);
        CHECK_RESULT_TYPE(sum(adcomplex), std::complex<double>);
        CHECK_RESULT_TYPE(mean(adcomplex), std::complex<double>);

        /***************
         * mixed types *
         ***************/
        CHECK_RESULT_TYPE(auchar + aint, int);
        CHECK_RESULT_TYPE(ashort + aint, int);
        CHECK_RESULT_TYPE(aulong + aint, unsigned long long);
        CHECK_RESULT_TYPE(afloat + aint, float);
        CHECK_RESULT_TYPE(adouble + aint, double);
        CHECK_RESULT_TYPE(adouble + adcomplex, std::complex<double>);
        CHECK_RESULT_TYPE(aulong + adouble, double);
    }


    /********************
     * Basic operations *
     ********************/

    TEST(xmath, abs)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        EXPECT_EQ(abs(a)(0, 0), std::abs(a(0, 0)));
    }

    TEST(xmath, fabs)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        EXPECT_EQ(fabs(a)(0, 0), std::fabs(a(0, 0)));
    }

    TEST(xmath, fmod)
    {
        shape_type shape = {3, 2};
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
        shape_type shape = {3, 2};
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
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        xarray<double> c(shape, 2.6);
        EXPECT_EQ(xt::fma(a, b, c)(0, 0), std::fma(a(0, 0), b(0, 0), c(0, 0)));

        double sb = 1.2;
        EXPECT_EQ(xt::fma(a, sb, c)(0, 0), std::fma(a(0, 0), sb, c(0, 0)));

        double sa = 4.6;
        EXPECT_EQ(xt::fma(sa, b, c)(0, 0), std::fma(sa, b(0, 0), c(0, 0)));

        EXPECT_EQ(xt::fma(sa, sb, c)(0, 0), std::fma(sa, sb, c(0, 0)));
    }

    TEST(xmath, fmax)
    {
        shape_type shape = {3, 2};
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
        shape_type shape = {3, 2};
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
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(fdim(a, b)(0, 0), std::fdim(a(0, 0), b(0, 0)));

        double sb = 1.2;
        EXPECT_EQ(fdim(a, sb)(0, 0), std::fdim(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(fdim(sa, b)(0, 0), std::fdim(sa, b(0, 0)));
    }

    TEST(xmath, clip)
    {
        shape_type shape = {3, 2};
        xarray<double> a = {1, 2, 3, 4, 5, 6};
        xarray<double> res = {2, 2, 3, 4, 4, 4};

        xarray<double> clipped = clip(a, 2.0, 4.0);
        EXPECT_EQ(res, clipped);
    }

    TEST(xmath, sign)
    {
        shape_type shape = {3, 2};
        xarray<float> a(shape, 1);
        a(0, 1) = -1;
        a(1, 1) = 0;
        a(2, 1) = -0;

        auto signs = sign(a);
        EXPECT_EQ(1.f, signs(0, 0));
        EXPECT_EQ(-1.f, signs(0, 1));
        EXPECT_EQ(0.f, signs(1, 1));
        EXPECT_EQ(0.f, signs(2, 1));

        xarray<unsigned int> b(shape, 1);
        b(1, 1) = static_cast<unsigned int>(-1);
        b(2, 1) = 0;

        auto signs_b = sign(b);
        EXPECT_EQ(1u, signs_b(0, 0));
        // sign from overflow
        EXPECT_EQ(1u, signs_b(1, 1));
        EXPECT_EQ(0u, signs_b(2, 1));

        xarray<double> c(shape, 1);
        c(0, 0) = std::numeric_limits<double>::infinity();
        c(0, 1) = -std::numeric_limits<double>::infinity();
        c(1, 0) = std::numeric_limits<double>::quiet_NaN();
        c(1, 1) = -std::numeric_limits<double>::quiet_NaN();

        auto signs_c = sign(c);
        EXPECT_EQ(1, signs_c(0, 0));
        EXPECT_EQ(-1, signs_c(0, 1));
        EXPECT_TRUE(std::isnan(signs_c(1, 0)));
        EXPECT_TRUE(std::isnan(signs_c(1, 1)));

        using ctype = std::complex<double>;
        xarray<ctype> d(shape, ctype(3, 2));
        d(0, 0) = ctype(1, 1);
        d(0, 1) = ctype(-1, 1);
        d(1, 0) = ctype(0, -1);
        d(1, 1) = ctype(-0, 1);

        auto signs_d = sign(d);
        EXPECT_EQ(ctype(1, 0), signs_d(0, 0));
        EXPECT_EQ(ctype(-1, 0), signs_d(0, 1));
        EXPECT_EQ(ctype(-1, 0), signs_d(1, 0));
        EXPECT_EQ(ctype(1, 0), signs_d(1, 1));
    }

    TEST(xmath, isnan)
    {
        xarray<double> arr
           {{1.0, std::numeric_limits<double>::quiet_NaN()},
            {std::numeric_limits<double>::quiet_NaN(), 0.0}};
        xarray<bool> expected
           {{false, true}, {true, false}};
        EXPECT_TRUE(all(equal(expected, xt::isnan(arr))));
    }

    /*************************
     * Exponential functions *
     *************************/

    TEST(xmath, exp)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(exp(a)(0, 0), std::exp(a(0, 0)));
    }

    TEST(xmath, exp2)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(exp2(a)(0, 0), std::exp2(a(0, 0)));
    }

    TEST(xmath, expm1)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(expm1(a)(0, 0), std::expm1(a(0, 0)));
    }

    TEST(xmath, log)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(log(a)(0, 0), std::log(a(0, 0)));
    }

    TEST(xmath, log2)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(log2(a)(0, 0), std::log2(a(0, 0)));
    }

    TEST(xmath, log1p)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(log1p(a)(0, 0), std::log1p(a(0, 0)));
    }

    /*******************
     * Power functions *
     *******************/

    TEST(xmath, pow)
    {
        shape_type shape = {3, 2};
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
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(sqrt(a)(0, 0), std::sqrt(a(0, 0)));
    }

    TEST(xmath, cbrt)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(cbrt(a)(0, 0), std::cbrt(a(0, 0)));
    }

    TEST(xmath, hypot)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(hypot(a, b)(0, 0), std::hypot(a(0, 0), b(0, 0)));

        double sb = 1.2;
        EXPECT_EQ(hypot(a, sb)(0, 0), std::hypot(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(hypot(sa, b)(0, 0), std::hypot(sa, b(0, 0)));
    }

    /***************************
     * Trigonometric functions *
     ***************************/

    TEST(xmath, sin)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(sin(a)(0, 0), std::sin(a(0, 0)));
    }

    TEST(xmath, cos)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(cos(a)(0, 0), std::cos(a(0, 0)));
    }

    TEST(xmath, tan)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(tan(a)(0, 0), std::tan(a(0, 0)));
    }

    TEST(xmath, asin)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(asin(a)(0, 0), std::asin(a(0, 0)));
    }

    TEST(xmath, acos)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(acos(a)(0, 0), std::acos(a(0, 0)));
    }

    TEST(xmath, atan)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(atan(a)(0, 0), std::atan(a(0, 0)));
    }

    TEST(xmath, atan2)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 4.5);
        xarray<double> b(shape, 1.3);
        EXPECT_EQ(atan2(a, b)(0, 0), std::atan2(a(0, 0), b(0, 0)));

        double sb = 1.2;
        EXPECT_EQ(atan2(a, sb)(0, 0), std::atan2(a(0, 0), sb));

        double sa = 4.6;
        EXPECT_EQ(atan2(sa, b)(0, 0), std::atan2(sa, b(0, 0)));
    }


    /************************
     * Hyperbolic functions *
     ************************/

    TEST(xmath, sinh)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(sinh(a)(0, 0), std::sinh(a(0, 0)));
    }

    TEST(xmath, cosh)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(cosh(a)(0, 0), std::cosh(a(0, 0)));
    }

    TEST(xmath, tanh)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 3.7);
        EXPECT_EQ(tanh(a)(0, 0), std::tanh(a(0, 0)));
    }

    TEST(xmath, asinh)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(asinh(a)(0, 0), std::asinh(a(0, 0)));
    }

    TEST(xmath, acosh)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 1.7);
        EXPECT_EQ(acosh(a)(0, 0), std::acosh(a(0, 0)));
    }

    TEST(xmath, atanh)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(atanh(a)(0, 0), std::atanh(a(0, 0)));
    }


    /*****************************
     * Error and gamma functions *
     *****************************/

    TEST(xmath, erf)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(erf(a)(0, 0), std::erf(a(0, 0)));
    }

    TEST(xmath, erfc)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(erfc(a)(0, 0), std::erfc(a(0, 0)));
    }

    TEST(xmath, tgamma)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(tgamma(a)(0, 0), std::tgamma(a(0, 0)));
    }

    TEST(xmath, lgamma)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        EXPECT_EQ(lgamma(a)(0, 0), std::lgamma(a(0, 0)));
    }

    TEST(xmath, ceil)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 2.3);
        EXPECT_EQ(ceil(a)(0, 0), std::ceil(a(0, 0)));
    }

    TEST(xmath, floor)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 2.3);
        EXPECT_EQ(floor(a)(0, 0), std::floor(a(0, 0)));
    }

    TEST(xmath, trunc)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 2.3);
        EXPECT_EQ(trunc(a)(0, 0), std::trunc(a(0, 0)));
    }

    TEST(xmath, round)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 2.3);
        EXPECT_EQ(round(a)(0, 0), std::round(a(0, 0)));
    }

    TEST(xmath, nearbyint)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 2.3);
        EXPECT_EQ(nearbyint(a)(0, 0), std::nearbyint(a(0, 0)));
    }

    TEST(xmath, rint)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 2.3);
        EXPECT_EQ(rint(a)(0, 0), std::rint(a(0, 0)));
    }

    TEST(xmath, isclose)
    {
        shape_type shape = {3, 2};
        xarray<double> a(shape, 0.7);
        xarray<double> b(shape, 0.70000000001);
        xarray<double> c(shape, 0.80000000001);
        EXPECT_TRUE(allclose(a, b));
        EXPECT_FALSE(allclose(a, c));
        b(1, 1) = 1;
        EXPECT_FALSE(allclose(a, b));
        EXPECT_TRUE(allclose(a, b, 10, 10));

        b = a;
        b(0, 0) = nan("n");
        EXPECT_FALSE(isclose(a, b)(0, 0));
        EXPECT_FALSE(isclose(a, b)(0, 0));
        a(0, 0) = nan("n");
        EXPECT_FALSE(isclose(a, b)(0, 0));
        EXPECT_TRUE(isclose(a, b, 1, 1, true)(0, 0));
    }

    TEST(xmath, isclose_int)
    {
        EXPECT_FALSE(isclose(1, 2)());
        EXPECT_TRUE(isclose(1, 2, 1)());
        EXPECT_TRUE(isclose(1u, 2u, 1u)());
        EXPECT_TRUE(isclose(1ul, 2ul, 1ul)());
        EXPECT_TRUE(isclose(1ul, 10ul, 1ul)());
        EXPECT_TRUE(isclose(1ul, 10ul, 1ul, 100ul)());
    }

    TEST(xmath, integer_abs)
    {
        EXPECT_EQ(math::abs(1ul), 1ul);
        EXPECT_EQ(math::abs(1u), 1u);
        EXPECT_EQ(math::abs(static_cast<unsigned char>(1)), static_cast<unsigned char>(1));
        EXPECT_EQ(math::abs(char(1)), 1);
        EXPECT_EQ(math::abs(int(1)), 1);
        EXPECT_EQ(math::abs(int(-1)), 1);
        EXPECT_EQ(math::abs(long(-1)), long(1));
        EXPECT_EQ(math::abs(-1.5), 1.5);
    }

    TEST(xmath, isinf_nan_int)
    {
        EXPECT_FALSE(math::isinf(132));
        EXPECT_FALSE(math::isinf(-123123));
        EXPECT_FALSE(math::isnan(132));
        EXPECT_FALSE(math::isnan(-123123));
        EXPECT_FALSE(math::isinf(132ul));
        EXPECT_FALSE(math::isnan(123123ul));
    }

    TEST(xmath, scalar_cast)
    {
        double arg = 1.0;
        double res = ::xt::atan(xscalar<double>(arg));
        EXPECT_EQ(res, std::atan(arg));
        bool close = ::xt::isclose(1.0, 1.0);
        EXPECT_EQ(close, true);

        auto p = ::xt::numeric_constants<>::PI;
        EXPECT_EQ(p, 3.141592653589793238463);
    }
}
