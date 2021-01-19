/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <complex>

#include "gtest/gtest.h"

// The following disables the conversion warnings. These warnings
// are legit and we don't want to avoid them with specific cast
// in xtensor implementation. However, we still want to check the
// results are correct and we don't want the warnings to pollute
// the output when building the tests suite.
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xrandom.hpp"
#pragma GCC diagnostic pop
#elif defined(_WIN32)
#pragma warning(push)
#pragma warning(disable: 4244)
#pragma warning(disable: 4267)
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xrandom.hpp"
#pragma warning(pop)
#else
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xrandom.hpp"
#endif

#include "xtl/xtype_traits.hpp"

namespace xt
{
    using std::size_t;
    using shape_type = dynamic_shape<size_t>;

    /*******************
     * type conversion *
     *******************/

#define CHECK_RESULT_TYPE(EXPRESSION, EXPECTED_TYPE)                                 \
    {                                                                                \
        using result_type = typename std::decay_t<decltype(EXPRESSION)>::value_type; \
        EXPECT_TRUE((std::is_same<result_type, EXPECTED_TYPE>::value));              \
    }

template <class T1, class T2>
using promote_t = xtl::promote_type_t<T1, T2>;

template <class T, class E>
void check_promoted_types(E&& e)
{
    using result_type = typename std::decay_t<decltype(e)>::value_type;
    EXPECT_TRUE((std::is_same<result_type, promote_t<T, result_type>>::value));
}

#define ARRAY_TYPE(VALUE_TYPE)  \
    std::array<VALUE_TYPE, 2>

#define CHECK_TEMPLATED_RESULT_TYPE(FUNC, INPUT)                                   \
        check_promoted_types<unsigned char>(FUNC<unsigned char>(INPUT));           \
        check_promoted_types<signed char>(FUNC<signed char>(INPUT));               \
        check_promoted_types<char>(FUNC<char>(INPUT));                             \
        check_promoted_types<unsigned short>(FUNC<unsigned short>(INPUT));         \
        check_promoted_types<signed short>(FUNC<signed short>(INPUT));             \
        check_promoted_types<short>(FUNC<short>(INPUT));                           \
        check_promoted_types<unsigned int>(FUNC<unsigned int>(INPUT));             \
        check_promoted_types<signed int>(FUNC<signed int>(INPUT));                 \
        check_promoted_types<int>(FUNC<int>(INPUT));                               \
        check_promoted_types<unsigned long long>(FUNC<unsigned long long>(INPUT)); \
        check_promoted_types<long long>(FUNC<signed long long>(INPUT));            \
        check_promoted_types<long long>(FUNC<long long>(INPUT));                   \
        check_promoted_types<float>(FUNC<float>(INPUT));                           \
        check_promoted_types<double>(FUNC<double>(INPUT));

#define CHECK_STDDEV_TEMPLATED_RESULT_TYPE(FUNC, INPUT)                            \
        {                                                                          \
        using result_type = typename std::decay_t<decltype(INPUT)>::value_type;    \
        using promo_type = std::conditional_t<std::is_integral<result_type>::value,\
                                              double,                              \
                                              float>;                              \
                                                                                   \
        check_promoted_types<promo_type>(FUNC<unsigned char>(INPUT));              \
        check_promoted_types<promo_type>(FUNC<signed char>(INPUT));                \
        check_promoted_types<promo_type>(FUNC<char>(INPUT));                       \
        check_promoted_types<promo_type>(FUNC<unsigned short>(INPUT));             \
        check_promoted_types<promo_type>(FUNC<signed short>(INPUT));               \
        check_promoted_types<promo_type>(FUNC<short>(INPUT));                      \
        check_promoted_types<promo_type>(FUNC<unsigned int>(INPUT));               \
        check_promoted_types<promo_type>(FUNC<signed int>(INPUT));                 \
        check_promoted_types<promo_type>(FUNC<int>(INPUT));                        \
        check_promoted_types<promo_type>(FUNC<unsigned long long>(INPUT));         \
        check_promoted_types<promo_type>(FUNC<signed long long>(INPUT));           \
        check_promoted_types<promo_type>(FUNC<long long>(INPUT));                  \
        check_promoted_types<float>(FUNC<float>(INPUT));                           \
        check_promoted_types<double>(FUNC<double>(INPUT));                         \
        }

#define CHECK_TEMPLATED_RESULT_TYPE_FOR_ALL(INPUT)        \
        CHECK_TEMPLATED_RESULT_TYPE(sum, INPUT)           \
        CHECK_TEMPLATED_RESULT_TYPE(mean, INPUT)          \
        CHECK_TEMPLATED_RESULT_TYPE(prod, INPUT)          \
        CHECK_TEMPLATED_RESULT_TYPE(variance, INPUT)      \
        CHECK_STDDEV_TEMPLATED_RESULT_TYPE(stddev, INPUT)

    TEST(xmath, uchar_result_type)
    {        
        shape_type shape = {3, 2};
        xarray<unsigned char> auchar(shape);

        CHECK_RESULT_TYPE(auchar + auchar, int);
        CHECK_RESULT_TYPE(2 * auchar, int);
        CHECK_RESULT_TYPE(2.0 * auchar, double);
        CHECK_RESULT_TYPE(sqrt(auchar), double);
        CHECK_RESULT_TYPE(abs(auchar), unsigned char);
        CHECK_RESULT_TYPE(sum(auchar), int);
        CHECK_RESULT_TYPE(mean(auchar), double);
        CHECK_RESULT_TYPE(minmax(auchar), ARRAY_TYPE(unsigned char));
        CHECK_TEMPLATED_RESULT_TYPE_FOR_ALL(auchar);
    }

    TEST(xmath, short_result_type)
    {        
        shape_type shape = {3, 2};
        xarray<short> ashort(shape);

        CHECK_RESULT_TYPE(ashort + ashort, int);
        CHECK_RESULT_TYPE(2 * ashort, int);
        CHECK_RESULT_TYPE(2.0 * ashort, double);
        CHECK_RESULT_TYPE(sqrt(ashort), double);
        CHECK_RESULT_TYPE(abs(ashort), decltype(std::abs(short{})));
        CHECK_RESULT_TYPE(sum(ashort), int);
        CHECK_RESULT_TYPE(mean(ashort), double);
        CHECK_RESULT_TYPE(minmax(ashort), ARRAY_TYPE(short));
        CHECK_TEMPLATED_RESULT_TYPE_FOR_ALL(ashort);
    }

    TEST(xmath, ushort_result_type)
    {        
        shape_type shape = {3, 2};
        xarray<unsigned short> aushort(shape);

        CHECK_RESULT_TYPE(aushort + aushort, int);
        CHECK_RESULT_TYPE(2u * aushort, unsigned int);
        CHECK_RESULT_TYPE(2.0 * aushort, double);
        CHECK_RESULT_TYPE(sqrt(aushort), double);
        CHECK_RESULT_TYPE(abs(aushort), unsigned short);
        CHECK_RESULT_TYPE(sum(aushort), int);
        CHECK_RESULT_TYPE(mean(aushort), double);
        CHECK_RESULT_TYPE(minmax(aushort), ARRAY_TYPE(unsigned short));
        CHECK_TEMPLATED_RESULT_TYPE_FOR_ALL(aushort);
    }

    TEST(xmath, int_result_type)
    {        
        shape_type shape = {3, 2};
        xarray<int> aint(shape);

        CHECK_RESULT_TYPE(aint + aint, int);
        CHECK_RESULT_TYPE(2 * aint, int);
        CHECK_RESULT_TYPE(2.0 * aint, double);
        CHECK_RESULT_TYPE(sqrt(aint), double);
        CHECK_RESULT_TYPE(abs(aint), int);
        CHECK_RESULT_TYPE(sum(aint), int);
        CHECK_RESULT_TYPE(mean(aint), double);
        CHECK_RESULT_TYPE(minmax(aint), ARRAY_TYPE(int));
        CHECK_TEMPLATED_RESULT_TYPE_FOR_ALL(aint);
    }

    TEST(xmath, uint_result_type)
    {        
        shape_type shape = {3, 2};
        xarray<unsigned int> auint(shape);
  
        CHECK_RESULT_TYPE(auint + auint, unsigned int);
        CHECK_RESULT_TYPE(2u * auint, unsigned int);
        CHECK_RESULT_TYPE(2.0 * auint, double);
        CHECK_RESULT_TYPE(sqrt(auint), double);
        CHECK_RESULT_TYPE(abs(auint), unsigned int);
        CHECK_RESULT_TYPE(sum(auint), unsigned int);
        CHECK_RESULT_TYPE(mean(auint), double);
        CHECK_RESULT_TYPE(minmax(auint), ARRAY_TYPE(unsigned int));
        CHECK_TEMPLATED_RESULT_TYPE_FOR_ALL(auint);
    }

    TEST(xmath, long_result_type)
    {        
        shape_type shape = {3, 2};
        xarray<long long> along(shape);

        CHECK_RESULT_TYPE(along + along, signed long long);
        CHECK_RESULT_TYPE(2 * along, signed long long);
        CHECK_RESULT_TYPE(2.0 * along, double);
        CHECK_RESULT_TYPE(sqrt(along), double);
        CHECK_RESULT_TYPE(abs(along), signed long long);
        CHECK_RESULT_TYPE(sum(along), signed long long);
        CHECK_RESULT_TYPE(mean(along), double);
        CHECK_RESULT_TYPE(minmax(along), ARRAY_TYPE(signed long long));
        CHECK_TEMPLATED_RESULT_TYPE_FOR_ALL(along);
    }

    TEST(xmath, ulong_result_type)
    {        
        shape_type shape = {3, 2};
        xarray<unsigned long long> aulong(shape);
    
        CHECK_RESULT_TYPE(aulong + aulong, unsigned long long);
        CHECK_RESULT_TYPE(2ul * aulong, unsigned long long);
        CHECK_RESULT_TYPE(2.0 * aulong, double);
        CHECK_RESULT_TYPE(sqrt(aulong), double);
        CHECK_RESULT_TYPE(abs(aulong), unsigned long long);
        CHECK_RESULT_TYPE(sum(aulong), unsigned long long);
        CHECK_RESULT_TYPE(mean(aulong), double);
        CHECK_RESULT_TYPE(minmax(aulong), ARRAY_TYPE(unsigned long long));
        CHECK_TEMPLATED_RESULT_TYPE_FOR_ALL(aulong);
    }

    TEST(xmath, float_result_type)
    {        
        shape_type shape = {3, 2};
        xarray<float> afloat(shape);

        CHECK_RESULT_TYPE(afloat + afloat, float);
        CHECK_RESULT_TYPE(2.0f * afloat, float);
        CHECK_RESULT_TYPE(2.0 * afloat, double);
        CHECK_RESULT_TYPE(sqrt(afloat), float);
        CHECK_RESULT_TYPE(abs(afloat), float);
        CHECK_RESULT_TYPE(sum(afloat), float);
        CHECK_RESULT_TYPE(mean(afloat), double);
        CHECK_RESULT_TYPE(minmax(afloat), ARRAY_TYPE(float));
        CHECK_TEMPLATED_RESULT_TYPE_FOR_ALL(afloat);
    }

    TEST(xmath, double_result_type)
    {        
        shape_type shape = {3, 2};
        xarray<double> adouble(shape);

        CHECK_RESULT_TYPE(adouble + adouble, double);
        CHECK_RESULT_TYPE(2.0 * adouble, double);
        CHECK_RESULT_TYPE(sqrt(adouble), double);
        CHECK_RESULT_TYPE(abs(adouble), double);
        CHECK_RESULT_TYPE(sum(adouble), double);
        CHECK_RESULT_TYPE(mean(adouble), double);
        CHECK_RESULT_TYPE(minmax(adouble), ARRAY_TYPE(double));
        CHECK_TEMPLATED_RESULT_TYPE_FOR_ALL(adouble);
    }

    TEST(xmath, float_complex_result_type)
    {        
        shape_type shape = {3, 2};
        xarray<std::complex<float>> afcomplex(shape);

        CHECK_RESULT_TYPE(afcomplex + afcomplex, std::complex<float>);
        CHECK_RESULT_TYPE(std::complex<float>(2.0) * afcomplex, std::complex<float>);
        CHECK_RESULT_TYPE(2.0f * afcomplex, std::complex<float>);
        CHECK_RESULT_TYPE(sqrt(afcomplex), std::complex<float>);
        CHECK_RESULT_TYPE(abs(afcomplex), float);
        CHECK_RESULT_TYPE(sum(afcomplex), std::complex<float>);
        CHECK_RESULT_TYPE(mean(afcomplex), std::complex<double>);
    }

    TEST(xmath, double_complex_result_type)
    {        
        shape_type shape = {3, 2};
        xarray<std::complex<double>> adcomplex(shape);

        CHECK_RESULT_TYPE(adcomplex + adcomplex, std::complex<double>);
        CHECK_RESULT_TYPE(std::complex<double>(2.0) * adcomplex, std::complex<double>);
        CHECK_RESULT_TYPE(2.0 * adcomplex, std::complex<double>);
        CHECK_RESULT_TYPE(sqrt(adcomplex), std::complex<double>);
        CHECK_RESULT_TYPE(abs(adcomplex), double);
        CHECK_RESULT_TYPE(sum(adcomplex), std::complex<double>);
        CHECK_RESULT_TYPE(mean(adcomplex), std::complex<double>);
    }

    TEST(xmath, mixed_result_type)
    {        
        shape_type shape = {3, 2};
        xarray<unsigned char> auchar(shape);
        xarray<short> ashort(shape);
        xarray<unsigned short> aushort(shape);
        xarray<int> aint(shape);
        xarray<unsigned int> auint(shape);
        xarray<long long> along(shape);
        xarray<unsigned long long> aulong(shape);
        xarray<float> afloat(shape);
        xarray<double> adouble(shape);
        xarray<std::complex<float>> afcomplex(shape);
        xarray<std::complex<double>> adcomplex(shape);

        CHECK_RESULT_TYPE(auchar + aint, int);
        CHECK_RESULT_TYPE(ashort + aint, int);
        CHECK_RESULT_TYPE(aulong + aint, unsigned long long);
        CHECK_RESULT_TYPE(afloat + aint, float);
        CHECK_RESULT_TYPE(adouble + aint, double);
        CHECK_RESULT_TYPE(adouble + adcomplex, std::complex<double>);
        CHECK_RESULT_TYPE(aulong + adouble, double);
        CHECK_RESULT_TYPE(afcomplex + adcomplex, std::complex<double>);
    }
}
