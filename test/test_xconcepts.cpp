/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xconcepts.hpp"

#include <type_traits>

namespace xt
{
    template <class T,
              XTENSOR_REQUIRE<std::is_integral<T>::value>>
    int test_concept_check(T) {}

    template <class T,
              XTENSOR_REQUIRE<!std::is_integral<T>::value>>
    void * test_concept_check(T) {}



    TEST(xconcepts, concept_check)
    {
        EXPECT_TRUE((std::is_same<decltype(test_concept_check(1)), int>::value));
        EXPECT_TRUE((std::is_same<decltype(test_concept_check(1.0)), void *>::value));
    }

    TEST(xconcepts, promote)
    {
        EXPECT_TRUE((std::is_same<promote_t<uint8_t>, int>::value));
        EXPECT_TRUE((std::is_same<promote_t<int>, int>::value));
        EXPECT_TRUE((std::is_same<promote_t<float>, float>::value));
        EXPECT_TRUE((std::is_same<promote_t<double>, double>::value));

        EXPECT_TRUE((std::is_same<real_promote_t<uint8_t>, double>::value));
        EXPECT_TRUE((std::is_same<real_promote_t<int>, double>::value));
        EXPECT_TRUE((std::is_same<real_promote_t<float>, float>::value));
        EXPECT_TRUE((std::is_same<real_promote_t<double>, double>::value));

        EXPECT_TRUE((std::is_same<bool_promote_t<bool>, uint8_t>::value));
        EXPECT_TRUE((std::is_same<bool_promote_t<int>, int>::value));
    }

    TEST(xconcepts, norm_traits)
    {
        EXPECT_TRUE((std::is_same<norm_t<uint8_t>, uint8_t>::value));
        EXPECT_TRUE((std::is_same<norm_t<int>, int>::value));
        EXPECT_TRUE((std::is_same<norm_t<double>, double>::value));
        EXPECT_TRUE((std::is_same<norm_t<std::vector<uint8_t>>, double>::value));
        EXPECT_TRUE((std::is_same<norm_t<std::vector<int>>, double>::value));
        EXPECT_TRUE((std::is_same<norm_t<std::vector<double>>, double>::value));
        EXPECT_TRUE((std::is_same<norm_t<std::vector<long double>>, long double>::value));

        EXPECT_TRUE((std::is_same<norm_sq_t<uint8_t>, int>::value));
        EXPECT_TRUE((std::is_same<norm_sq_t<int>, int>::value));
        EXPECT_TRUE((std::is_same<norm_sq_t<double>, double>::value));
        EXPECT_TRUE((std::is_same<norm_sq_t<std::vector<uint8_t>>, uint64_t>::value));
        EXPECT_TRUE((std::is_same<norm_sq_t<std::vector<int>>, uint64_t>::value));
        EXPECT_TRUE((std::is_same<norm_sq_t<std::vector<double>>, double>::value));
        EXPECT_TRUE((std::is_same<norm_sq_t<std::vector<long double>>, long double>::value));
    }
} // namespace xt