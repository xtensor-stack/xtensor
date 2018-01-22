/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xconcepts.hpp"

namespace xt
{
    template <class T,
              XTENSOR_REQUIRE<std::is_integral<T>::value>>
    int test_concept_check(T) {}

    template <class T,
              XTENSOR_REQUIRE<!std::is_integral<T>::value>>
    void* test_concept_check(T) {}

    TEST(concepts, concept_check)
    {
        EXPECT_TRUE((std::is_same<decltype(test_concept_check(1)), int>::value));
        EXPECT_TRUE((std::is_same<decltype(test_concept_check(1.0)), void*>::value));
    }

    TEST(concepts, iterator_concept)
    {
        {
            bool is_iter = iterator_concept<int>::value;
            EXPECT_FALSE(is_iter);
        }
        {
            bool is_iter = iterator_concept<int[3]>::value;
            EXPECT_TRUE(is_iter);
        }
        {
            bool is_iter = iterator_concept<int*>::value;
            EXPECT_TRUE(is_iter);
        }
        {
            bool is_iter = iterator_concept<decltype(std::vector<int>().begin())>::value;
            EXPECT_TRUE(is_iter);
        }
    }

    TEST(concepts, is_narrowing_conversion)
    {
        {
            bool is_narrowing = is_narrowing_conversion<int32_t, uint8_t>::value;
            EXPECT_TRUE(is_narrowing);
        }
        {
            bool is_narrowing = is_narrowing_conversion<double, uint32_t>::value;
            EXPECT_TRUE(is_narrowing);
        }
        {
            bool is_narrowing = is_narrowing_conversion<uint8_t, int32_t>::value;
            EXPECT_FALSE(is_narrowing);
        }
        {
            bool is_narrowing = is_narrowing_conversion<uint32_t, double>::value;
            EXPECT_FALSE(is_narrowing);
        }
    }

}  // namespace xt
