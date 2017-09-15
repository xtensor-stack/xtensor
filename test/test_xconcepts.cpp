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

    TEST(concepts, concept_check)
    {
        EXPECT_TRUE((std::is_same<decltype(test_concept_check(1)), int>::value));
        EXPECT_TRUE((std::is_same<decltype(test_concept_check(1.0)), void *>::value));
    }

    TEST(concepts, iterator_concept)
    {
        EXPECT_FALSE((iterator_concept<int>::value));
        EXPECT_TRUE((iterator_concept<int *>::value));
        EXPECT_TRUE((iterator_concept<decltype(std::vector<int>().begin())>::value));
    }

} // namespace xt