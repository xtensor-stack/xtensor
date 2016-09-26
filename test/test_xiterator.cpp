#include "gtest/gtest.h"
#include "xarray/xarray.hpp"
#include <numeric>

namespace qs
{
    TEST(xiterator, increment)
    {
        using shape_type = xarray<int>::shape_type;
        shape_type shape = {3, 2, 4};
        xarray<int> a(shape);
        std::iota(a.storage_begin(), a.storage_end(), 0);

        auto iter = a.begin();
        auto iter2 = a.begin();
        auto siter = a.storage_begin();

        for(size_t i = 0; i < 10; ++i)
        {
            ++iter;
            iter2++;
            ++siter;
        }

        ASSERT_EQ(*iter, *siter) << "preincrement operator doesn't give expected result";
        ASSERT_EQ(*iter2, *siter) << "postincrement operator doesn't give expected result";
    }

    TEST(xiterator, end)
    {
        using shape_type = xarray<int>::shape_type;
        shape_type shape = {3, 2, 4};
        xarray<int> a(shape, layout::column_major);
        std::iota(a.storage_begin(), a.storage_end(), 0);

        size_t size = a.size();
        auto iter = a.begin();
        auto last = a.end();
        for(size_t i = 0; i < size; ++i)
        {
            ++iter;
        }

        ASSERT_EQ(iter, last) << "iterator doesn't reach the end";
    }
}

