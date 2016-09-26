#include "gtest/gtest.h"
#include "xarray/xarray.hpp"
#include "xarray/xview.hpp"
#include <algorithm>

namespace qs
{

    TEST(xview, simple)
    {
        xshape<size_t> shape = {3, 4};
        xarray<double> a(shape);
        std::vector<double> data {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::copy(data.begin(), data.end(), a.storage_begin());

        auto view1 = make_xview(a, 1, range(1, 4));
        ASSERT_TRUE(view1(0) == a(1, 1));
        ASSERT_TRUE(view1(1) == a(1, 2));
        ASSERT_TRUE(view1.dimension() == 1);

        auto view2 = make_xview(a, 0, range(0, 3));
        ASSERT_TRUE(view2(0) == a(0, 0));
        ASSERT_TRUE(view2(1) == a(0, 1));
        ASSERT_TRUE(view2.dimension() == 1);

        auto view3 = make_xview(a, range(0, 2), 2);
        ASSERT_TRUE(view3(0) == a(0, 2));
        ASSERT_TRUE(view3(1) == a(1, 2));
        ASSERT_TRUE(view3.dimension() == 1);
    }

    TEST(xview, squeeze_count)
    {
        size_t squeeze1 = squeeze_count<size_t, size_t, size_t, xrange<size_t>>::value;
        ASSERT_TRUE(squeeze1 == 3);
        size_t squeeze2 = squeeze_count<size_t, xrange<size_t>, size_t>::value;
        ASSERT_TRUE(squeeze2 == 2);
        size_t squeeze3 = squeeze_count_before<3, size_t, size_t, size_t, xrange<size_t>>::value;
        ASSERT_TRUE(squeeze3 == 3);
        size_t squeeze4 = squeeze_count_before<2, size_t, xrange<size_t>, size_t>::value;
        ASSERT_TRUE(squeeze4 == 1);
    }
}

