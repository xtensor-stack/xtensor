/***************************************************************************
* Copyright (c) 2017, Johan Mabille, Sylvain Corlay Wolf Vollprecht and    *
* Martin Renou                                                             *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xoptional_assembly.hpp"
#include "xtensor/xmasked_view.hpp"
#include "xtensor/xio.hpp"

namespace xt
{
    using data_type = xoptional_assembly<xarray<double>, xarray<bool>>;

    // data = {{ 1. ,  2., N/A },
    //         { N/A,  5.,  6. },
    //         { 7. ,  8.,  9. }}
    inline data_type make_test_data()
    {
        data_type d = {{ 1., 2., 3.},
                       { 4., 5., 6.},
                       { 7., 8., 9.}};
        d(0, 2).has_value() = false;
        d(1, 0).has_value() = false;
        return d;
    }

    // masked_data = {{ 1. ,  2., N/A },
    //                { N/A, N/A, N/A },
    //                { 7. ,  8.,  9. }}
    inline auto make_masked_data(data_type& data)
    {
        xarray<bool> mask = {{ true,  true,  true},
                             {false, false, false},
                             { true,  true,  true}};

        return masked_view(data, std::move(mask));
    }

    TEST(xmasked_view, dimension)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(data.dimension(), masked_data.dimension());
    }

    TEST(xmasked_view, size)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(data.size(), masked_data.size());
    }

    TEST(xmasked_view, shape)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(data.shape(), masked_data.shape());
    }

    // masked_data = {{ 1. ,  2., N/A },
    //                { N/A, N/A, N/A },
    //                { 7. ,  8.,  9. }}
    TEST(xmasked_view, access)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(masked_data(0, 0), 1.);
        EXPECT_EQ(masked_data(0, 1), 2.);
        EXPECT_EQ(masked_data(0, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data(1, 0), xtl::missing<double>());
        EXPECT_EQ(masked_data(1, 1), xtl::missing<double>());
        EXPECT_EQ(masked_data(1, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data(2, 0), 7.);
        EXPECT_EQ(masked_data(2, 1), 8.);
        EXPECT_EQ(masked_data(2, 2), 9.);

#ifndef XTENSOR_ENABLE_ASSERT
        masked_data(3, 3);
#endif
    }

    TEST(xmasked_view, at)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(masked_data.at(0, 0), 1.);
        EXPECT_EQ(masked_data.at(0, 1), 2.);
        EXPECT_EQ(masked_data.at(0, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(1, 0), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(1, 1), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(1, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(2, 0), 7.);
        EXPECT_EQ(masked_data.at(2, 1), 8.);
        EXPECT_EQ(masked_data.at(2, 2), 9.);

        EXPECT_ANY_THROW(masked_data.at(3, 3));
    }

    TEST(xmasked_view, unchecked)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(masked_data.unchecked(0, 0), 1.);
        EXPECT_EQ(masked_data.unchecked(0, 1), 2.);
        EXPECT_EQ(masked_data.unchecked(0, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data.unchecked(1, 0), xtl::missing<double>());
        EXPECT_EQ(masked_data.unchecked(1, 1), xtl::missing<double>());
        EXPECT_EQ(masked_data.unchecked(1, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data.unchecked(2, 0), 7.);
        EXPECT_EQ(masked_data.unchecked(2, 1), 8.);
        EXPECT_EQ(masked_data.unchecked(2, 2), 9.);

#ifndef XTENSOR_ENABLE_ASSERT
        masked_data.unchecked(3, 3);
#endif
    }

    TEST(xmasked_view, access2)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        auto val = masked_data[{0, 0}];
        EXPECT_EQ(val, 1.);
        auto val2 = masked_data[{1, 0}];
        EXPECT_EQ(val2, xtl::missing<double>());

        auto index1 = std::array<int, 2>({0, 0});
        auto index2 = std::array<int, 2>({1, 0});
        EXPECT_EQ(masked_data[index1], 1.);
        EXPECT_EQ(masked_data[index2], xtl::missing<double>());
    }

    TEST(xmasked_view, element)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        auto index1 = std::array<int, 2>({0, 0});
        auto index2 = std::array<int, 2>({1, 0});
        EXPECT_EQ(masked_data.element(index1.begin(), index1.end()), 1.);
        EXPECT_EQ(masked_data.element(index2.begin(), index2.end()), xtl::missing<double>());
    }

    TEST(xmasked_view, storage)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        EXPECT_EQ(data.storage().value(), masked_data.storage().value());
        EXPECT_NE(data.storage().has_value(), masked_data.storage().has_value());
    }

    TEST(xmasked_view, fill)
    {
        auto data = make_test_data();
        auto masked_data = make_masked_data(data);

        masked_data.fill(2.);

        EXPECT_EQ(masked_data.at(0, 0), 2.);
        EXPECT_EQ(masked_data.at(0, 1), 2.);
        EXPECT_EQ(masked_data.at(0, 2), 2.);
        EXPECT_EQ(masked_data.at(1, 0), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(1, 1), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(1, 2), xtl::missing<double>());
        EXPECT_EQ(masked_data.at(2, 0), 2.);
        EXPECT_EQ(masked_data.at(2, 1), 2.);
        EXPECT_EQ(masked_data.at(2, 2), 2.);
    }
}
