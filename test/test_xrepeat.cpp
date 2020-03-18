/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xrepeat.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xio.hpp"

namespace xt
{

    TEST(xrepeat, const_array)
    {
        xarray<size_t> const array = {1, 2, 3};

        const auto repeated_array = xt::repeat(array, 1, 0);

        ASSERT_EQ(1, repeated_array(0));
        ASSERT_EQ(2, repeated_array(1));
        ASSERT_EQ(3, repeated_array(2));
    }

    TEST(xrepeat, stepper_begin)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 2, 0);
        const auto stepper = repeated_array.stepper_begin();

        ASSERT_EQ(1, *stepper);
    }

    TEST(xrepeat, stepper_end_row_major_with_repeat_1)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };

        const auto repeated_array = xt::repeat(array, 1, 0);
        auto stepper = repeated_array.stepper_end(xt::layout_type::row_major);

        stepper.step_back(1, 1);
        ASSERT_EQ(9, *stepper);
    }

    TEST(xrepeat, stepper_end_row_major_with_repeat_2)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };

        const auto repeated_array = xt::repeat(array, 2, 0);
        auto stepper = repeated_array.stepper_end(layout_type::row_major);

        stepper.step_back(1, 1);
        ASSERT_EQ(9, *stepper);
    }

    TEST(xrepeat, stepper_end_column_major_with_repeat_1)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };

        const auto repeated_array = xt::repeat(array, 1, 0);
        auto stepper = repeated_array.stepper_end(layout_type::column_major);

        stepper.step_back(0, 1);
        ASSERT_EQ(9, *stepper);
    }

    TEST(xrepeat, stepper_end_column_major_with_repeat_2)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 2, 0);
        auto stepper = repeated_array.stepper_end(layout_type::column_major);

        stepper.step_back(0, 1);
        ASSERT_EQ(9, *stepper);
    }

    TEST(xrepeat, assign)
    {
        xt::xarray<int> a = {{1, 2}, {3, 4}};
        xt::xarray<int> b = xt::repeat(a, 3, 1);
        xt::xarray<int> res = {{1, 1, 1, 2, 2, 2}, {3, 3, 3, 4, 4, 4}};
        EXPECT_EQ(b, res);
    }

    TEST(xrepeat_stepper, step_with_repeat_1)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 1, 0);
        auto stepper = repeated_array.stepper_begin();

        stepper.step(0, 2);
        stepper.step(1, 2);

        ASSERT_EQ(9, *stepper);
    }

    TEST(xrepeat_stepper, step_with_repeat_2)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 2, 0);
        auto stepper = repeated_array.stepper_begin();

        stepper.step(0, 2);
        stepper.step(1, 2);

        ASSERT_EQ(6, *stepper);
    }

    TEST(xrepeat_stepper, step_back_with_repeat_1)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 1, 0);
        auto stepper = repeated_array.stepper_begin();
        stepper.step(0, 2);
        stepper.step(1, 2);

        stepper.step_back(0, 1);
        stepper.step_back(1, 1);

        ASSERT_EQ(5, *stepper);
    }

    TEST(xrepeat_stepper, step_back_with_repeat_2)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 2, 0);
        auto stepper = repeated_array.stepper_begin();
        stepper.step(0, 2);
        stepper.step(1, 2);

        stepper.step_back(0, 1);
        stepper.step_back(1, 1);

        ASSERT_EQ(2, *stepper);
    }

    TEST(xrepeat_stepper, step_and_reset_with_repeat_1)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 1, 0);
        auto stepper = repeated_array.stepper_begin();

        stepper.step(0, 2);
        stepper.step(1, 2);
        stepper.reset(0);
        stepper.reset(1);

        ASSERT_EQ(1, *stepper);
    }

    TEST(xrepeat_stepper, step_and_reset_with_repeat_2)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 2, 0);
        auto stepper = repeated_array.stepper_begin();

        stepper.step(0, 5);
        stepper.step(1, 2);
        stepper.reset(0);
        stepper.reset(1);

        ASSERT_EQ(1, *stepper);
    }

    TEST(xrepeat_stepper, reset_back_with_repeat_1)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 1, 0);
        auto stepper = repeated_array.stepper_begin();

        stepper.reset_back(0);
        stepper.reset_back(1);

        ASSERT_EQ(9, *stepper);
    }

    TEST(xrepeat_stepper, reset_back_with_repeat_2)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 2, 0);
        auto stepper0 = repeated_array.stepper_begin();
        auto stepper1 = repeated_array.stepper_begin();

        stepper0.reset_back(0);
        stepper1.reset_back(1);

        ASSERT_EQ(7, *stepper0);
        ASSERT_EQ(3, *stepper1);
    }

    TEST(xrepeat_stepper, reset_back_and_step_back_with_repeat_2)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 2, 0);
        auto stepper = repeated_array.stepper_begin();

        stepper.reset_back(0);
        stepper.reset_back(1);
        stepper.step_back(0, 1);
        stepper.step_back(1, 1);

        ASSERT_EQ(8, *stepper);
    }

    TEST(xrepeat_stepper, to_begin_with_repeat_2)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 2, 0);
        auto stepper = repeated_array.stepper_end(layout_type::row_major);

        stepper.to_begin();

        ASSERT_EQ(1, *stepper);
    }

    TEST(xrepeat_stepper, to_begin_and_step_with_repeat_2)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 2, 0);
        auto stepper = repeated_array.stepper_end(layout_type::row_major);

        stepper.to_begin();
        stepper.step(0, 1);
        stepper.step(1, 1);

        ASSERT_EQ(2, *stepper);
    }

    TEST(xrepeat_stepper, to_end_row_major_with_repeat_2)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 2, 0);
        auto stepper = repeated_array.stepper_begin();

        stepper.to_end(layout_type::row_major);
        stepper.step_back(1, 2);

        ASSERT_EQ(8, *stepper);
    }

    TEST(xrepeat_stepper, to_end_column_major_with_repeat_2)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 2, 0);
        auto stepper = repeated_array.stepper_begin();

        stepper.to_end(layout_type::column_major);
        stepper.step_back(0, 2);

        ASSERT_EQ(9, *stepper);
    }

    TEST(xrepeat_stepper, step_leading_with_repeat_1)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 1, 0);
        auto stepper = repeated_array.stepper_begin();

        stepper.step_leading();
        stepper.step_leading();
        stepper.step_leading();

        ASSERT_EQ(4, *stepper);
    }

    TEST(xrepeat_stepper, step_leading_with_repeat_2)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 2, 0);
        auto stepper = repeated_array.stepper_begin();

        stepper.step_leading();
        stepper.step_leading();
        stepper.step_leading();

        ASSERT_EQ(1, *stepper);
    }

    TEST(xrepeat_stepper, huge_step_with_repeat_1)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 1, 0);
        auto stepper = repeated_array.stepper_begin();

        stepper.step(1, 8);

        ASSERT_EQ(9, *stepper);
    }

    TEST(xrepeat_stepper, huge_step_with_repeat_2)
    {
        xarray<size_t> array = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
        };
        const auto repeated_array = xt::repeat(array, 2, 0);
        auto stepper = repeated_array.stepper_begin();

        stepper.step(1, 17);

        ASSERT_EQ(9, *stepper);
    }
}
