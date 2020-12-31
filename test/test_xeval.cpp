/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xtensor_config.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

namespace xt
{
    TEST(xeval, array_tensor)
    {
        xarray<double> a = {1, 2, 3, 4};

        auto&& b = eval(a);

        EXPECT_EQ(a.storage().data(), b.storage().data());
        EXPECT_EQ(&a, &b);
        bool type_eq = std::is_same<decltype(b), xarray<double>&>::value;
        EXPECT_TRUE(type_eq);

        xtensor<double, 2> t({3, 3});

        auto&& i = eval(t);

        EXPECT_EQ(t.storage().data(), i.storage().data());
        EXPECT_EQ(&t, &i);
        bool type_eq_2 = std::is_same<decltype(i), xtensor<double, 2>&>::value;
        EXPECT_TRUE(type_eq_2);
    }

    TEST(xeval, funcs)
    {
        xarray<double> a = {1, 2, 3, 4};

        auto f = a * a - 2;
        auto&& b = eval(f);

        bool type_eq = std::is_same<decltype(b), xarray<double>&&>::value;
        EXPECT_TRUE(type_eq);

        xtensor<int, 2> k({3, 3});
        auto m = k * k - 4;
        auto&& n = eval(m);
        bool type_eq_3 = std::is_same<decltype(n), xtensor<int, 2>&&>::value;
        EXPECT_TRUE(type_eq_3);

        auto&& i = eval(linspace(0, 100));
        bool type_eq_2 = std::is_same<decltype(i), xtensor<int, 1>&&>::value;
        EXPECT_TRUE(type_eq_2);
    }


#define EXPECT_LAYOUT(EXPRESSION, LAYOUT)                         \
  EXPECT_TRUE((decltype(EXPRESSION)::static_layout == LAYOUT)) 

#define HAS_DATA_INTERFACE(EXPRESSION)                            \
  has_data_interface<std::decay_t<decltype(EXPRESSION)>>::value

#define EXPECT_XARRAY(EXPRESSION)                                    \
  EXPECT_TRUE(!detail::is_array<                                     \
                        typename std::decay_t<decltype(EXPRESSION)   \
                                >::shape_type>::value) 

#define EXPECT_XTENSOR(EXPRESSION)                                   \
  EXPECT_TRUE(detail::is_array<                                      \
                        typename std::decay_t<decltype(EXPRESSION)   \
                                >::shape_type>::value == true) 


    TEST(utils, has_same_layout)
    {
        xt::xtensor<double, 1, layout_type::row_major> ten1 {1., 2., 3.2};
        EXPECT_TRUE(detail::has_same_layout<layout_type::row_major>(ten1));
        EXPECT_FALSE(detail::has_same_layout<layout_type::column_major>(ten1));
        EXPECT_TRUE(detail::has_same_layout<layout_type::any>(ten1));

        xt::xtensor<double, 1, layout_type::column_major> ten2 {1., 2., 3.2};
        EXPECT_TRUE(detail::has_same_layout<layout_type::column_major>(ten2));
        EXPECT_FALSE(detail::has_same_layout<layout_type::row_major>(ten2));
        EXPECT_TRUE(detail::has_same_layout<layout_type::any>(ten2));
        
        EXPECT_FALSE((detail::has_same_layout(ten1, ten2)));
        EXPECT_TRUE((detail::has_same_layout(ten1, xt::xtensor<double, 1, layout_type::row_major>({1., 2., 3.2}))));
        EXPECT_TRUE((detail::has_same_layout(ten2, xt::xtensor<double, 1, layout_type::column_major>({1., 2., 3.2}))));
    }

    TEST(utils, has_fixed_dims)
    {
        xt::xtensor<double, 1> ten {1., 2., 3.2};
        EXPECT_TRUE((detail::has_fixed_dims<xt::xtensor<double, 1>>()));
        EXPECT_TRUE(detail::has_fixed_dims(ten));

        xt::xarray<double> arr {1., 2., 3.2};
        EXPECT_FALSE((detail::has_fixed_dims<xt::xarray<double>>()));
        EXPECT_FALSE(detail::has_fixed_dims(arr));
    }

    TEST(utils, as_xarray_container_t)
    {
        using array_type = xt::xarray<double, layout_type::row_major>;

        detail::as_xarray_container_t<array_type, layout_type::column_major> arr;
        EXPECT_XARRAY(arr);
        EXPECT_LAYOUT(arr, layout_type::column_major);
    }

    TEST(utils, as_xtensor_container_t)
    {
        using tensor_type = xt::xtensor<double, 1, layout_type::row_major>;

        detail::as_xtensor_container_t<tensor_type, layout_type::column_major> ten;
        EXPECT_XTENSOR(ten);
        EXPECT_LAYOUT(ten, layout_type::column_major);
    }

    namespace testing
    { // avoid collision with fixture class

        class as_strided: public ::testing::Test
        {
            protected:

                xt::xtensor<double, 1, layout_type::row_major> ten {1., 2., 3.2};
                xt::xarray<double, layout_type::row_major> arr {1., 2., 3.2};
        };

        TEST_F(as_strided, array_reference)
        {
            EXPECT_LAYOUT(arr, layout_type::row_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(arr));
            EXPECT_XARRAY(arr);
            EXPECT_EQ(arr(2), 3.2);
        }

        TEST_F(as_strided, tensor_reference)
        {
            EXPECT_LAYOUT(ten, layout_type::row_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(ten));
            EXPECT_XTENSOR(ten);
            EXPECT_EQ(ten(2), 3.2);
        }

        TEST_F(as_strided, array_layout_unchanged)
        {
            auto res_lvalue = xt::as_strided<layout_type::row_major>(arr);
            EXPECT_LAYOUT(res_lvalue, layout_type::row_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_lvalue));
            EXPECT_XARRAY(res_lvalue);
            EXPECT_EQ(res_lvalue(2), 3.2);

            auto res_rvalue = xt::as_strided<layout_type::row_major>(
                                xt::xarray<double, layout_type::row_major>({1., 2., 3.2})
                                );
            EXPECT_LAYOUT(res_rvalue, layout_type::row_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_rvalue));
            EXPECT_XARRAY(res_rvalue);
            EXPECT_EQ(res_rvalue(2), 3.2);
        }

        TEST_F(as_strided, tensor_layout_unchanged)
        {
            auto res_lvalue = xt::as_strided<layout_type::row_major>(ten);
            EXPECT_LAYOUT(res_lvalue, layout_type::row_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_lvalue));
            EXPECT_XTENSOR(res_lvalue);
            EXPECT_EQ(res_lvalue(2), 3.2);

            auto res_rvalue = xt::as_strided<layout_type::row_major>(
                                xt::xtensor<double, 1, layout_type::row_major>({1., 2., 3.2})
                                );
            EXPECT_LAYOUT(res_rvalue, layout_type::row_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_rvalue));
            EXPECT_XTENSOR(res_rvalue);
            EXPECT_EQ(res_rvalue(2), 3.2);
        }

        TEST_F(as_strided, array_layout_change)
        {
            auto res_lvalue = xt::as_strided<layout_type::column_major>(arr);
            EXPECT_LAYOUT(res_lvalue, layout_type::column_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_lvalue));
            EXPECT_XARRAY(res_lvalue);
            EXPECT_EQ(res_lvalue(2), 3.2);

            auto res_rvalue = xt::as_strided<layout_type::column_major>(
                                xt::xarray<double, layout_type::row_major>({1., 2., 3.2})
                                );
            EXPECT_LAYOUT(res_rvalue, layout_type::column_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_rvalue));
            EXPECT_XARRAY(res_rvalue);
            EXPECT_EQ(res_rvalue(2), 3.2);
        }

        TEST_F(as_strided, tensor_layout_changed)
        {
            auto res_lvalue = xt::as_strided<layout_type::column_major>(ten);
            EXPECT_LAYOUT(res_lvalue, layout_type::column_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_lvalue));
            EXPECT_XTENSOR(res_lvalue);
            EXPECT_EQ(res_lvalue(2), 3.2);

            auto res_rvalue = xt::as_strided<layout_type::column_major>(
                                xt::xtensor<double, 1, layout_type::row_major>({1., 2., 3.2})
                                );
            EXPECT_LAYOUT(res_rvalue, layout_type::column_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_rvalue));
            EXPECT_XTENSOR(res_rvalue);
            EXPECT_EQ(res_rvalue(2), 3.2);
        }

        TEST_F(as_strided, array_no_data_interface_layout_unchanged)
        {
            auto array_cast = xt::cast<int>(arr);
            EXPECT_FALSE(HAS_DATA_INTERFACE(array_cast));
            EXPECT_XARRAY(array_cast);

            auto res_lvalue = xt::as_strided<layout_type::row_major>(array_cast);
            EXPECT_LAYOUT(res_lvalue, layout_type::row_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_lvalue));
            EXPECT_XARRAY(res_lvalue);
            EXPECT_EQ(res_lvalue(2), 3);

            auto res_rvalue = xt::as_strided<layout_type::row_major>(
                                xt::cast<int>(arr)
                                );
            EXPECT_LAYOUT(res_rvalue, layout_type::row_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_rvalue));
            EXPECT_XARRAY(res_rvalue);
            EXPECT_EQ(res_rvalue(2), 3);
        }

        TEST_F(as_strided, tensor_no_data_interface_layout_unchanged)
        {
            auto tensor_cast = xt::cast<int>(ten);
            EXPECT_FALSE(HAS_DATA_INTERFACE(tensor_cast));
            EXPECT_XTENSOR(tensor_cast);

            auto res_lvalue = xt::as_strided<layout_type::row_major>(tensor_cast);
            EXPECT_LAYOUT(res_lvalue, layout_type::row_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_lvalue));
            EXPECT_XTENSOR(res_lvalue);
            EXPECT_EQ(res_lvalue(2), 3);

            auto res_rvalue = xt::as_strided<layout_type::row_major>(
                                xt::cast<int>(ten)
                                );
            EXPECT_LAYOUT(res_rvalue, layout_type::row_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_rvalue));
            EXPECT_XTENSOR(res_rvalue);
            EXPECT_EQ(res_rvalue(2), 3);
        }

        TEST_F(as_strided, array_no_data_interface_layout_changed)
        {
            auto array_cast = xt::cast<int>(arr);
            EXPECT_FALSE(HAS_DATA_INTERFACE(array_cast));
            EXPECT_XARRAY(array_cast);

            auto res_lvalue = xt::as_strided<layout_type::column_major>(array_cast);
            EXPECT_LAYOUT(res_lvalue, layout_type::column_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_lvalue));
            EXPECT_XARRAY(res_lvalue);
            EXPECT_EQ(res_lvalue(2), 3);

            auto res_rvalue = xt::as_strided<layout_type::column_major>(
                                xt::cast<int>(arr)
                                );
            EXPECT_LAYOUT(res_rvalue, layout_type::column_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_rvalue));
            EXPECT_XARRAY(res_rvalue);
            EXPECT_EQ(res_rvalue(2), 3);
        }

        TEST_F(as_strided, tensor_no_data_interface_layout_changed)
        {
            auto tensor_cast = xt::cast<int>(ten);
            EXPECT_FALSE(HAS_DATA_INTERFACE(tensor_cast));
            EXPECT_XTENSOR(tensor_cast);

            auto res_lvalue = xt::as_strided<layout_type::column_major>(tensor_cast);
            EXPECT_LAYOUT(res_lvalue, layout_type::column_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_lvalue));
            EXPECT_XTENSOR(res_lvalue);
            EXPECT_EQ(res_lvalue(2), 3);

            auto res_rvalue = xt::as_strided<layout_type::column_major>(
                                xt::cast<int>(ten)
                                );
            EXPECT_LAYOUT(res_rvalue, layout_type::column_major);
            EXPECT_TRUE(HAS_DATA_INTERFACE(res_rvalue));
            EXPECT_XTENSOR(res_rvalue);
            EXPECT_EQ(res_rvalue(2), 3);
        }
    }
}
