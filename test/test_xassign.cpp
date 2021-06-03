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
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"

#include "xtensor/xassign.hpp"
#include "xtensor/xnoalias.hpp"
#include "test_common.hpp"

#include <type_traits>
#include <vector>


// a dummy shape *not derived* from std::vector but compatible
template<class T>
class my_vector
{
private:
    using vector_type = std::vector<T>;
public:
    using value_type = T;
    using size_type = typename vector_type::size_type;
    template<class U>
    my_vector(std::initializer_list<U> vals)
    : m_data(vals.begin(), vals.end())
    {
    }
    my_vector(const std::size_t size = 0, const T & val = T())
    : m_data(size, val)
    {
    }
    auto resize(const std::size_t size)
    {
        return m_data.resize(size);
    }
    auto size()const
    {
        return m_data.size();
    }
    auto cend()const
    {
        return m_data.cend();
    }
    auto cbegin()const
    {
        return m_data.cbegin();
    }
    auto end()
    {
        return m_data.end();
    }
    auto end()const
    {
        return m_data.end();
    }
    auto begin()
    {
        return m_data.begin();
    }
    auto begin()const
    {
        return m_data.begin();
    }
    auto empty()const
    {
        return m_data.empty();
    }
    auto & back()
    {
        return m_data.back();
    }
    const auto & back()const
    {
        return m_data.back();
    }
    auto & front()
    {
        return m_data.front();
    }
    const auto & front()const
    {
        return m_data.front();
    }
    auto & operator[](const std::size_t i)
    {
        return m_data[i];
    }
    const auto & operator[](const std::size_t i)const
    {
        return m_data[i];
    }
private:
    std::vector<T> m_data;
};


namespace xt
{

    template <class T, class C_T>
    struct rebind_container<T, my_vector<C_T>>
    {
        using type = my_vector<T>;
    };

    TEST(xassign, mix_shape_types)
    {
        {
            // xarray like with custom shape
            using my_xarray = xt::xarray_container<
                std::vector<int>,
                xt::layout_type::row_major,
                my_vector<std::size_t>
            >;

            auto a = my_xarray::from_shape({1,3});
            auto b = xt::xtensor<int,2>::from_shape({2,3});
            xt::noalias(a) += b;
            EXPECT_EQ(a.dimension(), 2);
            EXPECT_EQ(a.shape(0), 2);
            EXPECT_EQ(a.shape(1), 3);
        }
        {
            // xarray like with custom shape
            using my_xarray = xt::xarray_container<
                std::vector<int>,
                xt::layout_type::row_major,
                my_vector<std::size_t>
            >;

            auto a = my_xarray::from_shape({3});
            auto b = xt::xtensor<int,2>::from_shape({2,3});
            xt::noalias(a) += b;
            EXPECT_EQ(a.dimension(), 2);
            EXPECT_EQ(a.shape(0), 2);
            EXPECT_EQ(a.shape(1), 3);
        }
    }

    TEST(xassign, fixed_shape)
    {
        // matching shape 1D
        {
            xt::xtensor_fixed<int, xt::xshape<2>> a = {2,3};
            xt::xtensor_fixed<int, xt::xshape<2>> b = {3,4};

            xt::noalias(a) += b;

            EXPECT_EQ(a(0), 5);
            EXPECT_EQ(a(1), 7);
        }
        //matching shape 2D
        {
            xt::xtensor_fixed<int, xt::xshape<2,2>> aa = {{1,2},{3,4}};
            xt::xtensor_fixed<int, xt::xshape<2,2>> a(aa);
            xt::xtensor_fixed<int, xt::xshape<2,2>> b = {{3,4},{5,6}};
            xt::noalias(a) += b;

            EXPECT_EQ(a(0,0),  aa(0,0) + b(0,0));
            EXPECT_EQ(a(0,1),  aa(0,1) + b(0,1));
            EXPECT_EQ(a(1,0),  aa(1,0) + b(1,0));
            EXPECT_EQ(a(1,1),  aa(1,1) + b(1,1));
        }
        // b is broadcasted with matching dimension (first axis is singleton)
        {
            xt::xtensor_fixed<int, xt::xshape<2,2>> aa = {{1,2},{3,4}};
            xt::xtensor_fixed<int, xt::xshape<2,2>> a(aa);
            xt::xtensor_fixed<int, xt::xshape<2,1>> b = {{5,6}};
            EXPECT_EQ(b.shape(0),2);
            EXPECT_EQ(b.shape(1),1);
            xt::noalias(a) += b;

            EXPECT_EQ(a(0,0),  aa(0,0) + b.at(0,0));
            EXPECT_EQ(a(0,1),  aa(0,1) + b.at(0,0));
            EXPECT_EQ(a(1,0),  aa(1,0) + b.at(1,0));
            EXPECT_EQ(a(1,1),  aa(1,1) + b.at(1,0));
        }
        // b is broadcasted with matching dimension (second axis is singleton)
        {
            xt::xtensor_fixed<int, xt::xshape<2,2>> aa = {{1,2},{3,4}};
            xt::xtensor_fixed<int, xt::xshape<2,2>> a(aa);
            xt::xtensor_fixed<int, xt::xshape<1,2>> b = {{3,4}};
            EXPECT_EQ(b.shape(0),1);
            EXPECT_EQ(b.shape(1),2);
            xt::noalias(a) += b;

            EXPECT_EQ(a(0,0),  aa(0,0) + b(0,0));
            EXPECT_EQ(a(0,1),  aa(0,1) + b(0,1));
            EXPECT_EQ(a(1,0),  aa(1,0) + b(0,0));
            EXPECT_EQ(a(1,1),  aa(1,1) + b(0,1));
        }
        // broadcast with non matching dimensions
        {
            xt::xtensor_fixed<int, xt::xshape<2,2>> aa = {{1,2},{3,4}};
            xt::xtensor_fixed<int, xt::xshape<2,2>> a(aa);
            xt::xtensor_fixed<int, xt::xshape<2>> b = {3,4};

            xt::noalias(a) += b;

            EXPECT_EQ(a(0,0),  aa(0,0) + b(0));
            EXPECT_EQ(a(0,1),  aa(0,1) + b(1));
            EXPECT_EQ(a(1,0),  aa(1,0) + b(0));
            EXPECT_EQ(a(1,1),  aa(1,1) + b(1));
        }
    }
    TEST(xassign, fixed_raises)
    {
        // cannot broadcast a  itself on assignment
        {
            xt::xtensor_fixed<int, xt::xshape<2>> a = {2,3};
            xt::xtensor_fixed<int, xt::xshape<2,2>> b = {{3,4},{3,4}};

            EXPECT_THROW(xt::noalias(a) += b, xt::broadcast_error);
        }

        // cannot broadcast a  itself on assignment
        {
            xt::xtensor_fixed<int, xt::xshape<1,2>> a = {{3,4}};
            xt::xtensor_fixed<int, xt::xshape<2,2>> b = {{3,4},{3,4}};

            EXPECT_THROW(xt::noalias(a) += b, xt::broadcast_error);
        }

        // cannot broadcast a  itself on assignment
        {
            xt::xtensor_fixed<int, xt::xshape<2,1>> a = {{3},{4}};
            xt::xtensor_fixed<int, xt::xshape<2,2>> b = {{3,4},{3,4}};

            EXPECT_THROW(xt::noalias(a) += b, xt::broadcast_error);
        }
    }
}
