/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <iterator>
#include <type_traits>
#include <vector>

#include "xtensor/xarray.hpp"
#include "xtensor/xassign.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"

#include "test_common.hpp"
#include "test_common_macros.hpp"

// a dummy shape *not derived* from std::vector but compatible
template <class T>
class my_vector
{
private:

    using vector_type = std::vector<T>;

public:

    using value_type = T;
    using size_type = typename vector_type::size_type;

    template <class U>
    my_vector(std::initializer_list<U> vals)
        : m_data(vals.begin(), vals.end())
    {
    }

    my_vector(const std::size_t size = 0, const T& val = T())
        : m_data(size, val)
    {
    }

    auto resize(const std::size_t size)
    {
        return m_data.resize(size);
    }

    auto size() const
    {
        return m_data.size();
    }

    auto cend() const
    {
        return m_data.cend();
    }

    auto cbegin() const
    {
        return m_data.cbegin();
    }

    auto end()
    {
        return m_data.end();
    }

    auto end() const
    {
        return m_data.end();
    }

    auto begin()
    {
        return m_data.begin();
    }

    auto begin() const
    {
        return m_data.begin();
    }

    auto rbegin() const
    {
        return std::make_reverse_iterator(end());
    }

    auto rend() const
    {
        return std::make_reverse_iterator(begin());
    }

    auto empty() const
    {
        return m_data.empty();
    }

    auto& back()
    {
        return m_data.back();
    }

    const auto& back() const
    {
        return m_data.back();
    }

    auto& front()
    {
        return m_data.front();
    }

    const auto& front() const
    {
        return m_data.front();
    }

    auto& operator[](const std::size_t i)
    {
        return m_data[i];
    }

    const auto& operator[](const std::size_t i) const
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
            using my_xarray = xt::xarray_container<std::vector<int>, xt::layout_type::row_major, my_vector<std::size_t>>;

            auto a = my_xarray::from_shape({1, 3});
            auto b = xt::xtensor<int, 2>::from_shape({2, 3});
            xt::noalias(a) += b;
            EXPECT_EQ(a.dimension(), 2);
            EXPECT_EQ(a.shape(0), 2);
            EXPECT_EQ(a.shape(1), 3);
        }
        {
            // xarray like with custom shape
            using my_xarray = xt::xarray_container<std::vector<int>, xt::layout_type::row_major, my_vector<std::size_t>>;

            auto a = my_xarray::from_shape({3});
            auto b = xt::xtensor<int, 2>::from_shape({2, 3});
            xt::noalias(a) += b;
            EXPECT_EQ(a.dimension(), 2);
            EXPECT_EQ(a.shape(0), 2);
            EXPECT_EQ(a.shape(1), 3);
        }
    }
}
