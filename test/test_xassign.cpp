/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <iterator>
#include <memory>
#include <type_traits>
#include <vector>

#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xassign.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor/generators/xrandom.hpp"
#include "xtensor/misc/xmanipulation.hpp"
#include "xtensor/views/xview.hpp"

#include "test_common.hpp"
#include "test_common_macros.hpp"

namespace
{
    // Counts the buffers allocated for the data of a container.
    template <class T>
    struct counting_allocator : std::allocator<T>
    {
        using value_type = T;
        inline static std::size_t allocations = 0;

        template <class U>
        struct rebind
        {
            using other = counting_allocator<U>;
        };

        counting_allocator() = default;

        template <class U>
        counting_allocator(const counting_allocator<U>&)
        {
        }

        T* allocate(std::size_t n)
        {
            ++allocations;
            return std::allocator<T>::allocate(n);
        }
    };

    // A generator functor that reads another expression, and therefore must
    // keep the overlap protection.
    struct reverse_reader
    {
        using value_type = int;
        const xt::xarray<int>& source;

        int operator()(std::size_t i) const
        {
            return source(source.size() - 1 - i);
        }

        template <class It>
        int element(It first, It) const
        {
            return (*this)(*first);
        }
    };
}

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

    TEST(xassign, memory_independent_generator_trait)
    {
        xarray<int> a = {1, 2, 3, 4};
        auto reverse = detail::make_xgenerator(reverse_reader{a}, {4});
        auto independent = [](const auto& e)
        {
            return detail::xgenerator_is_memory_independent<std::decay_t<decltype(e)>>::value;
        };

        EXPECT_TRUE(independent(arange<double>(4)));
        EXPECT_TRUE(independent(arange<int>(0, 8, 2)));
        EXPECT_TRUE(independent(linspace<double>(0., 1., 4)));
        EXPECT_TRUE(independent(eye<double>(3)));
        EXPECT_TRUE(independent(random::rand<double>({4})));
        EXPECT_TRUE(independent(random::randint<int>({4})));

        // Functors holding expressions, and unknown functors, are not.
        EXPECT_FALSE(independent(reverse));
        EXPECT_FALSE(independent(concatenate(xtuple(a, a))));
        EXPECT_FALSE(independent(stack(xtuple(a, a))));
        EXPECT_FALSE(independent(diag(a)));
        EXPECT_FALSE(independent(a));
    }

#ifndef XTENSOR_FORCE_TEMPORARY_MEMORY_IN_ASSIGNMENTS
    TEST(xassign, memory_independent_generator_reuses_storage)
    {
        using allocator = counting_allocator<double>;
        using vector_type = xtensor<double, 1, layout_type::row_major, allocator>;
        using matrix_type = xtensor<double, 2, layout_type::row_major, allocator>;

        vector_type a = vector_type::from_shape({16});
        allocator::allocations = 0;
        a = arange<double>(16);
        EXPECT_EQ(allocator::allocations, 0u);
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            EXPECT_EQ(a(i), double(i));
        }

        allocator::allocations = 0;
        a = linspace<double>(0., 30., 16);
        EXPECT_EQ(allocator::allocations, 0u);
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            EXPECT_EQ(a(i), double(2 * i));
        }

        allocator::allocations = 0;
        a = random::rand<double>({16}, 2., 3.);
        EXPECT_EQ(allocator::allocations, 0u);
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            EXPECT_GE(a(i), 2.);
            EXPECT_LT(a(i), 3.);
        }

        matrix_type m = matrix_type::from_shape({4, 4});
        allocator::allocations = 0;
        m = eye<double>(4);
        EXPECT_EQ(allocator::allocations, 0u);
        for (std::size_t i = 0; i < 4; ++i)
        {
            for (std::size_t j = 0; j < 4; ++j)
            {
                EXPECT_EQ(m(i, j), i == j ? 1. : 0.);
            }
        }
    }
#endif

    TEST(xassign, memory_independent_generator_resizes)
    {
        xarray<double> a = {{-1., -1.}, {-1., -1.}};
        a = arange<double>(5);
        EXPECT_EQ(a, (xarray<double>{0., 1., 2., 3., 4.}));

        a = arange<double>(0);
        EXPECT_EQ(a.size(), 0u);

        a = linspace<double>(2., 4., 3);
        EXPECT_EQ(a, (xarray<double>{2., 3., 4.}));

        a = eye<double>(2);
        EXPECT_EQ(a, (xarray<double>{{1., 0.}, {0., 1.}}));
    }

    TEST(xassign, memory_independent_generator_to_view)
    {
        xarray<double> a = {{-1., -1., -1.}, {-1., -1., -1.}, {-1., -1., -1.}};
        view(a, 1, all()) = arange<double>(3);
        EXPECT_EQ(a, (xarray<double>{{-1., -1., -1.}, {0., 1., 2.}, {-1., -1., -1.}}));

        view(a, all(), 2) = linspace<double>(4., 6., 3);
        EXPECT_EQ(a, (xarray<double>{{-1., -1., 4.}, {0., 1., 5.}, {-1., -1., 6.}}));

        view(a, range(0, 2), range(0, 2)) = eye<double>(2);
        EXPECT_EQ(a, (xarray<double>{{1., 0., 4.}, {0., 1., 5.}, {-1., -1., 6.}}));
    }

    TEST(xassign, reading_generators_keep_overlap_protection)
    {
        xarray<int> a = {1, 2, 3, 4};
        a = detail::make_xgenerator(reverse_reader{a}, {4});
        EXPECT_EQ(a, (xarray<int>{4, 3, 2, 1}));

        a = concatenate(xtuple(a, a));
        EXPECT_EQ(a, (xarray<int>{4, 3, 2, 1, 4, 3, 2, 1}));

        xarray<int> m = {{1, 2}, {3, 4}};
        m = diagonal(m);
        EXPECT_EQ(m, (xarray<int>{1, 4}));
    }

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

    TEST(xassign, fixed_dimension_mismatch)
    {
        auto a = xt::xtensor<float, 1>::from_shape({2});
        XT_ASSERT_THROW(a = xt::expand_dims(a, 0), std::runtime_error);
    }
}
