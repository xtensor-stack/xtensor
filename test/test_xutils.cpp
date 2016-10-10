#include "gtest/gtest.h"
#include "xarray/xutils.hpp"
#include <initializer_list>
#include <type_traits>
#include <tuple>

namespace qs
{
    using std::size_t;

    struct for_each_fn
    {
        short a;
        int b;
        float c;
        double d;

        template <class T>
        void operator()(T t)
        {
            if(std::is_same<T, short>::value)
                a = t;
            else if(std::is_same<T, int>::value)
                b = t;
            else if(std::is_same<T, float>::value)
                c = t;
            else if(std::is_same<T, double>::value)
                d = t;
        }
    };

    TEST(utils, for_each)
    {
        for_each_fn fn;
        short a = 1;
        int b = 4;
        float c = 1.2;
        double d = 2.3;
        auto t = std::make_tuple(a, b, c, d);
        for_each(fn, t);
        ASSERT_TRUE(a == fn.a && b == fn.b && c == fn.c && d == fn.d);
    }

    TEST(utils, accumulate)
    {
        auto func = [](int i, int j) { return i + j; };
        const std::tuple<int, int, int> t(3, 4, 1);
        ASSERT_EQ(8, accumulate(func, 0, t));
    }

    TEST(utils, or)
    {
        using true_t = std::true_type;
        using false_t = std::false_type;

        using t1 = or_<false_t, false_t, false_t>;
        using t2 = or_<false_t, true_t, false_t>;

        ASSERT_TRUE(!t1::value);
        ASSERT_TRUE(t2::value);
    }

    template <class... T >
    auto foo(T... t) {
        auto func = [](int i) { return i; };
        return apply<int>(1, func, t...);
    }

    TEST(utils, apply)
    {
        ASSERT_TRUE(foo(1, 2, 3)==2);
    }

    TEST(utils, initializer_dimension)
    {
        size_t d0 = initializer_dimension<double>::value;
        size_t d1 = initializer_dimension<std::initializer_list<double>>::value;
        size_t d2 = initializer_dimension<std::initializer_list<std::initializer_list<double>>>::value;
        ASSERT_EQ(0, d0);
        ASSERT_EQ(1, d1);
        ASSERT_EQ(2, d2);
    }

    TEST(utils, initializer_shape)
    {
        auto s0 = initializer_shape(3);
        auto s1 = initializer_shape({1, 2});
        auto s2 = initializer_shape({{1, 2, 4}, {1, 3, 5}});
        std::array<size_t, 0> e0 = {};
        std::array<size_t, 1> e1 = {2};
        std::array<size_t, 2> e2 = {2, 3};

        ASSERT_EQ(e0, s0);
        ASSERT_EQ(e1, s1);
        ASSERT_EQ(e2, s2);
    }

}

