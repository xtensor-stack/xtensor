#include "gtest/gtest.h"
#include "xarray/utils.hpp"
#include <type_traits>
#include <tuple>

namespace qs
{

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

    TEST(utils, for_each_arg)
    {
        for_each_fn fn;
        short a = 1;
        int b = 4;
        float c = 1.2;
        double d = 2.3;
        for_each_arg(fn, a, b, c, d);
        ASSERT_TRUE(a == fn.a && b == fn.b && c == fn.c && d == fn.d);
    }

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
        ASSERT_TRUE(accumulate(func, 0, t) == 8);
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

}

