/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ENABLE_ASSERT
#define XTENSOR_ENABLE_ASSERT
#endif

#include <numeric>
#include <limits>
#include <iostream>

#include <gtest/gtest.h>
#include <xtensor/xtiny.hpp>

namespace xt
{
    static const unsigned SIZE = 3;

    template <class T>
    struct xtiny_test_data
    {
        static T data[SIZE];
    };

    template <class T>
    T xtiny_test_data<T>::data[SIZE] = {1, 2, 4};

    template <>
    float xtiny_test_data<float>::data[SIZE] = { 1.2f, 2.4f, 4.6f};

    template <class T>
    class xtiny_test : public testing::Test
    {
    };

    typedef testing::Types<xtiny<uint8_t, SIZE>,
                           xtiny<int, SIZE>,
                           xtiny<float, SIZE>,
                           xtiny<int, runtime_size>,         // buffer_size > SIZE
                           xtiny<int, runtime_size, int[1]>  // buffer_size < SIZE
                          > xtiny_types;

    TYPED_TEST_CASE(xtiny_test, xtiny_types);

    TYPED_TEST(xtiny_test, construction)
    {
        using V = TypeParam;
        using T = typename V::value_type;
        using size_type = typename V::size_type;
        const bool fixed = V::has_fixed_size;

        T * data = xtiny_test_data<T>::data;

        V v0,
          v1(SIZE, 1),
          v3(data, data+SIZE);

        EXPECT_EQ(v0.size(), fixed ? SIZE : 0);
        EXPECT_EQ(v1.size(), SIZE);
        EXPECT_EQ(v3.size(), SIZE);
        EXPECT_EQ(v0.empty(), !fixed);
        EXPECT_FALSE(v1.empty());
        EXPECT_FALSE(v3.empty());
        EXPECT_EQ(v0.shape(), (std::array<size_t, 1>{fixed ? SIZE : 0}));
        EXPECT_EQ(v1.shape(), (std::array<size_t, 1>{SIZE}));
        EXPECT_EQ(v3.shape(), (std::array<size_t, 1>{SIZE}));

        EXPECT_EQ(v3.front(), data[0]);
        EXPECT_EQ(v3.back(), data[SIZE-1]);
        V const & cv3 = v3;
        EXPECT_EQ(cv3.front(), data[0]);
        EXPECT_EQ(cv3.back(), data[SIZE-1]);

        auto v3iter   = v3.begin();
        auto v3citer  = v3.cbegin();
        auto v3riter  = v3.rbegin();
        auto v3criter = v3.crbegin();
        for(size_type k=0; k<v3.size(); ++k, ++v3iter, ++v3citer, ++v3riter, ++v3criter)
        {
            if(fixed)
            {
                EXPECT_EQ(v0[k], 0);
            }
            EXPECT_EQ(v1[k], 1);
            EXPECT_EQ(v3[k], data[k]);
            EXPECT_EQ(v3.at(k), data[k]);
            EXPECT_EQ(*v3iter, data[k]);
            EXPECT_EQ(*v3citer, data[k]);
            EXPECT_EQ(*v3riter, data[SIZE-1-k]);
            EXPECT_EQ(*v3criter, data[SIZE-1-k]);
        }
        EXPECT_EQ(v3iter, v3.end());
        EXPECT_EQ(v3citer, v3.cend());
        EXPECT_EQ(v3riter, v3.rend());
        EXPECT_EQ(v3criter, v3.crend());
        EXPECT_THROW(v3.at(SIZE), std::out_of_range);

        EXPECT_EQ(v3, V(v3));
        EXPECT_EQ(v3, V(v3.begin(), v3.end()));
        if(fixed)
        {
            EXPECT_THROW(V(v3.begin(), v3.begin()+SIZE-1), std::runtime_error);
            EXPECT_THROW((V{1,2}), std::runtime_error);
        }

        if(std::is_integral<T>::value)
        {
            EXPECT_EQ(v3, (V{1, 2, 4}));
        }
        else
        {
            EXPECT_EQ(v3, (V{1.2f, 2.4f, 4.6f}));
        }
        if(fixed)
        {
            EXPECT_EQ(v1, (V{1}));
         }
        else
        {
            EXPECT_EQ((xtiny<T, 1>(1,1)), (V{1}));
        }

        V v;
        v.assign(SIZE, 1);
        EXPECT_EQ(v1, v);
        v.assign({1.2f, 2.4f, 4.6f});
        EXPECT_EQ(v3, v);

        v = 1;
        EXPECT_EQ(v, v1);
        v = v3;
        EXPECT_EQ(v, v3);

        V v2(v1), v4(v3);
        swap(v2, v4);
        EXPECT_EQ(v3, v2);
        EXPECT_EQ(v1, v4);

        // testing move constructor and assignment
        v4 = v3.push_back(0).pop_back();
        EXPECT_EQ(v4, v3);
        EXPECT_EQ(V(v3.push_back(0).pop_back()), v3);
    }

    TYPED_TEST(xtiny_test, subarray)
    {
        using V = TypeParam;
        using A = typename V::template rebind_size<runtime_size>;
        using T = typename V::value_type;

        T * data = xtiny_test_data<T>::data;
        V v3(data, data+SIZE);
        V const & cv3 = v3;

        EXPECT_EQ(v3, (v3.template subarray<0, SIZE>()));
        EXPECT_EQ(2u, (v3.template subarray<0, 2>().size()));
        EXPECT_EQ(v3[0], (v3.template subarray<0, 2>()[0]));
        EXPECT_EQ(v3[1], (v3.template subarray<0, 2>()[1]));
        EXPECT_EQ(2u, (v3.template subarray<1, 3>().size()));
        EXPECT_EQ(v3[1], (v3.template subarray<1, 3>()[0]));
        EXPECT_EQ(v3[2], (v3.template subarray<1, 3>()[1]));
        EXPECT_EQ(1u, (v3.template subarray<1, 2>().size()));
        EXPECT_EQ(v3[1], (v3.template subarray<1, 2>()[0]));
        EXPECT_EQ(1u, (v3.subarray(1, 2).size()));
        EXPECT_EQ(v3[1], (v3.subarray(1, 2)[0]));
        EXPECT_EQ(1u, (cv3.template subarray<1, 2>().size()));
        EXPECT_EQ(v3[1], (cv3.template subarray<1, 2>()[0]));
        EXPECT_EQ(1u, (cv3.subarray(1, 2).size()));
        EXPECT_EQ(v3[1], (cv3.subarray(1, 2)[0]));

        A r{ 2,3,4,5 };
        EXPECT_EQ(r, (A{ 2,3,4,5 }));
        EXPECT_EQ(r.subarray(1, 3).size(), 2u);
        EXPECT_EQ(r.subarray(1, 3), (A{ 3,4 }));
        EXPECT_EQ((r.template subarray<1, 3>().size()), 2u);
        EXPECT_EQ((r.template subarray<1, 3>()), (A{ 3,4 }));
    }

    TYPED_TEST(xtiny_test, erase_insert)
    {
        using V = TypeParam;
        using V1 = typename V::template rebind_size<SIZE - 1>;
        using T = typename V::value_type;

        T * data = xtiny_test_data<T>::data;
        V v3(data, data+SIZE);
        V1 v10(v3.begin(), v3.begin()+SIZE-1);

        EXPECT_EQ(v10, v3.erase(SIZE - 1));
        EXPECT_EQ(v3, v10.insert(SIZE - 1, v3[SIZE - 1]));
        EXPECT_EQ(v10, v3.pop_back());
        EXPECT_EQ(v3, v10.push_back(v3[SIZE - 1]));
        V1 v11(v3.begin() + 1, v3.begin() + SIZE);
        EXPECT_EQ(v11, v3.erase(0));
        EXPECT_EQ(v3, v11.insert(0, v3[0]));
        EXPECT_EQ(v11, v3.pop_front());
        EXPECT_EQ(v3, v11.push_front(v3[0]));
    }

    TYPED_TEST(xtiny_test, comparison)
    {
        using V = TypeParam;
        using T = typename V::value_type;
        const bool fixed = V::has_fixed_size;

        T * data = xtiny_test_data<T>::data;

        V v1{1},
          v2(SIZE, 1),
          v3(data, data+SIZE);

        EXPECT_TRUE(v3 == v3);
        EXPECT_EQ(v1 == v2, fixed);
        EXPECT_TRUE(v1 == v1);
        EXPECT_TRUE(v1 == 1);
        EXPECT_TRUE(1 == v1);
        EXPECT_TRUE(v1 != v3);
        EXPECT_TRUE(v2 != v3);
        EXPECT_TRUE(v1 != 0);
        EXPECT_TRUE(0 != v1);
        EXPECT_TRUE(v2 != 0);
        EXPECT_TRUE(0 != v2);
    }

    TYPED_TEST(xtiny_test, ostream)
    {
        using V = TypeParam;
        using T = typename V::value_type;

        T * data = xtiny_test_data<T>::data;
        V v3(data, data+SIZE);

        std::ostringstream out;
        out << v3;
        if(std::is_integral<T>::value)
        {
            std::string expected("{1, 2, 4}");
            EXPECT_EQ(expected, out.str());
        }
    }

    TEST(xtiny, conversion)
    {
        using IV = xtiny<int, SIZE>;
        using FV = xtiny<float, SIZE>;
        IV iv{1,2,3},
           iv1(SIZE, 1);
        FV fv{1.1f,2.2f,3.3f},
           fv1{1.0};

        EXPECT_TRUE(iv1 == fv1);
        EXPECT_TRUE(fv1 == iv1);
        EXPECT_TRUE(iv != fv);
        EXPECT_TRUE(fv != iv);
        EXPECT_TRUE(iv == IV(fv));
        EXPECT_TRUE(iv == (IV{1.1f,2.2f,3.3f}));
        EXPECT_TRUE(iv1 == IV{1.2});
        iv1 = fv;
        EXPECT_TRUE(iv1 == iv);
    }

    TEST(xtiny, interoperability)
    {
        using A = xtiny<int, 4>;
        using B = xtiny<int>;
        using C = xtiny<int, 4, int *>;
        using D = xtiny<int, runtime_size, int *>;
        using E = xtiny<int, runtime_size, xbuffer_adaptor<int *>>;

        EXPECT_TRUE(xtiny_concept<A>::value);
        EXPECT_TRUE(xtiny_concept<B>::value);
        EXPECT_TRUE(xtiny_concept<C>::value);
        EXPECT_TRUE(xtiny_concept<D>::value);
        EXPECT_TRUE(xtiny_concept<E>::value);

        static const size_t s = 4;
        std::array<int, s> data{1,2,3,4};
        A a(data.begin());
        B b(a);
        C c(a);
        D d(a);
        E e(data.data(), s);
        std::vector<int> v(data.begin(), data.end());

        EXPECT_EQ(b, a);
        EXPECT_EQ(c, a);
        EXPECT_EQ(d, a);
        EXPECT_EQ(e, a);

        EXPECT_EQ(A(data), a);
        EXPECT_EQ(B(data), a);
        EXPECT_EQ(C(data), a);
        EXPECT_EQ(D(data), a);

        EXPECT_EQ(A(v), a);
        EXPECT_EQ(B(v), a);
        EXPECT_EQ(C(v), a);
        EXPECT_EQ(D(v), a);

        EXPECT_EQ((A{1,2,3,4}), a);
        EXPECT_EQ((B{1,2,3,4}), a);

        data[0] = 0;
        EXPECT_EQ(e, (B{0,2,3,4}));
        EXPECT_EQ((a = e), e);
        EXPECT_EQ((b = e), e);
        EXPECT_EQ((c = e), e);
        EXPECT_EQ((d = e), e);

        EXPECT_EQ((a = 1), A{1});
        EXPECT_EQ((b = 1), A{1});
        EXPECT_EQ((c = 1), A{1});
        EXPECT_EQ((d = 1), A{1});
        EXPECT_EQ((e = 1), A{1});

        EXPECT_EQ((a = v), (A{1,2,3,4}));
        EXPECT_EQ((b = v), a);
        EXPECT_EQ((c = v), a);
        EXPECT_EQ((d = v), a);
        EXPECT_EQ((e = v), a);

        b[s-1] = 5;
        EXPECT_EQ(b, (A{1,2,3,5}));
        EXPECT_EQ((a = b), b);
        EXPECT_EQ((c = b), b);
        EXPECT_EQ((d = b), b);
        EXPECT_EQ((e = b), b);

        data[0] = 0;
        EXPECT_EQ((c = data), (A{0,2,3,5}));
        EXPECT_EQ((a = data), c);
        EXPECT_EQ((b = data), c);
        EXPECT_EQ((d = data), c);
        EXPECT_EQ((e = data), c);

        d.assign(v.begin(), v.end());
        EXPECT_EQ(d, (A{1,2,3,4}));
        a.assign(v.begin(), v.end());
        EXPECT_EQ(a, d);
        b.assign(v.begin(), v.end());
        EXPECT_EQ(b, d);
        c.assign(v.begin(), v.end());
        EXPECT_EQ(c, d);
        e.assign(v.begin(), v.end());
        EXPECT_EQ(e, d);
    }
} // namespace xt