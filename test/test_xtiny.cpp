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
    static const int SIZE = 3;
    using BV = xtiny<unsigned char, SIZE>;
    using IV = xtiny<int, SIZE>;
    using FV = xtiny<float, SIZE>;

    static float di[] = { 1, 2, 4};
    static float df[] = { 1.2f, 2.4f, 3.6f};
    BV bv0, bv1{1}, bv3(di);
    IV iv0, iv1{1}, iv3(di);
    FV fv0, fv1{1.0f}, fv3(df);

    TEST(xtiny, construction)
    {
        EXPECT_EQ(bv0.size(), SIZE);
        EXPECT_EQ(iv0.size(), SIZE);
        EXPECT_EQ(fv0.size(), SIZE);
        EXPECT_FALSE(bv0.empty());
        EXPECT_FALSE(iv0.empty());
        EXPECT_FALSE(fv0.empty());
        EXPECT_EQ(bv0.shape(), (std::array<size_t, 1>{SIZE}));
        EXPECT_EQ(iv0.shape(), (std::array<size_t, 1>{SIZE}));
        EXPECT_EQ(fv0.shape(), (std::array<size_t, 1>{SIZE}));

        auto iv3iter = iv3.begin();
        auto iv3citer = iv3.cbegin();
        auto iv3riter = iv3.rbegin();
        auto iv3criter = iv3.crbegin();
        for(int k=0; k<SIZE; ++k, ++iv3iter, ++iv3citer, ++iv3riter, ++iv3criter)
        {
            EXPECT_EQ(bv0[k], 0);
            EXPECT_EQ(iv0[k], 0);
            EXPECT_EQ(fv0[k], 0);
            EXPECT_EQ(bv1[k], 1);
            EXPECT_EQ(iv1[k], 1);
            EXPECT_EQ(fv1[k], 1);
            EXPECT_EQ(bv3[k], di[k]);
            EXPECT_EQ(iv3[k], di[k]);
            EXPECT_EQ(iv3.at(k), di[k]);
            EXPECT_EQ(fv3[k], df[k]);
            EXPECT_EQ(*iv3iter, di[k]);
            EXPECT_EQ(*iv3citer, di[k]);
            EXPECT_EQ(*iv3riter, di[SIZE-1-k]);
            EXPECT_EQ(*iv3criter, di[SIZE-1-k]);
        }
        EXPECT_EQ(iv3iter, iv3.end());
        EXPECT_EQ(iv3citer, iv3.cend());
        EXPECT_EQ(iv3riter, iv3.rend());
        EXPECT_EQ(iv3criter, iv3.crend());
        EXPECT_THROW(iv3.at(SIZE), std::out_of_range);

        EXPECT_EQ(iv3.front(), 1);
        EXPECT_EQ(iv3.back(), 4);

        EXPECT_EQ(iv3, IV(iv3));
        EXPECT_EQ(iv3, IV(bv3));
        EXPECT_EQ(iv3, IV(bv3.begin()));
        EXPECT_EQ(iv3, IV(bv3.begin(), bv3.end()));
        EXPECT_THROW(IV(bv3.begin(), bv3.begin()+2), std::runtime_error);
        EXPECT_EQ(iv1, IV(fv1));
        EXPECT_EQ(iv3, (IV{ 1, 2, 4 }));
        EXPECT_EQ(iv3, (IV{ 1.1, 2.2, 4.4 }));
        EXPECT_EQ(iv1, (IV{ 1 }));
        EXPECT_EQ(iv1, (IV{ 1.1 }));
        EXPECT_EQ(iv1, IV(SIZE, 1));
        EXPECT_EQ(iv3, IV(std::array<int, SIZE>{1, 2, 4}));
        EXPECT_EQ(iv1, IV(std::vector<int>(SIZE, 1)));

        IV iv;
        iv.assign(SIZE, 0);
        EXPECT_EQ(iv0, iv);
        iv.assign(SIZE, 1);
        EXPECT_EQ(iv1, iv);
        iv.assign({1,2,4});
        EXPECT_EQ(iv3, iv);

        FV fv(iv3);
        EXPECT_EQ(fv, iv3);
        fv = fv3;
        EXPECT_EQ(fv, fv3);
        fv = bv3;
        EXPECT_EQ(fv, bv3);

        EXPECT_EQ(iv3, (iv3.template subarray<0, SIZE>()));
        EXPECT_EQ(2, (iv3.template subarray<0, 2>().size()));
        EXPECT_EQ(iv3[0], (iv3.template subarray<0, 2>()[0]));
        EXPECT_EQ(iv3[1], (iv3.template subarray<0, 2>()[1]));
        EXPECT_EQ(2, (iv3.template subarray<1, 3>().size()));
        EXPECT_EQ(iv3[1], (iv3.template subarray<1, 3>()[0]));
        EXPECT_EQ(iv3[2], (iv3.template subarray<1, 3>()[1]));
        EXPECT_EQ(1, (iv3.template subarray<1, 2>().size()));
        EXPECT_EQ(iv3[1], (iv3.template subarray<1, 2>()[0]));
        EXPECT_EQ(1, (iv3.subarray(1, 2).size()));
        EXPECT_EQ(iv3[1], (iv3.subarray(1, 2)[0]));
        IV const & civ3 = iv3;
        EXPECT_EQ(1, (civ3.template subarray<1, 2>().size()));
        EXPECT_EQ(iv3[1], (civ3.template subarray<1, 2>()[0]));
        EXPECT_EQ(1, (civ3.subarray(1, 2).size()));
        EXPECT_EQ(iv3[1], (civ3.subarray(1, 2)[0]));

        using FV1 = FV::rebind_size<SIZE - 1>;
        FV1 fv10(fv3.begin());
        EXPECT_EQ(fv10, fv3.erase(SIZE - 1));
        EXPECT_EQ(fv3, fv10.insert(SIZE - 1, fv3[SIZE - 1]));
        EXPECT_EQ(fv10, fv3.pop_back());
        EXPECT_EQ(fv3, fv10.push_back(fv3[SIZE - 1]));
        FV1 fv11(fv3.begin() + 1);
        EXPECT_EQ(fv11, fv3.erase(0));
        EXPECT_EQ(fv3, fv11.insert(0, fv3[0]));
        EXPECT_EQ(fv11, fv3.pop_front());
        EXPECT_EQ(fv3, fv11.push_front(fv3[0]));

        BV bv2(bv1), bv4(bv3);
        swap(bv2, bv4);
        EXPECT_EQ(bv3, bv2);
        EXPECT_EQ(bv1, bv4);
    }

    TEST(xtiny, comparison)
    {
        EXPECT_TRUE(bv0 == bv0);
        EXPECT_TRUE(bv0 == 0);
        EXPECT_TRUE(0 == bv0);
        EXPECT_TRUE(iv0 == iv0);
        EXPECT_TRUE(fv0 == fv0);
        EXPECT_TRUE(fv0 == 0);
        EXPECT_TRUE(0 == fv0);
        EXPECT_TRUE(iv0 == bv0);
        EXPECT_TRUE(iv0 == fv0);
        EXPECT_TRUE(fv0 == bv0);

        EXPECT_TRUE(bv3 == bv3);
        EXPECT_TRUE(iv3 == iv3);
        EXPECT_TRUE(fv3 == fv3);
        EXPECT_TRUE(iv3 == bv3);
        EXPECT_TRUE(iv3 != fv3);
        EXPECT_TRUE(iv3 != 0);
        EXPECT_TRUE(0 != iv3);
        EXPECT_TRUE(fv3 != bv3);
        EXPECT_TRUE(fv3 != 0);
        EXPECT_TRUE(0 != fv3);
    }

    TEST(xtiny, ostream)
    {
        std::ostringstream out;
        out << iv3;
        std::string expected("{1, 2, 4}");
        EXPECT_EQ(expected, out.str());
        out << "Testing.." << fv3 << 42;
        out << bv3 << std::endl;
    }

    TEST(xtiny, runtime_size)
    {
        using A = xtiny<int>;
        using V1 = xtiny<int, 1>;

        EXPECT_TRUE(typeid(A) == typeid(xtiny<int, runtime_size>));

        A a{ 1,2,3 }, b{ 1,2,3 }, c = a, e(3, 0);
        EXPECT_EQ(a.size(), 3);
        EXPECT_EQ(b.size(), 3);
        EXPECT_EQ(c.size(), 3);
        EXPECT_EQ(e.size(), 3);
        EXPECT_EQ(a, b);
        EXPECT_EQ(a, c);
        EXPECT_TRUE(a != e);
        EXPECT_EQ(e, (A{ 0,0,0 }));

        EXPECT_EQ(iv3, (A{ 1, 2, 4 }));
        EXPECT_EQ(iv3, (A{ 1.1, 2.2, 4.4 }));
        EXPECT_EQ(iv3, (A({ 1, 2, 4 })));
        EXPECT_EQ(iv3, (A({ 1.1, 2.2, 4.4 })));
        EXPECT_EQ(V1{1}, (A{ 1 }));
        EXPECT_EQ(V1{1}, (A{ 1.1 }));
        EXPECT_EQ(V1{1}, (A({ 1 })));
        EXPECT_EQ(V1{1}, (A({ 1.1 })));
        EXPECT_EQ(iv0, A(SIZE, 0));
        EXPECT_EQ(iv1, A(SIZE, 1));

        c.assign({ 1,2,3 });
        EXPECT_EQ(a, c);

        EXPECT_EQ(a.erase(1), (A{ 1,3 }));
        EXPECT_EQ(a.insert(3, 4), (A{ 1,2,3,4 }));

        // testing move constructor and assignment
        EXPECT_EQ(std::move(A{ 1,2,3 }), (A{ 1,2,3 }));
        EXPECT_EQ(A(a.insert(3, 4)), (A{ 1,2,3,4 }));
        a = a.insert(3, 4);
        EXPECT_EQ(a, (A{ 1,2,3,4 }));

        A r{ 2,3,4,5 };
        EXPECT_EQ(r, (A{ 2,3,4,5 }));
        EXPECT_EQ(r.subarray(1, 3).size(), 2);
        EXPECT_EQ(r.subarray(1, 3), (A{ 3,4 }));
        EXPECT_EQ((r.template subarray<1, 3>().size()), 2);
        EXPECT_EQ((r.template subarray<1, 3>()), (A{ 3,4 }));
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