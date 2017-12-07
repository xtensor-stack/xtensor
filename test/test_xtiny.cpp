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

#include "gtest/gtest.h"
#include "xtensor/xtiny.hpp"

template <class VECTOR, class VALUE>
bool equalValue(VECTOR const & v, VALUE const & vv)
{
    for (unsigned int i = 0; i<v.size(); ++i)
        if (v[i] != vv)
            return false;
    return true;
}

template <class VECTOR1, class VECTOR2>
bool equalVector(VECTOR1 const & v1, VECTOR2 const & v2)
{
    for (unsigned int i = 0; i<v1.size(); ++i)
        if (v1[i] != v2[i])
            return false;
    return true;
}

template <class ITER1, class ITER2>
bool equalIter(ITER1 i1, ITER1 i1end, ITER2 i2, xt::index_t size)
{
    if (i1end - i1 != size)
        return false;
    for (; i1<i1end; ++i1, ++i2)
        if (*i1 != *i2)
            return false;
    return true;
}


namespace xt
{
    static const int SIZE = 3;
    using BV = tiny_array<unsigned char, SIZE>;
    using IV = tiny_array<int, SIZE>;
    using FV = tiny_array<float, SIZE>;

    static float di[] = { 1, 2, 4};
    static float df[] = { 1.2f, 2.4f, 3.6f};
    BV bv0, bv1{1}, bv3(di);
    IV iv0, iv1{1}, iv3(di);
    FV fv0, fv1{1.0f}, fv3(df);

    // TEST(xtiny, traits)
    // {
        // EXPECT_TRUE(BV::may_use_uninitialized_memory);
        // EXPECT_TRUE((tiny_array<BV, runtime_size>::may_use_uninitialized_memory));
        // EXPECT_FALSE((tiny_array<tiny_array<int, runtime_size>, SIZE>::may_use_uninitialized_memory));

        // EXPECT_TRUE((std::is_same<IV, promote_t<BV>>::value));
        // EXPECT_TRUE((std::is_same<tiny_array<double, 3>, real_promote_t<IV>>::value));
        // EXPECT_TRUE((std::is_same<typename IV::template as_type<double>, real_promote_t<IV>>::value));

        // EXPECT_TRUE((std::is_same<uint64_t, squared_norm_t<tiny_array<int, 1> > >::value));
        // EXPECT_TRUE((std::is_same<uint64_t, squared_norm_t<tiny_array<tiny_array<int, 1>, 1> > >::value));
        // EXPECT_TRUE((std::is_same<uint64_t, squared_norm_t<tiny_array<tiny_array<int, 1>, 1> > >::value));
        // EXPECT_TRUE((std::is_same<double, norm_t<tiny_array<int, 1> > >::value));
        // EXPECT_TRUE((std::is_same<double, norm_t<tiny_array<tiny_array<int, 1>, 1> > >::value));
        // EXPECT_TRUE((std::is_same<tiny_array<double, SIZE>, decltype(cos(iv3))>::value));
    // }

    TEST(xtiny, construction)
    {
        EXPECT_EQ(bv0.size(), SIZE);
        EXPECT_EQ(iv0.size(), SIZE);
        EXPECT_EQ(fv0.size(), SIZE);
        EXPECT_EQ(bv0.shape(), (std::array<index_t, 1>{SIZE}));
        EXPECT_EQ(iv0.shape(), (std::array<index_t, 1>{SIZE}));
        EXPECT_EQ(fv0.shape(), (std::array<index_t, 1>{SIZE}));

        for(int k=0; k<SIZE; ++k)
        {
            EXPECT_EQ(bv0[k], 0);
            EXPECT_EQ(iv0[k], 0);
            EXPECT_EQ(fv0[k], 0);
            EXPECT_EQ(bv1[k], 1);
            EXPECT_EQ(iv1[k], 1);
            EXPECT_EQ(fv1[k], 1);
            EXPECT_EQ(bv3[k], di[k]);
            EXPECT_EQ(iv3[k], di[k]);
            EXPECT_EQ(fv3[k], df[k]);
        }

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
        // these should not compile:
        // EXPECT_EQ(iv1, IV(1));
        // EXPECT_EQ(iv3, IV(1, 2, 4));

        IV iv;
        iv.init();
        EXPECT_EQ(iv0, iv);
        iv.init(1);
        EXPECT_EQ(iv1, iv);
        iv.init({1,2,4});
        EXPECT_EQ(iv3, iv);

        BV bv5 = reversed(bv3);
        auto rbv5 = bv3.rbegin();

        for(int k=0; k<SIZE; ++k, ++rbv5)
        {
            EXPECT_EQ(bv5[k], bv3[SIZE-1-k]);
            EXPECT_EQ(bv5[k], *rbv5);
        }

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

        for (int k = 0; k<SIZE; ++k)
        {
            IV iv = IV::unit_vector(k);
            EXPECT_EQ(iv[k], 1);
            iv[k] = 0;
            EXPECT_TRUE(iv == 0);
        }

        IV seq = IV::linear_sequence(), seq_ref;
        std::iota(seq_ref.begin(), seq_ref.end(), 0);
        EXPECT_EQ(seq, seq_ref);

        seq = IV::linear_sequence(2);
        std::iota(seq_ref.begin(), seq_ref.end(), 2);
        EXPECT_EQ(seq, seq_ref);
        EXPECT_EQ(seq, IV::range((int)seq.size() + 2));

        seq = IV::linear_sequence(20, -1);
        std::iota(seq_ref.rbegin(), seq_ref.rend(), 20 - (int)seq.size() + 1);
        EXPECT_EQ(seq, seq_ref);

        IV r = reversed(iv3);
        for (int k = 0; k<SIZE; ++k)
            EXPECT_EQ(iv3[k], r[SIZE - 1 - k]);

        EXPECT_EQ(transpose(r, IV::linear_sequence(SIZE - 1, -1)), iv3);

        r.reverse();
        EXPECT_EQ(r, iv3);

        using FV1 = FV::rebind_size<SIZE - 1>;
        FV1 fv10(fv3.begin());
        EXPECT_EQ(fv10, fv3.erase(SIZE - 1));
        EXPECT_EQ(fv3, fv10.insert(SIZE - 1, fv3[SIZE - 1]));
        FV1 fv11(fv3.begin() + 1);
        EXPECT_EQ(fv11, fv3.erase(0));

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

        EXPECT_TRUE(bv0 < bv1);

        EXPECT_TRUE(all_less(bv0, bv1));
        EXPECT_TRUE(!all_less(bv1, bv3));
        EXPECT_TRUE(all_greater(bv1, bv0));
        EXPECT_TRUE(!all_greater(bv3, bv1));
        EXPECT_TRUE(all_less_equal(bv0, bv1));
        EXPECT_TRUE(all_less_equal(0, bv0));
        EXPECT_TRUE(all_less_equal(bv0, 0));
        EXPECT_TRUE(all_less_equal(bv1, bv3));
        EXPECT_TRUE(!all_less_equal(bv3, bv1));
        EXPECT_TRUE(all_greater_equal(bv1, bv0));
        EXPECT_TRUE(all_greater_equal(bv3, bv1));
        EXPECT_TRUE(!all_greater_equal(bv1, bv3));

        EXPECT_TRUE(all_close(fv3, fv3));

        EXPECT_TRUE(!any(bv0) && !all(bv0) && any(bv1) && all(bv1));
        EXPECT_TRUE(!any(iv0) && !all(iv0) && any(iv1) && all(iv1));
        EXPECT_TRUE(!any(fv0) && !all(fv0) && any(fv1) && all(fv1));
        IV iv;
        iv = IV(); iv[0] = 1;
        EXPECT_TRUE(any(iv) && !all(iv));
        iv = IV(); iv[1] = 1;
        EXPECT_TRUE(any(iv) && !all(iv));
        iv = IV(); iv[SIZE - 1] = 1;
        EXPECT_TRUE(any(iv) && !all(iv));
        iv = IV{ 1 }; iv[0] = 0;
        EXPECT_TRUE(any(iv) && !all(iv));
        iv = IV{ 1 }; iv[1] = 0;
        EXPECT_TRUE(any(iv) && !all(iv));
        iv = IV{ 1 }; iv[SIZE - 1] = 0;
        EXPECT_TRUE(any(iv) && !all(iv));
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
        using A = tiny_array<int>;
        using V1 = tiny_array<int, 1>;

        EXPECT_TRUE(typeid(A) == typeid(tiny_array<int, runtime_size>));

        A a{ 1,2,3 }, b{ 1,2,3 }, c = a, e(3, 0);
        A d = a + b;
        EXPECT_EQ(a.size(), 3);
        EXPECT_EQ(b.size(), 3);
        EXPECT_EQ(c.size(), 3);
        EXPECT_EQ(d.size(), 3);
        EXPECT_EQ(e.size(), 3);
        EXPECT_EQ(a, b);
        EXPECT_EQ(a, c);
        EXPECT_TRUE(a != d);
        EXPECT_TRUE(a != e);
        EXPECT_TRUE(a < d);
        EXPECT_TRUE(e < a);
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

        c.init({ 1,2,3 });
        EXPECT_EQ(a, c);
        c = 2 * a;
        EXPECT_EQ(d, c);
        c.reverse();
        EXPECT_EQ(c, (A{ 6,4,2 }));
        EXPECT_EQ(c, reversed(d));
        c = c - 2;
        EXPECT_TRUE(all(d));
        EXPECT_TRUE(!all(c));
        EXPECT_TRUE(any(c));
        EXPECT_FALSE(c == 0);
        EXPECT_TRUE(!all(e));
        EXPECT_TRUE(!any(e));
        EXPECT_TRUE(e == 0);

        EXPECT_EQ(prod(a), 6);
        EXPECT_EQ(prod(A()), 0);

        EXPECT_EQ(dot(a, a), 14);
        EXPECT_EQ(cross(a, a), e);
        // EXPECT_EQ(norm_sq(a), 14u);

        EXPECT_EQ(a.erase(1), (A{ 1,3 }));
        EXPECT_EQ(a.insert(3, 4), (A{ 1,2,3,4 }));

        // testing move constructor and assignment
        EXPECT_EQ(std::move(A{ 1,2,3 }), (A{ 1,2,3 }));
        EXPECT_EQ(A(a.insert(3, 4)), (A{ 1,2,3,4 }));
        a = a.insert(3, 4);
        EXPECT_EQ(a, (A{ 1,2,3,4 }));

        A r = A::range(2, 6);
        EXPECT_EQ(r, (A{ 2,3,4,5 }));
        EXPECT_EQ(r.subarray(1, 3).size(), 2);
        EXPECT_EQ(r.subarray(1, 3), (A{ 3,4 }));
        EXPECT_EQ((r.template subarray<1, 3>().size()), 2);
        EXPECT_EQ((r.template subarray<1, 3>()), (A{ 3,4 }));

        EXPECT_EQ(A::range(0, 6, 3), (A{ 0,3 }));
        EXPECT_EQ(A::range(0, 7, 3), (A{ 0,3,6 }));
        EXPECT_EQ(A::range(0, 8, 3), (A{ 0,3,6 }));
        EXPECT_EQ(A::range(0, 9, 3), (A{ 0,3,6 }));
        EXPECT_EQ(A::range(10, 2, -2), (A{ 10, 8, 6, 4 }));
        EXPECT_EQ(A::range(10, 1, -2), (A{ 10, 8, 6, 4, 2 }));
        EXPECT_EQ(A::range(10, 0, -2), (A{ 10, 8, 6, 4, 2 }));

        EXPECT_EQ(transpose(A::range(1, 4)), (A{ 3,2,1 }));
        EXPECT_EQ(transpose(A::range(1, 4), A{ 1,2,0 }), (A{ 2,3,1 }));

        EXPECT_THROW(A(3, 0) / A(2, 0), std::runtime_error);

        using TA = tiny_array<int, 3>;
        TA s(A{ 1,2,3 });
        EXPECT_EQ(s, (TA{ 1,2,3 }));
        s = A{ 3,4,5 };
        EXPECT_EQ(s, (TA{ 3,4,5 }));

        EXPECT_THROW({ TA(A{ 1,2,3,4 }); }, std::runtime_error);

        EXPECT_EQ(A::unit_vector(3, 1), TA::unit_vector(1));
    }

    TEST(xtiny, arithmetic)
    {
        EXPECT_EQ(+iv3, iv3);
        EXPECT_EQ(-bv3, (IV{-1, -2, -4}));
        EXPECT_EQ(-iv3, (IV{-1, -2, -4}));
        EXPECT_EQ(-fv3, (FV{-1.2, -2.4, -3.6}));

        EXPECT_EQ(bv0 + bv1, bv1);
        EXPECT_EQ(bv0 + 1.0, fv1);
        EXPECT_EQ(1.0 + bv0, fv1);
        EXPECT_EQ(bv3 + bv1, (BV{2, 3, 5}));
        BV bv { 1, 2, 200};
        EXPECT_EQ(bv + bv, (IV{2, 4, 400}));

        EXPECT_EQ(bv1 - bv1, bv0);
        EXPECT_EQ(bv3 - iv3, bv0);
        EXPECT_EQ(fv3 - fv3, fv0);
        EXPECT_EQ(bv1 - 1.0, fv0);
        EXPECT_EQ(1.0 - bv1, fv0);
        EXPECT_EQ(bv3 - bv1, (BV{0, 1, 3}));
        EXPECT_EQ(bv0 - iv1, -iv1);

        EXPECT_EQ(bv1 * bv1, bv1);
        EXPECT_EQ(bv1 * 1.0, fv1);
        EXPECT_EQ(bv3 * 0.5, (FV{0.5, 1.0, 2.0}));
        EXPECT_EQ(1.0 * bv1, fv1);
        EXPECT_EQ(bv3 * bv3, (BV{1, 4, 16}));

        EXPECT_EQ(bv3 / bv3, bv1);
        EXPECT_EQ(bv1 / 1.0, fv1);
        EXPECT_EQ(1.0 / bv1, fv1);
        EXPECT_EQ(1.0 / bv3, (FV{1.0, 0.5, 0.25}));
        EXPECT_EQ(bv3 / 2, (IV{0, 1, 2}));
        EXPECT_EQ(bv3 / 2.0, (FV{0.5, 1.0, 2.0}));
        EXPECT_EQ(fv3 / 2.0, (FV{0.6, 1.2, 1.8}));
        EXPECT_EQ((2.0 * fv3) / 2.0, fv3);

        EXPECT_EQ(bv3 % 2, (BV{1, 0, 0}));
        EXPECT_EQ(iv3 % iv3, iv0);
        EXPECT_EQ(3   % bv3, (BV{0, 1, 3}));
        EXPECT_EQ(iv3 % (iv3 + iv1), iv3);

        BV bvp = (bv3 + bv3)*0.5;
        FV fvp = (fv3 + fv3)*0.5;
        EXPECT_EQ(bvp, bv3);
        EXPECT_EQ(fvp, fv3);
        bvp = 2.0*bv3 - bv3;
        fvp = 2.0*fv3 - fv3;
        EXPECT_EQ(bvp, bv3);
        EXPECT_EQ(fvp, fv3);
    }

    TEST(xtiny, algebraic)
    {
        EXPECT_EQ(abs(bv3), bv3);
        EXPECT_EQ(abs(iv3), iv3);
        EXPECT_EQ(abs(fv3), fv3);

        EXPECT_EQ(floor(fv3), (FV{1.0, 2.0, 3.0}));
        EXPECT_EQ(-ceil(-fv3), (FV{1.0, 2.0, 3.0}));
        EXPECT_EQ(round(fv3), (FV{1.0, 2.0, 4.0}));
        EXPECT_EQ(sqrt(fv3*fv3), fv3);
        EXPECT_EQ(cbrt(pow(fv3, 3)), fv3);

        tiny_array<int, 4> src{ 1, 2, -3, -4 }, signs{ 2, -3, 4, -5 };
        EXPECT_EQ(copysign(src, signs), (tiny_array<int, 4>{1, -2, 3, -4}));

        tiny_array<double, 3> left{ 3., 5., 8. }, right{ 4., 12., 15. };
        EXPECT_EQ(hypot(left, right), (tiny_array<double, 3>{5., 13., 17.}));

        EXPECT_EQ(sum(iv3),  7);
        EXPECT_EQ(sum(fv3),  7.2f);
        EXPECT_EQ(prod(iv3), 8);
        EXPECT_EQ(prod(fv3), 10.368f);
        EXPECT_NEAR(mean(iv3), 7.0 / SIZE, 1e-7);
        EXPECT_EQ(cumsum(bv3), (IV{1, 3, 7}));
        EXPECT_EQ(cumprod(bv3), (IV{1, 2, 8}));

        EXPECT_EQ(min(iv3, fv3), (FV{1.0, 2.0, 3.6}));
        EXPECT_EQ(min(3.0, fv3), (FV{1.2, 2.4, 3.0}));
        EXPECT_EQ(min(fv3, 3.0), (FV{1.2, 2.4, 3.0}));
        EXPECT_EQ(min(iv3), 1);
        EXPECT_EQ(min(fv3), 1.2f);
        EXPECT_EQ(max(iv3, fv3), (FV{1.2, 2.4, 4.0}));
        EXPECT_EQ(max(3.0, fv3), (FV{3.0, 3.0, 3.6}));
        EXPECT_EQ(max(fv3, 3.0), (FV{3.0, 3.0, 3.6}));
        EXPECT_EQ(max(iv3), 4);
        EXPECT_EQ(max(fv3), 3.6f);

        EXPECT_EQ(clip_lower(iv3, 0), iv3);
        EXPECT_EQ(clip_lower(iv3, 11), IV{ 11 });
        EXPECT_EQ(clip_upper(iv3, 0), IV{ 0 });
        EXPECT_EQ(clip_upper(iv3, 11), iv3);
        EXPECT_EQ(clip(iv3, 0, 11), iv3);
        EXPECT_EQ(clip(iv3, 11, 12), IV{ 11 });
        EXPECT_EQ(clip(iv3, -1, 0), IV{ 0 });
        EXPECT_EQ(clip(iv3, IV{0 }, IV{11}), iv3);
        EXPECT_EQ(clip(iv3, IV{11}, IV{12}), IV{11});
        EXPECT_EQ(clip(iv3, IV{-1}, IV{0 }), IV{0 });

        EXPECT_EQ(dot(iv3, iv3), 21);
        EXPECT_EQ(dot(fv1, fv3), sum(fv3));

        EXPECT_EQ(cross(bv3, bv3), IV{0});
        EXPECT_EQ(cross(iv3, bv3), IV{0});
        EXPECT_TRUE(all_close(cross(fv3, fv3), FV{ 0.0f }, 1e-6f));
        EXPECT_TRUE(all_close(cross(fv1, fv3), (FV{ 1.2f, -2.4f, 1.2f }), 1e-6f));

        // int oddRef[] = { 1, 0, 0, 1, 0, 0 };
        // EXPECT_TRUE(equalIter(oddRef, oddRef + SIZE, odd(iv3).begin(), SIZE));
        // EXPECT_TRUE(equalIter(oddRef, oddRef + SIZE, (iv3 & 1).begin(), SIZE));
    }

    TEST(xtiny, norm)
    {
        using math::sqrt;

        EXPECT_TRUE(norm_sq(bv1) == SIZE);
        EXPECT_TRUE(norm_sq(iv1) == SIZE);
        EXPECT_TRUE(norm_sq(fv1) == (float)SIZE);

        EXPECT_EQ(norm_sq(bv3), dot(bv3, bv3));
        EXPECT_EQ(norm_sq(iv3), dot(iv3, iv3));
        EXPECT_NEAR(norm_sq(fv3), sum(fv3*fv3), 1e-6);
        EXPECT_NEAR(norm_sq(fv3), dot(fv3, fv3), 1e-6);

        tiny_array<IV, 3> ivv{ iv3, iv3, iv3 };
        EXPECT_EQ(norm_sq(ivv), 3 * norm_sq(iv3));
        EXPECT_EQ(norm_l2(ivv), sqrt(3.0*norm_sq(iv3)));
        // EXPECT_EQ(elementwise_norm(iv3), iv3);
        // EXPECT_EQ(elementwise_squared_norm(iv3), (IV{ 1, 4, 16 }));

        EXPECT_NEAR(norm_l2(bv3), sqrt(dot(bv3, bv3)), 1e-6);
        EXPECT_NEAR(norm_l2(iv3), sqrt(dot(iv3, iv3)), 1e-6);
        EXPECT_NEAR(norm_l2(fv3), sqrt(dot(fv3, fv3)), 1e-6);

        BV bv { 1, 2, 200};
        EXPECT_EQ(norm_sq(bv), 40005);
    }

} // namespace xt