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
bool equalIter(ITER1 i1, ITER1 i1end, ITER2 i2, xt::ArrayIndex size)
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
    using BV = TinyArray<unsigned char, SIZE>;
    using IV = TinyArray<int, SIZE>;
    using FV = TinyArray<float, SIZE>;

    static float di[] = { 1, 2, 4, 5, 8, 10 };
    static float df[] = { 1.2f, 2.4f, 3.6f, 4.8f, 8.1f, 9.7f };
    BV bv0, bv1(1), bv3(di);
    IV iv0, iv1(1), iv3(di);
    FV fv0, fv1(1), fv3(df);

    TEST(xtiny, promote)
    {
        EXPECT_TRUE((std::is_same<unsigned long long, SquaredNormType<TinyArray<int, 1> > >::value));
        EXPECT_TRUE((std::is_same<unsigned long long, SquaredNormType<TinyArray<TinyArray<int, 1>, 1> > >::value));
        EXPECT_TRUE((std::is_same<double, NormType<TinyArray<int, 1> > >::value));
        EXPECT_TRUE((std::is_same<double, NormType<TinyArray<TinyArray<int, 1>, 1> > >::value));
    }

    TEST(xtiny, construct)
    {
        EXPECT_TRUE(BV::may_use_uninitialized_memory);
        EXPECT_TRUE((TinyArray<BV, SIZE>::may_use_uninitialized_memory == (SIZE != runtime_size)));
        EXPECT_TRUE(UninitializedMemoryTraits<BV>::value == (SIZE != runtime_size));
        EXPECT_TRUE((UninitializedMemoryTraits<TinyArray<TinyArray<int, runtime_size>, SIZE>>::value == false));
        //EXPECT_TRUE(ValueTypeTraits<BV>::value);
        //EXPECT_TRUE((std::is_same<unsigned char, typename ValueTypeTraits<BV>::type>::value));

        EXPECT_TRUE(bv0.size() == SIZE);
        EXPECT_TRUE(iv0.size() == SIZE);
        EXPECT_TRUE(fv0.size() == SIZE);

        EXPECT_TRUE(equalValue(bv0, 0));
        EXPECT_TRUE(equalValue(iv0, 0));
        EXPECT_TRUE(equalValue(fv0, 0.0f));

        EXPECT_TRUE(equalValue(bv1, 1));
        EXPECT_TRUE(equalValue(iv1, 1));
        EXPECT_TRUE(equalValue(fv1, 1.0f));

        EXPECT_TRUE(equalIter(bv3.begin(), bv3.end(), di, SIZE));
        EXPECT_TRUE(equalIter(iv3.begin(), iv3.end(), di, SIZE));
        EXPECT_TRUE(equalIter(fv3.begin(), fv3.end(), df, SIZE));

        EXPECT_TRUE(!equalVector(bv3, fv3));
        EXPECT_TRUE(!equalVector(iv3, fv3));

        BV bv(round(fv3));
        EXPECT_TRUE(equalIter(bv3.begin(), bv3.end(), bv.begin(), SIZE));
        EXPECT_TRUE(equalVector(bv3, bv));

        BV bv4(bv3.begin());
        EXPECT_TRUE(equalIter(bv3.begin(), bv3.end(), bv4.begin(), SIZE));
        EXPECT_TRUE(equalVector(bv3, bv4));

        BV bv5(bv3.begin(), ReverseCopy);
        EXPECT_TRUE(equalIter(bv3.begin(), bv3.end(),
            std::reverse_iterator<typename BV::iterator>(bv5.end()), SIZE));

        FV fv(iv3);
        EXPECT_TRUE(equalIter(iv3.begin(), iv3.end(), fv.begin(), SIZE));
        EXPECT_TRUE(equalVector(iv3, fv));

        fv = fv3;
        EXPECT_TRUE(equalIter(fv3.begin(), fv3.end(), fv.begin(), SIZE));
        EXPECT_TRUE(equalVector(fv3, fv));

        fv = bv3;
        EXPECT_TRUE(equalIter(bv3.begin(), bv3.end(), fv.begin(), SIZE));
        EXPECT_TRUE(equalVector(bv3, fv));

        TinyArray<double, 5> fv5;
        fv5.init(fv3.begin(), fv3.end());
        EXPECT_EQ(fv5[0], fv3[0]);
        EXPECT_EQ(fv5[1], fv3[1]);
        EXPECT_EQ(fv5[2], fv3[2]);
        EXPECT_EQ(fv5[3], SIZE <= 3 ? 0.0 : fv3[3]);
        EXPECT_EQ(fv5[4], SIZE <= 4 ? 0.0 : fv3[4]);

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

        for (int k = 0; k<SIZE; ++k)
        {
            IV iv = IV::unitVector(k);
            EXPECT_EQ(iv[k], 1);
            iv[k] = 0;
            EXPECT_TRUE(!any(iv));
        }

        IV seq = IV::linearSequence(), seq_ref(tags::size = seq.size());
        std::iota(seq_ref.begin(), seq_ref.end(), 0);
        EXPECT_EQ(seq, seq_ref);

        seq = IV::linearSequence(2);
        std::iota(seq_ref.begin(), seq_ref.end(), 2);
        EXPECT_EQ(seq, seq_ref);
        EXPECT_EQ(seq, IV::range((int)seq.size() + 2));

        seq = IV::linearSequence(20, -1);
        std::iota(seq_ref.rbegin(), seq_ref.rend(), 20 - (int)seq.size() + 1);
        EXPECT_EQ(seq, seq_ref);

        IV r = reversed(iv3);
        for (int k = 0; k<SIZE; ++k)
            EXPECT_EQ(iv3[k], r[SIZE - 1 - k]);

        EXPECT_EQ(transpose(r, IV::linearSequence(SIZE - 1, -1)), iv3);

        r.reverse();
        EXPECT_EQ(r, iv3);

        typedef TinyArray<typename FV::value_type, SIZE - 1> FV1;
        FV1 fv10(fv3.begin());
        EXPECT_EQ(fv10, fv3.erase(SIZE - 1));
        EXPECT_EQ(fv3, fv10.insert(SIZE - 1, fv3[SIZE - 1]));
        FV1 fv11(fv3.begin() + 1);
        EXPECT_EQ(fv11, fv3.erase(0));
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

        EXPECT_TRUE(allLess(bv0, bv1));
        EXPECT_TRUE(!allLess(bv1, bv3));
        EXPECT_TRUE(allGreater(bv1, bv0));
        EXPECT_TRUE(!allGreater(bv3, bv1));
        EXPECT_TRUE(allLessEqual(bv0, bv1));
        EXPECT_TRUE(allLessEqual(0, bv0));
        EXPECT_TRUE(allLessEqual(bv0, 0));
        EXPECT_TRUE(allLessEqual(bv1, bv3));
        EXPECT_TRUE(!allLessEqual(bv3, bv1));
        EXPECT_TRUE(allGreaterEqual(bv1, bv0));
        EXPECT_TRUE(allGreaterEqual(bv3, bv1));
        EXPECT_TRUE(!allGreaterEqual(bv1, bv3));

        EXPECT_TRUE(isclose(fv3, fv3));

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
        iv = IV(1); iv[0] = 0;
        EXPECT_TRUE(any(iv) && !all(iv));
        iv = IV(1); iv[1] = 0;
        EXPECT_TRUE(any(iv) && !all(iv));
        iv = IV(1); iv[SIZE - 1] = 0;
        EXPECT_TRUE(any(iv) && !all(iv));
    }

    TEST(xtiny, arithmetic)
    {
        EXPECT_TRUE((std::is_same<IV, PromoteType<IV>>::value));
        EXPECT_TRUE((std::is_same<TinyArray<double, 3>, RealPromoteType<IV>>::value));
        EXPECT_TRUE((std::is_same<typename IV::template AsType<double>, RealPromoteType<IV>>::value));

        IV ivm3 = -iv3;
        FV fvm3 = -fv3;

        int mi[] = { -1, -2, -4, -5, -8, -10 };
        float mf[] = { -1.2f, -2.4f, -3.6f, -4.8f, -8.1f, -9.7f };

        EXPECT_TRUE(equalIter(ivm3.begin(), ivm3.end(), mi, SIZE));
        EXPECT_TRUE(equalIter(fvm3.begin(), fvm3.end(), mf, SIZE));

        IV iva3 = abs(ivm3);
        FV fva3 = abs(fvm3);
        EXPECT_TRUE(equalVector(iv3, iva3));
        EXPECT_TRUE(equalVector(fv3, fva3));

        int fmi[] = { -2, -3, -4, -5, -9, -10 };
        int fpi[] = { 1, 2, 3, 4, 8, 9 };
        int ri[] = { 1, 2, 4, 5, 8, 10 };
        IV ivi3 = floor(fvm3);
        EXPECT_TRUE(equalIter(ivi3.begin(), ivi3.end(), fmi, SIZE));
        ivi3 = -ceil(fv3);
        EXPECT_TRUE(equalIter(ivi3.begin(), ivi3.end(), fmi, SIZE));
        ivi3 = round(fv3);
        EXPECT_TRUE(equalIter(ivi3.begin(), ivi3.end(), ri, SIZE));
        ivi3 = floor(fv3);
        EXPECT_TRUE(equalIter(ivi3.begin(), ivi3.end(), fpi, SIZE));
        ivi3 = roundi(fv3);
        EXPECT_TRUE(equalIter(ivi3.begin(), ivi3.end(), ri, SIZE));
        ivi3 = -ceil(fvm3);
        EXPECT_TRUE(equalIter(ivi3.begin(), ivi3.end(), fpi, SIZE));
        ivi3 = -round(fvm3);
        EXPECT_TRUE(equalIter(ivi3.begin(), ivi3.end(), ri, SIZE));

        EXPECT_EQ(clipLower(iv3), iv3);
        EXPECT_EQ(clipLower(iv3, 11), IV(11));
        EXPECT_EQ(clipUpper(iv3, 0), IV(0));
        EXPECT_EQ(clipUpper(iv3, 11), iv3);
        EXPECT_EQ(clip(iv3, 0, 11), iv3);
        EXPECT_EQ(clip(iv3, 11, 12), IV(11));
        EXPECT_EQ(clip(iv3, -1, 0), IV(0));
        EXPECT_EQ(clip(iv3, IV(0), IV(11)), iv3);
        EXPECT_EQ(clip(iv3, IV(11), IV(12)), IV(11));
        EXPECT_EQ(clip(iv3, IV(-1), IV(0)), IV(0));

        EXPECT_TRUE(squaredNorm(bv1) == SIZE);
        EXPECT_TRUE(squaredNorm(iv1) == SIZE);
        EXPECT_TRUE(squaredNorm(fv1) == (float)SIZE);

        float expectedSM = 1.2f*1.2f + 2.4f*2.4f + 3.6f*3.6f;
        EXPECT_TRUE(mathfunctions::isclose(squaredNorm(fv3), expectedSM, 1e-7f));

        EXPECT_EQ(dot(bv3, bv3), squaredNorm(bv3));
        EXPECT_EQ(dot(iv3, bv3), squaredNorm(iv3));
        EXPECT_TRUE(mathfunctions::isclose(dot(fv3, fv3), squaredNorm(fv3)));

        TinyArray<IV, 3> ivv(iv3, iv3, iv3);
        EXPECT_EQ(squaredNorm(ivv), 3 * squaredNorm(iv3));
        EXPECT_EQ(norm(ivv), sqrt(3.0*squaredNorm(iv3)));

        EXPECT_TRUE(mathfunctions::isclose(sqrt(dot(bv3, bv3)), norm(bv3), 0.0));
        EXPECT_TRUE(mathfunctions::isclose(sqrt(dot(iv3, bv3)), norm(iv3), 0.0));
        EXPECT_TRUE(mathfunctions::isclose(sqrt(dot(fv3, fv3)), norm(fv3), 0.0f));

        BV bv = bv3;
        bv[2] = 200;
        int expectedSM2 = 40005;
        if (SIZE == 6)
            expectedSM2 += 189;
        EXPECT_EQ(dot(bv, bv), expectedSM2);
        EXPECT_EQ(squaredNorm(bv), expectedSM2);

        EXPECT_TRUE(equalVector(bv0 + 1.0, fv1));
        EXPECT_TRUE(equalVector(1.0 + bv0, fv1));
        EXPECT_TRUE(equalVector(bv1 - 1.0, fv0));
        EXPECT_TRUE(equalVector(1.0 - bv1, fv0));
        EXPECT_TRUE(equalVector(bv3 - iv3, bv0));
        EXPECT_TRUE(equalVector(fv3 - fv3, fv0));
        BV bvp = (bv3 + bv3)*0.5;
        FV fvp = (fv3 + fv3)*0.5;
        EXPECT_TRUE(equalVector(bvp, bv3));
        EXPECT_TRUE(equalVector(fvp, fv3));
        bvp = 2.0*bv3 - bv3;
        fvp = 2.0*fv3 - fv3;
        EXPECT_TRUE(equalVector(bvp, bv3));
        EXPECT_TRUE(equalVector(fvp, fv3));

        IV ivp = bv + bv;
        int ip1[] = { 2, 4, 400, 10, 16, 20 };
        EXPECT_TRUE(equalIter(ivp.begin(), ivp.end(), ip1, SIZE));
        EXPECT_TRUE(equalVector(bv0 - iv1, -iv1));

        bvp = bv3 / 2.0;
        fvp = bv3 / 2.0;
        int ip[] = { 0, 1, 2, 3, 4, 5 }; 
        float fp[] = { 0.5, 1.0, 2.0, 2.5, 4.0, 5.0 };
        EXPECT_TRUE(equalIter(bvp.begin(), bvp.end(), ip, SIZE));
        EXPECT_TRUE(equalIter(fvp.begin(), fvp.end(), fp, SIZE));
        fvp = fv3 / 2.0;
        float fp1[] = { 0.6f, 1.2f, 1.8f, 2.4f, 4.05f, 4.85f };
        EXPECT_TRUE(equalIter(fvp.begin(), fvp.end(), fp1, SIZE));
        EXPECT_EQ(2.0 / fv1, 2.0 * fv1);
        float fp2[] = { 1.0f, 0.5f, 0.25f, 0.2f, 0.125f, 0.1f };
        fvp = 1.0 / bv3;
        EXPECT_TRUE(equalIter(fvp.begin(), fvp.end(), fp2, SIZE));

        int ivsq[] = { 1, 4, 16, 25, 64, 100 };
        ivp = iv3*iv3;
        EXPECT_TRUE(equalIter(ivp.begin(), ivp.end(), ivsq, SIZE));
        EXPECT_EQ(iv3 * iv1, iv3);
        EXPECT_EQ(iv0 * iv3, iv0);
        EXPECT_EQ(iv3 / iv3, iv1);
        EXPECT_EQ(iv3 % iv3, iv0);
        EXPECT_EQ(iv3 % (iv3 + iv1), iv3);

        float minRef[] = { 1.0f, 2.0f, 3.6f, 4.8f, 8.0f, 9.7f };
        float minRefScalar[] = { 1.2f, 2.4f, 3.6f, 4.0f, 4.0f, 4.0f };
        auto minRes = min(iv3, fv3);
        EXPECT_TRUE(equalIter(minRef, minRef + SIZE, minRes.cbegin(), SIZE));
        minRes = min(4.0f, fv3);
        EXPECT_TRUE(equalIter(minRefScalar, minRefScalar + SIZE, minRes.cbegin(), SIZE));
        minRes = min(fv3, 4.0f);
        EXPECT_TRUE(equalIter(minRefScalar, minRefScalar + SIZE, minRes.cbegin(), SIZE));
        IV ivmin = floor(fv3);
        ivmin[1] = 3;
        int minRef2[] = { 1, 2, 3, 4, 8, 9 };
        auto minRes2 = min(iv3, ivmin);
        EXPECT_TRUE(equalIter(minRef2, minRef2 + SIZE, minRes2.cbegin(), SIZE));
        EXPECT_EQ(min(iv3), di[0]);
        EXPECT_EQ(min(fv3), df[0]);
        EXPECT_EQ(max(iv3), di[SIZE - 1]);
        EXPECT_EQ(max(fv3), df[SIZE - 1]);

        float maxRef[] = { 1.2f, 2.4f, 4.0f, 5.0f, 8.1f, 10.0f };
        EXPECT_TRUE(equalIter(maxRef, maxRef + SIZE, max(iv3, fv3).begin(), SIZE));
        float maxRefScalar[] = { 4.0f, 4.0f, 4.0f, 4.8f, 8.1f, 9.7f };
        EXPECT_TRUE(equalIter(maxRefScalar, maxRefScalar + SIZE, max(4.0f, fv3).begin(), SIZE));
        EXPECT_TRUE(equalIter(maxRefScalar, maxRefScalar + SIZE, max(fv3, 4.0f).begin(), SIZE));
        IV ivmax = floor(fv3);
        ivmax[1] = 3;
        int maxRef2[] = { 1, 3, 4, 5, 8, 10 };
        EXPECT_TRUE(equalIter(maxRef2, maxRef2 + SIZE, max(iv3, ivmax).begin(), SIZE));

        EXPECT_EQ(sqrt(iv3 * iv3), iv3);
        EXPECT_EQ(sqrt(pow(iv3, 2)), iv3);

        EXPECT_EQ(sum(iv3), SIZE == 3 ? 7 : 30);
        EXPECT_EQ(sum(fv3), SIZE == 3 ? 7.2f : 29.8f);
        EXPECT_EQ(prod(iv3), SIZE == 3 ? 8 : 3200);
        EXPECT_EQ(prod(fv3), SIZE == 3 ? 10.368f : 3910.15f);
        EXPECT_TRUE(mathfunctions::isclose(mean(iv3), 7.0 / SIZE, 1e-15));

        float cumsumRef[] = { 1.2f, 3.6f, 7.2f, 12.0f, 20.1f, 29.8f };
        FV cs = cumsum(fv3), csr(cumsumRef);
        EXPECT_TRUE(isclose(cs, csr, 1e-6f));
        float cumprodRef[] = { 1.2f, 2.88f, 10.368f, 49.7664f, 403.108f, 3910.15f };
        FV cr = cumprod(fv3), crr(cumprodRef);
        EXPECT_TRUE(isclose(cr, crr, 1e-6f));

        TinyArray<int, 4> src{ 1, 2, -3, -4 }, signs{ 2, -3, 4, -5 };
        EXPECT_EQ(copysign(src, signs), (TinyArray<int, 4>{1, -2, 3, -4}));

        TinyArray<double, 3> left{ 3., 5., 8. }, right{ 4., 12., 15. };
        EXPECT_EQ(hypot(left, right), (TinyArray<double, 3>{5., 13., 17.}));

        int oddRef[] = { 1, 0, 0, 1, 0, 0 };
        EXPECT_TRUE(equalIter(oddRef, oddRef + SIZE, odd(iv3).begin(), SIZE));
        EXPECT_TRUE(equalIter(oddRef, oddRef + SIZE, (iv3 & 1).begin(), SIZE));
    }

    TEST(xtiny, cross_product)
    {
        EXPECT_EQ(cross(bv3, bv3), IV(0));
        EXPECT_EQ(cross(iv3, bv3), IV(0));
        EXPECT_TRUE(isclose(cross(fv3, fv3), FV(0.0), 1e-6f));

        FV cr = cross(fv1, fv3), crr{ 1.2f, -2.4f, 1.2f };
        EXPECT_TRUE(isclose(cr, crr, 1e-6f));
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

    TEST(xtiny, 2D)
    {
        using Array = TinyArray<int, 2, 3>;
        using Index = TinyArray<ArrayIndex, 2>;

        EXPECT_EQ(Array::static_ndim, 2);
        EXPECT_EQ(Array::static_size, 6);
        EXPECT_TRUE((std::is_same<Index, Array::index_type>::value));

        int adata[] = { 4,5,6,7,8,9 };
        Array a{ adata };
        EXPECT_EQ(a.ndim(), 2);
        EXPECT_EQ(a.size(), 6);
        EXPECT_EQ(a.shape(), Index(2, 3));

        int count = 0, i, j;
        Index idx;
        for (i = 0, idx[0] = 0; i<2; ++i, ++idx[0])
        {
            for (j = 0, idx[1] = 0; j<3; ++j, ++count, ++idx[1])
            {
                EXPECT_EQ(a[count], adata[count]);
                EXPECT_EQ((a[{i, j}]), adata[count]);
                EXPECT_EQ(a[idx], adata[count]);
                EXPECT_EQ(a(i, j), adata[count]);
            }
        }
        {
            std::string s = "{4, 5, 6,\n 7, 8, 9}";
            std::stringstream ss;
            ss << a;
            EXPECT_EQ(s, ss.str());
        }

        TinySymmetricView<int, 3> sym(a.data());
        EXPECT_EQ(sym.shape(), Index(3, 3));
        {
            std::string s = "{4, 5, 6,\n 5, 7, 8,\n 6, 8, 9}";
            std::stringstream ss;
            ss << sym;
            EXPECT_EQ(s, ss.str());
        }

        Array::AsType<float> b = a;
        EXPECT_EQ(a, b);

        int adata2[] = { 0,1,2,3,4,5 };
        a = { 0,1,2,3,4,5 };
        EXPECT_TRUE(equalIter(a.begin(), a.end(), adata2, a.size()));
        Array c = reversed(a);
        EXPECT_TRUE(equalIter(c.rbegin(), c.rend(), adata2, c.size()));

        EXPECT_TRUE(a == a);
        EXPECT_TRUE(a != b);
        EXPECT_TRUE(a < b);
        EXPECT_TRUE(any(a));
        EXPECT_TRUE(!all(a));
        EXPECT_TRUE(any(b));
        EXPECT_TRUE(all(b));
        EXPECT_TRUE(!allZero(a));
        EXPECT_TRUE(allLess(a, b));
        EXPECT_TRUE(allLessEqual(a, b));
        EXPECT_TRUE(!allGreater(a, b));
        EXPECT_TRUE(!allGreaterEqual(a, b));
        EXPECT_TRUE(isclose(a, b, 10.0f));

        EXPECT_EQ(squaredNorm(a), 55);
        EXPECT_TRUE(mathfunctions::isclose(norm(a), sqrt(55.0), 1e-15));
        EXPECT_EQ(min(a), 0);
        EXPECT_EQ(max(a), 5);
        EXPECT_EQ(max(a, b), b);

        swap(b, c);
        EXPECT_TRUE(equalIter(c.cbegin(), c.cend(), adata, c.size()));
        EXPECT_TRUE(equalIter(b.crbegin(), b.crend(), adata2, b.size()));

        int eyedata[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        auto eye = Array::eye<3>();
        EXPECT_TRUE(equalIter(eye.begin(), eye.end(), eyedata, eye.size()));
    }

    TEST(xtiny, runtime_size)
    {
        using A = TinyArray<int>;

        EXPECT_TRUE(typeid(A) == typeid(TinyArray<int, runtime_size>));

        A a{ 1,2,3 }, b{ 1,2,3 }, c = a, d = a + b, e(3);
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
        c.init(2, 4, 6);
        EXPECT_EQ(d, c);
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
        EXPECT_TRUE(!allZero(c));
        EXPECT_TRUE(!all(e));
        EXPECT_TRUE(!any(e));
        EXPECT_TRUE(allZero(e));

        EXPECT_EQ(prod(a), 6);
        EXPECT_EQ(prod(A()), 0);

        EXPECT_EQ(cross(a, a), e);
        EXPECT_EQ(dot(a, a), 14);
        EXPECT_EQ(squaredNorm(a), 14);

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

        EXPECT_THROW(A(3) / A(2), std::runtime_error);

        using TA = TinyArray<int, 3>;
        TA s(A{ 1,2,3 });
        EXPECT_EQ(s, (TA{ 1,2,3 }));
        s = A{ 3,4,5 };
        EXPECT_EQ(s, (TA{ 3,4,5 }));

        EXPECT_THROW({ TA(A{ 1,2,3,4 }); }, std::runtime_error);

        EXPECT_EQ((A{ 0,0,0 }), A(tags::size = 3));
        EXPECT_EQ((A{ 1,1,1 }), A(tags::size = 3, 1));

        EXPECT_EQ(A::unitVector(tags::size = 3, 1), TA::unitVector(1));
    }

} // namespace xt