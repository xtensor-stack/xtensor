/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

#include "xtensor/containers/xadapt.hpp"
#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xfixed.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xmath.hpp"
#include "xtensor/generators/xrandom.hpp"
#include "xtensor/io/xinfo.hpp"
#include "xtensor/io/xio.hpp"
#include "xtensor/misc/xsort.hpp"
#include "xtensor/views/xslice.hpp"
#include "xtensor/views/xview.hpp"

#include "test_common_macros.hpp"

namespace xt
{
    namespace
    {
        struct median_axes_range
        {
            std::array<int, 2> axes;
        };

        const int* begin(const median_axes_range& range)
        {
            return range.axes.data();
        }

        const int* end(const median_axes_range& range)
        {
            return range.axes.data() + range.axes.size();
        }

        // Enumerate logical coordinates and fully sort each output lane. This
        // oracle does not transpose, reshape, partition, or call xt::median.
        template <class E>
        xarray<double, layout_type::row_major>
        grouped_median_reference(const E& e, const std::vector<std::size_t>& axes)
        {
            std::vector<std::size_t> shape;
            std::size_t lane_count = 1;
            for (std::size_t ax = 0; ax < e.dimension(); ++ax)
            {
                if (std::find(axes.begin(), axes.end(), ax) == axes.end())
                {
                    shape.push_back(e.shape()[ax]);
                    lane_count *= e.shape()[ax];
                }
            }
            std::vector<std::vector<double>> lanes(lane_count);
            std::vector<std::size_t> index(e.dimension());
            for (std::size_t linear = 0; linear < e.size(); ++linear)
            {
                auto remainder = linear;
                for (std::size_t ax = e.dimension(); ax != 0; --ax)
                {
                    index[ax - 1] = remainder % e.shape()[ax - 1];
                    remainder /= e.shape()[ax - 1];
                }
                std::size_t lane = 0;
                for (std::size_t ax = 0; ax < e.dimension(); ++ax)
                {
                    if (std::find(axes.begin(), axes.end(), ax) == axes.end())
                    {
                        lane = lane * e.shape()[ax] + index[ax];
                    }
                }
                lanes[lane].push_back(static_cast<double>(e.element(index.begin(), index.end())));
            }
            auto result = xarray<double, layout_type::row_major>::from_shape(shape);
            for (std::size_t lane = 0; lane < lane_count; ++lane)
            {
                auto& values = lanes[lane];
                std::sort(values.begin(), values.end());
                auto middle = values.size() / 2;
                result.flat(lane) = values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.
                                                           : values[middle];
            }
            return result;
        }
    }

    TEST(xsort, argsort)
    {
        xarray<double> a1 = {2, 3, 1};
        xarray<double, xt::layout_type::column_major> a2_c = {{2, 3, 1}, {4, 6, 5}};
        xarray<double, xt::layout_type::row_major> a2_r = {{2, 3, 1}, {4, 6, 5}};
        xarray<float> a3 = {{{1, 3, 2}, {4, 2, 1}}, {{5, 1, 3}, {4, 2, 6}}};

        xarray<std::size_t> ex = {2, 0, 1};
        EXPECT_EQ(ex, argsort(a1, 0));
        EXPECT_EQ(ex, argsort(a1));

        xarray<std::size_t> ex2_0 = {{0, 0, 0}, {1, 1, 1}};
        xarray<std::size_t> ex2_1 = {{2, 0, 1}, {0, 2, 1}};

        EXPECT_EQ(ex2_1, argsort(a2_c));
        EXPECT_EQ(ex2_1, argsort(a2_r));
        EXPECT_EQ(ex2_0, argsort(a2_c, 0));
        EXPECT_EQ(ex2_0, argsort(a2_r, 0));
        EXPECT_EQ(ex2_1, argsort(a2_c, 1));
        EXPECT_EQ(ex2_1, argsort(a2_r, 1));

        xarray<std::size_t> ex3_0 = {{{0, 1, 0}, {0, 0, 0}}, {{1, 0, 1}, {1, 1, 1}}};
        xarray<std::size_t> ex3_1 = {{{0, 1, 1}, {1, 0, 0}}, {{1, 0, 0}, {0, 1, 1}}};
        xarray<std::size_t> ex3_2 = {{{0, 2, 1}, {2, 1, 0}}, {{1, 2, 0}, {1, 0, 2}}};

        EXPECT_EQ(ex3_2, argsort(a3));
        EXPECT_EQ(ex3_0, argsort(a3, 0));
        EXPECT_EQ(ex3_1, argsort(a3, 1));
        EXPECT_EQ(ex3_2, argsort(a3, 2));

        xtensor<double, 1> t1 = {2, 3, 1};
        xtensor<std::size_t, 1> ex4 = {2, 0, 1};
        EXPECT_EQ(ex4, argsort(t1, 0));
        EXPECT_EQ(ex4, argsort(t1));

        if (XTENSOR_DEFAULT_TRAVERSAL == layout_type::row_major)
        {
            xtensor_fixed<double, xt::xshape<2, 3>> tf1 = a2_r;
            EXPECT_EQ(ex2_1, argsort(tf1));
        }
        else
        {
            xtensor_fixed<double, xt::xshape<2, 3>> tf1 = a2_c;
            EXPECT_EQ(ex2_1, argsort(tf1));
        }
    }

    TEST(xsort, argsort_zero_stride)
    {
        {
            xt::xtensor<double, 2> A = {{1.4, 1.3, 1.2, 1.1}};
            xt::xtensor<size_t, 2> bsort = {{0, 0, 0, 0}};
            xt::xtensor<size_t, 2> fsort = {{3, 2, 1, 0}};
            EXPECT_EQ(bsort, xt::argsort(A, 0));
            EXPECT_EQ(fsort, xt::argsort(A, 1));
            EXPECT_EQ(fsort, xt::argsort(A));
        }
        {
            xt::xtensor<double, 3> A = {{{1.4, 1.3, 1.2, 1.1}}};
            xt::xtensor<size_t, 3> bsort = {{{0, 0, 0, 0}}};
            xt::xtensor<size_t, 3> fsort = {{{3, 2, 1, 0}}};
            EXPECT_EQ(bsort, xt::argsort(A, 0));
            EXPECT_EQ(bsort, xt::argsort(A, 1));
            EXPECT_EQ(fsort, xt::argsort(A, 2));
            EXPECT_EQ(fsort, xt::argsort(A));
        }
    }

    TEST(xsort, argsort_zero_stride_column_major)
    {
        {
            xt::xtensor<double, 2, xt::layout_type::column_major> A = {{1.4, 1.3, 1.2, 1.1}};
            xt::xtensor<size_t, 2, xt::layout_type::column_major> bsort = {{0, 0, 0, 0}};
            xt::xtensor<size_t, 2, xt::layout_type::column_major> fsort = {{3, 2, 1, 0}};
            EXPECT_EQ(bsort, xt::argsort(A, 0));
            EXPECT_EQ(fsort, xt::argsort(A, 1));
            EXPECT_EQ(fsort, xt::argsort(A));
        }
        {
            xt::xtensor<double, 3, xt::layout_type::column_major> A = {{{1.4, 1.3, 1.2, 1.1}}};
            xt::xtensor<size_t, 3, xt::layout_type::column_major> bsort = {{{0, 0, 0, 0}}};
            xt::xtensor<size_t, 3, xt::layout_type::column_major> fsort = {{{3, 2, 1, 0}}};
            EXPECT_EQ(bsort, xt::argsort(A, 0));
            EXPECT_EQ(bsort, xt::argsort(A, 1));
            EXPECT_EQ(fsort, xt::argsort(A, 2));
            EXPECT_EQ(fsort, xt::argsort(A));
        }
    }

    TEST(xsort, flatten_argsort)
    {
        {
            xarray<int, layout_type::row_major> a = {{1, 2, 3}, {4, 5, 6}};
            xarray<std::size_t, layout_type::row_major> res = argsort(a, placeholders::xtuph());
            xarray<std::size_t, layout_type::row_major> expected = {0, 1, 2, 3, 4, 5};
            EXPECT_EQ(res, expected);
        }
        {
            xarray<int, layout_type::column_major> a = {{1, 2, 3}, {4, 5, 6}};
            xarray<std::size_t, layout_type::column_major> res = argsort(a, placeholders::xtuph());
            xarray<std::size_t, layout_type::column_major> expected = {0, 1, 2, 3, 4, 5};
            EXPECT_EQ(res, expected);
        }
    }

    TEST(xsort, argsort_stable)
    {
        xt::xarray<int> a = {
            2247,  2044, 2037, 1825, 1699, 1600, 1501, 1432, 1440, 1388, 1299, 1259,  1211, 1177, 1124, 1121,
            1399,  1179, 1102, 1055, 1017, 1001, 979,  953,  925,  927,  1899, 4782,  2601, 3050, 1998, 3478,
            5762,  8745, 7777, 4086, 2968, 2456, 2138, 3199, 7187, 7165, 3942, 11588, 4643, 3618, 3184, 6052,
            3723,  3356, 2645, 2330, 2103, 1928, 1890, 3624, 3647, 5821, 2949, 4161,  3855, 5200, 3162, 3896,
            19818, 5228, 3711, 4874, 4868, 4267, 2978, 2748, 2500, 2276, 2107, 1969,  1815, 1714, 1649, 1561,
            1491,  1428, 1347, 1288, 1245, 1207, 1163, 1189, 1558, 1313, 1186, 1147,  1143, 1099, 1036, 1008,
            982,   953,  924,  899,  880,  858,  836,  818,  791,  762,  742,  720,   701,  685,  674,  658,
            643,   625,  609,  602,  594,  594,  625,  616,  707,  759,  929,  755,   684,  737,  804,  680,
            639,   620,  706,  2474, 1406, 2686, 2037, 1410, 1130, 992,  901,  847,   798,  766,  750,  2793,
            1574,  1087, 960,  883,  823,  782,  755,  730,  705,  677,  744,  932,   770,  794,  804,  728,
            721,   765,  826,  795,  784,  875,  711,  795,  2295, 1265, 1115, 936,   817,  742,  713,  680,
            647,   647,  671,  651,  692,  605,  576,  612,  564,  536,  526,  516,   506,  494,  473,  459,
            451,   450,  441,  428,  430,  453,  467,  444,  413,  390,  376,  373,   368,  360,  359,  351,
            348,   345,  343,  340,  337,  341,  529,  476,  448,  397,  367,  354,   342,  332,  320,  324,
            327,   332,  340,  359,  331,  382,  371,  356,  333,  318,  316,  318,   312,  308,  300,  298,
            399,   448,  669,  534,  402,  384,  361,  347,  337,  342,  349,  339,   333,  325,  324,  326,
            324,   322,  320,  318,  322,  327,  327,  333,  329,  333,  353,  436,   516,  412,  402,  398,
            551,   449,  416,  484,  801,  537,  457,  483,  534,  477,  483,  582,   575,  557,  504,  477,
            515,   643,  627,  627,  546,  515,  497,  577,  1439, 1198, 1591, 1815,  1448, 1117, 960,  847,
            809,   788,  749,  896,  870,  801,  752
        };

        xt::xarray<std::size_t> ex = {
            239, 238, 237, 236, 234, 233, 235, 259, 222, 258, 257, 260, 223, 254, 256, 253, 255, 224, 261,
            262, 264, 228, 221, 225, 232, 252, 263, 265, 212, 248, 251, 211, 226, 213, 220, 249, 210, 209,
            247, 208, 250, 207, 266, 219, 231, 206, 227, 205, 246, 218, 204, 230, 203, 202, 229, 245, 201,
            217, 271, 240, 244, 270, 269, 200, 274, 195, 196, 267, 194, 199, 216, 241, 273, 193, 192, 197,
            278, 191, 198, 190, 215, 281, 287, 279, 282, 275, 189, 294, 286, 188, 288, 293, 187, 268, 186,
            214, 243, 280, 185, 277, 292, 272, 285, 184, 284, 182, 295, 283, 116, 117, 115, 181, 114, 183,
            119, 129, 113, 118, 290, 291, 128, 112, 289, 176, 177, 179, 111, 242, 178, 110, 153, 127, 175,
            124, 109, 180, 108, 152, 130, 120, 166, 174, 107, 160, 159, 151, 125, 106, 173, 154, 306, 142,
            310, 123, 150, 121, 105, 161, 141, 156, 149, 164, 305, 104, 157, 163, 167, 140, 276, 309, 126,
            158, 304, 172, 103, 148, 162, 102, 139, 303, 101, 308, 165, 100, 147, 307, 99,  138, 98,  24,
            25,  122, 155, 171, 23,  97,  146, 302, 22,  96,  137, 21,  95,  20,  94,  19,  145, 93,  18,
            170, 301, 15,  14,  136, 92,  91,  86,  13,  17,  90,  87,  297, 85,  12,  84,  11,  169, 83,
            10,  89,  82,  9,   16,  132, 135, 81,  7,   296, 8,   300, 80,  6,   88,  79,  144, 298, 5,
            78,  4,   77,  76,  299, 3,   54,  26,  53,  75,  30,  2,   134, 1,   52,  74,  38,  0,   73,
            168, 51,  37,  131, 72,  28,  50,  133, 71,  143, 58,  36,  70,  29,  62,  46,  39,  49,  31,
            45,  55,  56,  66,  48,  60,  63,  42,  35,  59,  69,  44,  27,  68,  67,  61,  65,  32,  57,
            47,  41,  40,  34,  33,  43,  64
        };

        EXPECT_EQ(ex, xt::argsort(a, 0, xt::sorting_method::stable));
    }

    TEST(xsort, sort_easy)
    {
        xarray<double> a = {{5, 3, 1}, {4, 4, 4}};

        xarray<double> ex = {1, 3, 4, 4, 4, 5};
        EXPECT_EQ(ex, sort(a, xnone()));

        auto n_ex = sort(ex);
        EXPECT_EQ(n_ex, ex);

        auto n_ex_0 = sort(ex, 0);
        EXPECT_EQ(n_ex, ex);

        xarray<double> ex_2 = {{4, 3, 1}, {5, 4, 4}};
        EXPECT_EQ(ex_2, sort(a, 0));

        xarray<double> ex_3 = {{1, 3, 5}, {4, 4, 4}};
        EXPECT_EQ(ex_3, sort(a, 1));

#ifndef XTENSOR_DISABLE_EXCEPTIONS
        xt::xarray<double> xarr = xt::eval(xt::arange(0, 16));
        xarr.reshape({4, 4});
        auto view = xt::reshape_view(xt::transpose(xarr, {1, 0}), {2, 8});
        EXPECT_NO_THROW(xt::sort(view, 0));
#endif
    }

    TEST(xsort, fixed)
    {
        xtensor_fixed<double, xshape<4, 3>> a = {{5, 3, 1}, {4, 4, 4}, {5, 9, 1}, {2, 4, 2}};
        xarray<double> b = {{5, 3, 1}, {4, 4, 4}, {5, 9, 1}, {2, 4, 2}};

        EXPECT_EQ(sort(a, xnone()), sort(b, xnone()));
        EXPECT_EQ(sort(a), sort(b));
        EXPECT_EQ(sort(a, 1), sort(b, 1));
        EXPECT_EQ(sort(a, 0), sort(b, 0));
    }

    TEST(xsort, argmin)
    {
        xarray<double> a = {{5, 3, 1}, {4, 4, 4}};
        xarray<double> b = {1, 3, 4, -100};
        xarray<double, layout_type(int(XTENSOR_DEFAULT_LAYOUT) & 0x03)> ar = {{5, 3, 1}, {4, 4, 4}};

        xarray<std::size_t> ex = 2ul;

        EXPECT_EQ(ex, argmin(a));

        EXPECT_EQ(size_t(3), argmin(b)());
        EXPECT_EQ(size_t(3), argmin(b, 0)());

        xarray<std::size_t> ex_2 = {1, 0, 0};
        EXPECT_EQ(ex_2, argmin(a, 0));

        xarray<std::size_t> ex_3 = {2, 0};
        EXPECT_EQ(ex_3, argmin(a, 1));

        xtensor<double, 2> xa = {{5, 3, 1}, {4, 4, 4}};
        EXPECT_EQ(ex, argmin(xa));
        EXPECT_EQ(ex_2, argmin(xa, 0));
        EXPECT_EQ(ex_3, argmin(xa, 1));

        xtensor<double, 1> ya = {1, 0, 3, 2, 2};
        EXPECT_EQ(1, argmin(ya)());
        EXPECT_EQ(1, argmin(ya, 0)());
    }

    TEST(xsort, argmax)
    {
        xarray<double> a = {{5, 3, 1}, {4, 4, 4}};

        EXPECT_EQ(0ul, argmax(a)());

        xarray<std::size_t> ex_2 = {0, 1, 1};
        EXPECT_EQ(ex_2, argmax(a, 0));

        xarray<std::size_t> ex_3 = {0, 0};
        EXPECT_EQ(ex_3, argmax(a, 1));

        xtensor<int, 2> b = {{1, 2}};
        auto res = xt::eval(xt::argmax(b, 1));
        EXPECT_EQ(res(), 1u);

        xtensor<int, 3> c = {{{1, 2, 3, 4}}, {{4, 3, 2, 1}}};
        xtensor<std::size_t, 2> ex_4 = {{3}, {0}};
        EXPECT_EQ(ex_4, argmax(c, 2));

        xtensor<std::size_t, 2> ex_5 = {{1, 1, 0, 0}};
        EXPECT_EQ(ex_5, argmax(c, 0));

        xtensor<std::size_t, 2> ex_6 = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        EXPECT_EQ(ex_6, argmax(c, 1));

        xtensor<double, 1> ya = {1, 0, 3, 2, 2};
        EXPECT_EQ(2, argmax(ya)());
        EXPECT_EQ(2, argmax(ya, 0)());

        // xtensor#2568
        xarray<double> d = {0, 1, 0};
        xtensor<size_t, 0> d_ex_1 = {1};
        EXPECT_EQ(d_ex_1, argmax(d));
        EXPECT_EQ(1, argmax(d)(0));
    }

    TEST(xsort, sort_large_prob)
    {
        for (std::size_t i = 0; i < 20; ++i)
        {
            xarray<double> a = xt::random::rand<double>({5, 5, 100, 10});

            auto a_s0 = sort(a, 0);
            auto va_s0 = view(a_s0, xt::all(), 3, 45, 3);
            EXPECT_TRUE(std::is_sorted(va_s0.begin(), va_s0.end()));

            auto a_s1 = sort(a, 1);
            auto va_s1 = view(a_s1, 0, xt::all(), 0, 0);
            EXPECT_TRUE(std::is_sorted(va_s1.begin(), va_s1.end()));

            auto a_s2 = sort(a, 2);
            auto va_s2 = view(a_s2, 3, 1, xt::all(), 3);
            EXPECT_TRUE(std::is_sorted(va_s2.begin(), va_s2.end()));

            auto a_s3 = sort(a, 3);
            auto va_s3 = view(a_s3, 4, 3, 45, xt::all());
            EXPECT_TRUE(std::is_sorted(va_s3.begin(), va_s3.end()));
        }
    }

    TEST(xsort, argmax_prob)
    {
        for (std::size_t i = 0; i < 20; ++i)
        {
            xarray<double> a = xt::random::rand<double>({5, 4, 6, 7});

            auto a_s0 = argmin<XTENSOR_DEFAULT_LAYOUT>(a, 0);
            auto va_s0 = view(a, xt::all(), 3, 2, 3);
            auto m0_idx = a_s0(3, 2, 3);
            auto it0 = std::min_element(va_s0.begin(), va_s0.end());
            auto c0_idx = std::distance(va_s0.begin(), it0);
            EXPECT_EQ(static_cast<size_t>(c0_idx), m0_idx);

            auto a_s1 = argmin<XTENSOR_DEFAULT_LAYOUT>(a, 1);
            auto va_s1 = view(a, 3, xt::all(), 2, 3);
            auto m1_idx = a_s1(3, 2, 3);
            auto it1 = std::min_element(va_s1.begin(), va_s1.end());
            auto c1_idx = std::distance(va_s1.begin(), it1);
            EXPECT_EQ(static_cast<size_t>(c1_idx), m1_idx);

            auto a_s2 = argmin<XTENSOR_DEFAULT_LAYOUT>(a, 2);
            auto va_s2 = view(a, 3, 2, xt::all(), 3);
            auto m2_idx = a_s2(3, 2, 3);
            auto it2 = std::min_element(va_s2.begin(), va_s2.end());
            auto c2_idx = std::distance(va_s2.begin(), it2);
            EXPECT_EQ(static_cast<size_t>(c2_idx), m2_idx);

            auto a_s3 = argmin<XTENSOR_DEFAULT_LAYOUT>(a, 3);
            auto va_s3 = view(a, 3, 2, 3, xt::all());
            auto m3_idx = a_s3(3, 2, 3);
            auto it3 = std::min_element(va_s3.begin(), va_s3.end());
            auto c3_idx = std::distance(va_s3.begin(), it3);
            EXPECT_EQ(static_cast<size_t>(c3_idx), m3_idx);
        }
    }

    TEST(xsort, unique)
    {
        xarray<double> a = {1, 2, 3, 5, 3, 2, 1, 2, 2, 2, 2, 2, 2, 45};
        xarray<double> ax = {1, 2, 3, 5, 45};
        EXPECT_EQ(unique(a), ax);

        xarray<double> b = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> bx = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        EXPECT_EQ(unique(b), bx);

        xarray<double> bb = {{1, 2, 3}, {7, 8, 9}, {4, 5, 6}, {7, 8, 9}};
        xarray<double> bbx = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        EXPECT_EQ(unique(bb), bbx);

        auto c = view(b, range(0, 3), range(0, 3));
        auto d = unique(c);
        EXPECT_EQ(d, bx);

        auto e = xt::unique(xt::where(xt::greater(b, 2), 1, 0));
        xarray<double> ex = {0, 1};
        EXPECT_EQ(e, ex);
    }

    TEST(xsort, setdiff1d)
    {
        {
            xarray<size_t> ar1 = {1, 2, 3, 4, 4, 4, 5};
            xarray<size_t> ar2 = {4, 5};
            xarray<size_t> out = {1, 2, 3};
            EXPECT_EQ(setdiff1d(ar1, ar2), out);
        }

        {
            xarray<size_t> ar1 = {{5, 6, 7}, {4, 4, 4}, {1, 2, 3}};
            xarray<size_t> ar2 = {4, 1};
            xarray<size_t> out = {2, 3, 5, 6, 7};
            EXPECT_EQ(setdiff1d(ar1, ar2), out);
        }
    }

    template <class T>
    bool check_partition(T& arr, std::size_t pos)
    {
        bool res = true;
        for (std::size_t i = 0; i < pos; ++i)
        {
            res = res && arr(i) <= arr(pos);
        }
        for (std::size_t i = pos; i < arr.size(); ++i)
        {
            res = res && arr(i) >= arr(pos);
        }
        return res;
    }

    TEST(xsort, partition)
    {
        SUBCASE("simple")
        {
            xt::xarray<int> a = {3, 4, 2, 1};
            auto part = xt::partition(a, 2);
            EXPECT_TRUE(check_partition(part, 2));
        }

        SUBCASE("complex")
        {
            xt::xarray<double> a = {927.,  932.,  935.,  943.,  944.,  944.,  945.,  948.,  952.,
                                    962.,  968.,  968.,  969.,  969.,  974.,  981.,  993.,  994.,
                                    994.,  1003., 1007., 1008., 1008., 1012., 1013., 1014., 1014.,
                                    1017., 1019., 1020., 1023., 1026., 1026., 1028., 1030., 1032.,
                                    1039., 1047., 1071., 1080., 1085., 1088., 1111., 1112., 1117.,
                                    1119., 1128., 1130., 1209., 1309., 1426.};
            xt::xtensor_fixed<std::size_t, xt::xshape<4>> kth = {17, 32, 18, 33};

            SUBCASE("1D")
            {
                auto part = xt::partition(a, kth);
                for (std::size_t k : kth)
                {
                    CAPTURE(k);
                    EXPECT_TRUE(check_partition(part, k));
                }
            }

            SUBCASE("2D")
            {
                a.reshape({1, a.size()});
                auto part = xt::partition(a, kth, 1);
                for (std::size_t k : kth)
                {
                    CAPTURE(k);
                    EXPECT_TRUE(check_partition(part, k));
                }
            }
        }

        SUBCASE("exception")
        {
#ifndef XTENSOR_DISABLE_EXCEPTIONS
            xt::xarray<double> a = xt::eval(xt::arange(0, 16));
            a.reshape({4, 4});
            auto view = xt::transpose(a, {1, 0});
            EXPECT_NO_THROW(xt::partition(view, 1, 0));
#endif
        }
    }

    template <class T, class U>
    bool check_argpartition(T& arr, U& idxs, std::size_t pos)
    {
        bool res = true;
        for (std::size_t i = 0; i < pos; ++i)
        {
            res = res && arr(idxs[i]) <= arr(idxs[pos]);
        }
        for (std::size_t i = pos; i < arr.size(); ++i)
        {
            res = res && arr(idxs[i]) >= arr(idxs[pos]);
        }
        return res;
    }

    TEST(xsort, argpartition)
    {
        SUBCASE("simple")
        {
            xt::xarray<int> a = {3, 4, 2, 1};
            auto r1 = xt::argpartition(a, 2);
            EXPECT_TRUE(check_argpartition(a, r1, 2));

            SUBCASE("adapt")
            {
                std::size_t s = a.size();
                int* arr = a.data();
                dynamic_shape<std::size_t> sh = {s};
                auto b = xt::adapt(arr, s, xt::no_ownership(), sh);
                auto r2 = xt::argpartition(b, 2);
                EXPECT_TRUE(check_argpartition(b, r2, 2));
            }
        }

        SUBCASE("complex")
        {
            const xt::xarray<double> data = {1014., 1017., 1019., 1020., 1023., 1026., 1026., 1028., 1030.,
                                             1032., 1039., 1047., 1071., 927.,  932.,  935.,  943.,  944.,
                                             944.,  945.,  948.,  952.,  962.,  968.,  968.,  969.,  969.,
                                             974.,  981.,  993.,  994.,  994.,  1003., 1007., 1008., 1008.,
                                             1012., 1013., 1014., 1080., 1085., 1088., 1111., 1112., 1117.,
                                             1119., 1128., 1130., 1209., 1309., 1426.};
            const xt::xtensor_fixed<std::size_t, xt::xshape<4>> kth = {17, 32, 18, 33};

            SUBCASE("1D")
            {
                const auto& a = data;
                const auto argpart = xt::argpartition(a, kth);
                for (std::size_t k : kth)
                {
                    CAPTURE(k);
                    EXPECT_TRUE(check_argpartition(a, argpart, k));
                }
            }

            SUBCASE("2D")
            {
                auto a = data;
                a.reshape({1, a.size()});
                const auto argpart = xt::argpartition(a, kth, 1);
                for (std::size_t k : kth)
                {
                    CAPTURE(k);
                    EXPECT_TRUE(check_argpartition(a, argpart, k));
                }
            }
        }
    }

    TEST(xsort, quantile)
    {
        const xt::xtensor_fixed<double, xt::xshape<4, 2, 2>> data = {
            {{3., 4.}, {2., 1.}},
            {{-1., 1.}, {3., 2.}},
            {{-9., -0.}, {-9., -3.}},
            {{12., 12.}, {12., 12.}},
        };

        {
            auto q = xt::quantile(data, {.3, .0, 1.});
            const xt::xtensor_fixed<double, xt::xshape<3>> expected = {0.5, -9., 12.};
            EXPECT_TRUE(xt::equal(q, expected)(0));
        }

        const xt::xtensor_fixed<double, xt::xshape<3, 4, 2>> expected1 = {
            {{2.3, 1.9}, {0.2, 1.3}, {-9., -2.1}, {12., 12.}},
            {{2., 1.}, {-1., 1.}, {-9., -3.}, {12., 12.}},
            {{3., 4.}, {3., 2.}, {-9., 0.}, {12., 12.}},
        };
        {
            auto q1 = xt::quantile(data, {.3, .0, 1.}, 1);
            EXPECT_TRUE(xt::equal(q1, expected1)(0));
        }
        {
            auto q1 = xt::quantile(xt::xtensor<float, 3>(data), {.3, .0, 1.}, 1);
            EXPECT_TRUE(xt::equal(q1, expected1)(0));
        }
        {
            auto q1 = xt::quantile(xt::xarray<float>(data), {.3, .0, 1.}, 1);
            EXPECT_TRUE(xt::equal(q1, expected1)(0));
        }
    }

    TEST(xsort, median)
    {
        xt::xtensor<float, 2> a = {{3, 4, 2, 1}, {1, 1, 3, 2}, {9, 9, 9, 9}, {12, 12, 12, 12}, {5, 5, 5, 5}};
        auto mall = median(a);
        auto ma0 = median(a, 0);
        auto ma1 = median(a, 1);

        EXPECT_EQ(mall, 5.f);

        xt::xtensor<float, 1> ma0_exp = {5, 5, 5, 5};
        xt::xtensor<float, 1> ma1_exp = {2.5, 1.5, 9., 12., 5.};
        EXPECT_EQ(ma0, ma0_exp);
        EXPECT_EQ(ma1, ma1_exp);
    }

    TEST(xsort, median_grouped_axes)
    {
        const xtensor<double, 3> a = {
            {{0., 0., 0., 100.}, {1., 2., 99., 100.}, {-9., -8., -7., -6.}},
            {{10., 10., 10., 10.}, {3., 4., 5., 6.}, {-5., 30., 40., 50.}}
        };
        const auto original = a;
        const xarray<double> expected = {10., 4.5, -5.5};
        EXPECT_EQ(grouped_median_reference(a, {0, 2}), expected);
        EXPECT_EQ(median(a, {0, 2}), expected);
        EXPECT_EQ(median(a, {2, 0}), expected);
        EXPECT_EQ(median(a, {-1, -3}), expected);
        const int array_axes[] = {0, 2};
        EXPECT_EQ(median(a, array_axes), expected);
        const std::initializer_list<int> list_axes = {0, 2};
        EXPECT_EQ(median(a, list_axes), expected);
        xarray<double> last_axis = median(a, 2);
        xarray<double> first_axis = median(a, 0);
        EXPECT_NE(median(last_axis, 0), expected);
        EXPECT_NE(median(first_axis, 1), expected);

        const std::vector<std::size_t> axes = {0, 2};
        const std::array<int, 2> reversed = {2, 0};
        const int raw_axes[] = {0, 2};
        EXPECT_EQ(median(a, axes), expected);
        EXPECT_EQ(median(a, reversed), expected);
        EXPECT_EQ(median(a, raw_axes), expected);
        EXPECT_EQ(median(a, {0, 1}), grouped_median_reference(a, {0, 1}));
        EXPECT_EQ(median(a, {1, 2}), grouped_median_reference(a, {1, 2}));
        auto all_axes = median(a, {2, 0, 1});
        EXPECT_EQ(all_axes.dimension(), 0);
        EXPECT_EQ(all_axes(), median(a));
        EXPECT_EQ(a, original);
    }

    TEST(xsort, median_grouped_axes_layouts_and_views)
    {
        xtensor<int, 3> a = xtensor<int, 3>::from_shape({3, 2, 3});
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            a.flat(i) = static_cast<int>((i * 11) % 29) - 14;
        }
        const auto original = a;
        const auto expected = grouped_median_reference(a, {0, 2});
        xarray<int, layout_type::column_major> column_major = a;
        EXPECT_EQ(median(a, {0, 2}), expected);
        EXPECT_EQ(median(column_major, {2, 0}), expected);
        auto lazy = a + 2;
        static_assert(!strided_expression<decltype(lazy)>);
        EXPECT_EQ(median(lazy, {0, 2}), expected + 2);
        EXPECT_EQ(median(a + 2, {0, 2}), expected + 2);

        auto transposed = transpose(a, {2, 0, 1});
        EXPECT_EQ(median(transposed, {0, 1}), expected);
        auto strided = view(a, range(0, 3, 2), all(), range(0, 3, 2));
        EXPECT_EQ(median(strided, {0, 2}), grouped_median_reference(strided, {0, 2}));
        auto reversed = view(a, all(), all(), range(placeholders::_, placeholders::_, -1));
        EXPECT_EQ(median(reversed, {0, 2}), expected);
        auto broadcasted = broadcast(a, {2, 3, 2, 3});
        static_assert(!strided_expression<decltype(broadcasted)>);
        EXPECT_EQ(median(broadcasted, {0, 1, 3}), grouped_median_reference(broadcasted, {0, 1, 3}));
        EXPECT_EQ(median(broadcast(a, {2, 3, 2, 3}), {3, 1, 0}), grouped_median_reference(broadcasted, {0, 1, 3}));
        EXPECT_EQ(a, original);
        EXPECT_EQ(column_major, original);
    }

    TEST(xsort, median_grouped_axes_types_and_legacy)
    {
        xarray<int> integers = {{1, 2}, {5, 8}};
        auto grouped = median(integers, {0, 1});
        auto single = median(integers, 0);
        static_assert(std::is_same<decltype(grouped)::value_type, decltype(single)::value_type>::value, "");
        EXPECT_EQ(grouped(), 3.5);
        EXPECT_EQ(median(integers), 3);  // The existing scalar API keeps its input type.
        EXPECT_EQ(median(integers, {0}), single);
        EXPECT_EQ(median(integers, {1}), median(integers, 1));
        EXPECT_EQ(median(integers, {-1}), median(integers, -1));

        enum axis
        {
            rows = 0
        };

        EXPECT_EQ(median(integers, rows), single);
        EXPECT_EQ(median(integers, std::integral_constant<int, 1>{}), median(integers, 1));
        static_assert(std::ranges::range<const median_axes_range>);
        EXPECT_EQ(median(integers, median_axes_range{{0, 1}}), grouped);
        EXPECT_EQ(median(integers, std::vector<int>{}), integers);

        xarray<float> floats = {{1.f, 2.f}, {5.f, 8.f}};
        auto grouped_float = median(floats, {0, 1});
        auto single_float = median(floats, 0);
        static_assert(
            std::is_same<decltype(grouped_float)::value_type, decltype(single_float)::value_type>::value,
            ""
        );
        EXPECT_EQ(grouped_float(), 3.5f);
        EXPECT_EQ(median(floats, std::array<int, 0>{}), floats);
        xarray<std::uint64_t> large_integers = {9007199254740993ULL};
        auto empty_axes = median(large_integers, std::vector<int>{});
        auto one_axis = median(large_integers, 0);
        static_assert(std::is_same<decltype(empty_axes)::value_type, decltype(one_axis)::value_type>::value, "");
        static_assert(std::is_same<decltype(empty_axes)::value_type, double>::value, "");
        EXPECT_EQ(empty_axes.dimension(), 1);
        EXPECT_EQ(empty_axes.shape()[0], 1);
        EXPECT_EQ(empty_axes(0), 9007199254740992.);
        EXPECT_EQ(one_axis(), empty_axes(0));
        EXPECT_EQ(large_integers(0), 9007199254740993ULL);
        xarray<double> scalar = 7.;
        auto scalar_result = median(scalar, std::vector<int>{});
        EXPECT_EQ(scalar_result.dimension(), 0);
        EXPECT_EQ(scalar_result(), 7.);

        auto empty_lanes = xarray<double>::from_shape({0, 3, 2});
        auto empty_result = median(empty_lanes, {1, 2});
        EXPECT_EQ(empty_result.dimension(), 1);
        EXPECT_EQ(empty_result.shape()[0], 0);
    }

    TEST(xsort, median_grouped_axes_invalid)
    {
        xarray<double> a = {{1., 2.}, {3., 4.}};
        EXPECT_THROW(median(a, {0, 0}), std::runtime_error);
        EXPECT_THROW(median(a, {0, -2}), std::runtime_error);
        EXPECT_THROW(median(a, {2}), std::runtime_error);
        EXPECT_THROW(median(a, {-3}), std::runtime_error);
        EXPECT_THROW(median(a, {std::numeric_limits<std::ptrdiff_t>::min()}), std::runtime_error);
        EXPECT_THROW(median(a, {std::numeric_limits<std::size_t>::max()}), std::runtime_error);
        EXPECT_THROW(median(a, {std::numeric_limits<long long>::min()}), std::runtime_error);
        if constexpr (sizeof(std::size_t) < sizeof(long long))
        {
            EXPECT_THROW(median(a, {4294967296LL}), std::runtime_error);
            EXPECT_THROW(median(a, {-4294967297LL}), std::runtime_error);
            EXPECT_THROW(median(a, {4294967296ULL}), std::runtime_error);
        }
        EXPECT_EQ(median(a, std::array<signed char, 1>{-1}), (xarray<double>{1.5, 3.5}));
        EXPECT_EQ(median(a, std::array<unsigned char, 1>{0}), (xarray<double>{2., 3.}));
        EXPECT_EQ(median(a, std::array<bool, 1>{true}), (xarray<double>{1.5, 3.5}));
        EXPECT_THROW(median(a, {std::numeric_limits<signed char>::min()}), std::runtime_error);
        auto high_rank = xarray<double>::from_shape(std::vector<std::size_t>(130, 1));
        high_rank.fill(7.);
        xarray<double> narrow_axis = median(high_rank, {std::numeric_limits<signed char>::min()});
        EXPECT_EQ(narrow_axis.dimension(), 129);
        EXPECT_EQ(narrow_axis.size(), 1);
        EXPECT_EQ(narrow_axis.flat(0), 7.);
        auto empty = xarray<double>::from_shape({2, 0, 3});
        EXPECT_THROW(median(empty, {0, 1}), std::runtime_error);
        xarray<double> scalar = 1.;
        EXPECT_THROW(median(scalar, {0}), std::runtime_error);
    }
}
