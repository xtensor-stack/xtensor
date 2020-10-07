/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xbroadcast.hpp"
#include "xtensor/xchunked_array.hpp"
#include "xtensor/xchunk_store_manager.hpp"
#include "xtensor/xfile_array.hpp"
#include "xtensor/xdisk_io_handler.hpp"
#include "xtensor/xcsv.hpp"

namespace xt
{
    using in_memory_chunked_array = xchunked_array<xarray<xarray<double>>>;

    TEST(xchunked_array, indexed_access)
    {
        std::vector<size_t> shape = {10, 10, 10};
        std::vector<size_t> chunk_shape = {2, 3, 4};
        in_memory_chunked_array a(shape, chunk_shape);

        std::vector<size_t> idx = {3, 9, 8};
        double val;

        val = 1.;
        a[idx] = val;
        ASSERT_EQ(a[idx], val);
        ASSERT_EQ(a(3, 9, 8), val);

        val = 2.;
        a(3, 9, 8) = val;
        ASSERT_EQ(a(3, 9, 8), val);
        ASSERT_EQ(a[idx], val);

        val = 3.;
        for (auto& it: a)
            it = val;
        for (auto it: a)
            ASSERT_EQ(it, val);
    }

    TEST(xchunked_array, assign_expression)
    {
#ifdef _MSC_FULL_VER
        std::cout << "MSC_FULL_VER = " << _MSC_FULL_VER << std::endl;
#endif
        std::vector<size_t> shape1 = {2, 2, 2};
        std::vector<size_t> chunk_shape1 = {2, 3, 4};
        in_memory_chunked_array a1(shape1, chunk_shape1);
        double val;

        val = 3.;
        a1 = broadcast(val, a1.shape());
        for (const auto& v: a1)
        {
            EXPECT_EQ(v, val);
        }

        std::vector<size_t> shape2 = {32, 10, 10};
        in_memory_chunked_array a2(shape2, chunk_shape1);

        a2 = broadcast(val, a2.shape());
        for (const auto& v: a2)
        {
            EXPECT_EQ(v, val);
        }

        a2 += a2;
        for (const auto& v: a2)
        {
            EXPECT_EQ(v, 2. * val);
        }

        xarray<double> a3
          {{1., 2., 3.},
           {4., 5., 6.},
           {7., 8., 9.}};

        EXPECT_EQ(is_chunked(a3), false);

        std::vector<size_t> chunk_shape4 = {2, 2};
        auto a4 = in_memory_chunked_array(a3, chunk_shape4);

        EXPECT_EQ(is_chunked(a4), true);

        double i = 1.;
        for (const auto& v: a4)
        {
            EXPECT_EQ(v, i);
            i += 1.;
        }

        auto a5 = in_memory_chunked_array(a4);
        EXPECT_EQ(is_chunked(a5), true);
        for (const auto& v: a5.chunk_shape())
        {
            EXPECT_EQ(v, 2);
        }

        auto a6 = in_memory_chunked_array(a3);
        EXPECT_EQ(is_chunked(a6), true);
        for (const auto& v: a6.chunk_shape())
        {
            EXPECT_EQ(v, 3);
        }
    }

    TEST(xchunked_array, disk_array)
    {
        std::vector<size_t> shape = {4, 4};
        std::vector<size_t> chunk_shape = {2, 2};
        std::string chunk_dir = "files";
        std::size_t pool_size = 2;
        xchunked_array<xchunk_store_manager<xfile_array<double, xdisk_io_handler<xcsv_config>>>> a1(shape, chunk_shape, chunk_dir, pool_size);
        std::vector<size_t> idx = {1, 2};
        double v1 = 3.4;
        double v2 = 5.6;
        double v3 = 7.8;
        a1(2, 1) = v1;
        a1[idx] = v2;
        a1(0, 0) = v3; // this should unload chunk 1.0
        ASSERT_EQ(a1(2, 1), v1);
        ASSERT_EQ(a1[idx], v2);
        ASSERT_EQ(a1(0, 0), v3);

        std::ifstream in_file;
        xt::xarray<double> ref;
        xt::xarray<double> data;
        in_file.open(chunk_dir + "/1.0");
        data = xt::load_csv<double>(in_file);
        ref = {{0, v1}, {0, 0}};
        EXPECT_EQ(data, ref);
        in_file.close();

        a1.chunks().flush();
        in_file.open(chunk_dir + "/0.1");
        data = xt::load_csv<double>(in_file);
        ref = {{0, 0}, {v2, 0}};
        EXPECT_EQ(data, ref);
        in_file.close();

        in_file.open(chunk_dir + "/0.0");
        data = xt::load_csv<double>(in_file);
        ref = {{v3, 0}, {0, 0}};
        EXPECT_EQ(data, ref);
        in_file.close();
    }

    TEST(xfile_array, indexed_access)
    {
        std::vector<size_t> shape = {2, 2, 2};
        xfile_array<double, xdisk_io_handler<xcsv_config>> a;
        a.ignore_empty_path(true);
        a.resize(shape);
        double val = 3.;
        for (auto it: a)
            it = val;
        for (auto it: a)
            ASSERT_EQ(it, val);
    }

    TEST(xfile_array, assign_expression)
    {
        double v1 = 3.;
        auto a1 = xfile_array<double, xdisk_io_handler<xcsv_config>>(broadcast(v1, {2, 2}), "a1");
        a1.ignore_empty_path(true);
        for (const auto& v: a1)
        {
            EXPECT_EQ(v, v1);
        }

        double v2 = 2. * v1;
        auto a2 = xfile_array<double, xdisk_io_handler<xcsv_config>>(a1 + a1, "a2");
        a2.ignore_empty_path(true);
        for (const auto& v: a2)
        {
            EXPECT_EQ(v, v2);
        }

        a1.flush();
        a2.flush();

        std::ifstream in_file;
        in_file.open("a1");
        auto data = load_csv<double>(in_file);
        xarray<double> ref = {{v1, v1}, {v1, v1}};
        EXPECT_EQ(data, ref);
        in_file.close();

        in_file.open("a2");
        data = load_csv<double>(in_file);
        ref = {{v2, v2}, {v2, v2}};
        EXPECT_EQ(data, ref);
        in_file.close();
    }
}
