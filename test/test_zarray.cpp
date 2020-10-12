/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/zarray.hpp"
#include "xtensor/zfunction.hpp"
#include "xtensor/xfile_array.hpp"
#include "xtensor/xdisk_io_handler.hpp"

#ifndef XTENSOR_DISABLE_EXCEPTIONS
namespace xt
{
    using namespace xt;
    TEST(zarray, value_semantics)
    {
        xarray<double> a = {{1., 2.}, {3., 4.}};
        xarray<double> ra = {{2., 2.}, {3., 4.}};
        zarray da(a);
        da.get_array<double>()(0, 0) = 2.;

        EXPECT_EQ(a, ra);
    }

    // TODO : move to dedicated test file
    TEST(zarray, dispatching)
    {
        using dispatcher_type = zdispatcher_t<math::exp_fun, 1>;
        dispatcher_type::init();

        xarray<double> a = {{0.5, 1.5}, {2.5, 3.5}};
        xarray<double> expa = {{std::exp(0.5), std::exp(1.5)}, {std::exp(2.5), std::exp(3.5)}};
        xarray<double> res;
        zarray za(a);
        zarray zres(res);

        dispatcher_type::dispatch(za.get_implementation(), zres.get_implementation());

        EXPECT_EQ(expa, res);
    }

    // TODO: move to dedicated test file
    TEST(zarray, zfunction)
    {
        using exp_dispatcher_type = zdispatcher_t<math::exp_fun, 1>;
        exp_dispatcher_type::init();

        using add_dispatcher_type = zdispatcher_t<detail::plus, 2>;
        add_dispatcher_type::init();

        using nested_zfunction_type = zfunction<math::exp_fun, const zarray&>;
        using zfunction_type = zfunction<detail::plus, const zarray&, nested_zfunction_type>;

        xarray<double> a = {{0.5, 1.5}, {2.5, 3.5}};
        xarray<double> b = {{-0.2, 2.4}, {1.3, 4.7}};
        xarray<double> res;

        zarray za(a);
        zarray zb(b);
        zarray zres(res);

        zfunction_type f(zplus(), za, nested_zfunction_type(zexp(), zb));
        f.assign_to(zres.get_implementation());

        auto expected = xarray<double>::from_shape({2, 2});
        std::transform(a.cbegin(), a.cend(), b.cbegin(), expected.begin(),
                       [](const double& lhs, const double& rhs) { return lhs + std::exp(rhs); });

        EXPECT_TRUE(all(isclose(res, expected)));

        size_t res_index = f.get_result_type_index();
        EXPECT_EQ(res_index, ztyped_array<double>::get_class_static_index());
    }

    TEST(zarray, operations)
    {
        using exp_dispatcher_type = zdispatcher_t<math::exp_fun, 1>;
        exp_dispatcher_type::init();

        using add_dispatcher_type = zdispatcher_t<detail::plus, 2>;
        add_dispatcher_type::init();

        xarray<double> a = {{0.5, 1.5}, {2.5, 3.5}};
        xarray<double> b = {{-0.2, 2.4}, {1.3, 4.7}};
        xarray<double> res;

        zarray za(a);
        zarray zb(b);
        zarray zres(res);

        auto f = za + xt::exp(zb);
        f.assign_to(zres.get_implementation());

        auto expected = xarray<double>::from_shape({2, 2});
        std::transform(a.cbegin(), a.cend(), b.cbegin(), expected.begin(),
                       [](const double& lhs, const double& rhs) { return lhs + std::exp(rhs); });

        EXPECT_TRUE(all(isclose(res, expected)));
    }

    TEST(zarray, assign)
    {
        using exp_dispatcher_type = zdispatcher_t<math::exp_fun, 1>;
        exp_dispatcher_type::init();

        using add_dispatcher_type = zdispatcher_t<detail::plus, 2>;
        add_dispatcher_type::init();

        xarray<double> a = {{0.5, 1.5}, {2.5, 3.5}};
        xarray<double> b = {{-0.2, 2.4}, {1.3, 4.7}};

        zarray za(a);
        zarray zb(b);

        zarray zres = za + xt::exp(zb);
        auto expected = xarray<double>::from_shape({2, 2});
        std::transform(a.cbegin(), a.cend(), b.cbegin(), expected.begin(),
                       [](const double& lhs, const double& rhs) { return lhs + std::exp(rhs); });

        const auto& res = zres.get_array<double>();
        EXPECT_TRUE(all(isclose(res, expected)));

        zarray zres2;
        zres2 = za + xt::exp(zb);

        const auto& res2 = zres2.get_array<double>();
        EXPECT_TRUE(all(isclose(res2, expected)));
    }

    TEST(zarray, chunked_array)
    {
        using shape_type = std::vector<size_t>;
        shape_type shape = {10, 10, 10};
        shape_type chunk_shape = {2, 3, 4};
        xchunked_array<xarray<xarray<double>>> a(shape, chunk_shape);

        zarray za(a);
        shape_type res = za.as_chunked_array().chunk_shape();
        EXPECT_EQ(res, chunk_shape);
    }

    TEST(zarray, from_zarray)
    {
        xarray<double> a = {{0.5, 1.5}, {2.5, 3.5}};

        zarray za(a);
        zarray zb(za);

        const auto& b = zb.get_array<double>();
        EXPECT_EQ(b, a);
    }

    TEST(zarray, assign_chunked)
    {
        std::vector<size_t> shape = {4, 4};
        std::vector<size_t> chunk_shape = {2, 2};
        std::string chunk_dir = "files";
        std::size_t pool_size = 2;
        using chunked_array = xchunked_array<xchunk_store_manager<xfile_array<double, xdisk_io_handler<xcsv_config>>>>;
        chunked_array a1(shape, chunk_shape, chunk_dir, pool_size);

        double v = 489.1;
        zarray z1(a1);
        z1(2, 2) = v;
        z1(1, 1) = 1;  // will flush previous chunk assignment

        using file_array = xfile_array<double, xdisk_io_handler<xcsv_config>>;
        file_array a2;
        a2.set_path("files/1.1");

        EXPECT_EQ(a2(0, 0), v);
    }
}
#endif

