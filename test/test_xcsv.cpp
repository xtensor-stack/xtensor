/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <sstream>

#include "xtensor/io/xcsv.hpp"
#include "xtensor/io/xio.hpp"

#include "test_common_macros.hpp"

namespace xt
{
    TEST(xcsv, load_1D)
    {
        const std::string source = "1, 2, 3, 4";

        std::stringstream source_stream(source);

        const xtensor<int, 2> res = load_csv<int>(source_stream);

        const xtensor<int, 2> exp{{1, 2, 3, 4}};

        ASSERT_TRUE(all(equal(res, exp)));
    }

    TEST(xcsv, load_double)
    {
        std::string source = "1.0, 2.0, 3.0, 4.0\n"
                             "10.0, 12.0, 15.0, 18.0";

        std::stringstream source_stream(source);

        xtensor<double, 2> res = load_csv<double>(source_stream);

        xtensor<double, 2> exp{{1.0, 2.0, 3.0, 4.0}, {10.0, 12.0, 15.0, 18.0}};

        ASSERT_TRUE(all(equal(res, exp)));
    }

    TEST(xcsv, load_binary_matrix)
    {
        const std::string source = "1,0,1\n"
                                   "0,1,1";

        std::stringstream source_stream(source);

        const xtensor<uint8_t, 2> res = load_csv<uint8_t>(source_stream);

        const xtensor<uint8_t, 2> exp{{1, 0, 1}, {0, 1, 1}};

        ASSERT_TRUE(all(equal(res, exp)));
    }

    TEST(xcsv, load_double_with_options)
    {
        std::string source = "A B C D\n"
                             "#0.0 1.0 1.1 1.2\n"
                             "1.0 2.0 3.0 4.0\n"
                             "10.0 12.0 15.0 18.0\n"
                             "9.0, 8.0, 7.0, 6.";

        std::stringstream source_stream(source);

        auto res = load_csv<double>(source_stream, ' ', 1, 2, "#");

        xtensor<double, 2> exp{{1.0, 2.0, 3.0, 4.0}, {10.0, 12.0, 15.0, 18.0}};

        ASSERT_TRUE(all(equal(res, exp)));
    }

    TEST(xcsv, load_string_trims_cells)
    {
        const std::string source = "  alpha  , beta,gamma   \n delta,  epsilon , zeta  ";

        std::stringstream source_stream(source);

        const xtensor<std::string, 2> res = load_csv<std::string>(source_stream);

        ASSERT_EQ(res.shape()[0], std::size_t(2));
        ASSERT_EQ(res.shape()[1], std::size_t(3));
        ASSERT_EQ(res(0, 0), "alpha");
        ASSERT_EQ(res(0, 1), "beta");
        ASSERT_EQ(res(0, 2), "gamma");
        ASSERT_EQ(res(1, 0), "delta");
        ASSERT_EQ(res(1, 1), "epsilon");
        ASSERT_EQ(res(1, 2), "zeta");
    }

    TEST(xcsv, load_file_uses_config)
    {
        const std::string source = "metadata\n"
                                   "//ignore this row\n"
                                   "1;2;3\n"
                                   "4;5;6\n"
                                   "7;8;9";

        std::stringstream source_stream(source);
        xcsv_config config;
        config.delimiter = ';';
        config.skip_rows = 1;
        config.max_rows = 2;
        config.comments = "//";

        xtensor<int, 2> res = {{0, 0, 0}};
        load_file(source_stream, res, config);

        const xtensor<int, 2> exp{{1, 2, 3}, {4, 5, 6}};

        ASSERT_TRUE(all(equal(res, exp)));
    }

    TEST(xcsv, load_inconsistent_rows_throws)
    {
        const std::string source = "1,2,3\n4,5";

        std::stringstream source_stream(source);

        XT_EXPECT_THROW(load_csv<int>(source_stream), std::runtime_error);
    }

    TEST(xcsv, dump_1D)
    {
        xtensor<double, 1> data{{1.0, 2.0, 3.0, 4.0}};

        std::stringstream res;

        dump_csv(res, data);
        ASSERT_EQ("1,2,3,4\n", res.str());
    }

    TEST(xcsv, dump_double)
    {
        xtensor<double, 2> data{{1.0, 2.0, 3.0, 4.0}, {10.0, 12.0, 15.0, 18.0}};

        std::stringstream res;

        dump_csv(res, data);
        ASSERT_EQ("1,2,3,4\n10,12,15,18\n", res.str());
    }

    TEST(xcsv, dump_file_matches_dump_csv)
    {
        xtensor<int, 2> data{{1, 2}, {3, 4}};
        std::stringstream res;
        xcsv_config config;

        dump_file(res, data, config);

        ASSERT_EQ("1,2\n3,4\n", res.str());
    }

    TEST(xcsv, dump_higher_dimension_throws)
    {
        xtensor<double, 3> data{{{1.0, 2.0}, {3.0, 4.0}}};
        std::stringstream res;

        XT_EXPECT_THROW(dump_csv(res, data), std::runtime_error);
    }
}
