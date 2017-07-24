/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef BENCHMARK_ASSIGN_HPP
#define BENCHMARK_ASSIGN_HPP

#include <chrono>
#include <cstddef>
#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"

namespace xt
{
    namespace assign
    {
        using duration_type = std::chrono::duration<double, std::milli>;

        /****************************
         * Benchmark initialization *
         ****************************/

        template <class V>
        inline void init_benchmark_data(V& lhs, V& rhs, std::size_t size0, std::size_t size1)
        {
            using T = typename V::value_type;
            for (std::size_t i = 0; i < size0; ++i)
            {
                for (std::size_t j = 0; j < size1; ++j)
                {
                    lhs(i, j) = T(0.5) * T(j) / T(j + 1) + std::sqrt(T(i)) * T(9.) / T(size1);
                    rhs(i, j) = T(10.2) / T(i + 2) + T(0.25) * T(j);
                }
            }
        }

        template <class V>
        inline void init_xtensor_benchmark(V& lhs, V& rhs, V& res,
                                           std::size_t size0, size_t size1)
        {
            lhs.reshape({ size0, size1 });
            rhs.reshape({ size0, size1 });
            res.reshape({ size0, size1 });
            init_benchmark_data(lhs, rhs, size0, size1);
        }

        template <class V>
        inline void init_dl_xtensor_benchmark(V& lhs, V& rhs, V& res,
                                              std::size_t size0, size_t size1)
        {
            using strides_type = typename V::strides_type;
            strides_type str = { size1, 1 };
            lhs.reshape({ size0, size1 }, str);
            rhs.reshape({ size0, size1 }, str);
            res.reshape({ size0, size1 }, str);
            init_benchmark_data(lhs, rhs, size0, size1);
        }

        template <class E>
        inline auto benchmark_c_loop(const E& x, const E& y, E& res, std::size_t count)
        {
            duration_type t_res = duration_type::max();
            using size_type = typename E::size_type;
            for (std::size_t i = 0; i < count; ++i)
            {
                auto start = std::chrono::steady_clock::now();
                size_type csize = x.size();
                for (size_type i = 0; i < csize; ++i)
                {
                    res.data()[i] = 3 * x.data()[i] - 2 * y.data()[i];
                }
                auto end = std::chrono::steady_clock::now();
                auto tmp = end - start;
                t_res = tmp < t_res ? tmp : t_res;
            }
            return t_res;
        }

        template <class E>
        inline auto benchmark_xtensor(const E& x, const E& y, E& res, std::size_t count)
        {
            duration_type t_res = duration_type::max();
            for (std::size_t i = 0; i < count; ++i)
            {
                auto start = std::chrono::steady_clock::now();
                xt::noalias(res) = 3 * x - 2 * y;
                auto end = std::chrono::steady_clock::now();
                auto tmp = end - start;
                t_res = tmp < t_res ? tmp : t_res;
            }
            return t_res;
        }

        template <class OS>
        inline void benchmark(OS& out)
        {
            std::size_t count = 10;
            std::size_t size0 = 20, size1 = 500;
            xtensor<double, 2> lhs_r, rhs_r, res_r;
            init_xtensor_benchmark(lhs_r, rhs_r, res_r, size0, size1);

            xtensor<double, 2, layout_type::dynamic> lhs_d, rhs_d, res_d;
            init_dl_xtensor_benchmark(lhs_d, rhs_d, res_d, size0, size1);

            duration_type c_loop_assign = benchmark_c_loop(lhs_d, rhs_d, res_d, count);
            duration_type iterator_assign = benchmark_xtensor(lhs_d, rhs_d, res_d, count);
            duration_type index_assign = benchmark_xtensor(lhs_r, rhs_r, res_r, count);

            out << "********************" << std::endl;
            out << "* ASSIGN BENCHMARK *" << std::endl;
            out << "********************" << std::endl;

            out << "c loop assign  : " << c_loop_assign.count() << "ms" << std::endl;
            out << "iterator assign: " << iterator_assign.count() << "ms" << std::endl;
            out << "index assign   : " << index_assign.count() << "ms" << std::endl;
            out << std::endl;
        }
    }
}

#endif
