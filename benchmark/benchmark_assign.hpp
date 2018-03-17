/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef BENCHMARK_ASSIGN_HPP
#define BENCHMARK_ASSIGN_HPP

#include <benchmark/benchmark.h>

#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"

namespace xt
{
    namespace assign
    {

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
            lhs.resize({ size0, size1 });
            rhs.resize({ size0, size1 });
            res.resize({ size0, size1 });
            init_benchmark_data(lhs, rhs, size0, size1);
        }

        template <class V>
        inline void init_dl_xtensor_benchmark(V& lhs, V& rhs, V& res,
                                              std::size_t size0, size_t size1)
        {
            using strides_type = typename V::strides_type;
            strides_type str = { size1, 1 };
            lhs.resize({ size0, size1 }, str);
            rhs.resize({ size0, size1 }, str);
            res.resize({ size0, size1 }, str);
            init_benchmark_data(lhs, rhs, size0, size1);
        }

        template <class E>
        inline auto benchmark_c_assign(benchmark::State& state)
        {
            using size_type = typename E::size_type;

            E x, y, res;
            init_xtensor_benchmark(x, y, res, state.range(0), state.range(0));

            while (state.KeepRunning())
            {
                size_type csize = x.size();
                for (size_type i = 0; i < csize; ++i)
                {
                    res.data()[i] = 3.0 * x.data()[i] - 2.0 * y.data()[i];
                }
                benchmark::DoNotOptimize(res.data().data());
            }
        }

        template <class E>
        inline auto benchmark_assign(benchmark::State& state)
        {
            E x, y, res;
            init_xtensor_benchmark(x, y, res, state.range(0), state.range(0));
            while (state.KeepRunning())
            {
                xt::noalias(res) = 3.0 * x - 2.0 * y;
            }
            benchmark::DoNotOptimize(res.data().data());
        }

        BENCHMARK_TEMPLATE(benchmark_c_assign, xt::xtensor<double, 2>)->Range(32, 32<<3);
        BENCHMARK_TEMPLATE(benchmark_assign, xt::xarray<double>)->Range(32, 32<<3);
        BENCHMARK_TEMPLATE(benchmark_assign, xt::xtensor<double, 2>)->Range(32, 32<<3);
        BENCHMARK_TEMPLATE(benchmark_assign, xt::xarray<double, layout_type::dynamic>)->Range(32, 32<<3);
        BENCHMARK_TEMPLATE(benchmark_assign, xt::xtensor<double, 2, layout_type::dynamic>)->Range(32, 32<<3);
    }
}

#endif
