/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/


#ifndef BENCHMARK_SHAPE_HPP
#define BENCHMARK_SHAPE_HPP

#include <benchmark/benchmark.h>

#include "xtensor/xshape.hpp"
#include "xtensor/xstorage.hpp"


namespace xt
{
    namespace benchmark_xshape
    {
        template <class T>
        void xshape_initializer(benchmark::State& state)
        {
            for (auto _ : state)
            {
                T sv({2, 3, 1});
                benchmark::DoNotOptimize(sv.data());
            }
        }

        template <class T>
        void xshape_initializer_long(benchmark::State& state)
        {
            for (auto _ : state)
            {
                T sv({2, 3, 1, 2, 6, 1, 2, 3, 45, 6, 12, 3, 5, 45, 5, 6});
                benchmark::DoNotOptimize(sv.data());
            }
        }

        template <class T>
        void xshape_access(benchmark::State& state)
        {
            T a({3,2,1,3});
            for (auto _ : state)
            {
                a[0] = a[1] * a[2] + a[3];
                a[3] = a[1];
                a[1] = a[2] + a[3];
                a[2] = a[3];
                benchmark::DoNotOptimize(a.data());
            }
        }

        BENCHMARK_TEMPLATE(xshape_initializer, std::vector<std::size_t>);
        BENCHMARK_TEMPLATE(xshape_initializer, xt::svector<std::size_t, 4>);
        BENCHMARK_TEMPLATE(xshape_initializer, std::array<std::size_t, 4>);
        BENCHMARK_TEMPLATE(xshape_initializer_long, xt::svector<std::size_t, 4>);
        BENCHMARK_TEMPLATE(xshape_initializer_long, xt::uvector<std::size_t>);
        BENCHMARK_TEMPLATE(xshape_initializer_long, std::vector<std::size_t>);
        BENCHMARK_TEMPLATE(xshape_access, xt::uvector<double>);
        BENCHMARK_TEMPLATE(xshape_access, std::vector<double>);
        BENCHMARK_TEMPLATE(xshape_access, xt::svector<std::size_t, 4>);
        BENCHMARK_TEMPLATE(xshape_access, std::array<std::size_t, 4>);
    }
}

#endif