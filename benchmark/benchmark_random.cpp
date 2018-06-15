/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef BENCHMARK_RANDOM_HPP
#define BENCHMARK_RANDOM_HPP

#include <benchmark/benchmark.h>

#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"

namespace xt
{
    namespace random_bench
    { 
        void random_assign_xtensor(benchmark::State& state)
        {
            for (auto _ : state)
            {
                xtensor<double, 2> result = xt::random::rand<double>({20, 20});
                benchmark::DoNotOptimize(result.data());
            }
        }

        void random_assign_forloop(benchmark::State& state)
        {
            for (auto _ : state)
            {
                xtensor<double, 2> result;
                result.resize({20, 20});
                std::uniform_real_distribution<double> dist(0, 1);
                auto& engine = xt::random::get_default_random_engine();
                for (auto& el : result.storage())
                {
                    el = dist(engine);
                }
                benchmark::DoNotOptimize(result.data());
            }
        }

        void random_assign_xarray(benchmark::State& state)
        {
            for (auto _ : state)
            {
                xarray<double> result = xt::random::rand<double>({20, 20});
                benchmark::DoNotOptimize(result.data());
            }
        }

        BENCHMARK(random_assign_xarray);
        BENCHMARK(random_assign_xtensor);
        BENCHMARK(random_assign_forloop);
    }
}

#endif
