/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef BENCHMARK_EQUAL_HPP
#define BENCHMARK_EQUAL_HPP

#include <benchmark/benchmark.h>

#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"

namespace xt
{
    template <class T>
    inline bool eq1(T& a, T& b)
    {
        return a.data().size() == b.data().size() && std::equal(a.data().begin(), a.data().end(), b.data().begin());
    }

    template <class T>
    inline bool eq2(T& a, T& b)
    {
        if(a.data().size() != b.data().size())
            return false;
        for(decltype(a.data().size()) k=0; k < a.data().size(); ++k)
            if(a.data()[k] != b.data()[k])
                return false;
        return true;

    }

    template <class T>
    inline void range_fill(T& a, std::size_t val)
    {
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            a[i] = val;
        }
    }

    void bench_compare_eq1(benchmark::State& state)
    {
        xarray<std::size_t> a = xt::arange(state.range(0));
        xarray<std::size_t> b = xt::arange(state.range(0));
        xarray<std::size_t> idxs = xt::random::randint({10000}, 0, state.range(0) - 1);
        std::size_t i = 0;

        for (auto _ : state)
        {
            xarray<std::size_t> bc = b;
            bc[idxs[i % idxs.size()]] = 0;
            ++i;

            bool result = eq1(a, b);
            benchmark::DoNotOptimize(result);
        }
    }

    void bench_compare_eq2(benchmark::State& state)
    {
        xarray<std::size_t> a = xt::arange(state.range(0));
        xarray<std::size_t> b = xt::arange(state.range(0));
        xarray<std::size_t> idxs = xt::random::randint({10000}, 0, state.range(0) - 1);
        std::size_t i = 0;
        for (auto _ : state)
        {
            xarray<std::size_t> bc = b;
            bc[idxs[i % idxs.size()]] = 0;
            ++i;

            bool result = eq2(a, b);
            benchmark::DoNotOptimize(result);
        }
    }

    void bench_compare_fill(benchmark::State& state)
    {
        std::vector<std::size_t> a(state.range(0));
        for (auto _ : state)
        {
            std::fill(a.begin(), a.end(), state.range(0));
            benchmark::DoNotOptimize(a.data());
        }
    }

    void bench_compare_fill2(benchmark::State& state)
    {
        std::vector<std::size_t> a(state.range(0));
        for (auto _ : state)
        {
            range_fill(a, state.range(0));
            benchmark::DoNotOptimize(a.data());
        }
    }

    // BENCHMARK(bench_compare_eq1)->Range(4, 1000);
    // BENCHMARK(bench_compare_eq2)->Range(4, 1000);
    BENCHMARK(bench_compare_fill)->Range(4, 1000);
    BENCHMARK(bench_compare_fill2)->Range(4, 1000);
}

#endif