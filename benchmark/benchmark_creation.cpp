/****************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *                                                                                                               *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *                                                                                                                *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <benchmark/benchmark.h>

#include "xtensor/xbuilder.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"

namespace xt
{
    void benchmark_empty(benchmark::State& state)
    {
        for (auto _ : state)
        {
            auto e = xt::empty<double>({5, 5});
        }
    }

    template <class T>
    void benchmark_from_shape(benchmark::State& state)
    {
        for (auto _ : state)
        {
            T e = T::from_shape({5, 5});
        }
    }

    template <class T>
    void benchmark_creation(benchmark::State& state)
    {
        for (auto _ : state)
        {
            T e(typename T::shape_type({5, 5}));
        }
    }

    void benchmark_empty_to_xtensor(benchmark::State& state)
    {
        for (auto _ : state)
        {
            xtensor<double, 2> e = xt::empty<double>({5, 5});
        }
    }

    void benchmark_empty_to_xarray(benchmark::State& state)
    {
        for (auto _ : state)
        {
            xarray<double> e = xt::empty<double>({5, 5});
        }
    }

    BENCHMARK(benchmark_empty);
    BENCHMARK(benchmark_empty_to_xtensor);
    BENCHMARK(benchmark_empty_to_xarray);
    BENCHMARK_TEMPLATE(benchmark_from_shape, xarray<double>);
    BENCHMARK_TEMPLATE(benchmark_from_shape, xtensor<double, 2>);
    BENCHMARK_TEMPLATE(benchmark_creation, xarray<double>);
    BENCHMARK_TEMPLATE(benchmark_creation, xtensor<double, 2>);
}