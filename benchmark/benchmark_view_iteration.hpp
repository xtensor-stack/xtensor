/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef BENCHMARK_VIEW_ITERATION_HPP
#define BENCHMARK_VIEW_ITERATION_HPP

#include <benchmark/benchmark.h>

#include <cstddef>
#include <chrono>
#include <string>

#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xstrided_view.hpp"

namespace xt
{
    // Thanks to Ullrich Koethe for these benchmarks
    // https://github.com/QuantStack/xtensor/issues/695
    namespace view_benchmarks
    {
        constexpr int SIZE = 1000;

        template <class V>
        void dynamic_iterator(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::dynamic_view(data, xt::slice_vector{xt::all(), SIZE/2});
            for (auto _ : state)
            {
                std::copy(v.begin(), v.end(), res.begin());
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void iterator(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::view(data, xt::all(), SIZE/2);
            for (auto _ : state)
            {
                std::copy(v.begin(), v.end(), res.begin());
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void loop(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::dynamic_view(data, xt::slice_vector{xt::all(), SIZE/2});
            for (auto _ : state)
            {
                for(std::size_t k = 0; k < v.shape()[0]; ++k)
                {
                    res(k) = v(k);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void loop_view(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::view(data, xt::all(), SIZE / 2);
            for (auto _ : state)
            {
                for(std::size_t k = 0; k < v.shape()[0]; ++k)
                {
                    res(k) = v(k);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void loop_raw(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            for (auto _ : state)
            {
                std::size_t j = SIZE / 2;
                for(std::size_t k = 0; k < SIZE; ++k)
                {
                    res(k) = data(k, j);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void assign(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::dynamic_view(data, xt::slice_vector{xt::all(), SIZE/2});
            for (auto _ : state)
            {
                xt::noalias(res) = v;
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void assign_view(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::view(data, xt::all(), SIZE/2);
            auto r = xt::view(res, xt::all());
            for (auto _ : state)
            {
                r = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void assign_dynamic_view(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::dynamic_view(data, xt::slice_vector{xt::all(), SIZE/2});
            auto r = xt::dynamic_view(res, xt::slice_vector{xt::all()});

            for (auto _ : state)
            {
                r = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void assign_view_noalias(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::view(data, xt::all(), SIZE/2);
            auto r = xt::view(res, xt::all());
            for (auto _ : state)
            {
                xt::noalias(r) = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void assign_dynamic_view_noalias(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE,SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::dynamic_view(data, xt::slice_vector{xt::all(), SIZE/2});
            auto r = xt::dynamic_view(res, xt::slice_vector{xt::all()});

            for (auto _ : state)
            {
                xt::noalias(r) = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        BENCHMARK_TEMPLATE(dynamic_iterator, float);
        BENCHMARK_TEMPLATE(iterator, float);
        BENCHMARK_TEMPLATE(loop, float);
        BENCHMARK_TEMPLATE(loop_view, float);
        BENCHMARK_TEMPLATE(loop_raw, float);
        BENCHMARK_TEMPLATE(assign, float);
        BENCHMARK_TEMPLATE(assign_view, float);
        BENCHMARK_TEMPLATE(assign_dynamic_view, float);
        BENCHMARK_TEMPLATE(assign_view_noalias, float);
        BENCHMARK_TEMPLATE(assign_dynamic_view_noalias, float);
    }
}

#endif