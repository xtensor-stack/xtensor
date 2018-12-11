/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <benchmark/benchmark.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xreducer.hpp"

namespace xt
{
    namespace reducer
    {
        template <class E, class X>
        void reducer_reducer(benchmark::State& state, const E& x, E& res, const X& axes)
        {
            for (auto _ : state)
            {
                res = sum(x, axes);
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class E, class X>
        void reducer_immediate_reducer(benchmark::State& state, const E& x, E& res, const X& axes)
        {
            for (auto _ : state)
            {
                res = sum(x, axes, evaluation_strategy::immediate);
                benchmark::DoNotOptimize(res.data());
            }
        }

        xarray<double> u = ones<double>({ 10, 100000 });
        xarray<double> v = ones<double>({ 100000, 10 });
        xarray<double> res2 = ones<double>({ 1 });

        std::vector<std::size_t> axis0 = { 0 };
        std::vector<std::size_t> axis1 = { 1 };
        std::vector<std::size_t> axis_both = { 0, 1 };

        static auto res0 = xarray<double>::from_shape({ 100000 });
        static auto res1 = xarray<double>::from_shape({ 10 });

        BENCHMARK_CAPTURE(reducer_reducer, 10x100000/axis 0, u, res0, axis0);
        BENCHMARK_CAPTURE(reducer_reducer, 10x100000/axis 1, u, res1, axis1);
        BENCHMARK_CAPTURE(reducer_reducer, 100000x10/axis 1, v, res1, axis0);
        BENCHMARK_CAPTURE(reducer_reducer, 100000x10/axis 0, v, res0, axis1);
        BENCHMARK_CAPTURE(reducer_reducer, 100000x10/axis both, v, res2, axis_both);

        BENCHMARK_CAPTURE(reducer_immediate_reducer, 10x100000/axis 0, u, res0, axis0);
        BENCHMARK_CAPTURE(reducer_immediate_reducer, 10x100000/axis 1, u, res1, axis1);
        BENCHMARK_CAPTURE(reducer_immediate_reducer, 100000x10/axis 1, v, res1, axis0);
        BENCHMARK_CAPTURE(reducer_immediate_reducer, 100000x10/axis 0, v, res0, axis1);
        BENCHMARK_CAPTURE(reducer_immediate_reducer, 100000x10/axis both, v, res2, axis_both);

        template <class E, class X>
        inline auto reducer_manual_strided_reducer(benchmark::State& state, const E& x, E& res, const X& axes)
        {
            using value_type = typename E::value_type;
            std::size_t stride = x.strides()[axes[0]];
            std::size_t offset_end = x.strides()[axes[0]] * x.shape()[axes[0]];
            std::size_t offset_iter = 0;
            if (axes[0] == 1)
            {
                offset_iter = x.strides()[0];
            }
            else if (axes[0] == 0)
            {
                offset_iter = x.strides()[1];
            }

            for (auto _ : state)
            {
                for (std::size_t j = 0; j < res.shape()[0]; ++j)
                {
                    auto begin = x.data() + (offset_iter * j);
                    auto end = begin + offset_end;
                    value_type temp = *begin;
                    begin += stride;
                    for (; begin < end; begin += stride)
                    {
                        temp += *begin;
                    }
                    res(j) = temp;
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        BENCHMARK_CAPTURE(reducer_manual_strided_reducer, 10x100000/axis 0, u, res0, axis0);
        BENCHMARK_CAPTURE(reducer_manual_strided_reducer, 10x100000/axis 1, u, res1, axis1);
        BENCHMARK_CAPTURE(reducer_manual_strided_reducer, 100000x10/axis 1, v, res1, axis0);
        BENCHMARK_CAPTURE(reducer_manual_strided_reducer, 100000x10/axis 0, v, res0, axis1);
    }
}
