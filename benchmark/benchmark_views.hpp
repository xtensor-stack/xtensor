/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef BENCHMARK_VIEWS_HPP
#define BENCHMARK_VIEWS_HPP

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
    namespace reducer
    {
        template <class E, class X>
        void benchmark_reducer(benchmark::State& state, const E& x, E& res, const X& axes)
        {
            while (state.KeepRunning())
            {
                res = sum(x, axes);
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class E, class X>
        void benchmark_immediate_reducer(benchmark::State& state, const E& x, E& res, const X& axes)
        {
            while (state.KeepRunning())
            {
                res = sum(x, axes, evaluation_strategy::immediate());
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

        BENCHMARK_CAPTURE(benchmark_reducer, 10x100000/axis 0, u, res0, axis0);
        BENCHMARK_CAPTURE(benchmark_reducer, 10x100000/axis 1, u, res1, axis1);
        BENCHMARK_CAPTURE(benchmark_reducer, 100000x10/axis 1, v, res1, axis0);
        BENCHMARK_CAPTURE(benchmark_reducer, 100000x10/axis 0, v, res0, axis1);
        BENCHMARK_CAPTURE(benchmark_reducer, 100000x10/axis both, v, res2, axis_both);

        BENCHMARK_CAPTURE(benchmark_immediate_reducer, 10x100000/axis 0, u, res0, axis0);
        BENCHMARK_CAPTURE(benchmark_immediate_reducer, 10x100000/axis 1, u, res1, axis1);
        BENCHMARK_CAPTURE(benchmark_immediate_reducer, 100000x10/axis 1, v, res1, axis0);
        BENCHMARK_CAPTURE(benchmark_immediate_reducer, 100000x10/axis 0, v, res0, axis1);
        BENCHMARK_CAPTURE(benchmark_immediate_reducer, 100000x10/axis both, v, res2, axis_both);

        template <class E, class X>
        inline auto benchmark_strided_reducer(benchmark::State& state, const E& x, E& res, const X& axes)
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

            while (state.KeepRunning())
            {
                for (std::size_t j = 0; j < res.shape()[0]; ++j)
                {
                    auto begin = x.raw_data() + (offset_iter * j);
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

        BENCHMARK_CAPTURE(benchmark_strided_reducer, 10x100000/axis 0, u, res0, axis0);
        BENCHMARK_CAPTURE(benchmark_strided_reducer, 10x100000/axis 1, u, res1, axis1);
        BENCHMARK_CAPTURE(benchmark_strided_reducer, 100000x10/axis 1, v, res1, axis0);
        BENCHMARK_CAPTURE(benchmark_strided_reducer, 100000x10/axis 0, v, res0, axis1);
    }

    namespace stridedview
    {

        template <layout_type L1, layout_type L2>
        inline auto benchmark_stridedview(benchmark::State& state, std::vector<std::size_t> shape)
        {
            xarray<double, L1> x = xt::arange<double>(compute_size(shape));
            x.resize(shape);

            xarray<double, L2> res;
            res.resize(std::vector<std::size_t>(shape.rbegin(), shape.rend()));

            while (state.KeepRunning())
            {
                res = transpose(x);
            }
        }

        auto benchmark_stridedview_rm_rm = benchmark_stridedview<layout_type::row_major, layout_type::row_major>;
        auto benchmark_stridedview_cm_cm = benchmark_stridedview<layout_type::column_major, layout_type::column_major>;
        auto benchmark_stridedview_rm_cm = benchmark_stridedview<layout_type::row_major, layout_type::column_major>;
        auto benchmark_stridedview_cm_rm = benchmark_stridedview<layout_type::column_major, layout_type::row_major>;

        BENCHMARK_CAPTURE(benchmark_stridedview_rm_rm, 10x20x500, {10, 20, 500});
        BENCHMARK_CAPTURE(benchmark_stridedview_cm_cm, 10x20x500, {10, 20, 500});
        BENCHMARK_CAPTURE(benchmark_stridedview_rm_cm, 10x20x500, {10, 20, 500});
        BENCHMARK_CAPTURE(benchmark_stridedview_cm_rm, 10x20x500, {10, 20, 500});
    }
}

#endif
