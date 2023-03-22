/***************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <chrono>
#include <cstddef>
#include <string>

#include <benchmark/benchmark.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xstrides.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

namespace xt
{
    // Thanks to Ullrich Koethe for these benchmarks
    // https://github.com/xtensor-stack/xtensor/issues/695
    namespace view_benchmarks
    {
        constexpr int SIZE = 1000;

        template <class V>
        void view_dynamic_iterator(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::strided_view(data, xt::xstrided_slice_vector{xt::all(), SIZE / 2});
            for (auto _ : state)
            {
                std::copy(v.begin(), v.end(), res.begin());
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void view_iterator(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::view(data, xt::all(), SIZE / 2);
            for (auto _ : state)
            {
                std::copy(v.begin(), v.end(), res.begin());
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void view_loop(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::strided_view(data, xt::xstrided_slice_vector{xt::all(), SIZE / 2});
            for (auto _ : state)
            {
                for (std::size_t k = 0; k < v.shape()[0]; ++k)
                {
                    res(k) = v(k);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void view_loop_view(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::view(data, xt::all(), SIZE / 2);
            for (auto _ : state)
            {
                for (std::size_t k = 0; k < v.shape()[0]; ++k)
                {
                    res(k) = v(k);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void view_loop_raw(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            for (auto _ : state)
            {
                std::size_t j = SIZE / 2;
                for (std::size_t k = 0; k < SIZE; ++k)
                {
                    res(k) = data(k, j);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void view_assign(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::strided_view(data, xt::xstrided_slice_vector{xt::all(), SIZE / 2});
            for (auto _ : state)
            {
                xt::noalias(res) = v;
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class V>
        void view_assign_view(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::view(data, xt::all(), SIZE / 2);
            auto r = xt::view(res, xt::all());
            for (auto _ : state)
            {
                r = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void view_assign_strided_view(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::strided_view(data, xt::xstrided_slice_vector{xt::all(), SIZE / 2});
            auto r = xt::strided_view(res, xt::xstrided_slice_vector{xt::all()});

            for (auto _ : state)
            {
                r = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void view_assign_view_noalias(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::view(data, xt::all(), SIZE / 2);
            auto r = xt::view(res, xt::all());
            for (auto _ : state)
            {
                xt::noalias(r) = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        template <class V>
        void view_assign_strided_view_noalias(benchmark::State& state)
        {
            xt::xtensor<V, 2> data = xt::ones<V>({SIZE, SIZE});
            xt::xtensor<V, 1> res = xt::ones<V>({SIZE});

            auto v = xt::strided_view(data, xt::xstrided_slice_vector{xt::all(), SIZE / 2});
            auto r = xt::strided_view(res, xt::xstrided_slice_vector{xt::all()});

            for (auto _ : state)
            {
                xt::noalias(r) = v;
                benchmark::DoNotOptimize(r.data());
            }
        }

        BENCHMARK_TEMPLATE(view_dynamic_iterator, float);
        BENCHMARK_TEMPLATE(view_iterator, float);
        BENCHMARK_TEMPLATE(view_loop, float);
        BENCHMARK_TEMPLATE(view_loop_view, float);
        BENCHMARK_TEMPLATE(view_loop_raw, float);
        BENCHMARK_TEMPLATE(view_assign, float);
        BENCHMARK_TEMPLATE(view_assign_view, float);
        BENCHMARK_TEMPLATE(view_assign_strided_view, float);
        BENCHMARK_TEMPLATE(view_assign_view_noalias, float);
        BENCHMARK_TEMPLATE(view_assign_strided_view_noalias, float);
    }

    namespace finite_diff
    {
        inline auto stencil_threedirections(benchmark::State& state, size_t size)
        {
            for (auto _ : state)
            {
                const std::array<size_t, 3> shape = {size, size, size};
                xt::xtensor<double, 3> a(shape), b(shape);
                auto core = xt::range(1, size - 1);
                xt::noalias(xt::view(b, core, core, core)
                ) = 1.0 / 7.0
                    * (xt::view(a, core, core, core) + xt::view(a, core, core, xt::range(2, size))
                       + xt::view(a, core, core, xt::range(0, size - 2))
                       + xt::view(a, core, xt::range(2, size), core)
                       + xt::view(a, core, xt::range(0, size - 2), core)
                       + xt::view(a, xt::range(2, size), core, core)
                       + xt::view(a, xt::range(0, size - 2), core, core));
                benchmark::DoNotOptimize(b);
            }
        }

        inline auto stencil_twodirections(benchmark::State& state, size_t size)
        {
            for (auto _ : state)
            {
                const std::array<size_t, 3> shape = {size, size, size};
                xt::xtensor<double, 3> a(shape), b(shape);
                auto core = xt::range(1, size - 1);
                xt::noalias(xt::view(b, core, core, core)
                ) = 1.0 / 7.0
                    * (xt::view(a, core, core, core) + xt::view(a, core, xt::range(2, size), core)
                       + xt::view(a, core, xt::range(0, size - 2), core)
                       + xt::view(a, xt::range(2, size), core, core)
                       + xt::view(a, xt::range(0, size - 2), core, core));
                benchmark::DoNotOptimize(b);
            }
        }

        inline auto stencil_onedirection(benchmark::State& state, size_t size)
        {
            for (auto _ : state)
            {
                const std::array<size_t, 3> shape = {size, size, size};
                xt::xtensor<double, 3> a(shape), b(shape);
                auto core = xt::range(1, size - 1);
                xt::noalias(xt::view(b, core, core, core)
                ) = 1.0 / 2.0
                    * (xt::view(a, xt::range(2, size), core, core)
                       - xt::view(a, xt::range(0, size - 2), core, core));
                benchmark::DoNotOptimize(b);
            }
        }

        BENCHMARK_CAPTURE(stencil_threedirections, stencil_threedirections_50, 50);
        BENCHMARK_CAPTURE(stencil_threedirections, stencil_threedirections_100, 100);
        BENCHMARK_CAPTURE(stencil_threedirections, stencil_threedirections_200, 200);
        BENCHMARK_CAPTURE(stencil_threedirections, stencil_threedirections_300, 300);
        BENCHMARK_CAPTURE(stencil_threedirections, stencil_threedirections_500, 500);
        BENCHMARK_CAPTURE(stencil_twodirections, stencil_twodirections_50, 50);
        BENCHMARK_CAPTURE(stencil_twodirections, stencil_twodirections_100, 100);
        BENCHMARK_CAPTURE(stencil_twodirections, stencil_twodirections_200, 200);
        BENCHMARK_CAPTURE(stencil_twodirections, stencil_twodirections_300, 300);
        BENCHMARK_CAPTURE(stencil_twodirections, stencil_twodirections_500, 500);
        BENCHMARK_CAPTURE(stencil_onedirection, stencil_onedirections_50, 50);
        BENCHMARK_CAPTURE(stencil_onedirection, stencil_onedirections_100, 100);
        BENCHMARK_CAPTURE(stencil_onedirection, stencil_onedirections_200, 200);
        BENCHMARK_CAPTURE(stencil_onedirection, stencil_onedirections_300, 300);
        BENCHMARK_CAPTURE(stencil_onedirection, stencil_onedirections_500, 500);
    }

    namespace stridedview
    {

        template <layout_type L1, layout_type L2>
        inline auto transpose_assign(benchmark::State& state, std::vector<std::size_t> shape)
        {
            xarray<double, L1> x = xt::arange<double>(compute_size(shape));
            x.resize(shape);

            xarray<double, L2> res;
            res.resize(std::vector<std::size_t>(shape.rbegin(), shape.rend()));

            for (auto _ : state)
            {
                res = transpose(x);
            }
        }

        auto transpose_assign_rm_rm = transpose_assign<layout_type::row_major, layout_type::row_major>;
        auto transpose_assign_cm_cm = transpose_assign<layout_type::column_major, layout_type::column_major>;
        auto transpose_assign_rm_cm = transpose_assign<layout_type::row_major, layout_type::column_major>;
        auto transpose_assign_cm_rm = transpose_assign<layout_type::column_major, layout_type::row_major>;

        BENCHMARK_CAPTURE(transpose_assign_rm_rm, 10x20x500, {10, 20, 500});
        BENCHMARK_CAPTURE(transpose_assign_cm_cm, 10x20x500, {10, 20, 500});
        BENCHMARK_CAPTURE(transpose_assign_rm_cm, 10x20x500, {10, 20, 500});
        BENCHMARK_CAPTURE(transpose_assign_cm_rm, 10x20x500, {10, 20, 500});
    }
}
