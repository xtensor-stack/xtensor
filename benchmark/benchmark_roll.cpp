/***************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <benchmark/benchmark.h>

#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xmath.hpp"
#include "xtensor/generators/xrandom.hpp"
#include "xtensor/misc/xmanipulation.hpp"

namespace xt
{
    namespace roll_bench
    {
        namespace detail
        {
            /*********************************************
             * Correctness verification helper
             *********************************************/

            template <class T>
            bool verify_correctness(const T& multi_result, const T& sequential_result)
            {
                return xt::allclose(multi_result, sequential_result);
            }

            /*********************************************
             * 2D roll benchmarks (2 axes)
             * shift = size * ratio for each axis
             *********************************************/

            inline void roll_2d_sequential(benchmark::State& state, std::size_t h, std::size_t w, double ratio)
            {
                xt::xtensor<double, 2> input = xt::random::rand<double>({h, w});
                auto shift_h = static_cast<std::ptrdiff_t>(h * ratio);
                auto shift_w = static_cast<std::ptrdiff_t>(w * ratio);

                for (auto _ : state)
                {
                    xt::xtensor<double, 2> temp = xt::roll(input, shift_h, 0);
                    xt::xtensor<double, 2> result = xt::roll(temp, shift_w, 1);
                    benchmark::DoNotOptimize(result.data());
                    benchmark::ClobberMemory();
                }

                state.SetItemsProcessed(state.iterations() * h * w);
                state.SetBytesProcessed(state.iterations() * h * w * sizeof(double));
            }

            inline void roll_2d_multi(benchmark::State& state, std::size_t h, std::size_t w, double ratio)
            {
                xt::xtensor<double, 2> input = xt::random::rand<double>({h, w});
                auto shift_h = static_cast<std::ptrdiff_t>(h * ratio);
                auto shift_w = static_cast<std::ptrdiff_t>(w * ratio);

                // Verify correctness once
                {
                    xt::xtensor<double, 2> temp = xt::roll(input, shift_h, 0);
                    xt::xtensor<double, 2> sequential = xt::roll(temp, shift_w, 1);
                    auto multi = xt::roll(input, {shift_h, shift_w}, {0, 1});
                    if (!verify_correctness(multi, sequential))
                    {
                        state.SkipWithError("Correctness check failed!");
                        return;
                    }
                }

                for (auto _ : state)
                {
                    auto result = xt::roll(input, {shift_h, shift_w}, {0, 1});
                    benchmark::DoNotOptimize(result.data());
                    benchmark::ClobberMemory();
                }

                state.SetItemsProcessed(state.iterations() * h * w);
                state.SetBytesProcessed(state.iterations() * h * w * sizeof(double));
            }

            /*********************************************
             * 3D roll benchmarks (2 axes - image spatial roll)
             * shift = size * ratio for H and W axes
             *********************************************/

            inline void roll_3d_2axes_sequential(benchmark::State& state, std::size_t h, std::size_t w, std::size_t c, double ratio)
            {
                xt::xtensor<double, 3> input = xt::random::rand<double>({h, w, c});
                auto shift_h = static_cast<std::ptrdiff_t>(h * ratio);
                auto shift_w = static_cast<std::ptrdiff_t>(w * ratio);

                for (auto _ : state)
                {
                    xt::xtensor<double, 3> temp = xt::roll(input, shift_h, 0);
                    xt::xtensor<double, 3> result = xt::roll(temp, shift_w, 1);
                    benchmark::DoNotOptimize(result.data());
                    benchmark::ClobberMemory();
                }

                state.SetItemsProcessed(state.iterations() * h * w * c);
                state.SetBytesProcessed(state.iterations() * h * w * c * sizeof(double));
            }

            inline void roll_3d_2axes_multi(benchmark::State& state, std::size_t h, std::size_t w, std::size_t c, double ratio)
            {
                xt::xtensor<double, 3> input = xt::random::rand<double>({h, w, c});
                auto shift_h = static_cast<std::ptrdiff_t>(h * ratio);
                auto shift_w = static_cast<std::ptrdiff_t>(w * ratio);

                // Verify correctness
                {
                    xt::xtensor<double, 3> temp = xt::roll(input, shift_h, 0);
                    xt::xtensor<double, 3> sequential = xt::roll(temp, shift_w, 1);
                    auto multi = xt::roll(input, {shift_h, shift_w}, {0, 1});
                    if (!verify_correctness(multi, sequential))
                    {
                        state.SkipWithError("Correctness check failed!");
                        return;
                    }
                }

                for (auto _ : state)
                {
                    auto result = xt::roll(input, {shift_h, shift_w}, {0, 1});
                    benchmark::DoNotOptimize(result.data());
                    benchmark::ClobberMemory();
                }

                state.SetItemsProcessed(state.iterations() * h * w * c);
                state.SetBytesProcessed(state.iterations() * h * w * c * sizeof(double));
            }

            /*********************************************
             * 3D roll benchmarks (3 axes - cube roll)
             * shift = size * ratio for all axes
             *********************************************/

            inline void roll_3d_3axes_sequential(benchmark::State& state, std::size_t size, double ratio)
            {
                xt::xtensor<double, 3> input = xt::random::rand<double>({size, size, size});
                auto shift = static_cast<std::ptrdiff_t>(size * ratio);

                for (auto _ : state)
                {
                    xt::xtensor<double, 3> temp1 = xt::roll(input, shift, 0);
                    xt::xtensor<double, 3> temp2 = xt::roll(temp1, shift, 1);
                    xt::xtensor<double, 3> result = xt::roll(temp2, shift, 2);
                    benchmark::DoNotOptimize(result.data());
                    benchmark::ClobberMemory();
                }

                auto total = size * size * size;
                state.SetItemsProcessed(state.iterations() * total);
                state.SetBytesProcessed(state.iterations() * total * sizeof(double));
            }

            inline void roll_3d_3axes_multi(benchmark::State& state, std::size_t size, double ratio)
            {
                xt::xtensor<double, 3> input = xt::random::rand<double>({size, size, size});
                auto shift = static_cast<std::ptrdiff_t>(size * ratio);

                // Verify correctness
                {
                    xt::xtensor<double, 3> temp1 = xt::roll(input, shift, 0);
                    xt::xtensor<double, 3> temp2 = xt::roll(temp1, shift, 1);
                    xt::xtensor<double, 3> sequential = xt::roll(temp2, shift, 2);
                    auto multi = xt::roll(input, {shift, shift, shift}, {0, 1, 2});
                    if (!verify_correctness(multi, sequential))
                    {
                        state.SkipWithError("Correctness check failed!");
                        return;
                    }
                }

                for (auto _ : state)
                {
                    auto result = xt::roll(input, {shift, shift, shift}, {0, 1, 2});
                    benchmark::DoNotOptimize(result.data());
                    benchmark::ClobberMemory();
                }

                auto total = size * size * size;
                state.SetItemsProcessed(state.iterations() * total);
                state.SetBytesProcessed(state.iterations() * total * sizeof(double));
            }

            /*********************************************
             * 4D roll benchmarks (4 axes)
             * shift = size * ratio for all axes
             *********************************************/

            inline void roll_4d_4axes_sequential(benchmark::State& state, std::size_t size, double ratio)
            {
                xt::xtensor<double, 4> input = xt::random::rand<double>({size, size, size, size});
                auto shift = static_cast<std::ptrdiff_t>(size * ratio);

                for (auto _ : state)
                {
                    xt::xtensor<double, 4> t1 = xt::roll(input, shift, 0);
                    xt::xtensor<double, 4> t2 = xt::roll(t1, shift, 1);
                    xt::xtensor<double, 4> t3 = xt::roll(t2, shift, 2);
                    xt::xtensor<double, 4> result = xt::roll(t3, shift, 3);
                    benchmark::DoNotOptimize(result.data());
                    benchmark::ClobberMemory();
                }

                auto total = size * size * size * size;
                state.SetItemsProcessed(state.iterations() * total);
                state.SetBytesProcessed(state.iterations() * total * sizeof(double));
            }

            inline void roll_4d_4axes_multi(benchmark::State& state, std::size_t size, double ratio)
            {
                xt::xtensor<double, 4> input = xt::random::rand<double>({size, size, size, size});
                auto shift = static_cast<std::ptrdiff_t>(size * ratio);

                // Verify correctness
                {
                    xt::xtensor<double, 4> t1 = xt::roll(input, shift, 0);
                    xt::xtensor<double, 4> t2 = xt::roll(t1, shift, 1);
                    xt::xtensor<double, 4> t3 = xt::roll(t2, shift, 2);
                    xt::xtensor<double, 4> sequential = xt::roll(t3, shift, 3);
                    auto multi = xt::roll(input, {shift, shift, shift, shift}, {0, 1, 2, 3});
                    if (!verify_correctness(multi, sequential))
                    {
                        state.SkipWithError("Correctness check failed!");
                        return;
                    }
                }

                for (auto _ : state)
                {
                    auto result = xt::roll(input, {shift, shift, shift, shift}, {0, 1, 2, 3});
                    benchmark::DoNotOptimize(result.data());
                    benchmark::ClobberMemory();
                }

                auto total = size * size * size * size;
                state.SetItemsProcessed(state.iterations() * total);
                state.SetBytesProcessed(state.iterations() * total * sizeof(double));
            }

            /*********************************************
             * 5D roll benchmarks (5 axes)
             * shift = size * ratio for all axes
             *********************************************/

            inline void roll_5d_5axes_sequential(benchmark::State& state, std::size_t size, double ratio)
            {
                xt::xtensor<double, 5> input = xt::random::rand<double>({size, size, size, size, size});
                auto shift = static_cast<std::ptrdiff_t>(size * ratio);

                for (auto _ : state)
                {
                    xt::xtensor<double, 5> t1 = xt::roll(input, shift, 0);
                    xt::xtensor<double, 5> t2 = xt::roll(t1, shift, 1);
                    xt::xtensor<double, 5> t3 = xt::roll(t2, shift, 2);
                    xt::xtensor<double, 5> t4 = xt::roll(t3, shift, 3);
                    xt::xtensor<double, 5> result = xt::roll(t4, shift, 4);
                    benchmark::DoNotOptimize(result.data());
                    benchmark::ClobberMemory();
                }

                auto total = size * size * size * size * size;
                state.SetItemsProcessed(state.iterations() * total);
                state.SetBytesProcessed(state.iterations() * total * sizeof(double));
            }

            inline void roll_5d_5axes_multi(benchmark::State& state, std::size_t size, double ratio)
            {
                xt::xtensor<double, 5> input = xt::random::rand<double>({size, size, size, size, size});
                auto shift = static_cast<std::ptrdiff_t>(size * ratio);

                // Verify correctness
                {
                    xt::xtensor<double, 5> t1 = xt::roll(input, shift, 0);
                    xt::xtensor<double, 5> t2 = xt::roll(t1, shift, 1);
                    xt::xtensor<double, 5> t3 = xt::roll(t2, shift, 2);
                    xt::xtensor<double, 5> t4 = xt::roll(t3, shift, 3);
                    xt::xtensor<double, 5> sequential = xt::roll(t4, shift, 4);
                    auto multi = xt::roll(input, {shift, shift, shift, shift, shift}, {0, 1, 2, 3, 4});
                    if (!verify_correctness(multi, sequential))
                    {
                        state.SkipWithError("Correctness check failed!");
                        return;
                    }
                }

                for (auto _ : state)
                {
                    auto result = xt::roll(input, {shift, shift, shift, shift, shift}, {0, 1, 2, 3, 4});
                    benchmark::DoNotOptimize(result.data());
                    benchmark::ClobberMemory();
                }

                auto total = size * size * size * size * size;
                state.SetItemsProcessed(state.iterations() * total);
                state.SetBytesProcessed(state.iterations() * total * sizeof(double));
            }
        }

        /*********************************************
         * Rate variation test (3D cube 128, 3 axes)
         * Demonstrates that rate does not affect performance
         * Rates: 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, -0.3
         *********************************************/

        BENCHMARK_CAPTURE(detail::roll_3d_3axes_sequential, 128/r0.01, 128, 0.01);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_multi, 128/r0.01, 128, 0.01);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_sequential, 128/r0.05, 128, 0.05);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_multi, 128/r0.05, 128, 0.05);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_sequential, 128/r0.1, 128, 0.1);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_multi, 128/r0.1, 128, 0.1);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_sequential, 128/r0.2, 128, 0.2);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_multi, 128/r0.2, 128, 0.2);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_sequential, 128/r0.3, 128, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_multi, 128/r0.3, 128, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_sequential, 128/r0.4, 128, 0.4);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_multi, 128/r0.4, 128, 0.4);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_sequential, 128/r-0.3, 128, -0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_multi, 128/r-0.3, 128, -0.3);

        /*********************************************
         * Main benchmarks (rate = 0.3)
         *********************************************/

        // 2D square tensors
        BENCHMARK_CAPTURE(detail::roll_2d_sequential, 64x64, 64, 64, 0.3);
        BENCHMARK_CAPTURE(detail::roll_2d_multi, 64x64, 64, 64, 0.3);
        BENCHMARK_CAPTURE(detail::roll_2d_sequential, 256x256, 256, 256, 0.3);
        BENCHMARK_CAPTURE(detail::roll_2d_multi, 256x256, 256, 256, 0.3);
        BENCHMARK_CAPTURE(detail::roll_2d_sequential, 1024x1024, 1024, 1024, 0.3);
        BENCHMARK_CAPTURE(detail::roll_2d_multi, 1024x1024, 1024, 1024, 0.3);

        // 3D cube - 2 axes
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_sequential, 64x64x64, 64, 64, 64, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_multi, 64x64x64, 64, 64, 64, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_sequential, 128x128x128, 128, 128, 128, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_multi, 128x128x128, 128, 128, 128, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_sequential, 256x256x256, 256, 256, 256, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_multi, 256x256x256, 256, 256, 256, 0.3);

        // 3D cube - 3 axes
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_sequential, 16, 16, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_multi, 16, 16, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_sequential, 32, 32, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_multi, 32, 32, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_sequential, 64, 64, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_3axes_multi, 64, 64, 0.3);

        // 4D - 4 axes
        BENCHMARK_CAPTURE(detail::roll_4d_4axes_sequential, 16, 16, 0.3);
        BENCHMARK_CAPTURE(detail::roll_4d_4axes_multi, 16, 16, 0.3);
        BENCHMARK_CAPTURE(detail::roll_4d_4axes_sequential, 32, 32, 0.3);
        BENCHMARK_CAPTURE(detail::roll_4d_4axes_multi, 32, 32, 0.3);
        BENCHMARK_CAPTURE(detail::roll_4d_4axes_sequential, 64, 64, 0.3);
        BENCHMARK_CAPTURE(detail::roll_4d_4axes_multi, 64, 64, 0.3);

        // 5D - 5 axes
        BENCHMARK_CAPTURE(detail::roll_5d_5axes_sequential, 16, 16, 0.3);
        BENCHMARK_CAPTURE(detail::roll_5d_5axes_multi, 16, 16, 0.3);
        BENCHMARK_CAPTURE(detail::roll_5d_5axes_sequential, 32, 32, 0.3);
        BENCHMARK_CAPTURE(detail::roll_5d_5axes_multi, 32, 32, 0.3);

        // 3D RGB images (H x W x 3)
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_sequential, rgb_1080p, 1080, 1920, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_multi, rgb_1080p, 1080, 1920, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_sequential, rgb_2K, 1440, 2560, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_multi, rgb_2K, 1440, 2560, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_sequential, rgb_4K, 2160, 3840, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_multi, rgb_4K, 2160, 3840, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_sequential, rgb_8K, 4320, 7680, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_multi, rgb_8K, 4320, 7680, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_sequential, rgb_256x256, 256, 256, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_multi, rgb_256x256, 256, 256, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_sequential, rgb_512x512, 512, 512, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_multi, rgb_512x512, 512, 512, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_sequential, rgb_1024x1024, 1024, 1024, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_multi, rgb_1024x1024, 1024, 1024, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_sequential, rgb_2048x2048, 2048, 2048, 3, 0.3);
        BENCHMARK_CAPTURE(detail::roll_3d_2axes_multi, rgb_2048x2048, 2048, 2048, 3, 0.3);
    }
}
