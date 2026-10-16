/***************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <algorithm>
#include <cstddef>
#include <vector>

#include <benchmark/benchmark.h>

#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xfixed.hpp"
#include "xtensor/core/xnoalias.hpp"
#include "xtensor/core/xoperation.hpp"
#include "xtensor/views/xstrided_view.hpp"

#ifdef XTENSOR_BENCHMARK_USE_EIGEN
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#endif

#ifdef XTENSOR_BENCHMARK_USE_ARMADILLO
#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#include <armadillo>
#endif

namespace xt::compare
{
    constexpr std::size_t dynamic_size = 64 * 64 * 16;
    constexpr std::size_t fixed_size = 16 * 16;

    template <class X, class Y, class Z>
    void init(X& x, Y& y, Z& z, std::size_t size)
    {
        for (std::size_t i = 0; i < size; ++i)
        {
            x[i] = 0.5 + double(i % 251) * 0.01;
            y[i] = 0.25 - double(i % 127) * 0.02;
            z[i] = 1.0 + double(i % 61) * 0.03;
        }
    }

    void linear_dynamic_xtensor(benchmark::State& state)
    {
        const std::vector<std::size_t> shape = {64, 64, 16};
        xarray<double> x, y, z, out;
        x.resize(shape);
        y.resize(shape);
        z.resize(shape);
        out.resize(shape);
        init(x, y, z, dynamic_size);
        for (auto _ : state)
        {
            noalias(out) = (x + y * 0.5) * (z - 0.25) + y / (x + 2.0);
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
    }

    void linear_dynamic_raw(benchmark::State& state)
    {
        std::vector<double> x(dynamic_size), y(dynamic_size), z(dynamic_size), out(dynamic_size);
        init(x, y, z, dynamic_size);
        for (auto _ : state)
        {
            for (std::size_t i = 0; i < dynamic_size; ++i)
            {
                out[i] = (x[i] + y[i] * 0.5) * (z[i] - 0.25) + y[i] / (x[i] + 2.0);
            }
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
    }

    void broadcast_dynamic_xtensor(benchmark::State& state)
    {
        const std::vector<std::size_t> shape = {64, 64, 16};
        xarray<double> x, row, column, out;
        x.resize(shape);
        row.resize({16});
        column.resize({64, 1, 1});
        out.resize(shape);
        xarray<double> unused;
        unused.resize(shape);
        init(x, unused, out, dynamic_size);
        for (std::size_t i = 0; i < row.size(); ++i)
        {
            row[i] = double(i) * 0.25;
        }
        for (std::size_t i = 0; i < column.size(); ++i)
        {
            column[i] = double(i) * 0.125;
        }
        for (auto _ : state)
        {
            noalias(out) = (x + row) * 1.5 - column;
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
    }

    void broadcast_dynamic_raw(benchmark::State& state)
    {
        std::vector<double> x(dynamic_size), row(16), column(64), out(dynamic_size), unused(dynamic_size);
        init(x, unused, out, dynamic_size);
        for (std::size_t i = 0; i < row.size(); ++i)
        {
            row[i] = double(i) * 0.25;
        }
        for (std::size_t i = 0; i < column.size(); ++i)
        {
            column[i] = double(i) * 0.125;
        }
        for (auto _ : state)
        {
            for (std::size_t i = 0; i < 64; ++i)
            {
                for (std::size_t j = 0; j < 64; ++j)
                {
                    for (std::size_t k = 0; k < 16; ++k)
                    {
                        const std::size_t n = (i * 64 + j) * 16 + k;
                        out[n] = (x[n] + row[k]) * 1.5 - column[i];
                    }
                }
            }
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
    }

    void transpose_cast_raw(benchmark::State& state)
    {
        constexpr std::size_t height = 128;
        constexpr std::size_t width = 256;
        constexpr std::size_t channels = 3;
        std::vector<unsigned char> input(height * width * channels);
        std::vector<float> out(input.size());
        for (std::size_t i = 0; i < input.size(); ++i)
        {
            input[i] = static_cast<unsigned char>(i % 251);
        }
        for (auto _ : state)
        {
            for (std::size_t c = 0; c < channels; ++c)
            {
                for (std::size_t i = 0; i < height; ++i)
                {
                    for (std::size_t j = 0; j < width; ++j)
                    {
                        out[(c * height + i) * width + j] = float(input[(i * width + j) * channels + c])
                                                            / 255.0f;
                    }
                }
            }
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
    }

    void transpose_cast_xtensor(benchmark::State& state)
    {
        xarray<unsigned char> input = xarray<unsigned char>::from_shape({128, 256, 3});
        xarray<float> out = xarray<float>::from_shape({3, 128, 256});
        for (std::size_t i = 0; i < input.size(); ++i)
        {
            input.storage()[i] = static_cast<unsigned char>(i % 251);
        }
        for (auto _ : state)
        {
            noalias(out) = cast<float>(transpose(input, {2, 0, 1})) / 255.0f;
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
    }

    void linear_fixed_xtensor(benchmark::State& state)
    {
        xtensor_fixed<double, xshape<16, 16>> x, y, z, out;
        init(x, y, z, fixed_size);
        for (auto _ : state)
        {
            noalias(out) = (x + y * 0.5) * (z - 0.25) + y / (x + 2.0);
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
    }

#ifdef XTENSOR_BENCHMARK_USE_EIGEN
    void linear_dynamic_eigen(benchmark::State& state)
    {
        Eigen::ArrayXd x(dynamic_size), y(dynamic_size), z(dynamic_size), out(dynamic_size);
        init(x, y, z, dynamic_size);
        for (auto _ : state)
        {
            out = (x + y * 0.5) * (z - 0.25) + y / (x + 2.0);
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
    }

    void linear_fixed_eigen(benchmark::State& state)
    {
        Eigen::Array<double, fixed_size, 1> x, y, z, out;
        init(x, y, z, fixed_size);
        for (auto _ : state)
        {
            out = (x + y * 0.5) * (z - 0.25) + y / (x + 2.0);
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
    }

    void broadcast_dynamic_eigen(benchmark::State& state)
    {
        using tensor = Eigen::Tensor<double, 3, Eigen::RowMajor>;
        tensor x(64, 64, 16), out(64, 64, 16);
        Eigen::Tensor<double, 1, Eigen::RowMajor> row(16), column(64);
        for (Eigen::Index i = 0; i < x.size(); ++i)
        {
            x.data()[i] = 0.5 + double(i % 251) * 0.01;
        }
        for (Eigen::Index i = 0; i < row.size(); ++i)
        {
            row(i) = double(i) * 0.25;
        }
        for (Eigen::Index i = 0; i < column.size(); ++i)
        {
            column(i) = double(i) * 0.125;
        }
        const Eigen::array<Eigen::Index, 3> row_shape = {1, 1, 16};
        const Eigen::array<Eigen::Index, 3> row_broadcast = {64, 64, 1};
        const Eigen::array<Eigen::Index, 3> column_shape = {64, 1, 1};
        const Eigen::array<Eigen::Index, 3> column_broadcast = {1, 64, 16};
        for (auto _ : state)
        {
            out = (x + row.reshape(row_shape).broadcast(row_broadcast)) * 1.5
                  - column.reshape(column_shape).broadcast(column_broadcast);
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
    }

    void transpose_cast_eigen(benchmark::State& state)
    {
        Eigen::Tensor<unsigned char, 3, Eigen::RowMajor> input(128, 256, 3);
        Eigen::Tensor<float, 3, Eigen::RowMajor> out(3, 128, 256);
        for (Eigen::Index i = 0; i < input.size(); ++i)
        {
            input.data()[i] = static_cast<unsigned char>(i % 251);
        }
        const Eigen::array<Eigen::Index, 3> permutation = {2, 0, 1};
        for (auto _ : state)
        {
            out = input.shuffle(permutation).cast<float>() / 255.0f;
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
    }
#endif

#ifdef XTENSOR_BENCHMARK_USE_ARMADILLO
    void linear_dynamic_armadillo(benchmark::State& state)
    {
        arma::cube x(64, 64, 16), y(64, 64, 16), z(64, 64, 16), out(64, 64, 16);
        init(x, y, z, dynamic_size);
        for (auto _ : state)
        {
            out = (x + y * 0.5) % (z - 0.25) + y / (x + 2.0);
            benchmark::DoNotOptimize(out.memptr());
            benchmark::ClobberMemory();
        }
    }

    void linear_fixed_armadillo(benchmark::State& state)
    {
        arma::vec::fixed<fixed_size> x, y, z, out;
        init(x, y, z, fixed_size);
        for (auto _ : state)
        {
            out = (x + y * 0.5) % (z - 0.25) + y / (x + 2.0);
            benchmark::DoNotOptimize(out.memptr());
            benchmark::ClobberMemory();
        }
    }
#endif

    BENCHMARK(linear_dynamic_raw);
    BENCHMARK(linear_dynamic_xtensor);
    BENCHMARK(broadcast_dynamic_raw);
    BENCHMARK(broadcast_dynamic_xtensor);
    BENCHMARK(transpose_cast_raw);
    BENCHMARK(transpose_cast_xtensor);
    BENCHMARK(linear_fixed_xtensor);
#ifdef XTENSOR_BENCHMARK_USE_EIGEN
    BENCHMARK(linear_dynamic_eigen);
    BENCHMARK(linear_fixed_eigen);
    BENCHMARK(broadcast_dynamic_eigen);
    BENCHMARK(transpose_cast_eigen);
#endif
#ifdef XTENSOR_BENCHMARK_USE_ARMADILLO
    BENCHMARK(linear_dynamic_armadillo);
    BENCHMARK(linear_fixed_armadillo);
#endif
}
