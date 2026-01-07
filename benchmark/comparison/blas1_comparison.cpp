/***************************************************************************
 * Comparison benchmarks: BLAS1 operations
 * Comparing raw C++ vs xtensor performance
 ****************************************************************************/

#include <cstddef>
#include <vector>
#include <benchmark/benchmark.h>

#include "xtensor/containers/xarray.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xnoalias.hpp"

namespace xt::comparison
{

    //========================================================================
    // Vector Addition: z = x + y
    //========================================================================

    // Raw C++ implementation with std::vector
    static void add_vector_raw_cpp(benchmark::State& state)
    {
        using value_type = double;
        using size_type = std::size_t;

        const size_type size = state.range(0);
        std::vector<value_type> x(size, 1.0);
        std::vector<value_type> y(size, 2.0);
        std::vector<value_type> z(size);

        for (auto _ : state)
        {
            for (size_type i = 0; i < size; ++i)
            {
                z[i] = x[i] + y[i];
            }
            benchmark::DoNotOptimize(z.data());
        }
    }

    // xtensor implementation with xarray
    static void add_vector_xtensor_xarray(benchmark::State& state)
    {
        using size_type = std::size_t;

        const size_type size = state.range(0);
        xarray<double> x = xt::ones<double>({size});
        xarray<double> y = 2.0 * xt::ones<double>({size});
        xarray<double> z;

        for (auto _ : state)
        {
            z = x + y;
            benchmark::DoNotOptimize(z.data());
        }
    }

    // xtensor implementation with xtensor (fixed size)
    static void add_vector_xtensor_xtensor(benchmark::State& state)
    {
        using size_type = std::size_t;

        const size_type size = state.range(0);
        xtensor<double, 1> x = xt::ones<double>({size});
        xtensor<double, 1> y = 2.0 * xt::ones<double>({size});
        xtensor<double, 1> z;

        for (auto _ : state)
        {
            z = x + y;
            benchmark::DoNotOptimize(z.data());
        }
    }

    // xtensor with noalias (avoids temporary allocation)
    static void add_vector_xtensor_noalias(benchmark::State& state)
    {
        using size_type = std::size_t;

        const size_type size = state.range(0);
        xtensor<double, 1> x = xt::ones<double>({size});
        xtensor<double, 1> y = 2.0 * xt::ones<double>({size});
        xtensor<double, 1> z = xt::zeros<double>({size});

        for (auto _ : state)
        {
            xt::noalias(z) = x + y;
            benchmark::DoNotOptimize(z.data());
        }
    }

    //========================================================================
    // Scalar Addition: z = x + a (add_scalar)
    //========================================================================

    // Raw C++
    static void add_scalar_raw_cpp(benchmark::State& state)
    {
        using value_type = double;
        using size_type = std::size_t;

        const size_type size = state.range(0);
        const value_type a = 5.0;
        std::vector<value_type> x(size, 1.0);
        std::vector<value_type> z(size);

        for (auto _ : state)
        {
            for (size_type i = 0; i < size; ++i)
            {
                z[i] = x[i] + a;
            }
            benchmark::DoNotOptimize(z.data());
        }
    }

    // xtensor
    static void add_scalar_xtensor(benchmark::State& state)
    {
        using size_type = std::size_t;

        const size_type size = state.range(0);
        const double a = 5.0;
        xtensor<double, 1> x = xt::ones<double>({size});
        xtensor<double, 1> z;

        for (auto _ : state)
        {
            z = x + a;
            benchmark::DoNotOptimize(z.data());
        }
    }

    // xtensor with noalias
    static void add_scalar_xtensor_noalias(benchmark::State& state)
    {
        using size_type = std::size_t;

        const size_type size = state.range(0);
        const double a = 5.0;
        xtensor<double, 1> x = xt::ones<double>({size});
        xtensor<double, 1> z = xt::zeros<double>({size});

        for (auto _ : state)
        {
            xt::noalias(z) = x + a;
            benchmark::DoNotOptimize(z.data());
        }
    }

    //========================================================================
    // Scalar Multiplication: y = a * x
    //========================================================================

    // Raw C++
    static void mul_scalar_raw_cpp(benchmark::State& state)
    {
        using value_type = double;
        using size_type = std::size_t;

        const size_type size = state.range(0);
        const value_type a = 2.5;
        std::vector<value_type> x(size, 1.0);
        std::vector<value_type> y(size);

        for (auto _ : state)
        {
            for (size_type i = 0; i < size; ++i)
            {
                y[i] = a * x[i];
            }
            benchmark::DoNotOptimize(y.data());
        }
    }

    // xtensor
    static void mul_scalar_xtensor(benchmark::State& state)
    {
        using size_type = std::size_t;

        const size_type size = state.range(0);
        const double a = 2.5;
        xtensor<double, 1> x = xt::ones<double>({size});
        xtensor<double, 1> y;

        for (auto _ : state)
        {
            y = a * x;
            benchmark::DoNotOptimize(y.data());
        }
    }

    //========================================================================
    // Register benchmarks with different sizes
    //========================================================================

    // Vector sizes to test: 64, 256, 1024, 4096, 16384
    BENCHMARK(add_vector_raw_cpp)->Range(64, 16384)->RangeMultiplier(4);
    BENCHMARK(add_vector_xtensor_xarray)->Range(64, 16384)->RangeMultiplier(4);
    BENCHMARK(add_vector_xtensor_xtensor)->Range(64, 16384)->RangeMultiplier(4);
    BENCHMARK(add_vector_xtensor_noalias)->Range(64, 16384)->RangeMultiplier(4);

    BENCHMARK(add_scalar_raw_cpp)->Range(64, 16384)->RangeMultiplier(4);
    BENCHMARK(add_scalar_xtensor)->Range(64, 16384)->RangeMultiplier(4);
    BENCHMARK(add_scalar_xtensor_noalias)->Range(64, 16384)->RangeMultiplier(4);

    BENCHMARK(mul_scalar_raw_cpp)->Range(64, 16384)->RangeMultiplier(4);
    BENCHMARK(mul_scalar_xtensor)->Range(64, 16384)->RangeMultiplier(4);

}
