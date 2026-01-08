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
    // Helpers
    //========================================================================

    // Benchmark range configuration
    constexpr std::size_t min_size = 8;
    constexpr std::size_t max_size = 16384;
    constexpr std::size_t multiplier = 4;

    // Helper to create xtensor vectors
    inline auto make_xtensor(std::size_t size, double val)
    {
        return xt::xtensor<double, 1>::from_shape({size}) * val;
    }

    inline auto make_xtensor_zeros(std::size_t size)
    {
        auto c = xt::xtensor<double, 1>::from_shape({size});
        c.fill(0);
        return c;
    }

    // Helper to create xarray
    inline auto make_xarray(std::size_t size, double val)
    {
        return xt::xarray<double>::from_shape({size}) * val;
    }

// Macro for benchmark loop (reduces boilerplate)
#define BENCHMARK_LOOP(state, container, ...)       \
    for (auto _ : state)                            \
    {                                               \
        __VA_ARGS__;                                \
        benchmark::DoNotOptimize(container.data()); \
    }

// Macro for registering benchmarks with standard sizes
#define REGISTER_BENCHMARK(func) BENCHMARK(func)->Range(min_size, max_size)->RangeMultiplier(multiplier)

    //========================================================================
    // Vector Addition: z = x + y
    //========================================================================

    static void add_vector_std(benchmark::State& state)
    {
        const std::size_t size = state.range(0);
        std::vector<double> x(size, 1.0);
        std::vector<double> y(size, 2.0);
        std::vector<double> z(size);

        BENCHMARK_LOOP(state, z, for (std::size_t i = 0; i < size; ++i) z[i] = x[i] + y[i];);
    }

    static void add_vector_xarray(benchmark::State& state)
    {
        const std::size_t size = state.range(0);
        auto x = make_xarray(size, 1.0);
        auto y = make_xarray(size, 2.0);
        xt::xarray<double> z;

        BENCHMARK_LOOP(state, z, z = x + y;);
    }

    static void add_vector_xtensor(benchmark::State& state)
    {
        const std::size_t size = state.range(0);
        auto x = make_xtensor(size, 1.0);
        auto y = make_xtensor(size, 2.0);
        xt::xtensor<double, 1> z;

        BENCHMARK_LOOP(state, z, z = x + y;);
    }

    static void add_vector_noalias(benchmark::State& state)
    {
        const std::size_t size = state.range(0);
        auto x = make_xtensor(size, 1.0);
        auto y = make_xtensor(size, 2.0);
        auto z = make_xtensor_zeros(size);

        BENCHMARK_LOOP(state, z, xt::noalias(z) = x + y;);
    }

    //========================================================================
    // Scalar Addition: z = x + a
    //========================================================================

    static void add_scalar_std(benchmark::State& state)
    {
        const std::size_t size = state.range(0);
        constexpr double a = 5.0;
        std::vector<double> x(size, 1.0);
        std::vector<double> z(size);

        BENCHMARK_LOOP(state, z, for (std::size_t i = 0; i < size; ++i) z[i] = x[i] + a;);
    }

    static void add_scalar_xtensor(benchmark::State& state)
    {
        const std::size_t size = state.range(0);
        constexpr double a = 5.0;
        auto x = make_xtensor(size, 1.0);
        xt::xtensor<double, 1> z;

        BENCHMARK_LOOP(state, z, z = x + a;);
    }

    static void add_scalar_noalias(benchmark::State& state)
    {
        const std::size_t size = state.range(0);
        constexpr double a = 5.0;
        auto x = make_xtensor(size, 1.0);
        auto z = make_xtensor_zeros(size);

        BENCHMARK_LOOP(state, z, xt::noalias(z) = x + a;);
    }

    //========================================================================
    // Scalar Multiplication: y = a * x
    //========================================================================

    static void mul_scalar_std(benchmark::State& state)
    {
        const std::size_t size = state.range(0);
        constexpr double a = 2.5;
        std::vector<double> x(size, 1.0);
        std::vector<double> y(size);

        BENCHMARK_LOOP(state, y, for (std::size_t i = 0; i < size; ++i) y[i] = a * x[i];);
    }

    static void mul_scalar_xtensor(benchmark::State& state)
    {
        const std::size_t size = state.range(0);
        constexpr double a = 2.5;
        auto x = make_xtensor(size, 1.0);
        xt::xtensor<double, 1> y;

        BENCHMARK_LOOP(state, y, y = a * x;);
    }

    //========================================================================
    // Register benchmarks
    //========================================================================

    // Vector + Vector
    REGISTER_BENCHMARK(add_vector_std);
    REGISTER_BENCHMARK(add_vector_xarray);
    REGISTER_BENCHMARK(add_vector_xtensor);
    REGISTER_BENCHMARK(add_vector_noalias);

    // Scalar operations
    REGISTER_BENCHMARK(add_scalar_std);
    REGISTER_BENCHMARK(add_scalar_xtensor);
    REGISTER_BENCHMARK(add_scalar_noalias);

    REGISTER_BENCHMARK(mul_scalar_std);
    REGISTER_BENCHMARK(mul_scalar_xtensor);

}
