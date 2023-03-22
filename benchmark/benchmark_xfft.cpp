#include <chrono>
#include <cstddef>
#include <random>
#include <string>

#include <benchmark/benchmark.h>

#include "xtensor/xfft.hpp"

namespace xt
{

    namespace benchmark_xfft
    {
        void xfft_double(benchmark::State& state)
        {
            size_t n = state.range(0);
            xt::xarray<double> x = xt::xarray<double>::from_shape({n});
            for (auto _ : state)
            {
                auto z = xt::fft::fft(x);
                benchmark::DoNotOptimize(z);
                x[0] += 8 * std::numeric_limits<double>::epsilon();
            }
            state.SetComplexityN(state.range(0));
        }

        void xfft_single(benchmark::State& state)
        {
            size_t n = state.range(0);
            xt::xarray<float> x = xt::xarray<float>::from_shape({n});
            for (auto _ : state)
            {
                auto z = xt::fft::fft(x);
                benchmark::DoNotOptimize(z);
                x[0] += 8 * std::numeric_limits<float>::epsilon();
            }
            state.SetComplexityN(state.range(0));
        }

        BENCHMARK(xfft_double)->Range(2, 65536)->Complexity();
        BENCHMARK(xfft_single)->Range(2, 65536)->Complexity();
    }
}
