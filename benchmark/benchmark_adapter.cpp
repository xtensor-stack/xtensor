/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <benchmark/benchmark.h>

// #include "xtensor/xshape.hpp"
#include "xtensor/xstorage.hpp"
#include "xtensor/xutils.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xnoalias.hpp"

namespace xt
{
    template <class V>
    void shape_array_adapter(benchmark::State& state)
    {
        const V a({1,2,3,4});
        const V b({1,2,3,4});
        using value_type = typename V::value_type;
    
        for (auto _ : state)
        {
            xtensor<value_type, 1> result(std::array<std::size_t, 1>({4}));
            auto aa = xt::adapt(a);
            auto ab = xt::adapt(b);
            xt::noalias(result) = aa + ab;
            benchmark::DoNotOptimize(result.data());
        }
    }

    template <class V>
    void shape_array_adapter_result(benchmark::State& state)
    {
        const V a({1, 2, 3, 4});
        const V b({1, 2, 3, 4});

        for (auto _ : state)
        {
            V res({0, 0, 0, 0});
            auto aa = xt::adapt(a);
            auto ab = xt::adapt(b);
            auto ar = xt::adapt(res);
            xt::noalias(ar) = aa + ab;
            benchmark::DoNotOptimize(ar.data());
        }
    }

    template <class V>
    void shape_array_adapter_result_copy(benchmark::State& state)
    {
        const V a({1, 2, 3, 4});
        const V b({1, 2, 3, 4});

        for (auto _ : state)
        {
            V res({0, 0, 0, 0});
            auto aa = xt::adapt(a);
            auto ab = xt::adapt(b);
            auto ar = xt::adapt(res);
            auto fun = aa + ab;
            std::copy(fun.storage_cbegin(), fun.storage_cend(), ar.storage_begin());
            benchmark::DoNotOptimize(ar.data());
        }
    }

    template <class V>
    void shape_array_adapter_result_transform(benchmark::State& state)
    {
        const V a({1, 2, 3, 4});
        const V b({1, 2, 3, 4});

        for (auto _ : state)
        {
            V res({0, 0, 0, 0});
            auto aa = xt::adapt(a);
            auto ab = xt::adapt(b);
            auto ar = xt::adapt(res);
            auto fun = aa + ab;
            std::transform(fun.storage_cbegin(), fun.storage_cend(), ar.storage_begin(),
                           [](typename decltype(fun)::value_type x) { return static_cast<typename decltype(ar)::value_type>(x); });
            benchmark::DoNotOptimize(ar.data());
        }
    }

    template <class V>
    void shape_no_adapter(benchmark::State& state)
    {
        V a({1, 2, 3, 4});
        V b({1, 2, 3, 4});

        for (auto _ : state)
        {
            V result({0, 0, 0, 0});
            auto n = std::distance(a.begin(), a.end());
            for (std::size_t i = 0; i < n; ++i)
            {
                result[i] = a[i] + b[i];
            }
            benchmark::DoNotOptimize(result.data());
        }
    }

    using array_type = std::array<std::size_t, 4>;
    // using array_type_ll = std::array<int64_t, 4>;
    using array_type_ll = std::array<double, 4>;
    using uvector_type = xt::uvector<std::size_t, xsimd::aligned_allocator<std::size_t, 32>>;
    using uvector_type_i64 = xt::uvector<int64_t, xsimd::aligned_allocator<int64_t, 32>>;
    using uvector_type_i64_16 = xt::uvector<int64_t, xsimd::aligned_allocator<int64_t, 16>>;
    using uvector_type_i64_ra = xt::uvector<int64_t, std::allocator<int64_t>>;

    using small_type = xt::svector<int64_t, 4, xsimd::aligned_allocator<int64_t, 32>>;
    using small_type_d = xt::svector<double, 4, xsimd::aligned_allocator<double, 32>>;

    // BENCHMARK_TEMPLATE(shape_array_adapter, array_type);
    // BENCHMARK_TEMPLATE(shape_array_adapter, uvector_type);
    // BENCHMARK_TEMPLATE(shape_array_adapter, uvector_type_i64);
    // BENCHMARK_TEMPLATE(shape_array_adapter, uvector_type_i64_ra);
    // BENCHMARK_TEMPLATE(shape_array_adapter, std::vector<std::size_t>);
    // BENCHMARK_TEMPLATE(shape_array_adapter, small_type);
    // BENCHMARK_TEMPLATE(shape_array_adapter, small_type_d);
    // BENCHMARK_TEMPLATE(shape_array_adapter_result, small_type);
    // BENCHMARK_TEMPLATE(shape_array_adapter_result, small_type_d);
    BENCHMARK_TEMPLATE(shape_array_adapter_result, array_type);
    // BENCHMARK_TEMPLATE(shape_array_adapter_result, array_type_ll);
    // BENCHMARK_TEMPLATE(shape_array_adapter_result_2, array_type);
    BENCHMARK_TEMPLATE(shape_array_adapter_result_copy, array_type);
    BENCHMARK_TEMPLATE(shape_array_adapter_result_transform, array_type);
    // // BENCHMARK_TEMPLATE(shape_array_adapter_result_2, array_type_ll);
    BENCHMARK_TEMPLATE(shape_no_adapter, array_type);
    // BENCHMARK_TEMPLATE(shape_no_adapter, std::vector<int64_t>);
    // BENCHMARK_TEMPLATE(shape_no_adapter, uvector_type_i64);
    // BENCHMARK_TEMPLATE(shape_no_adapter, uvector_type_i64_ra);
    // BENCHMARK_TEMPLATE(shape_no_adapter, uvector_type_i64_16);
}
