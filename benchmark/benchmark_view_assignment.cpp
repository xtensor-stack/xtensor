/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <benchmark/benchmark.h>

#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xrandom.hpp"

namespace xt
{
    inline void create_xview(benchmark::State& state)
    {
        xt::xtensor<double, 4> tens = xt::random::rand<double>({100, 100, 3, 3});
        for (auto _ : state)
        {
            auto v = xt::view(tens, 1, 2, all(), all());
        }
    }

    inline void create_strided_view_outofplace(benchmark::State& state)
    {
        xt::xtensor<double, 4> tens = xt::random::rand<double>({100, 100, 3, 3});
        xstrided_slice_vector sv = {1, 2, all(), all()};
        for (auto _ : state)
        {
            auto v = xt::strided_view(tens, sv);
        }
    }

    inline void create_strided_view_inplace(benchmark::State& state)
    {
        xt::xtensor<double, 4> tens = xt::random::rand<double>({100, 100, 3, 3});
        for (auto _ : state)
        {
            auto v = xt::strided_view(tens, {1, 2, all(), all()});
        }
    }

    inline void assign_create_view(benchmark::State& state)
    {
        xt::xtensor<double, 4> tens = xt::random::rand<double>({100, 100, 3, 3});
        for (auto _ : state)
        {
            for (std::size_t i = 0; i < tens.shape()[0]; ++i)
            {
                for (std::size_t j = 0; j < tens.shape()[1]; ++j)
                {
                    auto v = xt::view(tens, i, j, all(), all());
                    xt::xtensor<double, 2> vas = v;
                    benchmark::ClobberMemory();
                }
            }
        }
    }

    inline void assign_create_strided_view(benchmark::State& state)
    {
        xt::xtensor<double, 4> tens = xt::random::rand<double>({100, 100, 3, 3});
        for (auto _ : state)
        {
            for (std::size_t i = 0; i < tens.shape()[0]; ++i)
            {
                for (std::size_t j = 0; j < tens.shape()[1]; ++j)
                {
                    auto v = xt::strided_view(tens, {i, j, all(), all()});
                    xt::xtensor<double, 2> vas = v;
                    benchmark::ClobberMemory();
                }
            }
        }
    }

    inline void assign_create_manual_view(benchmark::State& state)
    {
        xt::xtensor<double, 4> tens = xt::random::rand<double>({100, 100, 3, 3});
        for (auto _ : state)
        {
            for (std::size_t i = 0; i < tens.shape()[0]; ++i)
            {
                for (std::size_t j = 0; j < tens.shape()[1]; ++j)
                {
                    auto v = xt::view(tens, i, j, all(), all());
                    xt::xtensor<double, 2> vas(std::array<std::size_t, 2>({3, 3}));
                    std::copy(v.data() + v.data_offset(), v.data() + v.data_offset() + vas.size(), vas.begin());
                    benchmark::ClobberMemory();
                }
            }
        }
    }

    inline void assign_create_manual_noview(benchmark::State& state)
    {
        xt::xtensor<double, 4> tens = xt::random::rand<double>({100, 100, 3, 3});
        for (auto _ : state)
        {
            for (std::size_t i = 0; i < tens.shape()[0]; ++i)
            {
                for (std::size_t j = 0; j < tens.shape()[1]; ++j)
                {
                    ptrdiff_t offset = i * tens.strides()[0] + j * tens.strides()[1];
                    xt::xtensor<double, 2> vas(std::array<std::size_t, 2>({3, 3}));
                    std::copy(tens.data() + offset, tens.data() + offset + vas.size(), vas.begin());
                    benchmark::ClobberMemory();
                }
            }
        }
    }

    inline void data_offset(benchmark::State& state)
    {
        xt::xtensor<double, 4> tens = xt::random::rand<double>({100, 100, 3, 3});
        for (auto _ : state)
        {
            for (std::size_t i = 0; i < tens.shape()[0]; ++i)
            {
                for (std::size_t j = 0; j < tens.shape()[1]; ++j)
                {
                    volatile ptrdiff_t offset = i * tens.strides()[0] + j * tens.strides()[1];
                    static_cast<void>(offset);
                }
            }
        }
    }

    inline void data_offset_view(benchmark::State& state)
    {
        xt::xtensor<double, 4> tens = xt::random::rand<double>({100, 100, 3, 3});
        for (auto _ : state)
        {
            for (std::size_t i = 0; i < tens.shape()[0]; ++i)
            {
                for (std::size_t j = 0; j < tens.shape()[1]; ++j)
                {
                    auto v = xt::view(tens, i, j, all(), all());
                    volatile ptrdiff_t offset = v.data_offset();
                    static_cast<void>(offset);
                }
            }
        }
    }

    BENCHMARK(create_xview);
    BENCHMARK(create_strided_view_outofplace);
    BENCHMARK(create_strided_view_inplace);
    BENCHMARK(assign_create_manual_noview);
    BENCHMARK(assign_create_strided_view);
    BENCHMARK(assign_create_view);
    BENCHMARK(assign_create_manual_view);
    BENCHMARK(data_offset);
    BENCHMARK(data_offset_view);
}