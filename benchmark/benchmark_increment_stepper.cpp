/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <benchmark/benchmark.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"

#define SHAPE 30, 30
#define RANGE 3, 100

namespace xt
{
    namespace benchmark_xstepper
    {
        void stepper_stepper(benchmark::State& state)
        {
            std::vector<std::size_t> shape = {SHAPE, std::size_t(state.range(0))};
            xt::xarray<double> a = xt::random::rand<double>(shape);
            xt::xarray<double> b = xt::random::rand<double>(shape);
            volatile double c = 0;
            for (auto _ : state)
            {
                auto end = compute_size(shape);
                auto it = a.stepper_begin(shape);
                auto bit = b.stepper_begin(shape);

                xindex index(shape.size());
                xindex bindex(shape.size());

                for (std::size_t i = 0; i < end; ++i)
                {
                    c += *it + *bit;
                    stepper_tools<layout_type::row_major>::increment_stepper(bit, bindex, shape);
                    stepper_tools<layout_type::row_major>::increment_stepper(it, index, shape);
                }
                benchmark::DoNotOptimize(c);
            }
        }
        BENCHMARK(stepper_stepper)->Range(RANGE);

        void stepper_stepper_ref(benchmark::State& state)
        {
            std::vector<std::size_t> shape = {SHAPE, std::size_t(state.range(0))};
            xt::xarray<double> a = xt::random::rand<double>(shape);
            xt::xarray<double> b = xt::random::rand<double>(shape);
            xindex index;
            xindex bindex;
            volatile double c = 0;
            for (auto _ : state)
            {
                auto it = a.storage().begin();
                auto bit = b.storage().begin();
                auto end = a.storage().end();
                for (; it != end; ++it)
                {
                    c += *it + *bit;
                    ++bit;
                }
                benchmark::DoNotOptimize(c);
            }
        }
        BENCHMARK(stepper_stepper_ref)->Range(RANGE);
    }
}
