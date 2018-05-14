/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <cstddef>
#include <chrono>
#include <string>

#include <benchmark/benchmark.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

namespace xt
{

    namespace axpy_1d
    {

        // BENCHMARK Initialization
        template <class E>
        inline void init_benchmark(E& x, E& y, E& res, typename E::size_type size)
        {
            x.resize({ size });
            y.resize({ size });
            res.resize({ size });

            using value_type = typename E::value_type;
            using size_type = typename E::size_type;
            for (size_type i = 0; i < size; ++i)
            {
                x(i) = 0.5 + value_type(i);
                y(i) = 0.25 * value_type(i);
            }
        }

        template <class E>
        inline auto container_iteration(benchmark::State& state)
        {
            using value_type = typename E::value_type;
            E x, y, res;
            init_benchmark(x, y, res, state.range(0));
            value_type a = value_type(2.7);
            for (auto _ : state)
            {
                auto iterx = x.begin();
                auto itery = y.begin();
                for (auto iter = res.begin(); iter != res.end(); ++iter, ++iterx, ++itery)
                {
                    *iter = a * (*iterx) + (*itery);
                }
            }
        }

        BENCHMARK_TEMPLATE(container_iteration, xarray_container<std::vector<double>>)->Arg(1000);
        BENCHMARK_TEMPLATE(container_iteration, xarray_container<xt::uvector<double>>)->Arg(1000);
        BENCHMARK_TEMPLATE(container_iteration, xtensor_container<std::vector<double>, 1>)->Arg(1000);
        BENCHMARK_TEMPLATE(container_iteration, xtensor_container<xt::uvector<double>, 1>)->Arg(1000);

        template <class E>
        inline auto container_xiteration(benchmark::State& state)
        {
            using value_type = typename E::value_type;
            E x, y, res;
            init_benchmark(x, y, res, state.range(0));
            value_type a = value_type(2.7);

            for (auto _ : state)
            {
                auto iterx = x.begin();
                auto itery = y.begin();
                for (auto iter = res.begin(); iter != res.end(); ++iter, ++iterx, ++itery)
                {
                    *iter = a * (*iterx) + (*itery);
                }
            }
        }

        BENCHMARK_TEMPLATE(container_xiteration, xarray_container<std::vector<double>>)->Arg(1000);
        BENCHMARK_TEMPLATE(container_xiteration, xarray_container<xt::uvector<double>>)->Arg(1000);
        BENCHMARK_TEMPLATE(container_xiteration, xtensor_container<std::vector<double>, 1>)->Arg(1000);
        BENCHMARK_TEMPLATE(container_xiteration, xtensor_container<xt::uvector<double>, 1>)->Arg(1000);

        template <class E>
        inline auto container_indexing(benchmark::State& state)
        {
            using size_type = typename E::size_type;
            using value_type = typename E::value_type;
            E x, y, res;
            init_benchmark(x, y, res, state.range(0));
            value_type a = value_type(2.7);

            for (auto _ : state)
            {
                size_type n = x.size();
                for (size_type i = 0; i < n; ++i)
                {
                    res(i) = a * x(i) + y(i);
                }
            }
        }

        BENCHMARK_TEMPLATE(container_indexing, xarray_container<std::vector<double>>)->Arg(1000);
        BENCHMARK_TEMPLATE(container_indexing, xarray_container<xt::uvector<double>>)->Arg(1000);
        BENCHMARK_TEMPLATE(container_indexing, xtensor_container<std::vector<double>, 1>)->Arg(1000);
        BENCHMARK_TEMPLATE(container_indexing, xtensor_container<xt::uvector<double>, 1>)->Arg(1000);
    }
}
