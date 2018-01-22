/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef BENCHMARK_CONTAINER_HPP
#define BENCHMARK_CONTAINER_HPP

#include <cstddef>
#include <chrono>
#include <string>

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
        inline auto benchmark_iteration(benchmark::State& state)
        {
            using value_type = typename E::value_type;
            E x, y, res;
            init_benchmark(x, y, res, state.range(0));
            value_type a = value_type(2.7);
            while (state.KeepRunning())
            {
                auto iterx = x.begin();
                auto itery = y.begin();
                for (auto iter = res.begin(); iter != res.end(); ++iter, ++iterx, ++itery)
                {
                    *iter = a * (*iterx) + (*itery);
                }
            }
        }

        BENCHMARK_TEMPLATE(benchmark_iteration, xarray_container<std::vector<double>>)->Arg(1000);
        BENCHMARK_TEMPLATE(benchmark_iteration, xarray_container<xt::uvector<double>>)->Arg(1000);
        BENCHMARK_TEMPLATE(benchmark_iteration, xtensor_container<std::vector<double>, 1>)->Arg(1000);
        BENCHMARK_TEMPLATE(benchmark_iteration, xtensor_container<xt::uvector<double>, 1>)->Arg(1000);

        template <class E>
        inline auto benchmark_xiteration(benchmark::State& state)
        {
            using value_type = typename E::value_type;
            E x, y, res;
            init_benchmark(x, y, res, state.range(0));
            value_type a = value_type(2.7);

            while (state.KeepRunning())
            {
                auto iterx = x.begin();
                auto itery = y.begin();
                for (auto iter = res.begin(); iter != res.end(); ++iter, ++iterx, ++itery)
                {
                    *iter = a * (*iterx) + (*itery);
                }
            }
        }

        BENCHMARK_TEMPLATE(benchmark_xiteration, xarray_container<std::vector<double>>)->Arg(1000);
        BENCHMARK_TEMPLATE(benchmark_xiteration, xarray_container<xt::uvector<double>>)->Arg(1000);
        BENCHMARK_TEMPLATE(benchmark_xiteration, xtensor_container<std::vector<double>, 1>)->Arg(1000);
        BENCHMARK_TEMPLATE(benchmark_xiteration, xtensor_container<xt::uvector<double>, 1>)->Arg(1000);

        template <class E>
        inline auto benchmark_indexing(benchmark::State& state)
        {
            using size_type = typename E::size_type;
            using value_type = typename E::value_type;
            E x, y, res;
            init_benchmark(x, y, res, state.range(0));
            value_type a = value_type(2.7);

            while (state.KeepRunning())
            {
                size_type n = x.size();
                for (size_type i = 0; i < n; ++i)
                {
                    res(i) = a * x(i) + y(i);
                }
            }
        }

        BENCHMARK_TEMPLATE(benchmark_indexing, xarray_container<std::vector<double>>)->Arg(1000);
        BENCHMARK_TEMPLATE(benchmark_indexing, xarray_container<xt::uvector<double>>)->Arg(1000);
        BENCHMARK_TEMPLATE(benchmark_indexing, xtensor_container<std::vector<double>, 1>)->Arg(1000);
        BENCHMARK_TEMPLATE(benchmark_indexing, xtensor_container<xt::uvector<double>, 1>)->Arg(1000);
    }

    namespace func
    {

        template <class E>
        inline void init_benchmark(E& x, E& y, E& z, E& res)
        {
            using value_type = typename E::value_type;
            using size_type = typename E::size_type;
            using shape_type = typename E::shape_type;

            shape_type shape = { 4, 3, 5 };

            x.resize(shape);
            y.resize(shape);
            z.resize(shape);
            res.resize(shape);

            for (size_type i = 0; i < shape[0]; ++i)
            {
                for (size_type j = 0; j < shape[1]; ++j)
                {
                    for (size_type k = 0; k < shape[2]; ++k)
                    {
                        x(i, j, k) = 0.25 * value_type(i) + 0.5 * value_type(j) - 0.01 * value_type(k);
                        y(i, j, k) = 0.31 * value_type(i) - 0.2 * value_type(j) + 0.07 * value_type(k);
                        z(i, j, k) = 0.27 * value_type(i) + 0.4 * value_type(j) - 0.03 * value_type(k);
                    }
                }
            }
        }

        template <class E>
        inline auto func(benchmark::State& state)
        {
            E x, y, z, res;
            init_benchmark(x, y, z, res);

            while (state.KeepRunning())
            {
                res = 3 * x - 2 * y * z;
            }
        }

        BENCHMARK_TEMPLATE(func, xarray_container<std::vector<double>>);
        BENCHMARK_TEMPLATE(func, xarray_container<xt::uvector<double>>);
        BENCHMARK_TEMPLATE(func, xtensor_container<std::vector<double>, 3>);
        BENCHMARK_TEMPLATE(func, xtensor_container<xt::uvector<double>, 3>);
    }

    namespace sum_assign
    {
        template <class E>
        inline void init_benchmark(E& x, E& y, E& res)
        {
            using value_type = typename E::value_type;
            using size_type = typename E::size_type;
            using shape_type = typename E::shape_type;

            shape_type shape = { 100, 100 };

            x.resize(shape);
            y.resize(shape);
            res.resize(shape);

            for (size_type i = 0; i < shape[0]; ++i)
            {
                for (size_type j = 0; j < shape[1]; ++j)
                {
                    x(i, j) = 0.25 * value_type(i) + 0.5 * value_type(j);
                    y(i, j) = 0.31 * value_type(i) - 0.2 * value_type(j);
                }
            }
        }

        template <class E>
        inline auto sum_assign(benchmark::State& state)
        {
            E x, y, res;
            init_benchmark(x, y, res);

            while (state.KeepRunning())
            {
                res = 3 * x - 2 * y;
            }
        }

        BENCHMARK_TEMPLATE(sum_assign, xarray_container<std::vector<double>>);
        BENCHMARK_TEMPLATE(sum_assign, xarray_container<xt::uvector<double>>);
        BENCHMARK_TEMPLATE(sum_assign, xtensor_container<std::vector<double>, 2>);
        BENCHMARK_TEMPLATE(sum_assign, xtensor_container<xt::uvector<double>, 2>);
    }
}

#endif
