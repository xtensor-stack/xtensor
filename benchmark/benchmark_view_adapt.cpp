/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef BENCHMARK_VIEW_ADAPT_HPP
#define BENCHMARK_VIEW_ADAPT_HPP

#include <benchmark/benchmark.h>

#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xadapt.hpp"

namespace xt
{
    namespace benchmark_view_adapt
    {
        using T2 = xt::xtensor_fixed<double, xt::xshape<2,2>>;

        T2 foo(const T2 &A)
        {
            return 2. * A;
        }

        void random_view(benchmark::State& state)
        {
            xt::xtensor<double,4> A = xt::random::randn<double>({2000,8,2,2});
            xt::xtensor<double,4> B = xt::empty<double>(A.shape());

            for (auto _ : state)
            {
                for ( size_t i = 0 ; i < A.shape()[0] ; ++i )
                {
                    for ( size_t j = 0 ; j < A.shape()[1] ; ++j )
                    {
                        auto a = xt::view(A, i, j);
                        auto b = xt::view(B, i, j);

                        xt::noalias(b) = foo(a);
                    }
                }
                benchmark::DoNotOptimize(B.data());
            }
        }

        void random_adapt(benchmark::State& state)
        {
            xt::xtensor<double,4> A = xt::random::randn<double>({2000,8,2,2});
            xt::xtensor<double,4> B = xt::empty<double>(A.shape());

            for (auto _ : state)
            {
                for ( size_t i = 0 ; i < A.shape()[0] ; ++i )
                {
                    for ( size_t j = 0 ; j < A.shape()[1] ; ++j )
                    {
                        auto a = xt::adapt(&A(i,j,0,0), xt::xshape<2,2>());
                        auto b = xt::adapt(&B(i,j,0,0), xt::xshape<2,2>());

                        xt::noalias(b) = foo(a);
                    }
                }
                benchmark::DoNotOptimize(B.data());
            }
        }

        BENCHMARK(random_view);
        BENCHMARK(random_adapt);
    }
}

#endif
