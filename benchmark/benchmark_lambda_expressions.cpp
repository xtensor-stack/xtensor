/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <benchmark/benchmark.h>

#include "xtensor/xnoalias.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"

namespace xt
{
	void lambda_cube(benchmark::State& state)
	{
		xtensor<double, 2> x = empty<double>({state.range(0), state.range(0)});
		for (auto _ : state)
		{
			xtensor<double, 2> res = xt::cube(x);
			benchmark::DoNotOptimize(res.data());
		}
	}

	void xexpression_cube(benchmark::State& state)
	{
		xtensor<double, 2> x = empty<double>({state.range(0), state.range(0)});
		for (auto _ : state)
		{
			xtensor<double, 2> res = x * x * x;
			benchmark::DoNotOptimize(res.data());
		}
	}

	BENCHMARK(lambda_cube)->Range(32, 32<<3);
	BENCHMARK(xexpression_cube)->Range(32, 32<<3);
}