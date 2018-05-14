/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <iostream>

#include <benchmark/benchmark.h>

#include "xtensor/xtensor.hpp"
#include "xtensor/xarray.hpp"

#ifdef XTENSOR_USE_XSIMD
#ifdef __GNUC__
template <class T>
void print_type(T&& /*t*/)
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;
}
#endif
void print_stats()
{
    std::cout << "USING XSIMD\nSIMD SIZE: " << xsimd::simd_traits<double>::size << "\n\n";
#ifdef __GNUC__
    print_type(xt::xarray<double>());
    print_type(xt::xtensor<double, 2>());
#endif
}
#else
void print_stats()
{
    std::cout << "NOT USING XSIMD\n\n";
};
#endif


// Custom main function to print SIMD config
int main(int argc, char** argv)
{
    print_stats();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
}
