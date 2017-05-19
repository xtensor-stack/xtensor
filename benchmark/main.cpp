/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "benchmark_container.hpp"
#include "benchmark_views.hpp"
#include <iostream>

template <class OS>
void benchmark_container(OS& out)
{
    xt::axpy_1d::benchmark<std::vector<double>>(out);
    xt::axpy_1d::benchmark<xt::uvector<double>>(out);
    xt::func::benchmark<std::vector<double>>(out);
    xt::func::benchmark<xt::uvector<double>>(out);
    xt::sum_assign::benchmark<std::vector<double>>(out);
    xt::sum_assign::benchmark<xt::uvector<double>>(out);
}

template <class OS>
void benchmark_views(OS& out)
{
    xt::reducer::benchmark(out);
    xt::stridedview::benchmark(out);
}

int main(int /*argc*/, char** /*argv*/)
{
    std::cout << "Using steady_clock" << std::endl;
    std::cout << "period num: " << std::chrono::steady_clock::period::num << std::endl;
    std::cout << "period den: " << std::chrono::steady_clock::period::den << std::endl;
    std::cout << "steady = " << std::boolalpha << std::chrono::steady_clock::is_steady << std::endl;
    std::cout << std::endl;

    //benchmark_container(std::cout);
    benchmark_views(std::cout);
    return 0;
}
