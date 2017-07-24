/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "benchmark_assign.hpp"
#include "benchmark_container.hpp"
#include "benchmark_math.hpp"
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

template <class OS>
void benchmark_assign(OS& out)
{
    
}
int main(int argc, char* argv[])
{
    std::cout << "Using steady_clock" << std::endl;
    std::cout << "period num: " << std::chrono::steady_clock::period::num << std::endl;
    std::cout << "period den: " << std::chrono::steady_clock::period::den << std::endl;
    std::cout << "steady = " << std::boolalpha << std::chrono::steady_clock::is_steady << std::endl;
    std::cout << std::endl;

    if (argc != 1)
    {
        if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")
        {
            std::cout << "Avalaible options:" << std::endl;
            std::cout << "assign    : run benchmark on tensor assign" << std::endl;
            std::cout << "container : run benchmark on container basic operations" << std::endl;
            std::cout << "view      : run benchmark on view basic operations" << std::endl;
            std::cout << "op        : run benchmark on arithmetic operations" << std::endl;
            std::cout << "exp       : run benchmark on exponential and logarithm functions" << std::endl;
            std::cout << "trigo     : run benchmark on trigonomeric functions" << std::endl;
            std::cout << "hyperbolic: run benchmark on hyperbolic functions" << std::endl;
            std::cout << "power     : run benchmark on power functions" << std::endl;
            std::cout << "rounding  : run benchmark on rounding functions" << std::endl;
        }
        else
        {
            for (int i = 1; i < argc; ++i)
            {
                std::string sarg = std::string(argv[i]);
                if (sarg == "assign")
                {
                    xt::assign::benchmark(std::cout);
                }
                else if (sarg == "container")
                {
                    benchmark_container(std::cout);
                }
                else if (sarg == "view")
                {
                    benchmark_views(std::cout);
                }
                else
                {
                    xt::math::benchmark_math(std::cout, sarg);
                }
            }
        }
    }
    else
    {
        xt::assign::benchmark(std::cout);
        benchmark_container(std::cout);
        benchmark_views(std::cout);
        xt::math::benchmark_math(std::cout);
    }
    return 0;
}
