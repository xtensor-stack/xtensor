#ifndef BENCHMARK_VIEWS_HPP
#define BENCHMARK_VIEWS_HPP

#include "xtensor/xarray.hpp"
#include <cstddef>
#include <chrono>
#include <string>

namespace xt
{
    namespace reducer
    {
        template <class E, class X>
        inline auto benchmark_reducer(const E& x, E& res, const X& axes, std::size_t number)
        {
            auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < number; ++i)
            {
                res = sum(x, axes);
            }
            auto end = std::chrono::steady_clock::now();
            auto diff = end - start;
            return diff;
        }

        template <class OS>
        void benchmark(OS& out)
        {
            using duration_type = std::chrono::duration<double, std::milli>;
            std::size_t number = 100;

            xarray<double> u = ones<double>({ 10, 100000 });
            xarray<double> v = ones<double>({ 100000, 10 });

            std::vector<std::size_t> axis0 = { 0 };
            std::vector<std::size_t> axis1 = { 1 };

            xarray<double> res0;
            res0.reshape({ 100000 });
            xarray<double> res1;
            res1.reshape({ 10 });

            duration_type du0 = benchmark_reducer(u, res0, axis0, number);
            duration_type du1 = benchmark_reducer(u, res1, axis1, number);
            duration_type dv0 = benchmark_reducer(v, res1, axis0, number);
            duration_type dv1 = benchmark_reducer(v, res0, axis1, number);

            out << "************************" << std::endl;
            out << "* REDUCER BENCHMARK : " << " *" << std::endl;
            out << "************************" << std::endl << std::endl;

            out << "sum((10, 1000000), 0): " << du0.count() << "ms" << std::endl;
            out << "sum((10, 1000000), 1): " << du1.count() << "ms" << std::endl;
            out << "sum((1000000, 10), 0): " << dv0.count() << "ms" << std::endl;
            out << "sum((1000000, 10), 1): " << dv1.count() << "ms" << std::endl;
            out << std::endl;
        }
    }
}

#endif
