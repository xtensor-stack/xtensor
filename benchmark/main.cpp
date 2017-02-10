#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

#include <cstddef>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace xt
{

    namespace axpy_1d
    {
        template <class E>
        inline auto benchmark_iteration(const E& x, const E& y, E& res, typename E::value_type a, std::size_t number)
        {
            auto start = std::chrono::steady_clock::now();
            for(std::size_t i = 0; i < number; ++i)
            {
                auto iterx = x.begin();
                auto itery = y.begin();
                for(auto iter = res.begin(); iter != res.end(); ++iter, ++iterx, ++itery)
                {
                    *iter = a * (*iterx) + (*itery);
                }
            }
            auto end = std::chrono::steady_clock::now();
            auto diff = end - start;
            return diff;
        }

        template <class E>
        inline auto benchmark_xiteration(const E& x, const E& y, E& res, typename E::value_type a, std::size_t number)
        {
            auto start = std::chrono::steady_clock::now();
            for(std::size_t i = 0; i < number; ++i)
            {
                auto iterx = x.xbegin();
                auto itery = y.xbegin();
                for(auto iter = res.xbegin(); iter != res.xend(); ++iter, ++iterx, ++itery)
                {
                    *iter = a * (*iterx) + (*itery);
                }
            }
            auto end = std::chrono::steady_clock::now();
            auto diff = end - start;
            return diff;
        }

        template <class E>
        inline auto benchmark_indexing(const E& x, const E& y, E& res, typename E::value_type a, std::size_t number)
        {
            using size_type = typename E::size_type;
            auto start = std::chrono::steady_clock::now();
            for(std::size_t i = 0; i < number; ++i)
            {
                size_type n = x.size();
                for(size_type i = 0; i < n; ++i)
                {
                    res(i) = a * x(i) + y(i);
                }
            }
            auto end = std::chrono::steady_clock::now();
            auto diff = end - start;
            return diff;
        }

        template <class E>
        inline void init_benchmark(E& x, E& y, E& res, typename E::size_type size)
        {
            x.reshape({ size });
            y.reshape({ size });
            res.reshape({ size });

            using value_type = typename E::value_type;
            using size_type = typename E::size_type;
            for(size_type i = 0; i < size; ++i)
            {
                x(i) = 0.5 + value_type(i);
                y(i) = 0.25 * value_type(i);
            }
        }

        void benchmark()
        {
            using duration_type = std::chrono::duration<double, std::milli>;
            using size_type = xarray<double>::size_type;
            constexpr size_type size = 1000;
            constexpr size_type number = 10000;
            double a = 2.7;

            xarray<double> ax, ay, ares;
            init_benchmark(ax, ay, ares, size);

            xtensor<double, 1> tx, ty, tres;
            init_benchmark(tx, ty, tres, size);

            benchmark_iteration(ax, ay, ares, a, 10);

            duration_type aiter = benchmark_iteration(ax, ay, ares, a, number);
            duration_type titer = benchmark_iteration(tx, ty, tres, a, number);
            duration_type axiter = benchmark_xiteration(ax, ay, ares, a, number);
            duration_type txiter = benchmark_xiteration(tx, ty, tres, a, number);
            duration_type aindex = benchmark_indexing(ax, ay, ares, a, number);
            duration_type tindex = benchmark_indexing(tx, ty, tres, a, number);

            std::cout << "***************************" << std::endl;
            std::cout << "*    AXPY 1D BENCHMARK    *" << std::endl;
            std::cout << "***************************" << std::endl << std::endl;

            std::cout << "xarray   iteration: " << aiter.count() << "ms" << std::endl;
            std::cout << "xtensor  iteration: " << titer.count() << "ms" << std::endl;
            std::cout << "xarray  xiteration: " << axiter.count() << "ms" << std::endl;
            std::cout << "xtensor xiteraiton: " << txiter.count() << "ms" << std::endl;
            std::cout << "xarray    indexing: " << aindex.count() << "ms" << std::endl;
            std::cout << "xtensor   indexing: " << tindex.count() << "ms" << std::endl;
        }
    }
}

int main(int argc, char* argv[])
{
    std::cout << "Using steady_clock" << std::endl;
    std::cout << "period num: " << std::chrono::steady_clock::period::num << std::endl;
    std::cout << "period den: " << std::chrono::steady_clock::period::den << std::endl;
    std::cout << "steady = " << std::boolalpha << std::chrono::steady_clock::is_steady << std::endl;
    std::cout << std::endl;

    xt::axpy_1d::benchmark();

    return 0;
}
