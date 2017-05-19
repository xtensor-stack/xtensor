/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef BENCHMARK_VIEWS_HPP
#define BENCHMARK_VIEWS_HPP

#include "xtensor/xarray.hpp"
#include "xtensor/xstridedview.hpp"
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

        template <class E, class X>
        inline auto benchmark_strided_reducer(const E& x, E& res, const X& axes, std::size_t number)
        {
            auto start = std::chrono::steady_clock::now();

            using value_type = typename E::value_type;
            std::size_t stride = x.strides()[axes[0]];
            std::size_t offset_end = x.strides()[axes[0]] * x.shape()[axes[0]];
            std::size_t offset_iter = 0;
            if (axes[0] == 1)
            {
                offset_iter = x.strides()[0];
            }
            else if (axes[0] == 0)
            {
                offset_iter = x.strides()[1];
            }

            for (std::size_t i = 0; i < number; ++i)
            {
                for (std::size_t j = 0; j < res.shape()[0]; ++j)
                {
                    auto begin = x.raw_data() + (offset_iter * j);
                    auto end = begin + offset_end;
                    value_type temp = *begin;
                    begin += stride;
                    for (; begin < end; begin += stride)
                    {
                        temp += *begin;
                    }
                    res(j) = temp;
                }
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
            duration_type dsu0 = benchmark_strided_reducer(u, res0, axis0, number);
            duration_type dsu1 = benchmark_strided_reducer(u, res1, axis1, number);
            duration_type dsv0 = benchmark_strided_reducer(v, res1, axis0, number);
            duration_type dsv1 = benchmark_strided_reducer(v, res0, axis1, number);

            out << "************************" << std::endl;
            out << "* REDUCER BENCHMARK : " << " *" << std::endl;
            out << "************************" << std::endl << std::endl;

            out << "sum((10, 100000), 0): " << du0.count() << "ms" << std::endl;
            out << "sum((10, 100000), 1): " << du1.count() << "ms" << std::endl;
            out << "sum((100000, 10), 0): " << dv0.count() << "ms" << std::endl;
            out << "sum((100000, 10), 1): " << dv1.count() << "ms" << std::endl;
            out << "strided sum(10, 100000), 0): " << dsu0.count() << "ms" << std::endl;
            out << "strided sum(10, 100000), 1): " << dsu1.count() << "ms" << std::endl;
            out << "strided sum(100000, 10), 0): " << dsv0.count() << "ms" << std::endl;
            out << "strided sum(100000, 10), 1): " << dsv1.count() << "ms" << std::endl;
            out << std::endl;
        }
    }

    namespace stridedview
    {
        template <class E1, class E2>
        inline auto benchmark_stridedview(const E1& x, E2& res, std::size_t number)
        {
            auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < number; ++i)
            {
                res = transpose(x);
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

            xarray<double, layout_type::row_major> ar = xt::arange<double>(100000);
            ar.reshape({ 10, 20, 500 });
            xarray<double, layout_type::column_major> ac = xt::arange<double>(100000);
            ac.reshape({ 10, 20, 500 });
            
            xarray<double, layout_type::row_major> resr;
            resr.reshape({ 500, 20, 10 });

            xarray<double, layout_type::column_major> resc;
            resc.reshape({ 500, 20, 10 });

            duration_type darr = benchmark_stridedview(ar, resr, number);
            duration_type dacr = benchmark_stridedview(ac, resr, number);
            duration_type darc = benchmark_stridedview(ar, resc, number);
            duration_type dacc = benchmark_stridedview(ac, resc, number);

            out << "*****************************" << std::endl;
            out << "* STRIDED VIEW BENCHMARK : " << " *" << std::endl;
            out << "*****************************" << std::endl << std::endl;

            out << "RM - transpose RM(10, 20, 500): " << darr.count() << "ms" << std::endl;
            out << "RM - transpose CM(10, 20, 500): " << dacr.count() << "ms" << std::endl;
            out << "CM - transpose RM(10, 20, 500): " << darc.count() << "ms" << std::endl;
            out << "CM - transpose CM(10, 20, 500): " << dacc.count() << "ms" << std::endl;
            out << std::endl;
        }
    }
}

#endif
