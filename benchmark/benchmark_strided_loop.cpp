/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef BENCHMARK_STRIDED_LOOP_HPP
#define BENCHMARK_STRIDED_LOOP_HPP

#include <benchmark/benchmark.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xreducer.hpp"

namespace xt
{
    namespace wip
    {
        template <class S1, class S2>
        std::size_t check_strides(const S1& s1, S2& s2)
        {
            // Indices are faster than reverse iterators
            std::size_t s1_index = s1.size();
            std::size_t s2_index = s2.size();

            for (; s2_index != 0; --s1_index, --s2_index)
            {
                if (s1[s1_index - 1] != s2[s2_index - 1])
                {
                    break;
                }
            }
            return s1_index;
        }

        template <class S>
        struct check_strides_fct
        {
            check_strides_fct(const S& strides)
                : max_strides(strides)
            {
            }

            template <class T>
            void operator()(const T& el)
            {
                auto var = check_strides(max_strides, el.strides());
                if (var > cut)
                {
                    cut = var;
                }
            }

            template <class T>
            void operator()(const xt::xscalar<T>& el)
            {
            }

            std::size_t cut = 0;
            const S& max_strides;
        };

        template <class E, class... T>
        auto get_loop_sizes(const E& res, const std::tuple<T...>& args)
        {
            check_strides_fct<typename E::strides_type> s_fct(res.strides());
            xt::for_each(s_fct, args);

            std::size_t cut = s_fct.cut;
            std::size_t inner_loop_size = 1;
            std::size_t outer_loop_size = 1;
            std::size_t i = 0;

            for (; i < cut; ++i)
            {
                outer_loop_size *= res.shape()[i];
            }
            for (; i < res.dimension(); ++i)
            {
                inner_loop_size *= res.shape()[i];
            }

            return std::make_tuple(inner_loop_size, outer_loop_size, cut);
        }

        template <class F, class E>
        struct super_stepper
        {
            super_stepper(F& f, E& ex)
                : m_f(f), res(ex)
            {
                std::fill(offsets.begin(), offsets.end(), 0);
                std::tie(inner_loop_size, outer_loop_size, cut) = get_loop_sizes(ex, m_f.arguments());

                std::size_t simd_size = 4;
                simd_end = inner_loop_size & ~(simd_size - 1);

                index.resize(cut);
            }

            inline void next_idx()
            {
                std::size_t i = index.size();
                for (; i > 0; --i)
                {
                    if (index[i - 1] > res.shape()[i - 1])
                    {
                        index[i - 1] = 0;
                    }
                    else
                    {
                        index[i - 1]++;
                        break;
                    }
                }
            }

            inline void outer_loop()
            {
                next_idx();
                xt::enumerate([&](std::size_t i, const auto& el)
                {
                    if (cut <= res.dimension() - el.dimension())
                    {
                        offsets[i] = 0;
                    }
                    else
                    {
                        int overlap = res.dimension() - el.dimension();
                        auto begin = index.begin() + overlap;
                        offsets[i] = std::inner_product(begin, index.end(), el.strides().begin(), std::size_t(0));
                    }
                },
                m_f.arguments());
            }
         
            inline void inner_loop()
            {
                std::size_t i = 0;

                // for (; i < simd_end; i += 4)
                // {
                //     res.template store_simd<xsimd::unaligned_mode, xsimd::batch<double, 4>>(out_idx, m_f.template load_simd<xsimd::unaligned_mode, xsimd::batch<double, 4>>(offsets));
                //     for (std::size_t ix = 0; ix < offsets.size(); ++ix)
                //     {
                //         offsets[ix] += 4;
                //     }
                //     out_idx += 4;
                // }
                for (; i < inner_loop_size; ++i)
                {
                    res.data_element(out_idx++) = m_f.data_element(offsets);
                    for (std::size_t ix = 0; ix < offsets.size(); ++ix)
                    {
                        offsets[ix] += 1;
                    }
                }
            }

            inline void run()
            {
                for (std::size_t i = 0; i < outer_loop_size; ++i)
                {
                    inner_loop();
                    outer_loop();
                }
            }

            xt::dynamic_shape<std::size_t> index;
            std::size_t inner_loop_size, outer_loop_size, cut, simd_end;

            std::size_t out_idx = 0;

            F& m_f;
            E& res;
            std::array<std::size_t, std::tuple_size<std::decay_t<decltype(m_f.arguments())>>::value> offsets;
        };

        template <class F, class E>
        auto get_super_stepper(F& f, E& e)
        {
            return super_stepper<F, E>(f, e);
        }

    }

    namespace strided
    {
        using namespace xt::wip;
        void benchmark_strided_loop(benchmark::State& state)
        {
            auto a = xt::xtensor<double, 3>::from_shape({5, 3, 2});
            auto b = xt::xtensor<double, 2>::from_shape({3, 2});
            while (state.KeepRunning())
            {
                auto f = a + b;
                auto c = xt::xtensor<double, 3>::from_shape(f.shape());
                auto s = get_super_stepper(f, c);
                s.run();
                benchmark::DoNotOptimize(c.data());
            }
        }

        void benchmark_strided_loop_xtensorf(benchmark::State& state)
        {
            auto a = xt::xtensorf<double, xt::xshape<5, 3, 2>>();
            auto b = xt::xtensorf<double, xt::xshape<3, 2>>();
            while (state.KeepRunning())
            {
                auto f = a + b;
                auto c = xt::xtensor<double, 3>::from_shape(f.shape());
                auto s = get_super_stepper(f, c);
                s.run();
                benchmark::DoNotOptimize(c.data());
            }
        }

        void benchmark_regular_bc(benchmark::State& state)
        {
            auto a = xt::xtensor<double, 3>::from_shape({5, 3, 2});
            auto b = xt::xtensor<double, 2>::from_shape({3, 2});
            while (state.KeepRunning())
            {
                auto f = a + b;
                xt::xtensor<double, 3> c = f;
                benchmark::DoNotOptimize(c.data());
            }
        }

        void benchmark_op(benchmark::State& state)
        {
            auto a = xt::xtensor<double, 1>::from_shape({5 * 3 * 2});
            auto b = xt::xtensor<double, 1>::from_shape({5 * 3 * 2});
            while (state.KeepRunning())
            {
                auto f = a + b;
                xt::xtensor<double, 1> c = f;
                benchmark::DoNotOptimize(c.data());
            }
        }

        void benchmark_manual_broadcast_xtensorf(benchmark::State& state)
        {
            auto a = xt::xtensorf<double, xt::xshape<5, 3, 2>>();
            auto b = xt::xtensorf<double, xt::xshape<3, 2>>();
            while (state.KeepRunning())
            {
                xt::xtensor<double, 3> c = xt::xtensor<double, 3>::from_shape({5, 3, 2});
                for (std::size_t i = 0; i < a.shape()[0]; ++i)
                    for (std::size_t j = 0; j < a.shape()[1]; ++j)
                        for (std::size_t k = 0; k < a.shape()[2]; ++k)
                            c(i, j, k) = a(i, j, k) + b(i, j, k);
            }
        }

        void benchmark_manual_broadcast(benchmark::State& state)
        {
            auto a = xt::xtensor<double, 3>::from_shape({5, 3, 2});
            auto b = xt::xtensor<double, 2>::from_shape({3, 2});
            while (state.KeepRunning())
            {
                xt::xtensor<double, 3> c = xt::xtensor<double, 3>::from_shape({5, 3, 2});
                for (std::size_t i = 0; i < a.shape()[0]; ++i)
                    for (std::size_t j = 0; j < a.shape()[1]; ++j)
                        for (std::size_t k = 0; k < a.shape()[2]; ++k)
                            c(i, j, k) = a(i, j, k) + b(i, j, k);
            }
        }

        BENCHMARK(benchmark_strided_loop);
        BENCHMARK(benchmark_regular_bc);
        BENCHMARK(benchmark_op);
        BENCHMARK(benchmark_manual_broadcast);
        BENCHMARK(benchmark_manual_broadcast_xtensorf);
        BENCHMARK(benchmark_strided_loop_xtensorf);
    }
}

#endif