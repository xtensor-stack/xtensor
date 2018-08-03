/***************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 *                                                                                                                                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                                 *
 *                                                                                                                                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <benchmark/benchmark.h>

// #include "xtensor/xshape.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xstorage.hpp"
#include "xtensor/xutils.hpp"
#include "xtensor/xview.hpp"

namespace xt
{
    template <class T, std::size_t N>
    class simple_array
    {
    public:

        explicit simple_array(const std::array<ptrdiff_t, N>& shape)
            : m_shape(shape)
        {
            ptrdiff_t data_size = 1;
            m_strides[m_strides.size() - 1] = 1;
            for (std::ptrdiff_t i = ptrdiff_t(m_strides.size()) - 1; i > 0; --i)
            {
                data_size *= static_cast<ptrdiff_t>(shape[i]);
                m_strides[i - 1] = data_size;
            }
            data_size *= shape[0];
            memory.resize(data_size);
        }

        template <class... Args>
        T& operator()(Args... args)
        {
            std::array<ptrdiff_t, sizeof...(Args)> idx({static_cast<long>(args)...});
            static_assert(sizeof...(Args) == N, "too few or too many indices!");
            ptrdiff_t offset = 0;
            for (std::size_t i = 0; i < N; ++i)
            {
                offset += m_strides[i] * idx[i];
            }
            return memory[offset];
        }
        xt::uvector<T> memory;
        std::array<ptrdiff_t, N> m_shape, m_strides;
    };

    void xview_access_calc(benchmark::State &state) {
        xt::xtensor<double, 4> A = xt::random::rand<double>({100, 100, 4, 4});
        xt::xtensor<double, 3> elemvec = xt::random::rand<double>({100, 4, 4});
        xt::xtensor<double, 2> eps = xt::empty<double>({2, 2});

        for (auto _ : state)
        {
            for (size_t e = 0; e < 100; ++e)
            {
                // alias element vector (e.g. nodal displacements)
                auto u = xt::view(elemvec, e, xt::all(), xt::all());
                for (size_t k = 0; k < 100; ++k)
                {
                    auto dNx = xt::view(A, e, k, xt::all(), xt::all());
                    // - evaluate symmetrized dyadic product (loops unrolled for efficiency)
                    //       grad(i,j) += dNx(m,i) * u(m,j)
                    //       eps (j,i)      = 0.5 * ( grad(i,j) + grad(j,i) )
                    eps(0, 0) = dNx(0, 0) * u(0, 0) + dNx(1, 0) * u(1, 0) +
                                dNx(2, 0) * u(2, 0) + dNx(3, 0) * u(3, 0);
                    eps(1, 1) = dNx(0, 1) * u(0, 1) + dNx(1, 1) * u(1, 1) +
                                dNx(2, 1) * u(2, 1) + dNx(3, 1) * u(3, 1);
                    eps(0, 1) =
                              (dNx(0, 1) * u(0, 0) + dNx(1, 1) * u(1, 0) + dNx(2, 1) * u(2, 0) +
                               dNx(3, 1) * u(3, 0) + dNx(0, 0) * u(0, 1) + dNx(1, 0) * u(1, 1) +
                               dNx(2, 0) * u(2, 1) + dNx(3, 0) * u(3, 1)) /
                              2.;
                    eps(1, 0) = eps(0, 1);
                    benchmark::DoNotOptimize(eps.storage());
                }
            }
        }
    }

    void raw_access_calc(benchmark::State &state) {
        xt::xtensor<double, 4> A = xt::random::rand<double>({100, 100, 4, 4});
        xt::xtensor<double, 3> elemvec = xt::random::rand<double>({100, 4, 4});
        xt::xtensor<double, 2> eps = xt::empty<double>({2, 2});

        for (auto _ : state)
        {
            for (size_t e = 0; e < 100; ++e)
            {
                for (size_t k = 0; k < 100; ++k)
                {
                    // - evaluate symmetrized dyadic product (loops unrolled for efficiency)
                    //       grad(i,j) += dNx(m,i) * u(m,j)
                    //       eps (j,i)      = 0.5 * ( grad(i,j) + grad(j,i) )
                    eps(0, 0) = A(e, k, 0, 0) * elemvec(e, 0, 0) + A(e, k, 1, 0) * elemvec(e, 1, 0) +
                                A(e, k, 2, 0) * elemvec(e, 2, 0) + A(e, k, 3, 0) * elemvec(e, 3, 0);
                    eps(1, 1) = A(e, k, 0, 1) * elemvec(e, 0, 1) + A(e, k, 1, 1) * elemvec(e, 1, 1) +
                                A(e, k, 2, 1) * elemvec(e, 2, 1) + A(e, k, 3, 1) * elemvec(e, 3, 1);
                    eps(0, 1) = (A(e, k, 0, 1) * elemvec(e, 0, 0) + A(e, k, 1, 1) * elemvec(e, 1, 0) +
                                 A(e, k, 2, 1) * elemvec(e, 2, 0) + A(e, k, 3, 1) * elemvec(e, 3, 0) +
                                 A(e, k, 0, 0) * elemvec(e, 0, 1) + A(e, k, 1, 0) * elemvec(e, 1, 1) +
                                 A(e, k, 2, 0) * elemvec(e, 2, 1) + A(e, k, 3, 0) * elemvec(e, 3, 1)) /
                                 2.;
                    eps(1, 0) = eps(0, 1);
                    benchmark::DoNotOptimize(eps.storage());
                }
            }
        }
    }

    void unchecked_access_calc(benchmark::State &state) {
        xt::xtensor<double, 4> A = xt::random::rand<double>({100, 100, 4, 4});
        xt::xtensor<double, 3> elemvec = xt::random::rand<double>({100, 4, 4});
        xt::xtensor<double, 2> eps = xt::empty<double>({2, 2});

        for (auto _ : state)
        {
            for (size_t e = 0; e < 100; ++e)
            {
                for (size_t k = 0; k < 100; ++k)
                {
                    // - evaluate symmetrized dyadic product (loops unrolled for efficiency)
                    //       grad(i,j) += dNx(m,i) * u(m,j)
                    //       eps (j,i)      = 0.5 * ( grad(i,j) + grad(j,i) )
                    eps.unchecked(0, 0) =
                        A.unchecked(e, k, 0, 0) * elemvec.unchecked(e, 0, 0) +
                        A.unchecked(e, k, 1, 0) * elemvec.unchecked(e, 1, 0) +
                        A.unchecked(e, k, 2, 0) * elemvec.unchecked(e, 2, 0) +
                        A.unchecked(e, k, 3, 0) * elemvec.unchecked(e, 3, 0);
                    eps.unchecked(1, 1) =
                        A.unchecked(e, k, 0, 1) * elemvec.unchecked(e, 0, 1) +
                        A.unchecked(e, k, 1, 1) * elemvec.unchecked(e, 1, 1) +
                        A.unchecked(e, k, 2, 1) * elemvec.unchecked(e, 2, 1) +
                        A.unchecked(e, k, 3, 1) * elemvec.unchecked(e, 3, 1);
                    eps.unchecked(0, 1) =
                        (A.unchecked(e, k, 0, 1) * elemvec.unchecked(e, 0, 0) +
                         A.unchecked(e, k, 1, 1) * elemvec.unchecked(e, 1, 0) +
                         A.unchecked(e, k, 2, 1) * elemvec.unchecked(e, 2, 0) +
                         A.unchecked(e, k, 3, 1) * elemvec.unchecked(e, 3, 0) +
                         A.unchecked(e, k, 0, 0) * elemvec.unchecked(e, 0, 1) +
                         A.unchecked(e, k, 1, 0) * elemvec.unchecked(e, 1, 1) +
                         A.unchecked(e, k, 2, 0) * elemvec.unchecked(e, 2, 1) +
                         A.unchecked(e, k, 3, 0) * elemvec.unchecked(e, 3, 1)) /
                        2.;
                    eps.unchecked(1, 0) = eps.unchecked(0, 1);
                    benchmark::DoNotOptimize(eps.storage());
                }
            }
        }
    }

    void simplearray_access_calc(benchmark::State &state) {
        simple_array<double, 4> A(std::array<ptrdiff_t, 4>{100, 100, 4, 2});
        simple_array<double, 3> elemvec(std::array<ptrdiff_t, 3>{100, 4, 2});
        simple_array<double, 2> eps(std::array<ptrdiff_t, 2>{2, 2});

        for (auto _ : state)
        {
            for (size_t e = 0; e < 100; ++e)
            {
                for (size_t k = 0; k < 100; ++k)
                {
                    // - evaluate sy mmetrized dyadic product (loops unrolled for efficiency)
                    //             grad(i,j) += dNx(m,i) * u(m,j)
                    //             eps (j,i)            = 0.5 * ( grad(i,j) + grad(j,i) )
                    eps(0, 0) = A(e, k, 0, 0) * elemvec(e, 0, 0) +
                                                        A(e, k, 1, 0) * elemvec(e, 1, 0) +
                                                        A(e, k, 2, 0) * elemvec(e, 2, 0) +
                                                        A(e, k, 3, 0) * elemvec(e, 3, 0);
                    eps(1, 1) = A(e, k, 0, 1) * elemvec(e, 0, 1) +
                                                        A(e, k, 1, 1) * elemvec(e, 1, 1) +
                                                        A(e, k, 2, 1) * elemvec(e, 2, 1) +
                                                        A(e, k, 3, 1) * elemvec(e, 3, 1);
                    eps(0, 1) = (A(e, k, 0, 1) * elemvec(e, 0, 0) +
                                                         A(e, k, 1, 1) * elemvec(e, 1, 0) +
                                                         A(e, k, 2, 1) * elemvec(e, 2, 0) +
                                                         A(e, k, 3, 1) * elemvec(e, 3, 0) +
                                                         A(e, k, 0, 0) * elemvec(e, 0, 1) +
                                                         A(e, k, 1, 0) * elemvec(e, 1, 1) +
                                                         A(e, k, 2, 0) * elemvec(e, 2, 1) +
                                                         A(e, k, 3, 0) * elemvec(e, 3, 1)) /
                                                        2.;
                    eps(1, 0) = eps(0, 1);
                    benchmark::DoNotOptimize(eps.memory);
                }
            }
        }
    }

    BENCHMARK(raw_access_calc);
    BENCHMARK(unchecked_access_calc);
    BENCHMARK(simplearray_access_calc);
    BENCHMARK(xview_access_calc);
}