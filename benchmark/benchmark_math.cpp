/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <chrono>
#include <cstddef>
#include <map>
#include <string>

#include <benchmark/benchmark.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"

// For how many sizes should math functions be tested?
#define MATH_RANGE 64, 64

namespace xt
{
    namespace math
    {
        // TODO use a fixture here to avoid initializing arrays everytime anew ...

        /****************************
         * Benchmark initialization *
         ****************************/

        template <class V>
        inline void init_benchmark_data(V& lhs, V& rhs, std::size_t size0, std::size_t size1)
        {
            using T = typename V::value_type;
            for (std::size_t i = 0; i < size0; ++i)
            {
                for (std::size_t j = 0; j < size1; ++j)
                {
                    lhs(i, j) = T(0.5) * T(j) / T(j + 1) + std::sqrt(T(i)) * T(9.) / T(size1);
                    rhs(i, j) = T(10.2) / T(i + 2) + T(0.25) * T(j);
                }
            }
        }

        template <class V>
        inline void init_xtensor_benchmark(V& lhs, V& rhs, V& res, std::size_t size0, size_t size1)
        {
            lhs.resize({ size0, size1 });
            rhs.resize({ size0, size1 });
            res.resize({ size0, size1 });
            init_benchmark_data(lhs, rhs, size0, size1);
        }

        template <class V>
        inline void init_ext_benchmark(V& lhs, V& rhs, V& res, std::size_t size0, size_t size1)
        {
            lhs.resize(size0, size1);
            rhs.resize(size0, size1);
            res.resize(size0, size1);
            init_benchmark_data(lhs, rhs, size0, size1);
        }

        /***********************
         * Benchmark functions *
         ***********************/

        template <class F, class V>
        inline void math_xtensor_2(benchmark::State& state)
        {
            xtensor<double, 2> lhs, rhs, res;
            init_xtensor_benchmark(lhs, rhs, res, state.range(0), state.range(0));

            auto f = F();

            for (auto _ : state)
            {
                xt::noalias(res) = f(lhs, rhs);
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class F, class V>
        inline void math_xtensor_cpy_2(benchmark::State& state)
        {
            xtensor<double, 2> lhs, rhs, res;
            init_xtensor_benchmark(lhs, rhs, res, state.range(0), state.range(0));

            auto f = F();

            for (auto _ : state)
            {
                auto fct = f(lhs, rhs);
                std::copy(fct.storage_begin(), fct.storage_end(), res.storage_begin());
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class F, class V>
        inline void math_xtensor_1(benchmark::State& state)
        {
            xtensor<double, 2> lhs, rhs, res;
            init_xtensor_benchmark(lhs, rhs, res, state.range(0), state.range(0));

            auto f = F();

            for (auto _ : state)
            {
                xt::noalias(res) = f(lhs);
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class F>
        inline auto math_ref_2(benchmark::State& state)
        {
            auto f = F();
            xtensor<double, 2> lhs, rhs, res;
            init_xtensor_benchmark(lhs, rhs, res, state.range(0), state.range(0));
            size_t size = lhs.shape()[0] * lhs.shape()[1];

            for (auto _ : state)
            {
                for (std::size_t i = 0; i < size; ++i)
                {
                    res.data()[i] = f(lhs.data()[i], res.data()[i]);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        template <class F>
        inline void math_ref_1(benchmark::State& state)
        {
            auto f = F();
            xtensor<double, 2> lhs, rhs, res;
            init_xtensor_benchmark(lhs, rhs, res, state.range(0), state.range(0));
            size_t size = lhs.shape()[0] * lhs.shape()[1];

            for (auto _ : state)
            {
                for (std::size_t i = 0; i < size; ++i)
                {
                    res.data()[i] = f(lhs.data()[i]);
                }
                benchmark::DoNotOptimize(res.data());
            }
        }

        /**********************
         * Benchmark functors *
         **********************/

#define DEFINE_OP_FUNCTOR_2OP(OP, NAME)\
    struct NAME##_fn {\
        template <class T>\
        inline auto operator()(const T& lhs, const T& rhs) const { return lhs OP rhs; }\
        inline static std::string name() { return #NAME; }\
    }

#define DEFINE_FUNCTOR_1OP(FN)\
    struct FN##_fn {\
        template <class T>\
        inline auto operator()(const T& x) const { using std::FN; using xt::FN; return FN(x); }\
        inline static std::string name() { return #FN; }\
    }

#define DEFINE_FUNCTOR_2OP(FN)\
    struct FN##_fn{\
        template <class T>\
        inline auto operator()(const T&lhs, const T& rhs) const { using std::FN; using xt::FN; return FN(lhs, rhs); }\
        inline static std::string name() { return #FN; }\
    }

        DEFINE_OP_FUNCTOR_2OP(+, add);
        DEFINE_OP_FUNCTOR_2OP(-, sub);
        DEFINE_OP_FUNCTOR_2OP(*, mul);
        DEFINE_OP_FUNCTOR_2OP(/ , div);

        DEFINE_FUNCTOR_1OP(exp);
        DEFINE_FUNCTOR_1OP(exp2);
        DEFINE_FUNCTOR_1OP(expm1);
        DEFINE_FUNCTOR_1OP(log);
        DEFINE_FUNCTOR_1OP(log10);
        DEFINE_FUNCTOR_1OP(log2);
        DEFINE_FUNCTOR_1OP(log1p);

        DEFINE_FUNCTOR_1OP(sin);
        DEFINE_FUNCTOR_1OP(cos);
        DEFINE_FUNCTOR_1OP(tan);
        DEFINE_FUNCTOR_1OP(asin);
        DEFINE_FUNCTOR_1OP(acos);
        DEFINE_FUNCTOR_1OP(atan);

        DEFINE_FUNCTOR_1OP(sinh);
        DEFINE_FUNCTOR_1OP(cosh);
        DEFINE_FUNCTOR_1OP(tanh);
        DEFINE_FUNCTOR_1OP(asinh);
        DEFINE_FUNCTOR_1OP(acosh);
        DEFINE_FUNCTOR_1OP(atanh);

        DEFINE_FUNCTOR_2OP(pow);
        DEFINE_FUNCTOR_1OP(sqrt);
        DEFINE_FUNCTOR_1OP(cbrt);
        DEFINE_FUNCTOR_2OP(hypot);

        DEFINE_FUNCTOR_1OP(ceil);
        DEFINE_FUNCTOR_1OP(floor);
        DEFINE_FUNCTOR_1OP(trunc);
        DEFINE_FUNCTOR_1OP(round);
        DEFINE_FUNCTOR_1OP(nearbyint);
        DEFINE_FUNCTOR_1OP(rint);

        /********************
         * benchmark groups *
         ********************/

        BENCHMARK_TEMPLATE(math_ref_2, add_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_2, add_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_cpy_2, add_fn, xtensor<double, 2>)->Range(MATH_RANGE);

        BENCHMARK_TEMPLATE(math_ref_2, sub_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_2, sub_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_cpy_2, sub_fn, xtensor<double, 2>)->Range(MATH_RANGE);

        BENCHMARK_TEMPLATE(math_ref_2, mul_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_2, mul_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_cpy_2, mul_fn, xtensor<double, 2>)->Range(MATH_RANGE);

        BENCHMARK_TEMPLATE(math_ref_2, div_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_2, div_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_cpy_2, div_fn, xtensor<double, 2>)->Range(MATH_RANGE);

        BENCHMARK_TEMPLATE(math_ref_1, exp_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, exp_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, exp2_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, exp2_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, expm1_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, expm1_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, log_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, log_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, log2_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, log2_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, log10_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, log10_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, log1p_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, log1p_fn, xtensor<double, 2>)->Range(MATH_RANGE);

        BENCHMARK_TEMPLATE(math_ref_1, sin_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, sin_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, cos_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, cos_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, tan_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, tan_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, asin_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, asin_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, acos_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, acos_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, atan_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, atan_fn, xtensor<double, 2>)->Range(MATH_RANGE);

        BENCHMARK_TEMPLATE(math_ref_1, sinh_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, sinh_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, cosh_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, cosh_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, tanh_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, tanh_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, asinh_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, asinh_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, acosh_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, acosh_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, atanh_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, atanh_fn, xtensor<double, 2>)->Range(MATH_RANGE);

        BENCHMARK_TEMPLATE(math_ref_2, pow_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_2, pow_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, sqrt_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, sqrt_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, cbrt_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, cbrt_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_2, hypot_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_2, hypot_fn, xtensor<double, 2>)->Range(MATH_RANGE);

        BENCHMARK_TEMPLATE(math_ref_1, ceil_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, ceil_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, floor_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, floor_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, trunc_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, trunc_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, round_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, round_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, nearbyint_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, nearbyint_fn, xtensor<double, 2>)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_ref_1, rint_fn)->Range(MATH_RANGE);
        BENCHMARK_TEMPLATE(math_xtensor_1, rint_fn, xtensor<double, 2>)->Range(MATH_RANGE);
    }

    template <class T>
    void scalar_assign(benchmark::State& state)
    {
        T res;
        std::size_t sz = static_cast<std::size_t>(state.range(0));
        res.resize({sz, sz});
        for (auto _ : state)
        {
            res += typename T::value_type(1);
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class T>
    void scalar_assign_ref(benchmark::State& state)
    {
        T res;
        std::size_t sz = static_cast<std::size_t>(state.range(0));
        res.resize({sz, sz});
        for (auto _ : state)
        {
            auto szt = res.size();
            for (std::size_t i = 0; i < szt; ++i)
            {
                res.data()[i] += typename T::value_type(1);
            }
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class T>
    void boolean_func(benchmark::State& state)
    {
        T a, b;
        std::size_t sz = static_cast<std::size_t>(state.range(0));

        a.resize({sz, sz});
        b.resize({sz, sz});
        xtensor<bool, 2> res; res.resize({sz, sz});

        for (auto _ : state)
        {
            res = equal(a, b);
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class T>
    void boolean_func_ref(benchmark::State& state)
    {
        T a, b;
        std::size_t sz = static_cast<std::size_t>(state.range(0));

        a.resize({sz, sz});
        b.resize({sz, sz});
        xtensor<bool, 2> res; res.resize({sz, sz});

        for (auto _ : state)
        {
            auto szt = res.size();
            for (std::size_t i = 0; i < szt; ++i)
            {
                res.data()[i] = (a.data()[i] == b.data()[i]);
            }
            benchmark::DoNotOptimize(res.data());
        }
    }

    BENCHMARK_TEMPLATE(scalar_assign, xtensor<double, 2>)->Range(MATH_RANGE);
    BENCHMARK_TEMPLATE(scalar_assign_ref, xtensor<double, 2>)->Range(MATH_RANGE);
    BENCHMARK_TEMPLATE(boolean_func, xtensor<double, 2>)->Range(MATH_RANGE);
    BENCHMARK_TEMPLATE(boolean_func_ref, xtensor<double, 2>)->Range(MATH_RANGE);
}
