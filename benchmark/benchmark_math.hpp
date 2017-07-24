/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef BENCHMARK_MATH_HPP
#define BENCHMARK_MATH_HPP

#include <chrono>
#include <cstddef>
#include <map>
#include <string>
#include "xtensor/xarray.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"

namespace xt
{
    namespace math
    {
        using duration_type = std::chrono::duration<double, std::milli>;

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
            lhs.reshape({ size0, size1 });
            rhs.reshape({ size0, size1 });
            res.reshape({ size0, size1 });
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
        inline duration_type benchmark_xtensor_ref(F f, const V& lhs, V& res, std::size_t number)
        {
            size_t size = lhs.shape()[0] * lhs.shape()[1];
            duration_type t_res = duration_type::max();
            for (std::size_t count = 0; count < number; ++count)
            {
                auto start = std::chrono::steady_clock::now();
                for (std::size_t i = 0; i < size; ++i)
                {
                    res.data()[i] = f(lhs.data()[i]);
                }
                auto end = std::chrono::steady_clock::now();
                auto tmp = end - start;
                t_res = tmp < t_res ? tmp : t_res;
            }
            return t_res;
        }

        template <class F, class V>
        inline duration_type benchmark_xtensor_ref(F f, const V& lhs, const V& rhs, V& res, std::size_t number)
        {
            size_t size = lhs.shape()[0] * lhs.shape()[1];
            duration_type t_res = duration_type::max();
            for (std::size_t count = 0; count < number; ++count)
            {
                auto start = std::chrono::steady_clock::now();
                for (std::size_t i = 0; i < size; ++i)
                {
                    res.data()[i] = f(lhs.data()[i], rhs.data()[i]);
                }
                auto end = std::chrono::steady_clock::now();
                auto tmp = end - start;
                t_res = tmp < t_res ? tmp : t_res;
            }
            return t_res;
        }

        template <class F, class V>
        inline duration_type benchmark_xtensor(F f, const V& lhs, V& res, std::size_t number)
        {
            size_t s0 = lhs.shape()[0];
            size_t s1 = lhs.shape()[1];
            duration_type t_res = duration_type::max();
            for (std::size_t count = 0; count < number; ++count)
            {
                auto start = std::chrono::steady_clock::now();
                xt::noalias(res) = f(lhs);
                auto end = std::chrono::steady_clock::now();
                auto tmp = end - start;
                t_res = tmp < t_res ? tmp : t_res;
            }
            return t_res;
        }

        template <class F, class V>
        inline duration_type benchmark_xtensor(F f, const V& lhs, const V& rhs, V& res, std::size_t number)
        {
            size_t s0 = lhs.shape()[0];
            size_t s1 = lhs.shape()[1];
            duration_type t_res = duration_type::max();
            for (std::size_t count = 0; count < number; ++count)
            {
                auto start = std::chrono::steady_clock::now();
                xt::noalias(res) = f(lhs, rhs);
                auto end = std::chrono::steady_clock::now();
                auto tmp = end - start;
                t_res = tmp < t_res ? tmp : t_res;
            }
            return t_res;
        }

        /*********************
         * Benchmark runners *
         *********************/

        template <class F, class OS>
        void run_benchmark_1op(F f, OS& out, std::size_t size0, std::size_t size1, std::size_t iter)
        {
            xtensor<double, 2> lhs, rhs, res;
            init_xtensor_benchmark(lhs, rhs, res, size0, size1);

            duration_type ref_time = benchmark_xtensor_ref(f, lhs, res, iter);
            duration_type xt_time = benchmark_xtensor(f, lhs, res, iter);

            out << "=====================" << std::endl;
            out << F::name() << std::endl;
            out << "reference: " << ref_time.count() << "ms" << std::endl;
            out << "xtensor  : " << xt_time.count() << "ms" << std::endl;
            out << "=====================" << std::endl;
        }

        template <class F, class OS>
        void run_benchmark_2op(F f, OS& out, std::size_t size0, std::size_t size1, std::size_t iter)
        {
            xtensor<double, 2> lhs, rhs, res;
            init_xtensor_benchmark(lhs, rhs, res, size0, size1);

            duration_type ref_time = benchmark_xtensor_ref(f, lhs, rhs, res, iter);
            duration_type xt_time = benchmark_xtensor(f, lhs, rhs, res, iter);

            out << "=====================" << std::endl;
            out << F::name() << std::endl;
            out << "reference: " << ref_time.count() << "ms" << std::endl;
            out << "xtensor  : " << xt_time.count() << "ms" << std::endl;
            out << "=====================" << std::endl;
        }

        /**********************
         * Benchmark functors *
         **********************/

#define DEFINE_OP_FUNCTOR_2OP(OP, NAME)\
    struct NAME##_fn {\
        template <class T>\
        inline T operator()(const T& lhs, const T& rhs) const { return lhs OP rhs; }\
        inline static std::string name() { return #NAME; }\
    }

#define DEFINE_FUNCTOR_1OP(FN)\
    struct FN##_fn {\
        template <class T>\
        inline T operator()(const T& x) const { using std::FN; using xt::FN; return FN(x); }\
        inline static std::string name() { return #FN; }\
    }

#define DEFINE_FUNCTOR_2OP(FN)\
    struct FN##_fn{\
        template <class T>\
        inline T operator()(const T&lhs, const T& rhs) const { using std::FN; using xt::FN; return FN(lhs, rhs); }\
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

        template <class OS>
        void benchmark_arithmetic(OS& out, std::size_t size0, std::size_t size1, std::size_t iter)
        {
            run_benchmark_2op(add_fn(), out, size0, size1, iter);
            run_benchmark_2op(sub_fn(), out, size0, size1, iter);
            run_benchmark_2op(mul_fn(), out, size0, size1, iter);
            run_benchmark_2op(div_fn(), out, size0, size1, iter);
        }

        template <class OS>
        void benchmark_exp_log(OS& out, std::size_t size0, std::size_t size1, std::size_t iter)
        {
            run_benchmark_1op(exp_fn(), out, size0, size1, iter);
            run_benchmark_1op(exp2_fn(), out, size0, size1, iter);
            run_benchmark_1op(expm1_fn(), out, size0, size1, iter);
            run_benchmark_1op(log_fn(), out, size0, size1, iter);
            run_benchmark_1op(log2_fn(), out, size0, size1, iter);
            run_benchmark_1op(log10_fn(), out, size0, size1, iter);
            run_benchmark_1op(log1p_fn(), out, size0, size1, iter);
        }

        template <class OS>
        void benchmark_trigo(OS& out, std::size_t size0, std::size_t size1, std::size_t iter)
        {
            run_benchmark_1op(sin_fn(), out, size0, size1, iter);
            run_benchmark_1op(cos_fn(), out, size0, size1, iter);
            run_benchmark_1op(tan_fn(), out, size0, size1, iter);
            run_benchmark_1op(asin_fn(), out, size0, size1, iter);
            run_benchmark_1op(acos_fn(), out, size0, size1, iter);
            run_benchmark_1op(atan_fn(), out, size0, size1, iter);
        }

        template <class OS>
        void benchmark_hyperbolic(OS& out, std::size_t size0, std::size_t size1, std::size_t iter)
        {
            run_benchmark_1op(sinh_fn(), out, size0, size1, iter);
            run_benchmark_1op(cosh_fn(), out, size0, size1, iter);
            run_benchmark_1op(tanh_fn(), out, size0, size1, iter);
            run_benchmark_1op(asinh_fn(), out, size0, size1, iter);
            run_benchmark_1op(acosh_fn(), out, size0, size1, iter);
            run_benchmark_1op(atanh_fn(), out, size0, size1, iter);
        }

        template <class OS>
        void benchmark_power(OS& out, std::size_t size0, std::size_t size1, std::size_t iter)
        {
            run_benchmark_2op(pow_fn(), out, size0, size1, iter);
            run_benchmark_1op(sqrt_fn(), out, size0, size1, iter);
            run_benchmark_1op(cbrt_fn(), out, size0, size1, iter);
            run_benchmark_2op(hypot_fn(), out, size0, size1, iter);
        }

        template <class OS>
        void benchmark_rounding(OS& out, std::size_t size0, std::size_t size1, std::size_t iter)
        {
            run_benchmark_1op(ceil_fn(), out, size0, size1, iter);
            run_benchmark_1op(floor_fn(), out, size0, size1, iter);
            run_benchmark_1op(trunc_fn(), out, size0, size1, iter);
            run_benchmark_1op(round_fn(), out, size0, size1, iter);
            run_benchmark_1op(nearbyint_fn(), out, size0, size1, iter);
            run_benchmark_1op(rint_fn(), out, size0, size1, iter);
        }

        template <class OS>
        using benchmark_function = void(*)(OS&, std::size_t, std::size_t, std::size_t);

        template <class OS>
        using benchmark_map = std::map < std::string, benchmark_function<OS>>;

        template <class OS>
        const benchmark_map<OS>& get_benchmark_map()
        {
            static benchmark_map<OS> bm;
            if (bm.empty())
            {
                bm["op"] = &benchmark_arithmetic<OS>;
                bm["exp"] = &benchmark_exp_log<OS>;
                bm["trigo"] = &benchmark_trigo<OS>;
                bm["hyperbolic"] = &benchmark_hyperbolic<OS>;
                bm["power"] = &benchmark_power<OS>;
                bm["rounding"] = &benchmark_rounding<OS>;
            }
            return bm;
        }

        template <class OS>
        void benchmark_math(OS& out, const std::string& meth = "")
        {
            std::size_t size0 = 20;
            std::size_t size1 = 500;
            std::size_t nb_iter = 1000;

            const auto& bm = get_benchmark_map<OS>();
            if (meth != "")
            {
                auto iter = bm.find(meth);
                if (iter != bm.end())
                    (iter->second)(out, size0, size1, nb_iter);
            }
            else
            {
                for (auto v : bm)
                {
                    (v.second)(out, size0, size1, nb_iter);
                }
            }
        }
    }

}

#endif
