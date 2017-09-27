/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XNORM_HPP
#define XNORM_HPP

#include <cmath>
#include <cstdlib>   // std::abs(int) prior to C++ 17
#include <complex>

#include "xconcepts.hpp"
#include "xutils.hpp"
#include "xmath.hpp"
#include "xoperation.hpp"

namespace xt
{
    /*************************************
     * norm functions for built-in types *
     *************************************/

        /** \brief The L2-norm of a numerical object.

            For scalar types: implemented as <tt>abs(t)</tt><br>
            otherwise: implemented as <tt>sqrt(norm_sq(t))</tt>.
        */
    template <class T>
    inline auto norm_l2(T && t)
    {
        using std::sqrt;
        return sqrt(norm_sq(std::forward<T>(t)));
    }

    #define XTENSOR_DEFINE_SIGNED_NORMS(T)                       \
        inline auto                                              \
        norm_lp(T t, double p) -> decltype(std::abs(t))          \
        {                                                        \
            return p == 0.0                                      \
                      ? (t != 0)                                 \
                      : std::abs(t);                             \
        }                                                        \
        inline size_t norm_l0(T t)   { return (t != 0); }        \
        inline auto   norm_l1(T t)   { return std::abs(t); }     \
        inline auto   norm_l2(T t)   { return std::abs(t); }     \
        inline auto   norm_linf(T t) { return std::abs(t); }     \
        inline auto   norm_sq(T t)   { return t*t; }

    XTENSOR_DEFINE_SIGNED_NORMS(signed char)
    XTENSOR_DEFINE_SIGNED_NORMS(short)
    XTENSOR_DEFINE_SIGNED_NORMS(int)
    XTENSOR_DEFINE_SIGNED_NORMS(long)
    XTENSOR_DEFINE_SIGNED_NORMS(long long)
    XTENSOR_DEFINE_SIGNED_NORMS(float)
    XTENSOR_DEFINE_SIGNED_NORMS(double)
    XTENSOR_DEFINE_SIGNED_NORMS(long double)

    #undef XTENSOR_DEFINE_SIGNED_NORMS

    #define XTENSOR_DEFINE_UNSIGNED_NORMS(T)                   \
        inline T norm_lp(T t, double p)                        \
        {                                                      \
            return p == 0.0                                    \
                      ? t != 0                                 \
                      : t;                                     \
        }                                                      \
        inline T    norm_l0(T t)   { return t != 0 ? 1 : 0; }  \
        inline T    norm_l1(T t)   { return t; }               \
        inline T    norm_l2(T t)   { return t; }               \
        inline T    norm_linf(T t) { return t; }               \
        inline auto norm_sq(T t)   { return t*t; }

    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned char)
    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned short)
    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned int)
    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned long)
    XTENSOR_DEFINE_UNSIGNED_NORMS(unsigned long long)

    #undef XTENSOR_DEFINE_UNSIGNED_NORMS

    /***********************************
     * norm functions for std::complex *
     ***********************************/

        /** \brief L2-norm of a complex number.

            Equivalent to <tt>std::abs(t)</tt>.
        */
    template <class T>
    inline auto norm_l2(std::complex<T> & t)
    {
        return std::abs(t);
    }
    template <class T>
    inline auto norm_l2(std::complex<T> const & t)
    {
        return std::abs(t);
    }

        /** \brief Squared norm of a complex number.

            Equivalent to <tt>std::norm(t)</tt> (yes, the C++ standard really defines
            <tt>norm()</tt> to compute the squared norm).
        */
    template <class T>
    inline auto norm_sq(std::complex<T> & t)
    {
        return std::norm(t);
    }
    template <class T>
    inline auto norm_sq(std::complex<T> const & t)
    {
        return std::norm(t);
    }

    template <class T>
    inline uint64_t norm_l0(std::complex<T> & t)
    {
        return t.real() != 0 || t.imag() != 0;
    }
    template <class T>
    inline uint64_t norm_l0(std::complex<T> const & t)
    {
        return t.real() != 0 || t.imag() != 0;
    }

    /***********************************
     * norm functions for xexpressions *
     ***********************************/

     // FIXME: support axes

    /**
     * Calculate L1 norm of an array-like argument.
     *
     * @param e array-like
     * @return scalar result
     *
     * @tparam type of array-like
     */
    template <class E>
    auto norm_l1(E && e)
    {
        using value_type = typename std::decay_t<E>::value_type;
        using result_type = big_promote_type_t<value_type>;

        auto norm_func = [](result_type const & r, result_type const & v)
                         {
                             return r + norm_l1(v);
                         };
        auto init_func = [](value_type const & v)
                         {
                             return norm_l1(v);
                         };
        return reduce(make_xreducer_functor<result_type>(std::move(norm_func), std::move(init_func), std::plus<result_type>()),
                      std::forward<E>(e));
    }

    template <class E>
    auto norm_sq(E && e)
    {
        using value_type = typename std::decay_t<E>::value_type;
        using result_type = big_promote_type_t<value_type>;

        auto norm_func = [](result_type const & r, value_type const & v)
                         {
                             return r + norm_sq(v);
                         };
        auto init_func = [](value_type const & v)
                         {
                             return norm_sq(v);
                         };
        return reduce(make_xreducer_functor<result_type>(std::move(norm_func), std::move(init_func), std::plus<result_type>()),
                      std::forward<E>(e));
    }

    template <class E>
    auto norm_linf(E && e)
    {
        using value_type = typename std::decay_t<E>::value_type;
        using result_type = decltype(norm_linf(*(value_type*)0));

        auto norm_func = [](result_type const & r, value_type const & v)
                         {
                             return std::max<result_type>(r, norm_linf(v));
                         };
        auto init_func = [](value_type const & v)
                         {
                             return norm_linf(v);
                         };
        auto merge_func = [](result_type const & r1, result_type const & r2)
                          {
                              return std::max(r1, r2);
                          };
        return reduce(make_xreducer_functor<result_type>(std::move(norm_func), std::move(init_func), std::move(merge_func)),
                      std::forward<E>(e));
    }

    template <class E>
    auto norm_l0(E && e)
    {
        using value_type = typename std::decay_t<E>::value_type;
        using result_type = unsigned long long;

        auto norm_func = [](result_type const & r, value_type const & v)
                         {
                             return r + norm_l0(v);
                         };
        auto init_func = [](value_type const & v)
                         {
                             return norm_l0(v);
                         };
        return reduce(make_xreducer_functor<result_type>(std::move(norm_func), std::move(init_func), std::plus<result_type>()),
                      std::forward<E>(e));
    }

    template <class E>
    auto norm_lp(E && e, double p)
    {
        return pow(sum(pow(abs(std::forward<E>(e)), p)), 1.0/p);
    }

} // namespace xt

#endif
