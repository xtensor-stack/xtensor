#ifndef XOPERATION_HPP
#define XOPERATION_HPP

#include "xfunction.hpp"

namespace qs
{

    /**************
     * Functors
     **************/

    template <class T>
    struct plus_functor
    {
        using result_type = T;

        inline static result_type apply(const T& t) noexcept
        {
            return t;
        }
    };

    template <class T>
    struct minus_functor
    {
        using result_type = T;

        inline static result_type apply(const T& t) noexcept
        {
            return -t;
        }
    };

    template <class T1, class T2>
    struct add_functor
    {
        using result_type = std::common_type_t<T1, T2>;

        inline static result_type apply(const T1& t1, const T2& t2) noexcept
        {
            return t1 + t2;
        }
    };

    template <class T1, class T2>
    struct sub_functor
    {
        using result_type = std::common_type_t<T1, T2>;

        inline static result_type apply(const T1& t1, const T2& t2) noexcept
        {
            return t1 - t2;
        }
    };

    template <class T1, class T2>
    struct mul_functor
    {
        using result_type = std::common_type_t<T1, T2>;

        inline static result_type apply(const T1& t1, const T2& t2) noexcept
        {
            return t1 * t2;
        }
    };

    template <class T1, class T2>
    struct div_functor
    {
        using result_type = std::common_type_t<T1, T2>;

        inline static result_type apply(const T1& t1, const T2& t2) noexcept
        {
            return t1 / t2;
        }
    };


    /***************
     * Operators
     ***************/

    template <class E>
    inline xfunction_op<plus_functor, E>
    operator+(const xexpression<E>& e) noexcept
    {
        using type = xfunction_op<plus_functor, E>;
        return type(e.derived_cast());
    }

    template <class E>
    inline xfunction_op<minus_functor, E>
    operator-(const xexpression<E>& e) noexcept
    {
        using type = xfunction_op<minus_functor, E>;
        return type(e.derived_cast());
    }

    template <class E1, class E2>
    inline xfunction_op<add_functor, E1, E2>
    operator+(const xexpression<E1>& e1, const xexpression<E2>& e2) noexcept
    {
        using type = xfunction_op<add_functor, E1, E2>;
        return type(e1.derived_cast(), e2.derived_cast());
    }

    template <class E1, class E2>
    inline xfunction_op<sub_functor, E1, E2>
    operator-(const xexpression<E1>& e1, const xexpression<E2>& e2) noexcept
    {
        using type = xfunction_op<sub_functor, E1, E2>;
        return type(e1.derived_cast(), e2.derived_cast());
    }

    template <class E1, class E2>
    inline xfunction_op<mul_functor, E1, E2>
    operator*(const xexpression<E1>& e1, const xexpression<E2>& e2) noexcept
    {
        using type = xfunction_op<mul_functor, E1, E2>;
        return type(e1.derived_cast(), e2.derived_cast());
    }

    template <class E1, class E2>
    inline xfunction_op<div_functor, E1, E2>
    operator/(const xexpression<E1>& e1, const xexpression<E2>& e2) noexcept
    {
        using type = xfunction_op<div_functor, E1, E2>;
        return type(e1.derived_cast(), e2.derived_cast());
    }

}

#endif

