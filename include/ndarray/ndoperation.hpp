#ifndef NDOPERATION_HPP
#define NDOPERATION_HPP

#include "ndoperation_expression.hpp"

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
    inline ndunary_op<E, plus_functor>
    operator+(const ndexpression<E>& e) noexcept
    {
        using type = ndunary_op<E, plus_functor>;
        return type(e());
    }

    template <class E>
    inline ndunary_op<E, minus_functor>
    operator-(const ndexpression<E>& e) noexcept
    {
        using type = ndunary_op<E, minus_functor>;
        return type(e());
    }

    template <class E1, class E2>
    inline ndbinary_op<E1, E2, add_functor>
    operator+(const ndexpression<E1>& e1, const ndexpression<E2>& e2) noexcept
    {
        using type = ndbinary_op<E1, E2, add_functor>;
        return type(e1(), e2());
    }

    template <class E1, class E2>
    inline ndbinary_op<E1, E2, sub_functor>
    operator-(const ndexpression<E1>& e1, const ndexpression<E2>& e2) noexcept
    {
        using type = ndbinary_op<E1, E2, sub_functor>;
        return type(e1(), e2());
    }

    template <class E1, class E2>
    inline ndbinary_op<E1, E2, mul_functor>
    operator*(const ndexpression<E1>& e1, const ndexpression<E2>& e2) noexcept
    {
        using type = ndbinary_op<E1, E2, mul_functor>;
        return type(e1(), e2());
    }

    template <class E1, class E2>
    inline ndbinary_op<E1, E2, div_functor>
    operator/(const ndexpression<E1>& e1, const ndexpression<E2>& e2) noexcept
    {
        using type = ndbinary_op<E1, E2, div_functor>;
        return type(e1(), e2());
    }

}

#endif

