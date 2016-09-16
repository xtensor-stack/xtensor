#ifndef XOPERATION_HPP
#define XOPERATION_HPP

#include <functional>

#include "xfunction.hpp"

namespace qs
{

    /**************
     * Helpers
     **************/

    template <class T>
    struct identity
    {
        using result_type = T;

        constexpr T operator()(const T& t) const
        {
            return +t;
        }
    };

    namespace detail
    {
        template <template <class...> class F, class... E>
        auto make_xfunction(const xexpression<E>&... e) noexcept
        {
            using functor_type = F<common_value_type<E...>>;
            using result_type = typename functor_type::result_type;
            using type = xfunction<functor_type, result_type, E...>;
            return type(functor_type(), e.derived_cast()...);
        }
    }


    /***************
     * Operators
     ***************/

    template <class E>
    inline auto operator+(const xexpression<E>& e) noexcept
    {
        return detail::make_xfunction<identity>(e);
    }

    template <class E>
    inline auto operator-(const xexpression<E>& e) noexcept
    {
        return detail::make_xfunction<std::negate>(e);
    }

    template <class E1, class E2>
    inline auto operator+(const xexpression<E1>& e1, const xexpression<E2>& e2) noexcept
    {
        return detail::make_xfunction<std::plus>(e1, e2);
    }

    template <class E1, class E2>
    inline auto operator-(const xexpression<E1>& e1, const xexpression<E2>& e2) noexcept
    {
        return detail::make_xfunction<std::minus>(e1, e2);
    }

    template <class E1, class E2>
    inline auto operator*(const xexpression<E1>& e1, const xexpression<E2>& e2) noexcept
    {
        return detail::make_xfunction<std::multiplies>(e1, e2);
    }

    template <class E1, class E2>
    inline auto operator/(const xexpression<E1>& e1, const xexpression<E2>& e2) noexcept
    {
        return detail::make_xfunction<std::divides>(e1, e2);
    }

}

#endif

