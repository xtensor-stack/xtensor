#ifndef XMATH_HPP
#define XMATH_HPP

#include <cmath>
#include "xfunction.hpp"

namespace qs
{

    /*************
     * Helpers
     *************/

    namespace detail
    {
        template <class R, class... Args, class... E>
        inline auto make_xfunction(R (*f) (Args...), const xexpression<E>&... e) noexcept
        {
            using type = xfunction<R (*) (Args...), R, E...>;
            return type(f, e.derived_cast()...);
        }

        template <class... E>
        using mf_type = common_value_type<E...> (*) (get_value_type<E>...);
    }


    /**********************
     * Basic operations
     **********************/

    template <class E>
    inline auto abs(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::abs, e);
    }

    template <class E>
    inline auto fabs(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::fabs, e);
    }

    template <class E1, class E2>
    inline auto fmod(const xexpression<E1>& e1, const xexpression<E2>& e2) noexcept
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fmod, e1, e2);
    }

    template <class E1, class E2>
    inline auto remainder(const xexpression<E1>& e1, const xexpression<E2>& e2) noexcept
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::remainder, e1, e2);
    }

    template <class E1, class E2, class E3>
    inline auto fma(const xexpression<E1>& e1, const xexpression<E2>& e2, const xexpression<E3>& e3) noexcept
    {
        using functor_type = detail::mf_type<E1, E2, E3>;
        return detail::make_xfunction((functor_type)std::fma, e1, e2, e3);
    }

    template <class E1, class E2>
    inline auto fmax(const xexpression<E1>& e1, const xexpression<E2>& e2) noexcept
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fmax, e1, e2);
    }

    template <class E1, class E2>
    inline auto fmin(const xexpression<E1>& e1, const xexpression<E2>& e2) noexcept
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fmin, e1, e2);
    }

    template <class E1, class E2>
    inline auto fdim(const xexpression<E1>& e1, const xexpression<E2>& e2) noexcept
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fdim, e1, e2);
    }


    /***************************
     * Exponential functions
     ***************************/

    template <class E>
    inline auto exp(const xexpression<E>& e) noexcept
    {
        using functor_type = detail::mf_type<E>;
        return detail::make_xfunction((functor_type)std::exp, e);
    }
}

#endif

