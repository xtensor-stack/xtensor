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
        inline auto make_xfunction(R (*f) (Args...), const E&... e) noexcept
        {
            using type = xfunction<R (*) (Args...), R, E...>;
            return type(f, get_xexpression(e)...);
        }

        template <class... E>
        using mf_type = common_value_type<E...> (*) (get_value_type<E>...);

        template <class... Args>
        using get_xfunction_free_type = std::enable_if_t<has_xexpression<Args...>::value,
                                                         xfunction<mf_type<Args...>,
                                                                   common_value_type<Args...>,
                                                                   Args...>>;

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
    inline auto fmod(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fmod, e1, e2);
    }

    template <class E1, class E2>
    inline auto remainder(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::remainder, e1, e2);
    }

    template <class E1, class E2, class E3>
    inline auto fma(const E1& e1, const E2& e2, const E3& e3) noexcept
        -> detail::get_xfunction_free_type<E1, E2, E3>
    {
        using functor_type = detail::mf_type<E1, E2, E3>;
        return detail::make_xfunction((functor_type)std::fma, e1, e2, e3);
    }

    template <class E1, class E2>
    inline auto fmax(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fmax, e1, e2);
    }

    template <class E1, class E2>
    inline auto fmin(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_free_type<E1, E2>
    {
        using functor_type = detail::mf_type<E1, E2>;
        return detail::make_xfunction((functor_type)std::fmin, e1, e2);
    }

    template <class E1, class E2>
    inline auto fdim(const E1& e1, const E2& e2) noexcept
        -> detail::get_xfunction_free_type<E1, E2>
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

