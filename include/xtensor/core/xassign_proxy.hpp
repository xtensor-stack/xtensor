/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_ASSIGN_PROXY_HPP
#define XTENSOR_ASSIGN_PROXY_HPP

#include <concepts>

#include "../core/xexpression.hpp"

namespace xt
{
    /*****************
     * assign_policy *
     *****************/

    struct alias_tag
    {
    };

    struct noalias_tag
    {
    };

    struct broadcast_tag
    {
    };

    struct nobroadcast_tag
    {
    };

    struct default_parallel_tag
    {
    };

    struct noparallel_tag
    {
    };

    struct tbb_tag
    {
    };

    struct openmp_tag
    {
    };

    template <class A>
    concept aliasing_policy = std::same_as<A, alias_tag> or std::same_as<A, noalias_tag>;

    template <class A>
    concept broadcasting_policy = std::same_as<A, broadcast_tag> or std::same_as<A, nobroadcast_tag>;

    template <class A>
    concept parallel_policy = std::same_as<A, default_parallel_tag> or std::same_as<A, noparallel_tag>
                              or std::same_as<A, tbb_tag> or std::same_as<A, openmp_tag>;

    template <aliasing_policy A1, aliasing_policy A2>
    constexpr auto operator|(A1, A2) noexcept;

    template <broadcasting_policy B1, broadcasting_policy B2>
    constexpr auto operator|(B1, B2) noexcept;

    template <parallel_policy P1, parallel_policy P2>
    constexpr auto operator|(P1, P2) noexcept;

    template <aliasing_policy A = alias_tag, broadcasting_policy B = broadcast_tag, parallel_policy P = default_parallel_tag>
    struct assign_policy
    {
        using aliasing = A;
        using broadcasting = B;
        using parallelism = P;

        template <class AT>
        auto operator()(AT&& arr) const noexcept;
    };

    template <aliasing_policy A1, broadcasting_policy B1, parallel_policy P1, aliasing_policy A2, broadcasting_policy B2, parallel_policy P2>
    constexpr auto operator|(assign_policy<A1, B1, P1>, assign_policy<A2, B2, P2>) noexcept;

    inline constexpr assign_policy<> default_assign{};
    inline constexpr assign_policy<noalias_tag> noalias{};
    inline constexpr assign_policy<alias_tag, nobroadcast_tag> nobroadcast{};
    inline constexpr assign_policy<alias_tag, broadcast_tag, noparallel_tag> noparallel{};
    inline constexpr assign_policy<alias_tag, broadcast_tag, tbb_tag> tbb{};
    inline constexpr assign_policy<alias_tag, broadcast_tag, openmp_tag> openmp{};

    template <class P>
    concept has_aliasing = std::same_as<typename P::aliasing, alias_tag>;

    template <class P>
    concept has_broadcasting = std::same_as<typename P::broadcasting, broadcast_tag>;

    /****************
     * assign_proxy *
     ****************/

    template <class A, class P>
    class xassign_proxy
    {
    public:

        xassign_proxy(A a) noexcept;

        template <class E>
        disable_xexpression<E, A> operator=(const E&);

        template <class E>
        disable_xexpression<E, A> operator+=(const E&);

        template <class E>
        disable_xexpression<E, A> operator-=(const E&);

        template <class E>
        disable_xexpression<E, A> operator*=(const E&);

        template <class E>
        disable_xexpression<E, A> operator/=(const E&);

        template <class E>
        disable_xexpression<E, A> operator%=(const E&);

        template <class E>
        disable_xexpression<E, A> operator&=(const E&);

        template <class E>
        disable_xexpression<E, A> operator|=(const E&);

        template <class E>
        disable_xexpression<E, A> operator^=(const E&);

        template <class E>
        A operator=(const xexpression<E>&);

        template <class E>
        A operator+=(const xexpression<E>&);

        template <class E>
        A operator-=(const xexpression<E>& e);

        template <class E>
        A operator*=(const xexpression<E>& e);

        template <class E>
        A operator/=(const xexpression<E>& e);

        template <class E>
        A operator%=(const xexpression<E>& e);

        template <class E>
        A operator&=(const xexpression<E>&);

        template <class E>
        A operator|=(const xexpression<E>&);

        template <class E>
        A operator^=(const xexpression<E>&);

    private:

        A m_array;
    };

    /********************************
     * assign_policy implementation *
     ********************************/

    template <aliasing_policy A1, aliasing_policy A2>
    constexpr auto operator|(A1, A2) noexcept
    {
        if constexpr (std::same_as<A1, alias_tag>)
        {
            return A2{};
        }
        else
        {
            return A1{};
        }
    }

    template <broadcasting_policy B1, broadcasting_policy B2>
    constexpr auto operator|(B1, B2) noexcept
    {
        if constexpr (std::same_as<B1, broadcast_tag>)
        {
            return B2{};
        }
        else
        {
            return B1{};
        }
    }

    template <parallel_policy P1, parallel_policy P2>
    constexpr auto operator|(P1, P2) noexcept
    {
        if constexpr (std::same_as<P1, default_parallel_tag>)
        {
            return P2{};
        }
        else
        {
            return P1{};
        }
    }

    template <aliasing_policy A, broadcasting_policy B, parallel_policy P>
    template <class AT>
    inline auto assign_policy<A, B, P>::operator()(AT&& arr) const noexcept
    {
        return xassign_proxy<xtl::closure_type_t<AT>, assign_policy>{std::forward<AT>(arr)};
    }

    template <aliasing_policy A1, broadcasting_policy B1, parallel_policy P1, aliasing_policy A2, broadcasting_policy B2, parallel_policy P2>
    constexpr auto operator|(assign_policy<A1, B1, P1>, assign_policy<A2, B2, P2>) noexcept
    {
        return assign_policy<
            decltype(std::declval<A1>() | std::declval<A2>()),
            decltype(std::declval<B1>() | std::declval<B2>()),
            decltype(std::declval<P1>() | std::declval<P2>())>{};
    }

    /*******************************
     * assign_proxy implementation *
     *******************************/

    template <class A, class P>
    inline xassign_proxy<A, P>::xassign_proxy(A a) noexcept
        : m_array(std::forward<A>(a))
    {
    }

    template <class A, class P>
    template <class E>
    inline auto xassign_proxy<A, P>::operator=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.assign(xscalar<E>(e), P{});
    }

    template <class A, class P>
    template <class E>
    inline auto xassign_proxy<A, P>::operator+=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::plus<>{}, P{});
    }

    template <class A, class P>
    template <class E>
    inline auto xassign_proxy<A, P>::operator-=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::minus<>{}, P{});
    }

    template <class A, class P>
    template <class E>
    inline auto xassign_proxy<A, P>::operator*=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::multiplies<>{}, P{});
    }

    template <class A, class P>
    template <class E>
    inline auto xassign_proxy<A, P>::operator/=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::divides<>{}, P{});
    }

    template <class A, class P>
    template <class E>
    inline auto xassign_proxy<A, P>::operator%=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::modulus<>{}, P{});
    }

    template <class A, class P>
    template <class E>
    inline auto xassign_proxy<A, P>::operator&=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::bit_and<>{}, P{});
    }

    template <class A, class P>
    template <class E>
    inline auto xassign_proxy<A, P>::operator|=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::bit_or<>{}, P{});
    }

    template <class A, class P>
    template <class E>
    inline auto xassign_proxy<A, P>::operator^=(const E& e) -> disable_xexpression<E, A>
    {
        return m_array.scalar_computed_assign(e, std::bit_xor<>{}, P{});
    }

    template <class A, class P>
    template <class E>
    A xassign_proxy<A, P>::operator=(const xexpression<E>& e)
    {
        return m_array.assign(e, P{});
    }

    template <class A, class P>
    template <class E>
    A xassign_proxy<A, P>::operator+=(const xexpression<E>& e)
    {
        return m_array.plus_assign(e, P{});
    }

    template <class A, class P>
    template <class E>
    A xassign_proxy<A, P>::operator-=(const xexpression<E>& e)
    {
        return m_array.minus_assign(e, P{});
    }

    template <class A, class P>
    template <class E>
    A xassign_proxy<A, P>::operator*=(const xexpression<E>& e)
    {
        return m_array.multiplies_assign(e, P{});
    }

    template <class A, class P>
    template <class E>
    A xassign_proxy<A, P>::operator/=(const xexpression<E>& e)
    {
        return m_array.divides_assign(e, P{});
    }

    template <class A, class P>
    template <class E>
    A xassign_proxy<A, P>::operator%=(const xexpression<E>& e)
    {
        return m_array.modulus_assign(e, P{});
    }

    template <class A, class P>
    template <class E>
    A xassign_proxy<A, P>::operator&=(const xexpression<E>& e)
    {
        return m_array.bit_and_assign(e, P{});
    }

    template <class A, class P>
    template <class E>
    A xassign_proxy<A, P>::operator|=(const xexpression<E>& e)
    {
        return m_array.bit_or_assign(e, P{});
    }

    template <class A, class P>
    template <class E>
    A xassign_proxy<A, P>::operator^=(const xexpression<E>& e)
    {
        return m_array.bit_xor_assign(e, P{});
    }
}

#endif
