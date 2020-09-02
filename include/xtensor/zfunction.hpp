/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZFUNCTION_HPP
#define XTENSOR_ZFUNCTION_HPP

#include <tuple>
#include <utility>

#include "zdispatcher.hpp"

namespace xt
{
    template <class F, class... CT>
    class zfunction
    {
    public:

        using self_type = zfunction<F, CT...>;
        using tuple_type = std::tuple<CT...>;
        using functor_type = F;

        template <class Func, class... CTA, class U = std::enable_if_t<!std::is_base_of<std::decay_t<Func>, self_type>::value>>
        zfunction(Func&& f, CTA&&... e) noexcept;

        zarray_impl& assign_to(zarray_impl& res) const;

    private:

        template <class E>
        std::enable_if_t<has_assign_to<zarray_impl, E>::value, const zarray_impl&>
        get_array_impl(const E& e, zarray_impl& res) const
        {
            return e.assign_to(res);
        }

        template <class E>
        std::enable_if_t<!has_assign_to<zarray_impl, E>::value, const zarray_impl&>
        get_array_impl(E& e, zarray_impl&) const
        {
            return e.get_implementation();
        }

        template <std::size_t... I>
        zarray_impl& assign_to_impl(std::index_sequence<I...>, zarray_impl& res) const;

        tuple_type m_e;
    };

    template <class F, class... CT>
    template <class Func, class... CTA, class U>
    inline zfunction<F, CT...>::zfunction(Func&&, CTA&&... e) noexcept
        : m_e(std::forward<CTA>(e)...)
    {
    }

    template <class F, class... CT>
    inline zarray_impl& zfunction<F, CT...>::assign_to(zarray_impl& res) const
    {
        return assign_to_impl(std::make_index_sequence<sizeof...(CT)>(), res);
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline zarray_impl& zfunction<F, CT...>::assign_to_impl(std::index_sequence<I...>, zarray_impl& res) const
    {
        // To do: call assign_to on zfunciton arguments
        zdispatcher_t<F, sizeof...(CT)>::dispatch(get_array_impl(std::get<I>(m_e), res)..., res);
        return res;
    }
}

#endif

