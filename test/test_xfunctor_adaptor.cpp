/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xfunctor_view.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xnoalias.hpp"

namespace xt
{
    template <class T>
    struct nooblean_proxy {
        nooblean_proxy(T& ref) : m_ref(ref) {}
        operator bool() { return !m_ref; };
        nooblean_proxy& operator=(bool rhs) {
            m_ref = !rhs;
            return *this;
        }

        T& m_ref;
    };

    template <class T>
    struct xproxy_inner_types<nooblean_proxy<T>>
    {
        // T is used for constness deduction
        using proxy = nooblean_proxy<T>;
        using reference = nooblean_proxy<T>;
        using pointer = nooblean_proxy<T>;
    };

    struct nooblean
    {
        using value_type = bool;
        using reference = nooblean_proxy<bool>;
        using const_reference = nooblean_proxy<const bool>;
        using pointer = bool*;
        using const_pointer = bool*;

        template <class value_type, class requested_type>
        using simd_return_type = xsimd::simd_return_type<value_type, requested_type>;

        const_reference operator()(const bool& in) const
        {
            return in;
        }

        reference operator()(bool& in)
        {
            return in;
        }
    };

    TEST(xfunctor_adaptor, basic)
    {
        using nooblean_adaptor = xt::xfunctor_adaptor<nooblean, xarray<bool>&>;
        xarray<bool> vals = {{1, 1, 1, 0, 0}, {1, 0, 1, 0, 1}};
        xarray<bool> xvals = !vals;

        nooblean_adaptor aptvals(vals);
        EXPECT_EQ(aptvals, xvals);
        auto begin = aptvals.storage_begin();

        *begin = true;
        EXPECT_EQ(bool(*begin), true);
        EXPECT_EQ(vals(0, 0), false);

        aptvals(0, 0) = false;
        EXPECT_EQ(vals(0, 0), true);
        EXPECT_EQ(bool(aptvals(0, 0)), false);

        bool execd = false;
        if (aptvals(0, 0) == false)
        {
            execd = true;
        }
        EXPECT_TRUE(execd);

        auto rhs1 = xt::xarray<bool>({true, false, true});
        aptvals = rhs1;
        EXPECT_EQ(rhs1, aptvals);
        EXPECT_EQ(!rhs1, vals);
    }

    TEST(xfunctor_adaptor, iterator)
    {
        using nooblean_adaptor = xt::xfunctor_adaptor<nooblean, xarray<bool>&>;
        xarray<bool> vals = {{1, 1, 1, 0, 0}, {1, 0, 1, 0, 1}};
        xarray<bool> xvals = !vals;

        nooblean_adaptor aptvals(vals);

        auto it_adapt = aptvals.begin();
        auto it_ref = xvals.begin();

        for (; it_adapt != aptvals.end(); ++it_adapt, ++it_ref)
        {
            EXPECT_EQ(static_cast<bool>(*it_adapt), *it_ref);
        }
    }
}