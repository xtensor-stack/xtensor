/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xcomplex.hpp"
#include "xtensor/xfunctor_view.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xnoalias.hpp"

namespace xt
{
    using namespace std::complex_literals;

    template <class T>
    struct nooblean_proxy
    {
        nooblean_proxy(T& ref) : m_ref(ref)
        {
        }

        operator bool()
        {
            return !m_ref;
        };

        nooblean_proxy& operator=(bool rhs)
        {
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

        /** cant implement yet -- need to figure out bool loading in xsimd **/
        // template <class align, class requested_type, std::size_t N, class E>
        // auto proxy_simd_load(const E& expr, std::size_t n) const
        // {
        //     using simd_value_type = xsimd::simd_type<value_type>;
        //     return !expr.template load_simd<align, requested_type, N>(n);
        // }
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

    TEST(xfunctor_adaptor, lhs_assignment)
    {
        using container_type = xarray<std::complex<double>>;
        
        container_type e = {{3.0       , 1.0 + 1.0i},
                            {1.0 - 1.0i, 2.0       }};

        // Assigning to a xfunctor_adaptor, which has a container semantics, resizes
        // the underlying container.
        auto radaptor = xt::xoffset_adaptor<container_type&, double, 0>(e);
        xt::xtensor<double, 1> rhs = {4.0, 5.0};
        radaptor = rhs;

        EXPECT_EQ(e.dimension(), 1u);
        EXPECT_EQ(xtl::real(e(0)), 4.0);
        EXPECT_EQ(xtl::real(e(1)), 5.0);
    }

#if defined(XTENSOR_USE_XSIMD) && XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION && XSIMD_X86_INSTR_SET < XSIMD_X86_AVX512_VERSION
    TEST(xfunctor_adaptor, simd)
    {
        xarray<std::complex<double>> e = {{3.0       , 1.0 + 1.0i},
                                          {1.0 - 1.0i, 2.0       }};
        auto iview = xt::imag(e);
        auto loaded_batch = iview.template load_simd<xsimd::aligned_mode, double, 4>(0);
        EXPECT_TRUE(xsimd::all(xsimd::batch<double, 4>(0, 1, -1, 0) == loaded_batch));
        auto newbatch = loaded_batch + 5;

        iview.template store_simd<xsimd::aligned_mode>(0, newbatch);
        xarray<std::complex<double>> exp1 = {{3.0 + 5.0i, 1.0 + 6.0i},
                                             {1.0 + 4.0i, 2.0 + 5.0i }};
        EXPECT_EQ(exp1, e);

        auto rview = xt::real(e);
        auto loaded_batch2 = rview.template load_simd<xsimd::aligned_mode, double, 4>(0);
        EXPECT_TRUE(xsimd::all(xsimd::batch<double, 4>(3, 1, 1, 2) == loaded_batch2));
        newbatch = loaded_batch2 + 5;
        rview.template store_simd<xsimd::aligned_mode>(0, newbatch);
        xarray<std::complex<double>> exp2 = {{8.0 + 5.0i, 6.0 + 6.0i},
                                             {6.0 + 4.0i, 7.0 + 5.0i }};
        EXPECT_EQ(exp2, e);

        auto f = xt::sin(xt::imag(e));
        auto b = f.load_simd<xsimd::aligned_mode>(0);
        static_cast<void>(b);
        using assign_to_view = xassign_traits<decltype(iview), decltype(f)>;
        EXPECT_TRUE(assign_to_view::convertible_types());
        EXPECT_TRUE(assign_to_view::simd_size());
        EXPECT_FALSE(assign_to_view::forbid_simd());
        EXPECT_TRUE(assign_to_view::simd_assign());
    }
#endif
}
