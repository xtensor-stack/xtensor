/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/
#if defined(_MSC_VER) && !defined(__clang__)
#define VS_SKIP_XFIXED 1
#endif

// xfixed leads to ICE in debug mode, this provides
// an easy way to prevent compilation
#ifndef VS_SKIP_XFIXED


#if (_MSC_VER < 1910 && _WIN64) || (_MSC_VER >= 1910 && !defined(DISABLE_VS2017)) || !defined(_MSC_VER)

#include <complex>
#include <limits>

#include "xtensor/xfixed.hpp"
#include "xtensor/xtensor_config.hpp"

#include "test_common_macros.hpp"

// On VS2015, when compiling in x86 mode, alignas(T) leads to C2718
// when used for a function parameter, even indirectly. This means that
// we cannot pass parameters whose class is declared with alignas specifier
// or any type wrapping or inheriting from such a type.
// The xtensor_fixed class internally uses aligned_array which is declared as
// alignas(something_different_from_0), hence the workaround.
#if _MSC_VER < 1910 && !_WIN64
#define VS_X86_WORKAROUND 1
#endif


template <typename T>
class alignas(XTENSOR_FIXED_ALIGN) Foo
{
public:

    using allocator_type = std::conditional_t<
        XTENSOR_FIXED_ALIGN != 0,
        xt_simd::aligned_allocator<T, XTENSOR_FIXED_ALIGN>,
        std::allocator<T>>;

    Foo(T fac)
        : m_fac(fac)
    {
        m_bar.fill(fac);
    }

    auto get() const
    {
        return m_bar;
    }

private:

    xt::xtensor_fixed<T, xt::xshape<10, 10>> m_bar;
    T m_fac;
};

namespace xt
{

    TEST(xsimd, alignas)
    {
        int fac = 10;
        Foo<int> foo(10);
        EXPECT_TRUE(xt::sum(foo.get())() == fac * 10 * 10);
    }
}

#endif
#endif  // VS_SKIP_XFIXED
