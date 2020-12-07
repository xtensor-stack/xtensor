/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <complex>
#include <limits>

#include "gtest/gtest.h"
#include "xtensor/xfixed.hpp"
#include "xtensor/xtensor_config.hpp"

template <typename T>
class alignas(XTENSOR_FIXED_ALIGN) Foo
{
public:

    using allocator_type = std::conditional_t<XTENSOR_FIXED_ALIGN != 0,
                                              xt_simd::aligned_allocator<T, XTENSOR_FIXED_ALIGN>,
                                              std::allocator<T>>;

    Foo(T fac) : m_fac(fac)
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

