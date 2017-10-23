/***************************************************************************
* Copyright (c) 2017, Patrick Bos                                          *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include <complex>
#include "xtensor/xarray.hpp"

namespace xt
{
  template <typename T>
  std::complex<T> i {0,1};

  template <typename T>
  auto generate_complex_data() {
    xt::xarray<T, xt::layout_type::row_major> a {0, 1, 2, 3};
    xt::xarray<T, xt::layout_type::row_major> b {10, 20, 30, 40};
    xt::xarray<std::complex<T>, xt::layout_type::row_major> c = (a/static_cast<T>(2)) + (b/static_cast<T>(2)) * i<T>;
    return c;
  }

  TEST(complex, returned_complex_array)
  {
    xt::xarray<std::complex<float>, xt::layout_type::row_major> c_returned = generate_complex_data<float>();
    xt::xarray<float, xt::layout_type::row_major> a {0, 1, 2, 3};
    xt::xarray<float, xt::layout_type::row_major> b {10, 20, 30, 40};
    xt::xarray<std::complex<float>, xt::layout_type::row_major> c = a + b * i<float>;

    EXPECT_EQ(c/static_cast<float>(2), c_returned);
  }

  template <typename T>
  auto generate_complex_data_with_division() {
    xt::xarray<T, xt::layout_type::row_major> a {0, 1, 2, 3};
    xt::xarray<T, xt::layout_type::row_major> b {10, 20, 30, 40};
    xt::xarray<std::complex<T>, xt::layout_type::row_major> c = a + b * i<T>;
    return c / static_cast<T>(2);
  }

  TEST(complex, returned_complex_array_with_division)
  {
    xt::xarray<std::complex<float>, xt::layout_type::row_major> c_returned = generate_complex_data_with_division<float>();
    xt::xarray<float, xt::layout_type::row_major> a {0, 1, 2, 3};
    xt::xarray<float, xt::layout_type::row_major> b {10, 20, 30, 40};
    xt::xarray<std::complex<float>, xt::layout_type::row_major> c = a + b * i<float>;

    EXPECT_EQ(c / static_cast<float>(2), c_returned);
  }
}