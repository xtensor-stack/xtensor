/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xinfo.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xslice.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xnobroadcast.hpp"

namespace xt
{
	TEST(xnobroadcast, A)
	{
		xt::xarray<int> a = { 1, 2 , 3 };
		xt::xarray<int> b = { 4, 5 , 6 };

		xt::xarray<int> res = a + b;
	
		auto res2 = nobroadcast(a);
		res2 = b;
	}
}