#include "gtest/gtest.h"
#include "xarray/xarray.hpp"
#include "test_xarray_base.hpp"

namespace qs
{
    TEST(xarray, reshape)
    {
        xarray<int> a;
        ASSERT_TRUE(test_xarray_reshape(a));
    }

    TEST(xarray, storage_iterator)
    {
        xarray<int> a;
        ASSERT_TRUE(test_xarray_storage_iterator(a));
    }

    TEST(xarray, access)
    {
        xarray<int> a;
        ASSERT_TRUE(test_xarray_access(a));
    }

    TEST(xarray, broadcast_shape)
    {
        xarray<int> a;
        ASSERT_TRUE(test_xarray_broadcast(a));
    }

    using vec_type = std::vector<int>;
    using adaptor_type = xarray_adaptor<vec_type>;
    
    TEST(xarray_adaptor, reshape)
    {
        vec_type v;
        adaptor_type a(v);
        ASSERT_TRUE(test_xarray_reshape(a));
    }

    TEST(xarray_adaptor, storage_iterator)
    {
        vec_type v;
        adaptor_type a(v);
        ASSERT_TRUE(test_xarray_storage_iterator(a));
    }

    TEST(xarray_adaptor, access)
    {
        vec_type v;
        adaptor_type a(v);
        ASSERT_TRUE(test_xarray_access(a));
    }

    TEST(xarray_adaptor, broadcast_shape)
    {
        vec_type v;
        adaptor_type a(v);
        ASSERT_TRUE(test_xarray_broadcast(a));
    }

}

