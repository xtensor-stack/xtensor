#include "gtest/gtest.h"
#include "ndarray/ndarray.hpp"
#include "test_ndarray_base.hpp"

namespace ndarray
{
    TEST(ndarray, reshape)
    {
        ndarray<int> a;
        ASSERT_TRUE(test_ndarray_reshape(a));
    }

    TEST(ndarray, iterator)
    {
        ndarray<int> a;
        ASSERT_TRUE(test_ndarray_iterator(a));
    }

    TEST(ndarray, access)
    {
        ndarray<int> a;
        ASSERT_TRUE(test_ndarray_access(a));
    }

    TEST(ndarray, broadcast_shape)
    {
        ndarray<int> a;
        ASSERT_TRUE(test_ndarray_broadcast(a));
    }

    using vec_type = std::vector<int>;
    using adaptor_type = ndarray_adaptor<vec_type>;
    
    TEST(ndarray_adaptor, reshape)
    {
        vec_type v;
        adaptor_type a(v);
        ASSERT_TRUE(test_ndarray_reshape(a));
    }

    TEST(ndarray_adaptor, iterator)
    {
        vec_type v;
        adaptor_type a(v);
        ASSERT_TRUE(test_ndarray_iterator(a));
    }

    TEST(ndarray_adaptor, access)
    {
        vec_type v;
        adaptor_type a(v);
        ASSERT_TRUE(test_ndarray_access(a));
    }

    TEST(ndarray_adaptor, broadcast_shape)
    {
        vec_type v;
        adaptor_type a(v);
        ASSERT_TRUE(test_ndarray_broadcast(a));
    }

}

