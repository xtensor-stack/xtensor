#include "gtest/gtest.h"
#include "xarray/xarray.hpp"
#include "test_common.hpp"

namespace qs
{
    using vec_type = std::vector<int>;
    using adaptor_type = xarray_adaptor<vec_type>;

    
    TEST(xarray_adaptor, reshape)
    {
        vec_type v;
        adaptor_type a(v);
        test_reshape(a);
    }

    TEST(xarray_adaptor, access)
    {
        vec_type v;
        adaptor_type a(v);
        test_access(a);
    }

    TEST(xarray_adaptor, broadcast_shape)
    {
        vec_type v;
        adaptor_type a(v);
        test_broadcast(a);
    }

    TEST(xarray_adaptor, storage_iterator)
    {
        vec_type v;
        adaptor_type a(v);
        test_storage_iterator(a);
    }
}

