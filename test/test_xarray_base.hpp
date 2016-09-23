#ifndef TEST_XARRAY_BASE_HPP
#define TEST_XARRAY_BASE_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include "xarray/xindex.hpp"

namespace qs
{
    struct layout_result
    {
        using vector_type = std::vector<int>;
        using size_type = vector_type::size_type;
        using shape_type = array_shape<size_type>;
        using strides_type = array_strides<size_type>;

        using assigner_type = std::vector<std::vector<vector_type>>;

        inline layout_result()
        {
            m_shape = { 3, 2, 4 };
            m_assigner.resize(m_shape[0]);
            for(size_t i = 0; i < m_shape[0]; ++i)
            {
                m_assigner[i].resize(m_shape[1]);
            }
            m_assigner[0][0] = { 0, 1, 2, 3 };
            m_assigner[0][1] = { 4, 5, 6, 7 };
            m_assigner[1][0] = { 8, 9, 10, 11 };
            m_assigner[1][1] = { 12, 13, 14, 15 };
            m_assigner[2][0] = { 16, 17, 18, 19 };
            m_assigner[2][1] = { 20, 21, 22, 23 };
        }

        shape_type m_shape;
        strides_type m_strides;
        strides_type m_backstrides;
        vector_type m_data;
        assigner_type m_assigner;
    };

    struct row_major_result : layout_result
    {
        inline row_major_result()
        {
            m_strides = { 8, 4, 1 };
            m_backstrides = {23, 7, 3};
            m_data = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                      10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                      20, 21, 22, 23 };
        }
    };

    struct column_major_result : layout_result
    {
        inline column_major_result()
        {
            m_strides = { 1, 3, 6 };
            m_backstrides = { 2, 5, 23 };
            m_data = { 0, 8, 16, 4, 12, 20,
                       1, 9, 17, 5, 13, 21,
                       2, 10, 18, 6, 14, 22,
                       3, 11, 19, 7, 15, 23 };
        }
    };

    struct central_major_result : layout_result
    {
        inline central_major_result()
        {
            m_strides = { 8, 1, 2 };
            m_backstrides = { 23, 1, 7};
            m_data = { 0, 4, 1, 5, 2, 6, 3, 7,
                       8, 12, 9, 13, 10, 14, 11, 15,
                      16, 20, 17, 21, 18, 22, 19, 23 };
        }
    };

    struct unit_shape_result
    {
        using vector_type = std::vector<int>;
        using size_type = vector_type::size_type;
        using shape_type = array_shape<size_type>;
        using strides_type = array_strides<size_type>;

        using assigner_type = std::vector<std::vector<vector_type>>;

        inline unit_shape_result()
        {
            m_shape = { 3, 1, 4 };
            m_strides = { 4, 0, 1 };
            m_backstrides = { 11, 0, 3 };
            m_data = { 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19 };
            m_assigner.resize(m_shape[0]);
            for(size_t i = 0; i < m_shape[0]; ++i)
            {
                m_assigner[i].resize(m_shape[1]);
            }
            m_assigner[0][0] = { 0, 1, 2, 3 };
            m_assigner[1][0] = { 8, 9, 10, 11 };
            m_assigner[2][0] = { 16, 17, 18, 19 };
        }

        shape_type m_shape;
        strides_type m_strides;
        strides_type m_backstrides;
        vector_type m_data;
        assigner_type m_assigner;
    };

    template <class V, class R>
    bool compare_shape(V& vec, const R& result)
    {
        return (vec.dimension() == result.m_shape.size()) &&
               (vec.shape() == result.m_shape) &&
               (vec.strides() == result.m_strides) &&
               (vec.backstrides() == result.m_backstrides) &&
               (vec.size() == result.m_data.size());
    }

    template <class V>
    bool test_xarray_reshape(V& vec)
    {
        row_major_result rm;
        vec.reshape(rm.m_shape, layout::row_major);
        bool result = compare_shape(vec, rm);

        column_major_result cm;
        vec.reshape(cm.m_shape, layout::column_major);
        result = result && compare_shape(vec, cm);

        central_major_result cem;
        vec.reshape(cem.m_shape, cem.m_strides);
        result = result && compare_shape(vec, cem);

        unit_shape_result usr;
        vec.reshape(usr.m_shape, layout::row_major);
        result = result && compare_shape(vec, usr);

        return result;
    }

    template <class V>
    bool test_xarray_storage_iterator(V& vec)
    {
        using result_type = central_major_result;
        using vector_type = result_type::vector_type;
        result_type res;
        vector_type tester(res.m_data.size());

        // begin/end test
        vec.reshape(res.m_shape);
        std::copy(res.m_data.begin(), res.m_data.end(), vec.storage_begin());
        bool b = (vec.data() == res.m_data) && (vec.storage_end() == vec.data().end());

        return b;
    }

    template <class V1, class V2>
    void assign_array(V1& dst, const V2& src)
    {
        for(size_t i = 0; i < dst.shape()[0]; ++i)
        {
            for(size_t j = 0; j < dst.shape()[1]; ++j)
            {
                for(size_t k = 0; k < dst.shape()[2]; ++k)
                {
                    dst(i, j, k) = src[i][j][k];
                }
            }
        }
    }

    template <class V>
    bool test_xarray_access(V& vec)
    {
        row_major_result rm;
        vec.reshape(rm.m_shape, layout::row_major);
        assign_array(vec, rm.m_assigner);
        bool res = (vec.data() == rm.m_data);

        column_major_result cm;
        vec.reshape(cm.m_shape, layout::column_major);
        assign_array(vec, cm.m_assigner);
        res = res && (vec.data() == cm.m_data);

        central_major_result cem;
        vec.reshape(cem.m_shape, cem.m_strides);
        assign_array(vec, cem.m_assigner);
        res = res && (vec.data() == cem.m_data);

        unit_shape_result usr;
        vec.reshape(usr.m_shape, layout::row_major);
        assign_array(vec, usr.m_assigner);
        res = res && (vec.data() == usr.m_data);

        return res;
    }

    template <class V>
    bool test_xarray_broadcast(V& vec)
    {
        using shape_type = typename V::shape_type;

        shape_type s1 = { 3, 1, 4, 2 };
        shape_type s2 = { 3, 5, 1, 2 };
        shape_type s3 = { 3, 5, 4, 2 };

        vec.reshape(s1);
        bool res = vec.broadcast_shape(s1);

        bool trivial = vec.broadcast_shape(s2);
        res = res && (s2 == s3) && !trivial;

        shape_type s4 = { 2, 1, 3, 2 };
        bool wit = false;
        try
        {
            vec.broadcast_shape(s4);
        }
        catch(...)
        {
            wit = true;
        }

        res = res && wit;

        return res;
    }
}

#endif

