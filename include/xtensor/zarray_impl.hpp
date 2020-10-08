/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZARRAY_IMPL_HPP
#define XTENSOR_ZARRAY_IMPL_HPP

#include "xarray.hpp"
#include "xchunked_array.hpp"

namespace xt
{

    /*************************
     * zarray_expression_tag *
     *************************/

    struct zarray_expression_tag {};

    namespace extension
    {
        template <>
        struct expression_tag_and<xtensor_expression_tag, zarray_expression_tag>
        {
            using type = zarray_expression_tag;
        };

        template <>
        struct expression_tag_and<zarray_expression_tag, xtensor_expression_tag>
            : expression_tag_and<xtensor_expression_tag, zarray_expression_tag>
        {
        };

        template <>
        struct expression_tag_and<zarray_expression_tag, zarray_expression_tag>
        {
            using type = zarray_expression_tag;
        };
    }

    /***************
     * zarray_impl *
     ***************/

    class zarray_impl
    {
    public:

        using self_type = zarray_impl;

        virtual ~zarray_impl() = default;

        zarray_impl(zarray_impl&&) = delete;
        zarray_impl& operator=(const zarray_impl&) = delete;
        zarray_impl& operator=(zarray_impl&&) = delete;

        virtual self_type* clone() const = 0;

        XTL_IMPLEMENT_INDEXABLE_CLASS()

    protected:

        zarray_impl() = default;
        zarray_impl(const zarray_impl&) = default;
    };

    /****************
     * ztyped_array *
     ****************/

    template <class T>
    class ztyped_array : public zarray_impl
    {
    public:

        virtual ~ztyped_array() = default;

        virtual xarray<T>& get_array() = 0;
        virtual const xarray<T>& get_array() const = 0;

        XTL_IMPLEMENT_INDEXABLE_CLASS()

    protected:

        ztyped_array() = default;
        ztyped_array(const ztyped_array&) = default;
    };

    /***********************
     * zexpression_wrapper *
     ***********************/

    template <class CTE>
    class zexpression_wrapper : public ztyped_array<typename std::decay_t<CTE>::value_type>
    {
    public:

        using self_type = zexpression_wrapper<CTE>;
        using value_type = typename std::decay_t<CTE>::value_type;
        using base_type = ztyped_array<value_type>;

        template <class E>
        zexpression_wrapper(E&& e);

        virtual ~zexpression_wrapper() = default;

        xarray<value_type>& get_array() override;
        const xarray<value_type>& get_array() const override;

        self_type* clone() const override;

    private:

        zexpression_wrapper(const zexpression_wrapper&) = default;

        void compute_cache() const;

        CTE m_expression;
        mutable xarray<value_type> m_cache;
        mutable bool m_cache_initialized;
    };

    /******************
     * zarray_wrapper *
     ******************/

    template <class CTE>
    class zarray_wrapper : public ztyped_array<typename std::decay_t<CTE>::value_type>
    {
    public:

        using self_type = zarray_wrapper;
        using value_type = typename std::decay_t<CTE>::value_type;
        using base_type = ztyped_array<value_type>;

        template <class E>
        zarray_wrapper(E&& e);

        virtual ~zarray_wrapper() = default;

        xarray<value_type>& get_array() override;
        const xarray<value_type>& get_array() const override;

        self_type* clone() const override;

    private:

        zarray_wrapper(const zarray_wrapper&) = default;

        CTE m_array;
    };

    /********************
     * zchunked_wrapper *
     ********************/

    class zchunked_array
    {
    public:

        using shape_type = std::vector<std::size_t>;

        virtual ~zchunked_array() = default;
        virtual const shape_type& chunk_shape() const = 0;
    };

    template <class CTE>
    class zchunked_wrapper : public ztyped_array<typename std::decay_t<CTE>::value_type>,
                             public zchunked_array
    {
    public:

        using self_type = zchunked_wrapper;
        using value_type = typename std::decay_t<CTE>::value_type;
        using base_type = ztyped_array<value_type>;
        using shape_type = typename zchunked_array::shape_type;

        template <class E>
        zchunked_wrapper(E&& e);

        virtual ~zchunked_wrapper() = default;

        xarray<value_type>& get_array() override;
        const xarray<value_type>& get_array() const override;

        self_type* clone() const override;

        const shape_type& chunk_shape() const override;

    private:

        zchunked_wrapper(const zchunked_wrapper&) = default;

        void compute_cache() const;

        CTE m_chunked_array;
        shape_type m_chunk_shape;
        mutable xarray<value_type> m_cache;
        mutable bool m_cache_initialized;

    };

    /***********************
     * zexpression_wrapper *
     ***********************/
        
    template <class CTE>
    template <class E>
    inline zexpression_wrapper<CTE>::zexpression_wrapper(E&& e)
        : base_type()
        , m_expression(std::forward<E>(e))
        , m_cache()
        , m_cache_initialized(false)
    {
    }

    template <class CTE>
    inline auto zexpression_wrapper<CTE>::get_array() -> xarray<value_type>&
    {
        compute_cache();
        return m_cache;
    }

    template <class CTE>
    inline auto zexpression_wrapper<CTE>::get_array() const -> const xarray<value_type>&
    {
        compute_cache();
        return m_cache;
    }

    template <class CTE>
    inline auto zexpression_wrapper<CTE>::clone() const -> self_type*
    {
        return new self_type(*this);
    }

    template <class CTE>
    inline void zexpression_wrapper<CTE>::compute_cache() const
    {
        if (!m_cache_initialized)
        {
            m_cache = m_expression;
            m_cache_initialized = true;
        }
    }

    /******************
     * zarray_wrapper *
     ******************/

    template <class CTE>
    template <class E>
    inline zarray_wrapper<CTE>::zarray_wrapper(E&& e)
        : base_type()
        , m_array(std::forward<E>(e))
    {
    }

    template <class CTE>
    inline auto zarray_wrapper<CTE>::get_array() -> xarray<value_type>&
    {
        return m_array;
    }

    template <class CTE>
    inline auto zarray_wrapper<CTE>::get_array() const -> const xarray<value_type>&
    {
        return m_array;
    }

    template <class CTE>
    inline auto zarray_wrapper<CTE>::clone() const -> self_type*
    {
        return new self_type(*this);
    }

    /********************
     * zchunked_wrapper *
     ********************/

    template <class CTE>
    template <class E>
    inline zchunked_wrapper<CTE>::zchunked_wrapper(E&& e)
        : base_type()
        , m_chunked_array(std::forward<E>(e))
        , m_chunk_shape(m_chunked_array.chunk_shape().size())
        , m_cache()
        , m_cache_initialized(false)
    {
        std::copy(m_chunked_array.chunk_shape().begin(),
                  m_chunked_array.chunk_shape().end(),
                  m_chunk_shape.begin());
    }

    template <class CTE>
    inline auto zchunked_wrapper<CTE>::get_array() -> xarray<value_type>&
    {
        compute_cache();
        return m_cache;
    }

    template <class CTE>
    inline auto zchunked_wrapper<CTE>::get_array() const -> const xarray<value_type>&
    {
        compute_cache();
        return m_cache;
    }

    template <class CTE>
    inline auto zchunked_wrapper<CTE>::clone() const -> self_type*
    {
        return new self_type(*this);
    }

    template <class CTE>
    inline auto zchunked_wrapper<CTE>::chunk_shape() const -> const shape_type&
    {
        return m_chunk_shape;
    }

    template <class CTE>
    inline void zchunked_wrapper<CTE>::compute_cache() const
    {
        if (!m_cache_initialized)
        {
            m_cache = m_chunked_array;
            m_cache_initialized = true;
        }
    }

    /******************
     * zarray builder *
     ******************/

    namespace detail
    {
        template <class E>
        struct is_xarray : std::false_type
        {
        };

        template <class T, layout_type L, class A, class SA>
        struct is_xarray<xarray<T, L, A, SA>> : std::true_type
        {
        };

        template <class E>
        struct is_chunked_array : std::false_type
        {
        };

        template <class CS, class E>
        struct is_chunked_array<xchunked_array<CS, E>> : std::true_type
        {
        };

        template <class E>
        struct zwrapper_builder
        {
            using closure_type = xtl::closure_type_t<E>;
            using wrapper_type = std::conditional_t<is_xarray<std::decay_t<E>>::value,
                                                    zarray_wrapper<closure_type>,
                                                    std::conditional_t<is_chunked_array<std::decay_t<E>>::value,
                                                                       zchunked_wrapper<closure_type>,
                                                                       zexpression_wrapper<closure_type>
                                                                      >
                                                    >;

            template <class OE>
            static wrapper_type* run(OE&& e)
            {
                return new wrapper_type(std::forward<OE>(e));
            }
        };

        template <class E>
        inline auto build_zarray(E&& e)
        {
            return zwrapper_builder<E>::run(std::forward<E>(e));
        }
    }
}

#endif

