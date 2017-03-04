/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XARRAY_HPP
#define XARRAY_HPP

#include <initializer_list>
#include <utility>
#include <algorithm>

#include "xcontainer.hpp"
#include "xsemantic.hpp"

namespace xt
{

    /**********************
     * xarray declaration *
     **********************/

    template <class T, class EA, class SA>
    struct xcontainer_inner_types<xarray<T, EA, SA>>
    {
        using container_type = std::vector<T, EA>;
        using shape_type = std::vector<typename container_type::size_type, SA>;
        using strides_type = shape_type;
        using temporary_type = xarray<T, EA, SA>;
    };

    /**
     * @class xarray
     * @brief Dense multidimensional container with tensor
     * semantic.
     *
     * The xarray class implements a dense multidimensional container
     * with tensor semantic.
     *
     * @tparam T The type of objects stored in the container.
     */
    template <class T, class EA, class SA>
    class xarray : public xcontainer<xarray<T, EA, SA>>,
                   public xcontainer_semantic<xarray<T, EA, SA>>
    {

    public:

        using self_type = xarray<T, EA, SA>;
        using base_type = xcontainer<self_type>;
        using semantic_base = xcontainer_semantic<self_type>;
        using container_type = typename base_type::container_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;

        xarray();
        explicit xarray(const shape_type& shape, layout l = layout::row_major);
        explicit xarray(const shape_type& shape, const_reference value, layout l = layout::row_major);
        explicit xarray(const shape_type& shape, const strides_type& strides);
        explicit xarray(const shape_type& shape, const strides_type& strides, const_reference value);

        xarray(const value_type& t);
        xarray(nested_initializer_list_t<T, 1> t);
        xarray(nested_initializer_list_t<T, 2> t);
        xarray(nested_initializer_list_t<T, 3> t);
        xarray(nested_initializer_list_t<T, 4> t);
        xarray(nested_initializer_list_t<T, 5> t);

        ~xarray() = default;

        xarray(const xarray&) = default;
        xarray& operator=(const xarray&) = default;

        xarray(xarray&&) = default;
        xarray& operator=(xarray&&) = default;

        template <class E>
        xarray(const xexpression<E>& e);

        template <class E>
        xarray& operator=(const xexpression<E>& e);

    private:

        container_type m_data;

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        friend class xcontainer<xarray<T, EA, SA>>;
    };

    /******************************
     * xarray_adaptor declaration *
     ******************************/

    template <class C, class EA = std::allocator<typename C::value_type>, class SA = std::allocator<typename C::size_type>>
    class xarray_adaptor;

    template <class C, class EA, class SA>
    struct xcontainer_inner_types<xarray_adaptor<C, EA, SA>>
    {
        using container_type = C;
        using shape_type = std::vector<typename container_type::size_type, SA>;
        using strides_type = shape_type;
        using temporary_type = xarray<typename C::value_type, EA, SA>;
    };

    /**
     * @class xarray_adaptor
     * @brief Dense multidimensional container adaptor with
     * tensor semantic.
     *
     * The xarray_adaptor class implements a dense multidimensional
     * container adaptor with tensor semantic. It is used to provide
     * a multidimensional container semantic and a tensor semantic to
     * stl-like containers.
     *
     * @tparam C The container type to adapt.
     */
    template <class C, class EA, class SA>
    class xarray_adaptor : public xcontainer<xarray_adaptor<C, EA, SA>>,
                           public xadaptor_semantic<xarray_adaptor<C, EA, SA>>
    {

    public:

        using self_type = xarray_adaptor<C, EA, SA>;
        using base_type = xcontainer<self_type>;
        using semantic_base = xadaptor_semantic<self_type>;
        using container_type = typename base_type::container_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;

        xarray_adaptor(container_type& data);
        xarray_adaptor(container_type& data, const shape_type& shape, layout l = layout::row_major);
        xarray_adaptor(container_type& data, const shape_type& shape, const strides_type& strides);

        ~xarray_adaptor() = default;

        xarray_adaptor(const xarray_adaptor&) = default;
        xarray_adaptor& operator=(const xarray_adaptor&);

        xarray_adaptor(xarray_adaptor&&) = default;
        xarray_adaptor& operator=(xarray_adaptor&&);

        template <class E>
        xarray_adaptor& operator=(const xexpression<E>& e);

    private:

        container_type& m_data;

        container_type& data_impl() noexcept;
        const container_type& data_impl() const noexcept;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        void assign_temporary_impl(temporary_type& tmp);

        friend class xcontainer<xarray_adaptor<C, EA, SA>>;
        friend class xadaptor_semantic<xarray_adaptor<C, EA, SA>>;
    };

    /*************************
     * xarray implementation *
     *************************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Allocates an uninitialized xarray that holds 0 element.
     */
    template <class T, class EA, class SA>
    inline xarray<T, EA, SA>::xarray()
        : base_type(), m_data(1, value_type())
    {
    }

    /**
     * Allocates an uninitialized xarray with the specified shape and
     * layout.
     * @param shape the shape of the xarray
     * @param l the layout of the xarray
     */
    template <class T, class EA, class SA>
    inline xarray<T, EA, SA>::xarray(const shape_type& shape, layout l)
        : base_type()
    {
        base_type::reshape(shape, l);
    }

    /**
     * Allocates an xarray with the specified shape and layout. Elements
     * are initialized to the specified value.
     * @param shape the shape of the xarray
     * @param value the value of the elements
     * @param l the layout of the xarray
     */
    template <class T, class EA, class SA>
    inline xarray<T, EA, SA>::xarray(const shape_type& shape, const_reference value, layout l)
        : base_type()
    {
        base_type::reshape(shape, l);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    /**
     * Allocates an uninitialized xarray with the specified shape and strides.
     * @param shape the shape of the xarray
     * @param strides the strides of the xarray
     */
    template <class T, class EA, class SA>
    inline xarray<T, EA, SA>::xarray(const shape_type& shape, const strides_type& strides)
        : base_type()
    {
        base_type::reshape(shape, strides);
    }

    /**
     * Allocates an uninitialized xarray with the specified shape and strides.
     * Elements are initialized to the specified value.
     * @param shape the shape of the xarray
     * @param strides the strides of the xarray
     * @param value the value of the elements
     */
    template <class T, class EA, class SA>
    inline xarray<T, EA, SA>::xarray(const shape_type& shape, const strides_type& strides, const_reference value)
        : base_type()
    {
        base_type::reshape(shape, strides);
        std::fill(m_data.begin(), m_data.end(), value);
    }

    /**
     * Allocates an xarray that holds a single element initialized to the
     * specified value.
     * @param t the value of the element
     */
    template <class T, class EA, class SA>
    inline xarray<T, EA, SA>::xarray(const value_type& t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }
    //@}

    /**
     * @name Constructors from initializer list
     */
    //@{
    /**
     * Allocates a one-dimensional xarray.
     * @param t the elements of the xarray
     */
    template <class T, class EA, class SA>
    inline xarray<T, EA, SA>::xarray(nested_initializer_list_t<T, 1> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    /**
     * Allocates a two-dimensional xarray.
     * @param t the elements of the xarray
     */
    template <class T, class EA, class SA>
    inline xarray<T, EA, SA>::xarray(nested_initializer_list_t<T, 2> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    /**
     * Allocates a three-dimensional xarray.
     * @param t the elements of the xarray
     */
    template <class T, class EA, class SA>
    inline xarray<T, EA, SA>::xarray(nested_initializer_list_t<T, 3> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    /**
     * Allocates a four-dimensional xarray.
     * @param t the elements of the xarray
     */
    template <class T, class EA, class SA>
    inline xarray<T, EA, SA>::xarray(nested_initializer_list_t<T, 4> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    /**
     * Allocates a five-dimensional xarray.
     * @param t the elements of the xarray
     */
    template <class T, class EA, class SA>
    inline xarray<T, EA, SA>::xarray(nested_initializer_list_t<T, 5> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }
    //@}


    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended copy constructor.
     */
    template <class T, class EA, class SA>
    template <class E>
    inline xarray<T, EA, SA>::xarray(const xexpression<E>& e)
        : base_type()
    {
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class T, class EA, class SA>
    template <class E>
    inline auto xarray<T, EA, SA>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class T, class EA, class SA>
    inline auto xarray<T, EA, SA>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class T, class EA, class SA>
    inline auto xarray<T, EA, SA>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }

    /******************
     * xarray_adaptor *
     ******************/

    /**
     * @name Constructors
     */
    //@{
    /**
     * Constructs an xarray_adaptor of the given stl-like container.
     * @param data the container to adapt
     */
    template <class C, class EA, class SA>
    inline xarray_adaptor<C, EA, SA>::xarray_adaptor(container_type& data)
        : base_type(), m_data(data)
    {
    }

    /**
     * Constructs an xarray_adaptor of the given stl-like container,
     * with the specified shape and layout.
     * @param data the container to adapt
     * @param shape the shape of the xarray_adaptor
     * @param l the layout of the xarray_adaptor
     */
    template <class C, class EA, class SA>
    inline xarray_adaptor<C, EA, SA>::xarray_adaptor(container_type& data, const shape_type& shape, layout l)
        : base_type(), m_data(data)
    {
        base_type::reshape(shape, l);
    }

    /**
     * Constructs an xarray_adaptor of the given stl-like container,
     * with the specified shape and strides.
     * @param data the container to adapt
     * @param shape the shape of the xarray_adaptor
     * @param strides the strides of the xarray_adaptor
     */
    template <class C, class EA, class SA>
    inline xarray_adaptor<C, EA, SA>::xarray_adaptor(container_type& data, const shape_type& shape, const strides_type& strides)
        : base_type(), m_data(data)
    {
        base_type::reshape(shape, strides);
    }
    //@}

    template <class C, class EA, class SA>
    inline xarray_adaptor<C, EA, SA>& xarray_adaptor<C, EA, SA>::operator=(const xarray_adaptor& rhs)
    {
        base_type::operator=(rhs);
        m_data = rhs.m_data;
        return *this;
    }

    template <class C, class EA, class SA>
    inline xarray_adaptor<C, EA, SA>& xarray_adaptor<C, EA, SA>::operator=(xarray_adaptor&& rhs)
    {
        base_type::operator=(std::move(rhs));
        m_data = rhs.m_data;
        return *this;
    }

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class C, class EA, class SA>
    template <class E>
    inline auto xarray_adaptor<C, EA, SA>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class C, class EA, class SA>
    inline auto xarray_adaptor<C, EA, SA>::data_impl() noexcept -> container_type&
    {
        return m_data;
    }

    template <class C, class EA, class SA>
    inline auto xarray_adaptor<C, EA, SA>::data_impl() const noexcept -> const container_type&
    {
        return m_data;
    }

    template <class C, class EA, class SA>
    inline void xarray_adaptor<C, EA, SA>::assign_temporary_impl(temporary_type& tmp)
    {
        // TODO (performance improvement) : consider moving tmps
        // shape and strides
        base_type::get_shape() = tmp.shape();
        base_type::get_strides() = tmp.strides();
        base_type::get_backstrides() = tmp.backstrides();
        m_data.resize(tmp.size());
        std::copy(tmp.data().cbegin(), tmp.data().cend(), m_data.begin());
    }
}

#endif
