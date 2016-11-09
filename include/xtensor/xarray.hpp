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
#include <vector>
#include <algorithm>

#include "xcontainer.hpp"
#include "xsemantic.hpp"

namespace xt
{

    /**********************
     * xarray declaration *
     **********************/

    template <class T>
    class xarray;

    template <class T>
    struct xcontainer_inner_types<xarray<T>>
    {
        using container_type = std::vector<T>;
        using shape_type = std::vector<typename container_type::size_type>;
        using strides_type = shape_type;
        using temporary_type = xarray<T>;
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
    template <class T>
    class xarray : public xcontainer<xarray<T>>,
                   public xcontainer_semantic<xarray<T>>
    {

    public:

        using self_type = xarray<T>;
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

        using closure_type = const self_type&;

        xarray();
        explicit xarray(const shape_type& shape, layout l = layout::row_major);
        explicit xarray(const shape_type& shape, const_reference value, layout l = layout::row_major);
        explicit xarray(const shape_type& shape, const strides_type& strides);
        explicit xarray(const shape_type& shape, const strides_type& strides, const_reference value);

        explicit xarray(const T& t);
        xarray(std::initializer_list<T> t);
        xarray(std::initializer_list<std::initializer_list<T>> t);
        xarray(std::initializer_list<std::initializer_list<std::initializer_list<T>>> t);
        xarray(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> t);
        xarray(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>>> t);

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

        container_type& data_impl();
        const container_type& data_impl() const;

        friend class xcontainer<xarray<T>>;
    };

    /******************************
     * xarray_adaptor declaration *
     ******************************/

    template <class C>
    class xarray_adaptor;

    template <class C>
    struct xcontainer_inner_types<xarray_adaptor<C>>
    {
        using container_type = C;
        using shape_type = std::vector<typename container_type::size_type>;
        using strides_type = shape_type;
        using temporary_type = xarray<typename C::value_type>;
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
    template <class C>
    class xarray_adaptor : public xcontainer<xarray_adaptor<C>>,
                           public xadaptor_semantic<xarray_adaptor<C>>
    {

    public:

        using self_type = xarray_adaptor<C>;
        using base_type = xcontainer<self_type>;
        using semantic_base = xadaptor_semantic<self_type>;
        using container_type = typename base_type::container_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;

        using closure_type = const self_type&;

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

        container_type& data_impl();
        const container_type& data_impl() const;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        void assign_temporary_impl(temporary_type& tmp);

        friend class xcontainer<xarray_adaptor<C>>;
        friend class xadaptor_semantic<xarray_adaptor<C>>;
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
    template <class T>
    inline xarray<T>::xarray()
        : base_type(), m_data(1, value_type())
    {
    }

    /**
     * Allocates an uninitialized xarray with the specified shape and
     * layout.
     * @param shape the shape of the xarray
     * @param l the layout of the xarray
     */
    template <class T>
    inline xarray<T>::xarray(const shape_type& shape, layout l)
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
    template <class T>
    inline xarray<T>::xarray(const shape_type& shape, const_reference value, layout l)
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
    template <class T>
    inline xarray<T>::xarray(const shape_type& shape, const strides_type& strides)
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
    template <class T>
    inline xarray<T>::xarray(const shape_type& shape, const strides_type& strides, const_reference value)
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
    template <class T>
    inline xarray<T>::xarray(const T& t)
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
    template <class T>
    inline xarray<T>::xarray(std::initializer_list<T> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    /**
     * Allocates a two-dimensional xarray.
     * @param t the elements of the xarray
     */
    template <class T>
    inline xarray<T>::xarray(std::initializer_list<std::initializer_list<T>> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    /**
     * Allocates a three-dimensional xarray.
     * @param t the elements of the xarray
     */
    template <class T>
    inline xarray<T>::xarray(std::initializer_list<std::initializer_list<std::initializer_list<T>>> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    /**
     * Allocates a four-dimensional xarray.
     * @param t the elements of the xarray
     */
    template <class T>
    inline xarray<T>::xarray(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> t)
        : base_type()
    {
        base_type::reshape(xt::shape<shape_type>(t), layout::row_major);
        nested_copy(m_data.begin(), t);
    }

    /**
     * Allocates a five-dimensional xarray.
     * @param t the elements of the xarray
     */
    template <class T>
    inline xarray<T>::xarray(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>>> t)
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
    template <class T>
    template <class E>
    inline xarray<T>::xarray(const xexpression<E>& e)
        : base_type()
    {
        semantic_base::assign(e);
    }

    /**
     * The extended assignment operator.
     */
    template <class T>
    template <class E>
    inline auto xarray<T>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class T>
    inline auto xarray<T>::data_impl() -> container_type&
    {
        return m_data;
    }

    template <class T>
    inline auto xarray<T>::data_impl() const -> const container_type&
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
    template <class C>
    inline xarray_adaptor<C>::xarray_adaptor(container_type& data)
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
    template <class C>
    inline xarray_adaptor<C>::xarray_adaptor(container_type& data, const shape_type& shape, layout l)
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
    template <class C>
    inline xarray_adaptor<C>::xarray_adaptor(container_type& data, const shape_type& shape, const strides_type& strides)
        : base_type(), m_data(data)
    {
        base_type::reshape(shape, strides);
    }
    //@}

    template <class C>
    inline xarray_adaptor<C>& xarray_adaptor<C>::operator=(const xarray_adaptor& rhs)
    {
        base_type::operator=(rhs);
        m_data = rhs.m_data;
        return *this;
    }

    template <class C>
    inline xarray_adaptor<C>& xarray_adaptor<C>::operator=(xarray_adaptor&& rhs)
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
    template <class C>
    template <class E>
    inline auto xarray_adaptor<C>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class C>
    inline auto xarray_adaptor<C>::data_impl() -> container_type&
    {
        return m_data;
    }

    template <class C>
    inline auto xarray_adaptor<C>::data_impl() const -> const container_type&
    {
        return m_data;
    }

    template <class C>
    inline void xarray_adaptor<C>::assign_temporary_impl(temporary_type& tmp)
    {
        // TODO (performance improvement) : consider moving tmps
        // shape and strides
        base_type::get_shape() = tmp.shape();
        base_type::get_strides() = tmp.strides();
        base_type::get_backstrides() = tmp.backstrides();
        m_data.resize(tmp.size());
        std::copy(tmp.data().begin(), tmp.data().end(), m_data.begin());
    }
}

#endif

