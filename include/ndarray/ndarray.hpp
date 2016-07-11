#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <utility>
#include "ndarray_base.hpp"

namespace ndarray
{

    /*************************
     * ndarray declaration
     *************************/

    template <class T>
    class ndarray;

    template <class T>
    struct array_inner_types<ndarray<T>>
    {
        using container_type = std::vector<T>;
    };

    template <class T>
    class ndarray : public ndarray_base<ndarray<T>>
    {

    public:

        using base_type = ndarray_base<ndarray<T>>;
        using container_type = typename base_type::container_type;
        using const_reference = typename base_type::const_reference;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;

        ndarray() = default;
        explicit ndarray(const shape_type& shape, layout l = layout::row_major);
        ndarray(const shape_type& shape, const_reference value, layout l = layout::row_major);
        ndarray(const shape_type& shape, const strides_type& strides);
        ndarray(const shape_type& shape, const strides_type& strides, const_reference value);

        ~ndarray() = default;

        ndarray(const ndarray&) = default;
        ndarray& operator=(const ndarray&) = default;

        ndarray(ndarray&&) = default;
        ndarray& operator=(ndarray&&) = default;

    private:

        container_type m_data;

        container_type& data_impl();
        const container_type& data_impl() const;

        friend class ndarray_base<ndarray<T>>;
    };


    /*********************************
     * ndarray_adaptor declaration
     *********************************/

    template <class C>
    class ndarray_adaptor;

    template <class C>
    struct array_inner_types<ndarray_adaptor<C>>
    {
        using container_type = C;
    };

    template <class C>
    class ndarray_adaptor : public ndarray_base<ndarray_adaptor<C>>
    {

    public:

        using base_type = ndarray_base<ndarray_adaptor<C>>;
        using container_type = typename base_type::container_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;

        ndarray_adaptor(container_type& data);
        ndarray_adaptor(container_type& data, const shape_type& shape, layout l = layout::row_major);
        ndarray_adaptor(container_type& data, const shape_type& shape, const strides_type& strides);

        ~ndarray_adaptor() = default;

        ndarray_adaptor(const ndarray_adaptor&) = default;
        ndarray_adaptor& operator=(const ndarray_adaptor&);

        ndarray_adaptor(ndarray_adaptor&&) = default;
        ndarray_adaptor& operator=(ndarray_adaptor&&);

    private:

        container_type& m_data;

        container_type& data_impl();
        const container_type& data_impl() const;

        friend class ndarray_base<ndarray_adaptor<C>>;
    };


    /****************************
     * ndarray implementation
     ****************************/

    template <class T>
    inline ndarray<T>::ndarray(const shape_type& shape, layout l)
        : base_type(shape, l)
    {
    }

    template <class T>
    inline ndarray<T>::ndarray(const shape_type& shape, const_reference value, layout l)
        : base_type(shape, value, l)
    {
    }

    template <class T>
    inline ndarray<T>::ndarray(const shape_type& shape, const strides_type& strides)
        : base_type(shape, strides)
    {
    }

    template <class T>
    inline ndarray<T>::ndarray(const shape_type& shape, const strides_type& strides, const_reference value)
        : base_type(shape, strides, value)
    {
    }

    template <class T>
    inline auto ndarray<T>::data_impl() -> container_type&
    {
        return m_data;
    }

    template <class T>
    inline auto ndarray<T>::data_impl() const -> const container_type&
    {
        return m_data;
    }


    /*********************
     * ndarray_adaptor
     *********************/

    template <class C>
    inline ndarray_adaptor<C>::ndarray_adaptor(container_type& data)
        : base_type(), m_data(data)
    {
    }

    template <class C>
    inline ndarray_adaptor<C>::ndarray_adaptor(container_type& data, const shape_type& shape, layout l)
        : base_type(), m_data(data)
    {
        base_type::reshape(shape, l);
    }

    template <class C>
    inline ndarray_adaptor<C>::ndarray_adaptor(container_type& data, const shape_type& shape, const strides_type& strides)
        : base_type(), m_data(data)
    {
        base_type::reshape(shape, strides);
    }

    template <class C>
    inline ndarray_adaptor<C>& ndarray_adaptor<C>::operator=(const ndarray_adaptor& rhs)
    {
        base_type::operator=(rhs);
        m_data = rhs.m_data;
        return *this;
    }

    template <class C>
    inline ndarray_adaptor<C>& ndarray_adaptor<C>::operator=(ndarray_adaptor&& rhs)
    {
        base_type::operator=(std::move(rhs));
        m_data = rhs.m_data;
        return *this;
    }

    template <class C>
    inline auto ndarray_adaptor<C>::data_impl() -> container_type&
    {
        return m_data;
    }

    template <class C>
    inline auto ndarray_adaptor<C>::data_impl() const -> const container_type&
    {
        return m_data;
    }

}

#endif

