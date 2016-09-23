#ifndef XARRAY_BASE_HPP
#define XARRAY_BASE_HPP

#include <functional>

#include "xindex.hpp"
#include "broadcast.hpp"
#include "xoperation.hpp"
#include "xmath.hpp"

namespace qs
{

    enum class layout
    {
        row_major,
        column_major
    };

    template <class D>
    class xarray_base
    {

    public:

        using derived_type = D;

        using inner_types = array_inner_types<D>;
        using container_type = typename inner_types::container_type;
        using value_type = typename container_type::value_type;
        using reference = typename container_type::reference;
        using const_reference = typename container_type::const_reference;
        using pointer = typename container_type::pointer;
        using const_pointer = typename container_type::const_pointer;
        using size_type = typename container_type::size_type;
        using difference_type = typename container_type::difference_type;

        using iterator = typename container_type::iterator;
        using const_iterator = typename container_type::const_iterator;
        using reverse_iterator = typename container_type::reverse_iterator;
        using const_reverse_iterator = typename container_type::const_reverse_iterator;

        using broadcasting_iterator = broadcast_iterator<D>;

        using shape_type = array_shape<size_type>;
        using strides_type = array_strides<size_type>;

        size_type size() const;

        size_type dimension() const;

        const shape_type& shape() const;
        const strides_type& strides() const;
        const strides_type& backstrides() const;

        void reshape(const shape_type& shape, layout l = layout::row_major);
        void reshape(const shape_type& shape, const strides_type& strides);

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        container_type& data();
        const container_type& data() const;

        bool broadcast_shape(shape_type& shape) const;

        iterator begin();
        iterator end();

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

        reverse_iterator rbegin();
        reverse_iterator rend();

        const_reverse_iterator rbegin() const;
        const_reverse_iterator rend() const;
        const_reverse_iterator crbegin() const;
        const_reverse_iterator crend() const;

        broadcasting_iterator begin(const shape_type& shape) const;
        broadcasting_iterator end(const shape_type& shape) const;

    protected:

        xarray_base() = default;
        ~xarray_base() = default;

        xarray_base(const xarray_base&) = default;
        xarray_base& operator=(const xarray_base&) = default;

        xarray_base(xarray_base&&) = default;
        xarray_base& operator=(xarray_base&&) = default;

    private:

        void adapt_strides();
        void adapt_strides(size_type i);

        shape_type m_shape;
        strides_type m_strides;
        strides_type m_backstrides;
    };


    /****************************
     * xarray_base implementation
     ****************************/

    template <class D>
    inline void xarray_base<D>::adapt_strides()
    {
        for(size_type i = 0; i < m_shape.size(); ++i)
        {
            adapt_strides(i);
        }
    }

    template <class D>
    inline void xarray_base<D>::adapt_strides(size_type i)
    {
        if(m_shape[i] == 1)
        {
            m_strides[i] = 0;
            m_backstrides[i] = 0;
        }
        else
        {
            m_backstrides[i] = m_strides[i] * m_shape[i] - 1;
        }
    }

    template <class D>
    inline auto xarray_base<D>::size() const -> size_type
    {
        return data().size();
    }

    template <class D>
    inline auto xarray_base<D>::dimension() const -> size_type
    {
        return m_shape.size();
    }
    
    template <class D>
    inline auto xarray_base<D>::shape() const -> const shape_type&
    {
        return m_shape;
    }

    template <class D>
    inline auto xarray_base<D>::strides() const -> const strides_type&
    {
        return m_strides;
    }

    template <class D>
    inline auto xarray_base<D>::backstrides() const -> const strides_type&
    {
        return m_backstrides;
    }

    template <class D>
    inline void xarray_base<D>::reshape(const shape_type& shape, layout l)
    {
        m_shape = shape;
        m_strides.resize(m_shape.size());
        m_backstrides.resize(m_shape.size());
        if(l == layout::row_major)
        {
            m_strides.back() = size_type(1);
            for(size_type i = m_strides.size() - 1; i != 0; --i)
            {
                m_strides[i - 1] = m_strides[i] * m_shape[i];
                adapt_strides(i);
            }
            data().resize(m_strides.front() * m_shape.front());
            adapt_strides(size_type(0));
        }
        else
        {
            m_strides.front() = size_type(1);
            for(size_type i = 1; i < m_strides.size(); ++i)
            {
                m_strides[i] = m_strides[i - 1] * m_shape[i - 1];
                adapt_strides(i - 1);
            }
            data().resize(m_strides.back() * m_shape.back());
            adapt_strides(m_strides.size() - 1);
        }
    }

    template <class D>
    inline void xarray_base<D>::reshape(const shape_type& shape, const strides_type& strides)
    {
        m_shape = shape;
        m_strides = strides;
        adapt_strides();
        data().resize(data_size(m_shape));
    }

    template <class D>
    template <class... Args>
    inline auto xarray_base<D>::operator()(Args... args) -> reference
    {
        size_type index = data_offset(m_strides, static_cast<size_type>(args)...);
        return data()[index];
    }

    template <class D>
    template <class... Args>
    inline auto xarray_base<D>::operator()(Args... args) const -> const_reference
    {
        size_type index = data_offset(m_strides, static_cast<size_type>(args)...);
        return data()[index];
    }

    template <class D>
    inline auto xarray_base<D>::data() -> container_type&
    {
        return static_cast<derived_type*>(this)->data_impl();
    }

    template <class D>
    inline auto xarray_base<D>::data() const -> const container_type&
    {
        return static_cast<const derived_type*>(this)->data_impl();
    }

    template <class D>
    inline bool xarray_base<D>::broadcast_shape(shape_type& shape) const
    {
        return qs::broadcast_shape(m_shape, shape);
    }

    template <class D>
    inline auto xarray_base<D>::begin() -> iterator
    {
        return data().begin();
    }

    template <class D>
    inline auto xarray_base<D>::end() -> iterator
    {
        return data().end();
    }

    template <class D>
    inline auto xarray_base<D>::begin() const -> const_iterator
    {
        return data().begin();
    }

    template <class D>
    inline auto xarray_base<D>::end() const -> const_iterator
    {
        return data().end();
    }

    template <class D>
    inline auto xarray_base<D>::cbegin() const -> const_iterator 
    {
        return data().cbegin();
    }

    template <class D>
    inline auto xarray_base<D>::cend() const -> const_iterator
    {
        return data().cend();
    }

    template <class D>
    inline auto xarray_base<D>::rbegin() -> reverse_iterator
    {
        return data().rbegin();
    }

    template <class D>
    inline auto xarray_base<D>::rend() -> reverse_iterator
    {
        return data().rend();
    }

    template <class D>
    inline auto xarray_base<D>::rbegin() const -> const_reverse_iterator
    {
        return data().rbegin();
    }

    template <class D>
    inline auto xarray_base<D>::rend() const -> const_reverse_iterator
    {
        return data().rend();
    }

    template <class D>
    inline auto xarray_base<D>::crbegin() const -> const_reverse_iterator
    {
        return data().rbegin();
    }

    template <class D>
    inline auto xarray_base<D>::crend() const -> const_reverse_iterator
    {
        return data().rend();
    }

    template <class D>
    inline auto xarray_base<D>::begin(const shape_type& shape) const -> broadcasting_iterator
    {
        return broadcasting_iterator(static_cast<const D*>(this), data().begin());
    }

    template <class D>
    inline auto xarray_base<D>::end(const shape_type& shape) const -> broadcasting_iterator
    {
        return broadcasting_iterator(static_cast<const D*>(this), data().end());
    }

}

#endif

