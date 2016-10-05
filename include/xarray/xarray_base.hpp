#ifndef XARRAY_BASE_HPP
#define XARRAY_BASE_HPP

#include <functional>

#include "xindex.hpp"
#include "xbroadcast.hpp"
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

        using shape_type = xshape<size_type>;
        using strides_type = xstrides<size_type>;

        using stepper = xstepper<D>;
        using const_stepper = xstepper<const D>;

        using iterator = xiterator<stepper>;
        using const_iterator = xiterator<const_stepper>;

        using storage_iterator = typename container_type::iterator;
        using const_storage_iterator = typename container_type::const_iterator;

        size_type size() const;

        size_type dimension() const;

        const shape_type& shape() const;
        const strides_type& strides() const;
        const strides_type& backstrides() const;

        void reshape(const shape_type& shape);
        void reshape(const shape_type& shape, layout l);
        void reshape(const shape_type& shape, const strides_type& strides);

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        container_type& data();
        const container_type& data() const;

        bool broadcast_shape(shape_type& shape) const;
        bool is_trivial_broadcast(const strides_type& strides) const;

        iterator begin();
        iterator end();

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

        iterator xbegin(const shape_type& shape);
        iterator xend(const shape_type& shape);

        const_iterator xbegin(const shape_type& shape) const;
        const_iterator xend(const shape_type& shape) const;
        const_iterator cxbegin(const shape_type& shape) const;
        const_iterator cxend(const shape_type& shape) const;

        stepper stepper_begin(const shape_type& shape);
        stepper stepper_end(const shape_type& shape);

        const_stepper stepper_begin(const shape_type& shape) const;
        const_stepper stepper_end(const shape_type& shape) const;

        storage_iterator storage_begin();
        storage_iterator storage_end();

        const_storage_iterator storage_begin() const;
        const_storage_iterator storage_end() const;

    protected:

        xarray_base() = default;
        ~xarray_base() = default;

        xarray_base(const xarray_base&) = default;
        xarray_base& operator=(const xarray_base&) = default;

        xarray_base(xarray_base&&) = default;
        xarray_base& operator=(xarray_base&&) = default;

        shape_type& get_shape();
        strides_type& get_strides();
        strides_type& get_backstrides();

    private:

        void adapt_strides();
        void adapt_strides(size_type i);

        shape_type m_shape;
        strides_type m_strides;
        strides_type m_backstrides;
    };

    template <class D1, class D2>
    bool operator==(const xarray_base<D1>& lhs, const xarray_base<D2>& rhs);

    template <class D1, class D2>
    bool operator!=(const xarray_base<D1>& lhs, const xarray_base<D2>& rhs);


    /****************************
     * xarray_base implementation
     ****************************/

    template <class D>
    inline auto xarray_base<D>::get_shape() -> shape_type&
    {
        return m_shape;
    }

    template <class D>
    inline auto xarray_base<D>::get_strides() -> strides_type&
    {
        return m_strides;
    }

    template <class D>
    inline auto xarray_base<D>::get_backstrides() -> strides_type&
    {
        return m_backstrides;
    }

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
            m_backstrides[i] = m_strides[i] * (m_shape[i] - 1);
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
    inline void xarray_base<D>::reshape(const shape_type& shape)
    {
        if(shape != m_shape)
        {
            reshape(shape, layout::row_major);
        }
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
        m_backstrides.resize(m_strides.size());
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
    inline bool xarray_base<D>::is_trivial_broadcast(const strides_type& str) const
    {
        return str == strides();
    }


    /******************
     * iterator api
     ******************/

    template <class D>
    inline auto xarray_base<D>::begin() -> iterator
    {
        return xbegin(shape());
    }

    template <class D>
    inline auto xarray_base<D>::end() -> iterator
    {
        return xend(shape());
    }

    template <class D>
    inline auto xarray_base<D>::begin() const -> const_iterator
    {
        return xbegin(shape());
    }

    template <class D>
    inline auto xarray_base<D>::end() const -> const_iterator
    {
        return xend(shape());
    }

    template <class D>
    inline auto xarray_base<D>::cbegin() const -> const_iterator
    {
        return begin();
    }

    template <class D>
    inline auto xarray_base<D>::cend() const -> const_iterator
    {
        return end();
    }

    template <class D>
    inline auto xarray_base<D>::xbegin(const shape_type& shape) -> iterator
    {
        return iterator(stepper_begin(shape), shape);
    }

    template <class D>
    inline auto xarray_base<D>::xend(const shape_type& shape) -> iterator
    {
        return iterator(stepper_end(shape), shape);
    }

    template <class D>
    inline auto xarray_base<D>::xbegin(const shape_type& shape) const -> const_iterator
    {
        return const_iterator(stepper_begin(shape), shape);
    }

    template <class D>
    inline auto xarray_base<D>::xend(const shape_type& shape) const -> const_iterator
    {
        return const_iterator(stepper_end(shape), shape);
    }

    template <class D>
    inline auto xarray_base<D>::cxbegin(const shape_type& shape) const -> const_iterator
    {
        return xbegin(shape);
    }

    template <class D>
    inline auto xarray_base<D>::cxend(const shape_type& shape) const -> const_iterator
    {
        return xend(shape);
    }


    /****************************
     * stepper api
     ****************************/

    template <class D>
    inline auto xarray_base<D>::stepper_begin(const shape_type& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(static_cast<derived_type*>(this), data().begin(), offset);
    }

    template <class D>
    inline auto xarray_base<D>::stepper_end(const shape_type& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(static_cast<derived_type*>(this), data().end(), offset);
    }

    template <class D>
    inline auto xarray_base<D>::stepper_begin(const shape_type& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(static_cast<const derived_type*>(this), data().begin(), offset);
    }

    template <class D>
    inline auto xarray_base<D>::stepper_end(const shape_type& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(static_cast<const derived_type*>(this), data().end(), offset);
    }


    /**************************
     * storage_iterator api
     **************************/

    template <class D>
    inline auto xarray_base<D>::storage_begin() -> storage_iterator
    {
        return data().begin();
    }

    template <class D>
    inline auto xarray_base<D>::storage_end() -> storage_iterator
    {
        return data().end();
    }

    template <class D>
    inline auto xarray_base<D>::storage_begin() const -> const_storage_iterator
    {
        return data().begin();
    }

    template <class D>
    inline auto xarray_base<D>::storage_end() const -> const_storage_iterator
    {
        return data().end();
    }


    /****************
     * Comparison
     ****************/

    template <class D1, class D2>
    inline bool operator==(const xarray_base<D1>& lhs, const xarray_base<D2>& rhs)
    {
        return lhs.shape() == rhs.shape() && lhs.strides() == rhs.strides()
            && lhs.data() == rhs.data();
    }

    template <class D1, class D2>
    inline bool operator!=(const xarray_base<D1>& lhs, const xarray_base<D2>& rhs)
    {
        return !(lhs == rhs);
    }

}

#endif

