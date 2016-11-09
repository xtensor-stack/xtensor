/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XCONTAINER_HPP
#define XCONTAINER_HPP

#include <numeric>
#include <functional>

#include "xiterator.hpp"
#include "xoperation.hpp"
#include "xmath.hpp"

namespace xt
{
    template <class C>
    struct xcontainer_inner_types;

    enum class layout
    {
        row_major,
        column_major
    };

    /**
     * @class xcontainer
     * @brief Base class for dense multidimensional containers.
     *
     * The xcontainer class defines the interface for dense multidimensional
     * container classes. It does not embed any data container, this responsibility
     * is delegated to the inheriting classes.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xcontainer
     *           provides the interface.
     */
    template <class D>
    class xcontainer
    {

    public:

        using derived_type = D;

        using inner_types = xcontainer_inner_types<D>;
        using container_type = typename inner_types::container_type;
        using value_type = typename container_type::value_type;
        using reference = typename container_type::reference;
        using const_reference = typename container_type::const_reference;
        using pointer = typename container_type::pointer;
        using const_pointer = typename container_type::const_pointer;
        using size_type = typename container_type::size_type;
        using difference_type = typename container_type::difference_type;

        using shape_type = typename inner_types::shape_type;
        using index_type = shape_type;
        using strides_type = typename inner_types::strides_type;

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

        reference operator[](const index_type& index);
        const_reference operator[](const index_type& index) const;

        container_type& data();
        const container_type& data() const;

        template <class S>
        bool broadcast_shape(S& shape) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const;

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

        template <class S>
        stepper stepper_begin(const S& shape);
        template <class S>
        stepper stepper_end(const S& shape);

        template <class S>
        const_stepper stepper_begin(const S& shape) const;
        template <class S>
        const_stepper stepper_end(const S& shape) const;

        storage_iterator storage_begin();
        storage_iterator storage_end();

        const_storage_iterator storage_begin() const;
        const_storage_iterator storage_end() const;

    protected:

        xcontainer() = default;
        ~xcontainer() = default;

        xcontainer(const xcontainer&) = default;
        xcontainer& operator=(const xcontainer&) = default;

        xcontainer(xcontainer&&) = default;
        xcontainer& operator=(xcontainer&&) = default;

        shape_type& get_shape();
        strides_type& get_strides();
        strides_type& get_backstrides();

    private:

        void adapt_strides();
        void adapt_strides(size_type i);

        size_type data_size() const;

        size_type data_offset_impl() const;

        template <class... Args>
        size_type data_offset_impl(size_type i, Args... args) const;

        template <class... Args>
        size_type data_offset(Args... args) const;

        size_type data_offset(const index_type& index) const;

        shape_type m_shape;
        strides_type m_strides;
        strides_type m_backstrides;
    };

    template <class D1, class D2>
    bool operator==(const xcontainer<D1>& lhs, const xcontainer<D2>& rhs);

    template <class D1, class D2>
    bool operator!=(const xcontainer<D1>& lhs, const xcontainer<D2>& rhs);

    /******************************
     * xcontainer implementation *
     ******************************/

    template <class D>
    inline auto xcontainer<D>::get_shape() -> shape_type&
    {
        return m_shape;
    }

    template <class D>
    inline auto xcontainer<D>::get_strides() -> strides_type&
    {
        return m_strides;
    }

    template <class D>
    inline auto xcontainer<D>::get_backstrides() -> strides_type&
    {
        return m_backstrides;
    }

    template <class D>
    inline void xcontainer<D>::adapt_strides()
    {
        for(size_type i = 0; i < m_shape.size(); ++i)
        {
            adapt_strides(i);
        }
    }

    template <class D>
    inline void xcontainer<D>::adapt_strides(size_type i)
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
    inline auto xcontainer<D>::data_size() const -> size_type
    {
        return std::accumulate(m_shape.begin(), m_shape.end(), size_type(1), std::multiplies<size_type>());
    }

    template <class D>
    inline auto xcontainer<D>::data_offset_impl() const -> size_type
    {
        return 0;
    }

    template <class D>
    template <class... Args>
    inline auto xcontainer<D>::data_offset_impl(size_type i, Args... args) const -> size_type
    {
        return i * m_strides[m_strides.size() - sizeof...(args)-1] + data_offset_impl(args...);
    }

    template <class D>
    template <class... Args>
    inline auto xcontainer<D>::data_offset(Args... args) const -> size_type
    {
        return data_offset_impl(args...);
    }

    template <class D>
    inline auto xcontainer<D>::data_offset(const index_type& index) const -> size_type
    {
        // VS2015 workaround : index.begin() + index.size() - m_strides.size()
        // doesn't compile
        auto iter = index.begin();
        iter += index.size() - m_strides.size();
        return std::inner_product(m_strides.begin(), m_strides.end(), iter, size_type(0));
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of element in the container.
     */
    template <class D>
    inline auto xcontainer<D>::size() const -> size_type
    {
        return data().size();
    }

    /**
     * Returns the number of dimensions of the container.
     */
    template <class D>
    inline auto xcontainer<D>::dimension() const -> size_type
    {
        return m_shape.size();
    }

    /**
     * Returns the shape of the container.
     */
    template <class D>
    inline auto xcontainer<D>::shape() const -> const shape_type&
    {
        return m_shape;
    }

    /**
     * Returns the strides of the container.
     */
    template <class D>
    inline auto xcontainer<D>::strides() const -> const strides_type&
    {
        return m_strides;
    }

    /**
     * Returns the backstrides of the container.
     */
    template <class D>
    inline auto xcontainer<D>::backstrides() const -> const strides_type&
    {
        return m_backstrides;
    }

    /**
     * Reshapes the container.
     * @param shape the new shape
     */
    template <class D>
    inline void xcontainer<D>::reshape(const shape_type& shape)
    {
        if(shape != m_shape)
        {
            reshape(shape, layout::row_major);
        }
    }

    /**
     * Reshapes the container.
     * @param shape the new shape
     * @param l the new layout
     */
    template <class D>
    inline void xcontainer<D>::reshape(const shape_type& shape, layout l)
    {
        m_shape = shape;
        resize_container(m_strides, m_shape.size());
        resize_container(m_backstrides, m_shape.size());
        size_type data_size = 1;
        if(l == layout::row_major)
        {
            for(size_type i = m_strides.size(); i != 0; --i)
            {
                m_strides[i - 1] = data_size;
                data_size = m_strides[i - 1] * m_shape[i - 1];
                adapt_strides(i - 1);
            }
        }
        else
        {
            for(size_type i = 0; i < m_strides.size(); ++i)
            {
                m_strides[i] = data_size;
                data_size = m_strides[i] * m_shape[i];
                adapt_strides(i);
            }
        }
        data().resize(data_size);
    }

    /**
     * Reshapes the container.
     * @param shape the new shape
     * @param strides the new strides
     */
    template <class D>
    inline void xcontainer<D>::reshape(const shape_type& shape, const strides_type& strides)
    {
        m_shape = shape;
        m_strides = strides;
        resize_container(m_backstrides, m_strides.size());
        adapt_strides();
        data().resize(data_size());
    }
    //@}

    /**
     * @name Data
     */
    //@{
    /**
     * Returns a reference to the element at the specified position in the container.
     * @param args a list of indices specifying the position in the container. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the container.
     */
    template <class D>
    template <class... Args>
    inline auto xcontainer<D>::operator()(Args... args) -> reference
    {
        size_type index = data_offset(static_cast<size_type>(args)...);
        return data()[index];
    }

    /**
     * Returns a constant reference to the element at the specified position in the container.
     * @param args a list of indices specifying the position in the container. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the container.
     */
    template <class D>
    template <class... Args>
    inline auto xcontainer<D>::operator()(Args... args) const -> const_reference
    {
        size_type index = data_offset(static_cast<size_type>(args)...);
        return data()[index];
    }

    /**
     * Returns a reference to the element at the specified position in the container.
     * @param index a list of indices specifying the position in the container. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the container.
     */
    template <class D>
    inline auto xcontainer<D>::operator[](const index_type& index) -> reference
    {
        return data()[data_offset(index)];
    }

    /**
    * Returns a constant reference to the element at the specified position in the container.
    * @param index a list of indices specifying the position in the container. Indices
    * must be unsigned integers, the number of indices in the list should be equal or greater
    * than the number of dimensions of the container.
    */
    template <class D>
    inline auto xcontainer<D>::operator[](const index_type& index) const -> const_reference
    {
        return data()[data_offset(index)];
    }

    /**
     * Returns a reference to the buffer containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::data() -> container_type&
    {
        return static_cast<derived_type*>(this)->data_impl();
    }

    /**
     * Returns a constant reference to the buffer containing the elements of the
     * container.
     */
    template <class D>
    inline auto xcontainer<D>::data() const -> const container_type&
    {
        return static_cast<const derived_type*>(this)->data_impl();
    }
    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the container to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcast is trivial
     */
    template <class D>
    template <class S>
    inline bool xcontainer<D>::broadcast_shape(S& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see wether
     * the broadcast is trivial.
     * @return a boolean indicating whether the broadcast is trivial
     */
    template <class D>
    template <class S>
    inline bool xcontainer<D>::is_trivial_broadcast(const S& str) const
    {
        return str.size() == strides().size() &&
            std::equal(str.cbegin(), str.cend(), strides().begin());
    }
    //@}

    /****************
     * iterator api *
     ****************/

    /**
     * @name Iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the container.
     */
    template <class D>
    inline auto xcontainer<D>::begin() -> iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns an iterator to the element following the last element
     * of the container.
     */
    template <class D>
    inline auto xcontainer<D>::end() -> iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the container.
     */
    template <class D>
    inline auto xcontainer<D>::begin() const -> const_iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the container.
     */
    template <class D>
    inline auto xcontainer<D>::end() const -> const_iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the container.
     */
    template <class D>
    inline auto xcontainer<D>::cbegin() const -> const_iterator
    {
        return begin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the container.
     */
    template <class D>
    inline auto xcontainer<D>::cend() const -> const_iterator
    {
        return end();
    }

    /**
     * Returns an iterator to the first element of the container. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class D>
    inline auto xcontainer<D>::xbegin(const shape_type& shape) -> iterator
    {
        return iterator(stepper_begin(shape), shape);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * container. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class D>
    inline auto xcontainer<D>::xend(const shape_type& shape) -> iterator
    {
        return iterator(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the container. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class D>
    inline auto xcontainer<D>::xbegin(const shape_type& shape) const -> const_iterator
    {
        return const_iterator(stepper_begin(shape), shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * container. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class D>
    inline auto xcontainer<D>::xend(const shape_type& shape) const -> const_iterator
    {
        return const_iterator(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the container. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class D>
    inline auto xcontainer<D>::cxbegin(const shape_type& shape) const -> const_iterator
    {
        return xbegin(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * container. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class D>
    inline auto xcontainer<D>::cxend(const shape_type& shape) const -> const_iterator
    {
        return xend(shape);
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class D>
    template <class S>
    inline auto xcontainer<D>::stepper_begin(const S& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(static_cast<derived_type*>(this), data().begin(), offset);
    }

    template <class D>
    template <class S>
    inline auto xcontainer<D>::stepper_end(const S& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(static_cast<derived_type*>(this), data().end(), offset);
    }

    template <class D>
    template <class S>
    inline auto xcontainer<D>::stepper_begin(const S& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(static_cast<const derived_type*>(this), data().begin(), offset);
    }

    template <class D>
    template <class S>
    inline auto xcontainer<D>::stepper_end(const S& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(static_cast<const derived_type*>(this), data().end(), offset);
    }

    /************************
     * storage_iterator api *
     ************************/

    /**
     * @name Storage iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the buffer containing
     * the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::storage_begin() -> storage_iterator
    {
        return data().begin();
    }

    /**
     * Returns an iterator to the element following the last element of
     * the buffer containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::storage_end() -> storage_iterator
    {
        return data().end();
    }

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::storage_begin() const -> const_storage_iterator
    {
        return data().begin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::storage_end() const -> const_storage_iterator
    {
        return data().end();
    }
    //@}

    /**************
     * comparison *
     **************/

    /**
     * @memberof xcontainer
     * Compares the content of two containers.
     * @param lhs the first container
     * @param rhs the second container
     * @return true if the container are equals
     */
    template <class D1, class D2>
    inline bool operator==(const xcontainer<D1>& lhs, const xcontainer<D2>& rhs)
    {
        return lhs.shape() == rhs.shape() && lhs.strides() == rhs.strides()
            && lhs.data() == rhs.data();
    }

    /**
     * @memberof xcontainer
     * Compares the content of two containers.
     * @param lhs the first container
     * @param rhs the second container
     * @return true if the container are different
     */
    template <class D1, class D2>
    inline bool operator!=(const xcontainer<D1>& lhs, const xcontainer<D2>& rhs)
    {
        return !(lhs == rhs);
    }
}

#endif

