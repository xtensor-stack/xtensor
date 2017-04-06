/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XCONTAINER_HPP
#define XCONTAINER_HPP

#include <functional>
#include <numeric>
#include <stdexcept>

#include "xiterable.hpp"
#include "xiterator.hpp"
#include "xmath.hpp"
#include "xoperation.hpp"
#include "xstrides.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    template <class D>
    struct xcontainer_iterable_types
    {
        using inner_shape_type = typename xcontainer_inner_types<D>::inner_shape_type;
        using container_type = typename xcontainer_inner_types<D>::container_type;
        using iterator = typename container_type::iterator;
        using const_iterator = typename container_type::const_iterator;
        using stepper = xstepper<D>;
        using const_stepper = xstepper<const D>;
        using broadcast_iterator = xiterator<stepper, inner_shape_type*>;
        using const_broadcast_iterator = xiterator<const_stepper, inner_shape_type*>;
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
    class xcontainer : public xiterable<D>
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
        using strides_type = typename inner_types::strides_type;
        using backstrides_type = typename inner_types::backstrides_type;

        using inner_shape_type = typename inner_types::inner_shape_type;
        using inner_strides_type = typename inner_types::inner_strides_type;
        using inner_backstrides_type = typename inner_types::inner_backstrides_type;

        using iterable_base = xiterable<D>;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using broadcast_iterator = typename iterable_base::broadcast_iterator;
        using const_broadcast_iterator = typename iterable_base::broadcast_iterator;

        size_type size() const noexcept;

        size_type dimension() const noexcept;

        const inner_shape_type& shape() const noexcept;
        const inner_strides_type& strides() const noexcept;
        const inner_backstrides_type& backstrides() const noexcept;

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        reference operator[](const xindex& index);
        reference operator[](size_type i);
        const_reference operator[](const xindex& index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        reference element(It first, It last);
        template <class It>
        const_reference element(It first, It last) const;

        container_type& data() noexcept;
        const container_type& data() const noexcept;

        value_type* raw_data() noexcept;
        const value_type* raw_data() const noexcept;
        const size_type raw_data_offset() const noexcept;

        template <class S>
        bool broadcast_shape(S& shape) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const noexcept;

        iterator begin() noexcept;
        iterator end() noexcept;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape) noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape) const noexcept;

    protected:

        xcontainer() = default;
        ~xcontainer() = default;

        xcontainer(const xcontainer&) = default;
        xcontainer& operator=(const xcontainer&) = default;

        xcontainer(xcontainer&&) = default;
        xcontainer& operator=(xcontainer&&) = default;

    private:

        inner_shape_type& mutable_shape();
        inner_strides_type& mutable_strides();
        inner_backstrides_type& mutable_backstrides();

        derived_type& derived_cast();
        const derived_type& derived_cast() const;
    };

    /**
     * @class xstrided_container
     * @brief Partial implementation of xcontainer that embeds the strides and the shape
     *
     * The xstrided_container class is a partial implementation of the xcontainer interface
     * that embed the strides and the shape of the multidimensional container. It does
     * not embed the data container, this responsibility is delegated to the inheriting
     * classes.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xstrided
     *           provides the partial imlpementation of xcontainer.
     */
    template <class D, layout L>
    class xstrided_container : public xcontainer<D>
    {

    public:

        using base_type = xcontainer<D>;
        using container_type = typename base_type::container_type;
        using value_type = typename base_type::value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using size_type = typename base_type::size_type;
        using shape_type = typename base_type::shape_type;
        using strides_type = typename base_type::strides_type;
        using inner_shape_type = typename base_type::inner_shape_type;
        using inner_strides_type = typename base_type::inner_strides_type;
        using inner_backstrides_type = typename base_type::inner_backstrides_type;

        void reshape(const shape_type& shape, bool force = false);
        void reshape(const shape_type& shape, xt::layout l);
        void reshape(const shape_type& shape, const strides_type& strides);

        xt::layout layout();

    protected:

        xstrided_container() noexcept;
        ~xstrided_container() = default;

        xstrided_container(const xstrided_container&) = default;
        xstrided_container& operator=(const xstrided_container&) = default;

        xstrided_container(xstrided_container&&) = default;
        xstrided_container& operator=(xstrided_container&&) = default;

        explicit xstrided_container(inner_shape_type&&, inner_strides_type&&) noexcept;

        inner_shape_type& shape_impl() noexcept;
        const inner_shape_type& shape_impl() const noexcept;

        inner_strides_type& strides_impl() noexcept;
        const inner_strides_type& strides_impl() const noexcept;

        inner_backstrides_type& backstrides_impl() noexcept;
        const inner_backstrides_type& backstrides_impl() const noexcept;

    private:
        inner_shape_type m_shape;
        inner_strides_type m_strides;
        inner_backstrides_type m_backstrides;
        xt::layout m_layout = L;
    };

    /******************************
     * xcontainer implementation *
     ******************************/

    template <class D>
    inline auto xcontainer<D>::mutable_shape() -> inner_shape_type&
    {
        return derived_cast().shape_impl();
    }

    template <class D>
    inline auto xcontainer<D>::mutable_strides() -> inner_strides_type&
    {
        return derived_cast().strides_impl();
    }

    template <class D>
    inline auto xcontainer<D>::mutable_backstrides() -> inner_backstrides_type&
    {
        return derived_cast().backstrides_impl();
    }

    template <class D>
    inline auto xcontainer<D>::derived_cast() -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xcontainer<D>::derived_cast() const -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of element in the container.
     */
    template <class D>
    inline auto xcontainer<D>::size() const noexcept -> size_type
    {
        return data().size();
    }

    /**
     * Returns the number of dimensions of the container.
     */
    template <class D>
    inline auto xcontainer<D>::dimension() const noexcept -> size_type
    {
        return shape().size();
    }

    /**
     * Returns the shape of the container.
     */
    template <class D>
    inline auto xcontainer<D>::shape() const noexcept -> const inner_shape_type&
    {
        return derived_cast().shape_impl();
    }

    /**
     * Returns the strides of the container.
     */
    template <class D>
    inline auto xcontainer<D>::strides() const noexcept -> const inner_strides_type&
    {
        return derived_cast().strides_impl();
    }

    /**
     * Returns the backstrides of the container.
     */
    template <class D>
    inline auto xcontainer<D>::backstrides() const noexcept -> const inner_backstrides_type&
    {
        return derived_cast().backstrides_impl();
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
        XTENSOR_ASSERT(check_index(shape(), args...));
        size_type index = data_offset<size_type>(strides(), static_cast<size_type>(args)...);
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
        XTENSOR_ASSERT(check_index(shape(), args...));
        size_type index = data_offset<size_type>(strides(), static_cast<size_type>(args)...);
        return data()[index];
    }

    /**
     * Returns a reference to the element at the specified position in the container.
     * @param index a sequence of indices specifying the position in the container. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the container.
     */
    template <class D>
    inline auto xcontainer<D>::operator[](const xindex& index) -> reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class D>
    inline auto xcontainer<D>::operator[](size_type i) -> reference
    {
        return operator()(i);
    }

    /**
     * Returns a constant reference to the element at the specified position in the container.
     * @param index a sequence of indices specifying the position in the container. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the container.
     */
    template <class D>
    inline auto xcontainer<D>::operator[](const xindex& index) const -> const_reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class D>
    inline auto xcontainer<D>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a reference to the element at the specified position in the container.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the squence should be equal to or greater
     * than the number of dimensions of the container.
     */
    template <class D>
    template <class It>
    inline auto xcontainer<D>::element(It first, It last) -> reference
    {
        XTENSOR_ASSERT(check_element_index(shape(), first, last));
        return data()[element_offset<size_type>(strides(), first, last)];
    }

    /**
     * Returns a reference to the element at the specified position in the container.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the squence should be equal to or greater
     * than the number of dimensions of the container.
     */
    template <class D>
    template <class It>
    inline auto xcontainer<D>::element(It first, It last) const -> const_reference
    {
        XTENSOR_ASSERT(check_element_index(shape(), first, last));
        return data()[element_offset<size_type>(strides(), first, last)];
    }

    /**
     * Returns a reference to the buffer containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::data() noexcept -> container_type&
    {
        return derived_cast().data_impl();
    }

    /**
     * Returns a constant reference to the buffer containing the elements of the
     * container.
     */
    template <class D>
    inline auto xcontainer<D>::data() const noexcept -> const container_type&
    {
        return derived_cast().data_impl();
    }

    /**
     * Returns the offset to the first element in the container.
     */
    template <class D>
    inline auto xcontainer<D>::raw_data() noexcept -> value_type*
    {
        return data().data();
    }

    template <class D>
    inline auto xcontainer<D>::raw_data() const noexcept -> const value_type*
    {
        return data().data();
    }

    /**
     * Returns the offset to the first element in the container.
     */
    template <class D>
    inline auto xcontainer<D>::raw_data_offset() const noexcept -> const size_type
    {
        return size_type(0);
    }
    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the container to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class D>
    template <class S>
    inline bool xcontainer<D>::broadcast_shape(S& shape) const
    {
        return xt::broadcast_shape(this->shape(), shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class D>
    template <class S>
    inline bool xcontainer<D>::is_trivial_broadcast(const S& str) const noexcept
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
     * Returns an iterator to the first element of the buffer containing
     * the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::begin() noexcept -> iterator
    {
        return data().begin();
    }

    /**
     * Returns an iterator to the element following the last element of
     * the buffer containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::end() noexcept -> iterator
    {
        return data().end();
    }

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::begin() const noexcept -> const_iterator
    {
        return cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::end() const noexcept -> const_iterator
    {
        return cend();
    }

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::cbegin() const noexcept -> const_iterator
    {
        return data().cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::cend() const noexcept -> const_iterator
    {
        return data().cend();
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class D>
    template <class S>
    inline auto xcontainer<D>::stepper_begin(const S& shape) noexcept -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(static_cast<derived_type*>(this), data().begin(), offset);
    }

    template <class D>
    template <class S>
    inline auto xcontainer<D>::stepper_end(const S& shape) noexcept -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(static_cast<derived_type*>(this), data().end(), offset);
    }

    template <class D>
    template <class S>
    inline auto xcontainer<D>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(static_cast<const derived_type*>(this), data().begin(), offset);
    }

    template <class D>
    template <class S>
    inline auto xcontainer<D>::stepper_end(const S& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(static_cast<const derived_type*>(this), data().end(), offset);
    }

    /*************************************
     * xstrided_container implementation *
     *************************************/

    template <class D, layout L>
    inline xstrided_container<D, L>::xstrided_container() noexcept
        : base_type()
    {
        m_shape = make_sequence<inner_shape_type>(base_type::dimension(), 1);
    }

    template <class D, layout L>
    inline xstrided_container<D, L>::xstrided_container(inner_shape_type&& shape, inner_strides_type&& strides) noexcept
        : base_type(), m_shape(std::move(shape)), m_strides(std::move(strides))
    {
        m_backstrides = make_sequence<inner_backstrides_type>(m_shape.size(), 0);
        adapt_strides(m_shape, m_strides, m_backstrides);
    }

    template <class D, layout L>
    inline auto xstrided_container<D, L>::shape_impl() noexcept -> inner_shape_type&
    {
        return m_shape;
    }

    template <class D, layout L>
    inline auto xstrided_container<D, L>::shape_impl() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    template <class D, layout L>
    inline auto xstrided_container<D, L>::strides_impl() noexcept -> inner_strides_type&
    {
        return m_strides;
    }

    template <class D, layout L>
    inline auto xstrided_container<D, L>::strides_impl() const noexcept -> const inner_strides_type&
    {
        return m_strides;
    }

    template <class D, layout L>
    inline auto xstrided_container<D, L>::backstrides_impl() noexcept -> inner_backstrides_type&
    {
        return m_backstrides;
    }

    template <class D, layout L>
    inline auto xstrided_container<D, L>::backstrides_impl() const noexcept -> const inner_backstrides_type&
    {
        return m_backstrides;
    }

    /**
     * Return the layout of the container
     * @return layout of the container
     */
    template <class D, layout L>
    xt::layout xstrided_container<D, L>::layout()
    {
        return m_layout;
    }

    /**
     * Reshapes the container.
     * @param shape the new shape
     * @param force force reshaping, even if the shape stays the same (default: false)
     */
    template <class D, layout L>
    inline void xstrided_container<D, L>::reshape(const shape_type& shape, bool force)
    {
        if (m_layout == layout::dynamic)
        {
            m_layout = layout::row_major;  // fall back to row major
        }
        if (shape != m_shape || force)
        {
            m_shape = shape;
            resize_container(m_strides, m_shape.size());
            resize_container(m_backstrides, m_shape.size());
            size_type data_size = compute_strides(m_shape, m_layout, m_strides, m_backstrides);
            this->data().resize(data_size);
        }
    }

    /**
     * Reshapes the container.
     * @param shape the new shape
     * @param l the new layout
     */
    template <class D, layout L>
    inline void xstrided_container<D, L>::reshape(const shape_type& shape, xt::layout l)
    {
        if (L != layout::dynamic && l != L)
        {
            throw std::runtime_error("Cannot change layout if template parameter not layout::dynamic.");
        }
        m_layout = l;
        reshape(shape, true);
    }

    /**
     * Reshapes the container.
     * @param shape the new shape
     * @param strides the new strides
     */
    template <class D, layout L>
    inline void xstrided_container<D, L>::reshape(const shape_type& shape, const strides_type& strides)
    {
        if (L != layout::dynamic)
        {
            throw std::runtime_error("Cannot reshape with custom strides when layout() is != layout::dynamic.");
        }
        m_shape = shape;
        m_strides = strides;
        resize_container(m_backstrides, m_strides.size());
        adapt_strides(m_shape, m_strides, m_backstrides);
        this->data().resize(compute_size(m_shape));
    }
}

#endif
