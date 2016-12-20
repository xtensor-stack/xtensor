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

    namespace check_policy
    {
        struct none {};
        struct full {};
    }

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
        using strides_type = typename inner_types::strides_type;

        using stepper = xstepper<D>;
        using const_stepper = xstepper<const D>;

        using iterator = xiterator<stepper, shape_type>;
        using const_iterator = xiterator<const_stepper, shape_type>;

        using storage_iterator = typename container_type::iterator;
        using const_storage_iterator = typename container_type::const_iterator;

        size_type size() const noexcept;

        size_type dimension() const noexcept;

        const shape_type& shape() const noexcept;
        const strides_type& strides() const noexcept;
        const strides_type& backstrides() const noexcept;

        void reshape(const shape_type& shape);
        void reshape(const shape_type& shape, layout l);
        void reshape(const shape_type& shape, const strides_type& strides);

        void transpose();

        template <class Tag = check_policy::none>
        void transpose(const std::vector<size_type>& permutation, Tag check_policy = Tag());

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        reference operator[](const xindex& index);
        const_reference operator[](const xindex& index) const;

        template <class It>
        reference element(It first, It last);
        template <class It>
        const_reference element(It first, It last) const;

        container_type& data() noexcept;
        const container_type& data() const noexcept;

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
        xiterator<stepper, S> xbegin(const S& shape) noexcept;
        template <class S>
        xiterator<stepper, S> xend(const S& shape) noexcept;

        template <class S>
        xiterator<const_stepper, S> xbegin(const S& shape) const noexcept;
        template <class S>
        xiterator<const_stepper, S> xend(const S& shape) const noexcept;
        template <class S>
        xiterator<const_stepper, S> cxbegin(const S& shape) const noexcept;
        template <class S>
        xiterator<const_stepper, S> cxend(const S& shape) const noexcept;

        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape) noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape) const noexcept;

        storage_iterator storage_begin() noexcept;
        storage_iterator storage_end() noexcept;

        const_storage_iterator storage_begin() const noexcept;
        const_storage_iterator storage_end() const noexcept;

        const_storage_iterator storage_cbegin() const noexcept;
        const_storage_iterator storage_cend() const noexcept;

    protected:

        xcontainer() = default;
        ~xcontainer() = default;

        xcontainer(const xcontainer&) = default;
        xcontainer& operator=(const xcontainer&) = default;

        xcontainer(xcontainer&&) = default;
        xcontainer& operator=(xcontainer&&) = default;

        shape_type& get_shape() noexcept;
        strides_type& get_strides() noexcept;
        strides_type& get_backstrides() noexcept;

    private:

        void adapt_strides() noexcept;
        void adapt_strides(size_type i) noexcept;

        void transpose_impl(const std::vector<size_type>& permutation, check_policy::none);
        void transpose_impl(const std::vector<size_type>& permutation, check_policy::full);

        size_type data_size() const noexcept;

        template <size_t dim = 0>
        size_type data_offset() const noexcept;

        template <size_t dim = 0, class... Args>
        size_type data_offset(size_type i, Args... args) const noexcept;

        template <class It>
        size_type element_offset(It, It last) const noexcept;

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
    inline auto xcontainer<D>::get_shape() noexcept -> shape_type&
    {
        return m_shape;
    }

    template <class D>
    inline auto xcontainer<D>::get_strides() noexcept -> strides_type&
    {
        return m_strides;
    }

    template <class D>
    inline auto xcontainer<D>::get_backstrides() noexcept -> strides_type&
    {
        return m_backstrides;
    }

    template <class D>
    inline void xcontainer<D>::adapt_strides() noexcept
    {
        for(size_type i = 0; i < m_shape.size(); ++i)
        {
            adapt_strides(i);
        }
    }

    template <class D>
    inline void xcontainer<D>::adapt_strides(size_type i) noexcept
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
    inline auto xcontainer<D>::data_size() const noexcept -> size_type
    {
        return std::accumulate(m_shape.begin(), m_shape.end(), size_type(1), std::multiplies<size_type>());
    }

    template <class D>
    template <size_t dim>
    inline auto xcontainer<D>::data_offset() const noexcept -> size_type
    {
        return 0;
    }

    template <class D>
    template <size_t dim, class... Args>
    inline auto xcontainer<D>::data_offset(size_type i, Args... args) const noexcept -> size_type
    {
        return i * m_strides[dim] + data_offset<dim + 1>(args...);
    }

    template <class D>
    template <class It>
    inline auto xcontainer<D>::element_offset(It, It last) const noexcept -> size_type
    {
        It first = last;
        first -= m_strides.size();
        return std::inner_product(m_strides.begin(), m_strides.end(), first, size_type(0));
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
        return m_shape.size();
    }

    /**
     * Returns the shape of the container.
     */
    template <class D>
    inline auto xcontainer<D>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    /**
     * Returns the strides of the container.
     */
    template <class D>
    inline auto xcontainer<D>::strides() const noexcept -> const strides_type&
    {
        return m_strides;
    }

    /**
     * Returns the backstrides of the container.
     */
    template <class D>
    inline auto xcontainer<D>::backstrides() const noexcept -> const strides_type&
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

    /**
     * Transposes the container inplace by reversing the dimensions.
     */
    template <class D>
    inline void xcontainer<D>::transpose()
    {
        // reverse stride and shape
        std::reverse(m_shape.begin(), m_shape.end());
        std::reverse(m_strides.begin(), m_strides.end());
        std::reverse(m_backstrides.begin(), m_backstrides.end());
    }

    /**
     * Transposes the container inplace by permuting the shape with @permutation.
     * @param permutation the vector containing permutation
     * @param check_policy the check level (check_policy::full() or check_policy::none())
     * @tparam Tag selects the level of error checking on permutation vector defaults to check_policy::none.
     */
    template <class D>
    template <class Tag>
    inline void xcontainer<D>::transpose(const std::vector<size_type>& permutation, Tag check_policy)
    {
        transpose_impl(permutation, check_policy);
    }

    template <class D>
    inline void xcontainer<D>::transpose_impl(const std::vector<size_type>& permutation, check_policy::full)
    {
        // check if axis appears twice in permutation
        for (size_type i = 0; i < permutation.size(); ++i)
        {
            for (size_type j = i + 1; j < permutation.size(); ++j)
            {
                if (permutation[i] == permutation[j])
                {
                    throw transpose_error("Permutation contains axis more than once");
                }
            }
        }
        transpose_impl(permutation, check_policy::none());
    }

    template <class D>
    inline void xcontainer<D>::transpose_impl(const std::vector<size_type>& permutation, check_policy::none)
    {
        if (permutation.size() != m_shape.size())
        {
            throw transpose_error("Permutation does not have the same size as shape");
        }

        // permute stride and shape
        strides_type temp_strides;
        resize_container(temp_strides, m_strides.size());

        shape_type temp_shape;
        resize_container(temp_shape, m_shape.size());

        shape_type temp_backstrides;
        resize_container(temp_backstrides, m_backstrides.size());

        for (size_type i = 0; i < m_shape.size(); ++i)
        {
            if (permutation[i] >= dimension())
            {
                throw transpose_error("Permutation contains wrong axis");
            }
            temp_shape[i] = m_shape[permutation[i]];
            temp_strides[i] = m_strides[permutation[i]];
            temp_backstrides[i] = m_backstrides[permutation[i]];
        }
        m_shape = std::move(temp_shape);
        m_strides = std::move(temp_strides);
        m_backstrides = std::move(temp_backstrides);
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
     * @param index a sequence of indices specifying the position in the container. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the container.
     */
    template <class D>
    inline auto xcontainer<D>::operator[](const xindex& index) -> reference
    {
        return element(index.cbegin(), index.cend());
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

    /**
     * Returns a reference to the element at the specified position in the container.
     * @param first iterator starting the sequence of indices
     * @param second iterator starting the sequence of indices
     * The number of indices in the squence should be equal or greater
     * than the number of dimensions of the container.
     */
    template <class D>
    template <class It>
    inline auto xcontainer<D>::element(It first, It last) -> reference
    {
        return data()[element_offset(first, last)];
    }

    /**
     * Returns a reference to the element at the specified position in the container.
     * @param first iterator starting the sequence of indices
     * @param second iterator starting the sequence of indices
     * The number of indices in the squence should be equal or greater
     * than the number of dimensions of the container.
     */
    template <class D>
    template <class It>
    inline auto xcontainer<D>::element(It first, It last) const -> const_reference
    {
        return data()[element_offset(first, last)];
    }

    /**
     * Returns a reference to the buffer containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::data() noexcept -> container_type&
    {
        return static_cast<derived_type*>(this)->data_impl();
    }

    /**
     * Returns a constant reference to the buffer containing the elements of the
     * container.
     */
    template <class D>
    inline auto xcontainer<D>::data() const noexcept -> const container_type&
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
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class D>
    template <class S>
    inline bool xcontainer<D>::broadcast_shape(S& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
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
     * Returns an iterator to the first element of the container.
     */
    template <class D>
    inline auto xcontainer<D>::begin() noexcept -> iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns an iterator to the element following the last element
     * of the container.
     */
    template <class D>
    inline auto xcontainer<D>::end() noexcept -> iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the container.
     */
    template <class D>
    inline auto xcontainer<D>::begin() const noexcept -> const_iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the container.
     */
    template <class D>
    inline auto xcontainer<D>::end() const noexcept -> const_iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the container.
     */
    template <class D>
    inline auto xcontainer<D>::cbegin() const noexcept -> const_iterator
    {
        return begin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the container.
     */
    template <class D>
    inline auto xcontainer<D>::cend() const noexcept -> const_iterator
    {
        return end();
    }

    /**
     * Returns an iterator to the first element of the container. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class D>
    template <class S>
    inline auto xcontainer<D>::xbegin(const S& shape) noexcept -> xiterator<stepper, S>
    {
        return xiterator<stepper, S>(stepper_begin(shape), shape);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * container. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class D>
    template <class S>
    inline auto xcontainer<D>::xend(const S& shape) noexcept -> xiterator<stepper, S>
    {
        return xiterator<stepper, S>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the container. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class D>
    template <class S>
    inline auto xcontainer<D>::xbegin(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xiterator<const_stepper, S>(stepper_begin(shape), shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * container. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class D>
    template <class S>
    inline auto xcontainer<D>::xend(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xiterator<const_stepper, S>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the container. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class D>
    template <class S>
    inline auto xcontainer<D>::cxbegin(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xbegin(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * container. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class D>
    template <class S>
    inline auto xcontainer<D>::cxend(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xend(shape);
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
    inline auto xcontainer<D>::storage_begin() noexcept -> storage_iterator
    {
        return data().begin();
    }

    /**
     * Returns an iterator to the element following the last element of
     * the buffer containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::storage_end() noexcept -> storage_iterator
    {
        return data().end();
    }

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::storage_begin() const noexcept -> const_storage_iterator
    {
        return data().cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::storage_end() const noexcept -> const_storage_iterator
    {
        return data().cend();
    }

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::storage_cbegin() const noexcept -> const_storage_iterator
    {
        return data().cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the container.
     */
    template <class D>
    inline auto xcontainer<D>::storage_cend() const noexcept -> const_storage_iterator
    {
        return data().cend();
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

