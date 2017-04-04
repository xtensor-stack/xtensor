/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSTRIDEDVIEW_HPP
#define XSTRIDEDVIEW_HPP

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "xexpression.hpp"
#include "xiterable.hpp"
#include "xstrides.hpp"
#include "xutils.hpp"

namespace xt
{

    template <class CT>
    class xstrided_view;

    template <class CT>
    struct xcontainer_inner_types<xstrided_view<CT>>
    {
        using xexpression_type = std::decay_t<CT>;
        using temporary_type = xarray<typename xexpression_type::value_type>;
    };

    template <class CT>
    struct xiterable_inner_types<xstrided_view<CT>>
    {
        using inner_shape_type = typename std::decay_t<CT>::shape_type;
        using const_stepper = xindexed_stepper<xstrided_view<CT>>;
        using stepper = xindexed_stepper<xstrided_view<CT>, false>;
        using const_broadcast_iterator = xiterator<const_stepper, inner_shape_type*>;
        using broadcast_iterator = xiterator<stepper, inner_shape_type*>;
        using const_iterator = const_broadcast_iterator;
        using iterator = broadcast_iterator;
    };

    /**************
     * xstrided_view *
     **************/

    /**
     * @class xstrided_view
     * @brief View of an xexpression from vector of indices.
     *
     * The xstrided_view class implements a flat (1D) view into a multidimensional
     * xexpression yielding the values at the indices of the index array.
     * xstrided_view is not meant to be used directly, but only with the \ref index_view
     * and \ref filter helper functions.
     *
     * @tparam CT the closure type of the \ref xexpression type underlying this view
     * @tparam I the index array type of the view
     *
     * @sa index_view, filter
     */
    template <class CT>
    class xstrided_view : public xview_semantic<xstrided_view<CT>>,
                          public xexpression_iterable<xstrided_view<CT>>
    {

    public:

        using self_type = xstrided_view<CT>;
        using xexpression_type = std::decay_t<CT>;
        using semantic_base = xview_semantic<self_type>;

        using value_type = typename xexpression_type::value_type;
        using reference = typename xexpression_type::reference;
        using const_reference = typename xexpression_type::const_reference;
        using pointer = typename xexpression_type::pointer;
        using const_pointer = typename xexpression_type::const_pointer;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using iterable_base = xexpression_iterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;
        using strides_type = shape_type;
        using closure_type = const self_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        using broadcast_iterator = typename iterable_base::broadcast_iterator;
        using const_broadcast_iterator = typename iterable_base::const_broadcast_iterator;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        using base_index_type = xindex_type_t<shape_type>;

        template <class I>
        xstrided_view(CT e, I&& shape, I&& strides) noexcept;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const shape_type& shape() const noexcept;
        const strides_type& strides() const noexcept;

        reference operator()();
        template <class... Args>
        reference operator()(Args... args);
        reference operator[](const xindex& index);
        reference operator[](size_type i);

        template <class It>
        reference element(It first, It last);

        const_reference operator()() const;
        template <class... Args>
        const_reference operator()(Args... args) const;
        const_reference operator[](const xindex& index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class O>
        bool broadcast_shape(O& shape) const;

        template <class O>
        bool is_trivial_broadcast(const O& /*strides*/) const noexcept;

        template <class ST>
        stepper stepper_begin(const ST& shape);
        template <class ST>
        stepper stepper_end(const ST& shape);

        template <class ST>
        const_stepper stepper_begin(const ST& shape) const;
        template <class ST>
        const_stepper stepper_end(const ST& shape) const;

    private:

        CT m_e;
        const shape_type m_shape;
        const strides_type m_strides;

        void assign_temporary_impl(temporary_type& tmp);

        friend class xview_semantic<xstrided_view<CT>>;
    };

    /*****************************
     * xstrided_view implementation *
     *****************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xstrided_view, selecting the indices specified by \a indices.
     * The resulting xexpression has a 1D shape with a length of n for n indices.
     * 
     * @param e the underlying xexpression for this view
     * @param indices the indices to select
     */
    template <class CT>
    template <class I>
    inline xstrided_view<CT>::xstrided_view(CT e, I&& shape, I&& strides) noexcept
        : m_e(e), m_shape(std::forward<I>(shape)), m_strides(std::forward<I>(strides))
    {
    }
    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class CT>
    template <class E>
    inline auto xstrided_view<CT>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class CT>
    template <class E>
    inline auto xstrided_view<CT>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(this->begin(), this->end(), e);
        return *this;
    }

    template <class CT>
    inline void xstrided_view<CT>::assign_temporary_impl(temporary_type& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), this->xbegin());
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the size of the xstrided_view.
     */
    template <class CT>
    inline auto xstrided_view<CT>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    /**
     * Returns the number of dimensions of the xstrided_view.
     */
    template <class CT>
    inline auto xstrided_view<CT>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    /**
     * Returns the shape of the xstrided_view.
     */
    template <class CT>
    inline auto xstrided_view<CT>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    template <class CT>
    inline auto xstrided_view<CT>::strides() const noexcept -> const strides_type&
    {
        return m_strides;
    }
    //@}

    /**
     * @name Data
     */
    template <class CT>
    inline auto xstrided_view<CT>::operator()() -> reference
    {
        return m_e();
    }

    template <class CT>
    inline auto xstrided_view<CT>::operator()() const -> const_reference
    {
        return m_e();
    }

    template <class CT>
    template <class... Args>
    inline auto xstrided_view<CT>::operator()(Args... args) -> reference
    {
        XTENSOR_ASSERT(check_index(shape(), args...));
        size_type index = data_offset<size_type>(strides(), static_cast<size_type>(args)...);
        return m_e.data()[index];
    }

    /**
     * Returns the element at the specified position in the xstrided_view. 
     * 
     * @param idx the position in the view
     */
    template <class CT>
    template <class... Args>
    inline auto xstrided_view<CT>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_ASSERT(check_index(shape(), args...));
        size_type index = data_offset<size_type>(strides(), static_cast<size_type>(args)...);
        return m_e.data()[index];
    }

    template <class CT>
    inline auto xstrided_view<CT>::operator[](const xindex& index) -> reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT>
    inline auto xstrided_view<CT>::operator[](size_type i) -> reference
    {
        return operator()(i);
    }

    template <class CT>
    inline auto xstrided_view<CT>::operator[](const xindex& index) const -> const_reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT>
    inline auto xstrided_view<CT>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a reference to the element at the specified position in the xstrided_view.
     * @param first iterator starting the sequence of indices
     * The number of indices in the squence should be equal to or greater 1.
     */
    template <class CT>
    template <class It>
    inline auto xstrided_view<CT>::element(It first, It last) -> reference
    {
        return m_e.data()[element_offset<size_type>(strides(), first, last)];
    }

    template <class CT>
    template <class It>
    inline auto xstrided_view<CT>::element(It first, It last) const -> const_reference
    {
        return m_e.data()[element_offset<size_type>(strides(), first, last)];
    }
    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the xstrided_view to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT>
    template <class O>
    inline bool xstrided_view<CT>::broadcast_shape(O& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT>
    template <class O>
    inline bool xstrided_view<CT>::is_trivial_broadcast(const O& /*strides*/) const noexcept
    {
        return false;
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class CT>
    template <class ST>
    inline auto xstrided_view<CT>::stepper_begin(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset);
    }

    template <class CT>
    template <class ST>
    inline auto xstrided_view<CT>::stepper_end(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, offset, true);
    }

    template <class CT>
    template <class ST>
    inline auto xstrided_view<CT>::stepper_begin(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset);
    }

    template <class CT>
    template <class ST>
    inline auto xstrided_view<CT>::stepper_end(const ST& shape) const -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset, true);
    }

    template <class E, class I>
    inline auto strided_view(E&& e, I&& shape, I&& strides) noexcept
    {
        using view_type = xstrided_view<xclosure_t<E>>;
        return view_type(std::forward<E>(e), std::forward<I>(shape), std::forward<I>(strides));
    }

    template <class E>
    inline auto transpose_view(E&& e) noexcept
    {
        using shape_type = typename std::decay_t<E>::shape_type;

        shape_type shape(e.shape().rbegin(), e.shape().rend());
        shape_type strides(e.strides().rbegin(), e.strides().rend());

        using view_type = xstrided_view<xclosure_t<E>>;
        return view_type(std::forward<E>(e), std::move(shape), std::move(strides));
    }

    template <class E>
    inline auto diag_view(E&& e) noexcept
    {
        using shape_type = typename std::decay_t<E>::shape_type;

        shape_type shape(e.shape().rbegin(), e.shape().rend() - 1);
        shape_type strides(e.strides().rbegin(), e.strides().rend() - 1);
        strides.back() += e.strides()[0];

        using view_type = xstrided_view<xclosure_t<E>>;
        return view_type(std::forward<E>(e), std::move(shape), std::move(strides));
    }
}

#endif
