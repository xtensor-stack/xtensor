/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XINDEXVIEW_HPP
#define XINDEXVIEW_HPP

#include <cstddef>
#include <type_traits>
#include <utility>
#include <tuple>
#include <algorithm>

#include "xexpression.hpp"
#include "xiterator.hpp"
#include "xutils.hpp"

namespace xt
{

    template <bool is_const, class V, class S>
    class xindexview_stepper;

    template <class CT, class S, class I>
    class xindexview;

    template <class CT, class S, class I>
    struct xcontainer_inner_types<xindexview<CT, S, I>>
    {
        using xexpression_type = std::decay_t<CT>;
        using temporary_type = xarray<typename xexpression_type::value_type>;
    };

    /**************
     * xindexview *
     **************/

    /**
     * @class xindexview
     * @brief View into xexpression from vector of indices.
     *
     * The xindexview class implements a flat (1D) view into a multidimensional
     * xexpression yielding the values at the indices of the index array.
     * xindexview is not meant to be used directly, but only with the \ref index_view
     * and \ref filter helper functions.
     *
     * @tparam CT the closure type of the \ref xexpression type underlying this view
     * @tparam S the shape type of the view
     * @tparam I the index array type of the view
     *
     * @sa index_view, filter
     */
    template <class CT, class S, class I>
    class xindexview : public xview_semantic<xindexview<CT, S, I>>
    {

    public:

        using self_type = xindexview<CT, S, I>;
        using xexpression_type = std::decay_t<CT>;
        using semantic_base = xview_semantic<self_type>;

        using value_type = typename xexpression_type::value_type;
        using reference = typename xexpression_type::reference;
        using const_reference = typename xexpression_type::const_reference;
        using pointer = typename xexpression_type::pointer;
        using const_pointer = typename xexpression_type::const_pointer;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using shape_type = S;
        using strides_type = S;
        using closure_type = const self_type;

        using indices_type = I;

        using stepper = xindexview_stepper<false, self_type, shape_type>;
        using iterator = xiterator<stepper, shape_type>;
        using storage_iterator = iterator;

        using const_stepper = xindexview_stepper<true, self_type, shape_type>;
        using const_iterator = xiterator<const_stepper, shape_type>;
        using const_storage_iterator = const_iterator;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        using base_index_type = xindex_type_t<shape_type>;

        template <class I2>
        xindexview(CT e, I2&& indices) noexcept;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        size_type dimension() const noexcept;
        const shape_type& shape() const;

        reference operator()();
        template <class... Args>
        reference operator()(std::size_t idx, Args... /*args*/);
        reference operator[](const xindex& index);

        template <class It>
        reference element(It first, It last);

        const_reference operator()() const;
        template <class... Args>
        const_reference operator()(std::size_t idx, Args... /*args*/) const;
        const_reference operator[](const xindex& index) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class O>
        bool broadcast_shape(O& shape) const;

        template <class O>
        bool is_trivial_broadcast(const O& /*strides*/) const noexcept;

        iterator begin();
        iterator end();

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

        template <class ST>
        xiterator<xindexview_stepper<false, self_type, ST>, ST> xbegin(const ST& shape);
        template <class ST>
        xiterator<xindexview_stepper<false, self_type, ST>, ST> xend(const ST& shape);

        template <class ST>
        xiterator<xindexview_stepper<true, self_type, ST>, ST> xbegin(const ST& shape) const;
        template <class ST>
        xiterator<xindexview_stepper<true, self_type, ST>, ST> xend(const ST& shape) const;
        template <class ST>
        xiterator<xindexview_stepper<true, self_type, ST>, ST> cxbegin(const ST& shape) const;
        template <class ST>
        xiterator<xindexview_stepper<true, self_type, ST>, ST> cxend(const ST& shape) const;

        template <class ST>
        xindexview_stepper<false, self_type, ST> stepper_begin(const ST& shape);
        template <class ST>
        xindexview_stepper<false, self_type, ST> stepper_end(const ST& shape);

        template <class ST>
        xindexview_stepper<true, self_type, ST> stepper_begin(const ST& shape) const;
        template <class ST>
        xindexview_stepper<true, self_type, ST> stepper_end(const ST& shape) const;

        storage_iterator storage_begin();
        storage_iterator storage_end();

        const_storage_iterator storage_begin() const;
        const_storage_iterator storage_end() const;

        const_storage_iterator storage_cbegin() const;
        const_storage_iterator storage_cend() const;

    private:

        CT m_e;
        const indices_type m_indices;
        const shape_type m_shape;
        
        void assign_temporary_impl(temporary_type& tmp);

        friend class xview_semantic<xindexview<CT, S, I>>;
    };

    /***************************
     * xindexview_stepper      *
     ***************************/

    template <bool is_const, class V, class S>
    class xindexview_stepper
    {

    public:

        using view_type = std::conditional_t<is_const, const V, V>;
        using self_type = xindexview_stepper<is_const, V, S>;

        using value_type = typename view_type::value_type;

        using reference = std::conditional_t<is_const,
                                             typename view_type::const_reference,
                                             typename view_type::reference>;

        using pointer = std::conditional_t<is_const,
                                           typename view_type::const_pointer,
                                           typename view_type::pointer>;
        using size_type = typename view_type::size_type;
        using difference_type = typename view_type::difference_type;
        using iterator_category = std::input_iterator_tag;

        using shape_type = S;
        using index_type = xindex_type_t<shape_type>;

        xindexview_stepper() = default;
        xindexview_stepper(view_type* func, size_type offset, bool end = false) noexcept;

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);

        void to_end();

        bool equal(const self_type& rhs) const;

    private:
        view_type* p_view;
        index_type m_index;
        size_type m_offset;
    };

    template <bool is_const, class V, class S>
    bool operator==(const xindexview_stepper<is_const, V, S>& it1,
                    const xindexview_stepper<is_const, V, S>& it2);

    template <bool is_const, class V, class S>
    bool operator!=(const xindexview_stepper<is_const, V, S>& it1,
                    const xindexview_stepper<is_const, V, S>& it2);

    /*****************************
     * xindexview implementation *
     *****************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xindexview, selecting the indices specified by \a indices.
     * The resulting xexpression has a 1D shape with a length of n for n indices.
     * 
     * @param e the underlying xepxression for this view
     * @param indices the indices to select
     */
    template <class CT, class S, class I>
    template <class I2>
    inline xindexview<CT, S, I>::xindexview(CT e, I2&& indices) noexcept
        : m_e(e), m_indices(std::forward<I2>(indices)), m_shape({m_indices.size()})
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
    template <class CT, class S, class I>
    template <class E>
    inline auto xindexview<CT, S, I>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}
    
    template <class CT, class S, class I>
    inline void xindexview<CT, S, I>::assign_temporary_impl(temporary_type& tmp)
    {
        std::copy(tmp.storage_cbegin(), tmp.storage_cend(), begin());
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of dimensions of the xindexview.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::dimension() const noexcept -> size_type
    {
        return 1;
    }

    /**
     * Returns the shape of the xindexview.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::shape() const -> const shape_type&
    {
        return m_shape;
    }
    //@}

    /**
     * @name Data
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::operator()() -> reference
    {
        return m_e();
    }

    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::operator()() const -> const_reference
    {
        return m_e();
    }

    template <class CT, class S, class I>
    template <class... Args>
    inline auto xindexview<CT, S, I>::operator()(std::size_t idx, Args... /*args*/) -> reference
    {
        return m_e[m_indices[idx]];
    }

    /**
     * Returns the element at the specified position in the xindexview. 
     * 
     * @param idx the position in the view
     */
    template <class CT, class S, class I>
    template <class... Args>
    inline auto xindexview<CT, S, I>::operator()(std::size_t idx, Args... /*args*/) const -> const_reference
    {
        return m_e[m_indices[idx]];
    }

    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::operator[](const xindex& index) -> reference
    {
        return m_e[m_indices[index[0]]];
    }

    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::operator[](const xindex& index) const -> const_reference
    {
        return m_e[m_indices[index[0]]];
    }

    /**
     * Returns a reference to the element at the specified position in the xindexview.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the squence should be equal to or greater 1.
     */
    template <class CT, class S, class I>
    template <class It>
    inline auto xindexview<CT, S, I>::element(It first, It /*last*/) -> reference
    {
        return m_e[m_indices[(*first)]];
    }

    template <class CT, class S, class I>
    template <class It>
    inline auto xindexview<CT, S, I>::element(It first, It /*last*/) const -> const_reference
    {
        return m_e[m_indices[(*first)]];
    }
    //@}
    
    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the xindexview to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class S, class I>
    template <class O>
    inline bool xindexview<CT, S, I>::broadcast_shape(O& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class S, class I>
    template <class O>
    inline bool xindexview<CT, S, I>::is_trivial_broadcast(const O& /*strides*/) const noexcept
    {
        return false;
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
     * Returns an iterator to the first element of the view.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::begin() -> iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns an iterator to the element following the last element
     * of the view.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::end() -> iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::begin() const -> const_iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the view.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::end() const -> const_iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::cbegin() const -> const_iterator
    {
        return begin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the view.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::cend() const -> const_iterator
    {
        return end();
    }

    /**
     * Returns an iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class CT, class S, class I>
    template <class ST>
    inline auto xindexview<CT, S, I>::xbegin(const ST& shape) -> xiterator<xindexview_stepper<false, self_type, ST>, ST>
    {
        return xiterator<stepper, ST>(stepper_begin(shape), shape);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * view. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class CT, class S, class I>
    template <class ST>
    inline auto xindexview<CT, S, I>::xend(const ST& shape) -> xiterator<xindexview_stepper<false, self_type, ST>, ST>
    {
        return xiterator<stepper, ST>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class CT, class S, class I>
    template <class ST>
    inline auto xindexview<CT, S, I>::xbegin(const ST& shape) const -> xiterator<xindexview_stepper<true, self_type, ST>, ST>
    {
        return xiterator<xindexview_stepper<true, self_type, ST>, ST>(stepper_begin(shape), shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * view. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class CT, class S, class I>
    template <class ST>
    inline auto xindexview<CT, S, I>::xend(const ST& shape) const -> xiterator<xindexview_stepper<true, self_type, ST>, ST>
    {
        return xiterator<xindexview_stepper<true, self_type, ST>, ST>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class CT, class S, class I>
    template <class ST>
    inline auto xindexview<CT, S, I>::cxbegin(const ST& shape) const -> xiterator<xindexview_stepper<true, self_type, ST>, ST>
    {
        return xbegin(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * container. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class CT, class S, class I>
    template <class ST>
    inline auto xindexview<CT, S, I>::cxend(const ST& shape) const -> xiterator<xindexview_stepper<true, self_type, ST>, ST>
    {
        return xend(shape);
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class CT, class S, class I>
    template <class ST>
    inline auto xindexview<CT, S, I>::stepper_begin(const ST& shape) -> xindexview_stepper<false, self_type, ST>
    {
        size_type offset = shape.size() - dimension();
        return xindexview_stepper<false, self_type, ST>(this, offset);
    }

    template <class CT, class S, class I>
    template <class ST>
    inline auto xindexview<CT, S, I>::stepper_end(const ST& shape) -> xindexview_stepper<false, self_type, ST>
    {
        size_type offset = shape.size() - dimension();
        return xindexview_stepper<false, self_type, ST>(this, offset, true);
    }

    template <class CT, class S, class I>
    template <class ST>
    inline auto xindexview<CT, S, I>::stepper_begin(const ST& shape) const -> xindexview_stepper<true, self_type, ST>
    {
        size_type offset = shape.size() - dimension();
        return xindexview_stepper<true, self_type, ST>(this, offset);
    }

    template <class CT, class S, class I>
    template <class ST>
    inline auto xindexview<CT, S, I>::stepper_end(const ST& shape) const -> xindexview_stepper<true, self_type, ST>
    {
        size_type offset = shape.size() - dimension();
        return xindexview_stepper<true, self_type, ST>(this, offset, true);
    }

    /************************
     * storage_iterator api *
     ************************/

    /**
     * @name Storage iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the view.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::storage_begin() -> storage_iterator
    {
        return begin();
    }

    /**
     * Returns an iterator to the element following the last element of
     * the view.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::storage_end() -> storage_iterator
    {
        return end();
    }

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::storage_begin() const -> const_storage_iterator
    {
        return cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the view.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::storage_end() const -> const_storage_iterator
    {
        return cend();
    }
    //@}

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::storage_cbegin() const -> const_storage_iterator
    {
        return cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the view.
     */
    template <class CT, class S, class I>
    inline auto xindexview<CT, S, I>::storage_cend() const -> const_storage_iterator
    {
        return cend();
    }
    //@}

    /*************************************
     * xindexview_stepper implementation *
     *************************************/

    template <bool is_const, class V, class S>
    inline xindexview_stepper<is_const, V, S>::xindexview_stepper(view_type* view, size_type offset, bool end) noexcept
        : p_view(view), m_index(make_sequence<index_type>(view->shape().size(), size_type(0))), m_offset(offset)
    {
        if (end)
        {
            m_index = p_view->shape();
        }
    }

    template <bool is_const, class V, class S>
    inline void xindexview_stepper<is_const, V, S>::step(size_type dim, size_type n)
    {
        if(dim >= m_offset)
            m_index[dim - m_offset] += n;
    }

    template <bool is_const, class V, class S>
    inline void xindexview_stepper<is_const, V, S>::step_back(size_type dim, size_type n)
    {
        if(dim >= m_offset) 
            m_index[dim - m_offset] -= n;
    }

    template <bool is_const, class V, class S>
    inline void xindexview_stepper<is_const, V, S>::reset(size_type dim)
    {
        if(dim >= m_offset)
            m_index[dim - m_offset] = 0;
    }

    template <bool is_const, class V, class S>
    inline void xindexview_stepper<is_const, V, S>::to_end()
    {
        m_index = p_view->shape();
    }

    template <bool is_const, class V, class S>
    inline bool xindexview_stepper<is_const, V, S>::equal(const self_type& rhs) const
    {
        return (p_view == rhs.p_view) && (m_index == rhs.m_index);
    }

    template <bool is_const, class V, class S>
    inline auto xindexview_stepper<is_const, V, S>::operator*() const -> reference
    {
        return p_view->element(m_index.begin(), m_index.end());
    }

    template <bool is_const, class V, class S>
    inline bool operator==(const xindexview_stepper<is_const, V, S>& it1,
                           const xindexview_stepper<is_const, V, S>& it2)
    {
        return it1.equal(it2);
    }

    template <bool is_const, class V, class S>
    inline bool operator!=(const xindexview_stepper<is_const, V, S>& it1,
                           const xindexview_stepper<is_const, V, S>& it2)
    {
        return !(it1.equal(it2));
    }

    /**
     * @brief creates an indexview from a container of indices.
     *        
     * Returns a 1D view with the elements at \a indices selected.
     *
     * @param e the underlying xexpression
     * @param indices the indices to select
     * 
     * \code{.cpp}
     * xarray<double> a = {{1,5,3}, {4,5,6}};
     * b = index_view(a, {{0, 0}, {1, 0}, {1, 1}});
     * std::cout << b << std::endl; // {1, 4, 5}
     * b += 100;
     * std::cout << a << std::endl; // {{101, 5, 3}, {104, 105, 6}}
     * \endcode
     */
    template <class E, class I>
    inline auto index_view(E&& e, I&& indices) noexcept
    {
        using view_type = xindexview<xclosure_t<E>, std::array<std::size_t, 1>, std::decay_t<I>>;
        return view_type(std::forward<E>(e), std::forward<I>(indices));
    }
#ifdef X_OLD_CLANG
    template <class E, class I>
    inline auto index_view(E&& e, std::initializer_list<I> indices) noexcept
    {
        using view_type = xindexview<xclosure_t<E>, std::array<std::size_t, 1>, std::vector<xindex>>;
        return view_type(std::forward<E>(e), std::vector<xindex>(indices));
    }
#else
    template <class E, std::size_t L>
    inline auto index_view(E&& e, const xindex(&indices)[L]) noexcept
    {
        using view_type = xindexview<xclosure_t<E>, std::array<std::size_t, 1>, std::array<xindex, L>>;
        return view_type(std::forward<E>(e), to_array(indices));
    }
#endif

    /**
     * @brief creates a view into \a e filtered by \a condition.
     *        
     * Returns a 1D view with the elements selected where \a condition evaluates to \em true.
     * This is equivalent to \verbatim{index_view(e, where(condition));}\endverbatim
     * 
     * @param e the underlying xexpression
     * @param condition xexpression with shape of \a e which selects indices
     *
     * \code{.cpp}
     * xarray<double> a = {{1,5,3}, {4,5,6}};
     * b = filter(a, a >= 5);
     * std::cout << b << std::endl; // {5, 5, 6}
     * \endcode
     */
    template <class E, class O>
    inline auto filter(E&& e, O&& condition) noexcept
    {
        auto indices = where(std::forward<O>(condition));
        using view_type = xindexview<xclosure_t<E>, std::vector<std::size_t>, decltype(indices)>;
        return view_type(std::forward<E>(e), std::move(indices));
    }

}

#endif