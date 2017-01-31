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

    template <class T, class S, class I>
    class xindexview;

    template <class T, class S, class I>
    struct xcontainer_inner_types<xindexview<T, S, I>>
    {
        using temporary_type = xarray<typename T::value_type>;
    };

    /**************
     * xindexview *
     **************/

    /**
     * @class xindexview
     * @brief View from vector of indices.
     *
     * Th xindexview class implements a flat view into a multidimensional
     * array yielding the values at the indices of the index array.
     *
     * @tparam T the function type
     * @tparam S the shape type of the view
     * @tparam I the index array type of the view
     */
    template <class T, class S, class I>
    class xindexview : public xview_semantic<xindexview<T, S, I>>
    {

    public:

        using self_type = xindexview<T, S, I>;
        using expression_type = T;
        using semantic_base = xview_semantic<self_type>;

        using value_type = typename T::value_type;
        using reference = typename T::reference;
        using const_reference = typename T::const_reference;
        using pointer = typename T::pointer;
        using const_pointer = typename T::const_pointer;
        using size_type = typename T::size_type;
        using difference_type = typename T::difference_type;

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
        using base_index_type = get_index_type<shape_type>;

        xindexview(T& f, const indices_type&& indices) noexcept;

        template <class OE>
        self_type& operator=(const xexpression<OE>& e);

        size_type dimension() const noexcept;
        const shape_type& shape() const;

        reference operator()();
        template <class... Args>
        reference operator()(std::size_t idx, Args... args);
        reference operator[](const xindex& index);

        template <class It>
        reference element(const It& first, const It& last);

        const_reference operator()() const;
        template <class... Args>
        const_reference operator()(std::size_t idx, Args... args) const;
        const_reference operator[](const xindex& index) const;

        template <class It>
        const_reference element(const It& first, const It& last) const;

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

        T& m_e;
        const indices_type m_indices;
        const shape_type m_shape;
        
        void assign_temporary_impl(temporary_type& tmp);

        friend class xview_semantic<xindexview<T, S, I>>;
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

        using pointer = typename view_type::pointer;
        using size_type = typename view_type::size_type;
        using difference_type = typename view_type::difference_type;
        using iterator_category = std::input_iterator_tag;

        using shape_type = S;
        using index_type = get_index_type<shape_type>;

        xindexview_stepper(view_type* func, const shape_type& shape) noexcept;

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);

        void to_end();

        bool equal(const self_type& rhs) const;

    private:
        view_type* p_view;
        shape_type m_shape;
        index_type m_index;
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
     * Constructs an xindexview applying the specified function over the 
     * given shape.
     * @param f the function to apply
     * @param shape the shape of the xindexview
     */
    template <class T, class S, class I>
    inline xindexview<T, S, I>::xindexview(T& e, const indices_type&& indices) noexcept
        : m_e(e), m_indices(indices), m_shape({indices.size()})
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
    template <class T, class S, class I>
    template <class OE>
    inline auto xindexview<T, S, I>::operator=(const xexpression<OE>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}
    
    template <class T, class S, class I>
    inline void xindexview<T, S, I>::assign_temporary_impl(temporary_type& tmp)
    {
        std::copy(tmp.storage_cbegin(), tmp.storage_cend(), begin());
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of dimensions of the function.
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    /**
     * Returns the shape of the xindexview.
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::shape() const -> const shape_type&
    {
        return m_shape;
    }
    //@}

    /**
     * @name Data
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::operator()() -> reference
    {
        return m_e();
    }

    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::operator()() const -> const_reference
    {
        return m_e();
    }

    template <class T, class S, class I>
    template <class... Args>
    inline auto xindexview<T, S, I>::operator()(std::size_t idx, Args... args) -> reference
    {
        return m_e[m_indices[idx]];
    }

    /**
     * Returns the evaluated element at the specified position in the function.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the function.
     */
    template <class T, class S, class I>
    template <class... Args>
    inline auto xindexview<T, S, I>::operator()(std::size_t idx, Args... args) const -> const_reference
    {
        return m_e[m_indices[idx]];
    }

    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::operator[](const xindex& index) -> reference
    {
        return m_e[m_indices[index[0]]];
    }

    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::operator[](const xindex& index) const -> const_reference
    {
        return m_e[m_indices[index[0]]];
    }

    /**
     * Returns a constant reference to the element at the specified position in the function.
     * @param first iterator starting the sequence of indices
     * @param second iterator starting the sequence of indices
     * The number of indices in the squence should be equal or greater
     * than the number of dimensions of the container.
     */
    template <class T, class S, class I>
    template <class It>
    inline auto xindexview<T, S, I>::element(const It& first, const It& /*last*/) -> reference
    {
        return m_e[m_indices[(*first)]];
    }

    template <class T, class S, class I>
    template <class It>
    inline auto xindexview<T, S, I>::element(const It& first, const It& /*last*/) const -> const_reference
    {
        return m_e[m_indices[(*first)]];
    }
    //@}
    
    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the function to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class T, class S, class I>
    template <class O>
    inline bool xindexview<T, S, I>::broadcast_shape(O& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class T, class S, class I>
    template <class O>
    inline bool xindexview<T, S, I>::is_trivial_broadcast(const O& /*strides*/) const noexcept
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
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::begin() -> iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns an iterator to the element following the last element
     * of the view.
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::end() -> iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::begin() const -> const_iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the view.
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::end() const -> const_iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::cbegin() const -> const_iterator
    {
        return begin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the view.
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::cend() const -> const_iterator
    {
        return end();
    }

    /**
     * Returns an iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class T, class S, class I>
    template <class ST>
    inline auto xindexview<T, S, I>::xbegin(const ST& shape) -> xiterator<xindexview_stepper<false, self_type, ST>, ST>
    {
        return xiterator<stepper, ST>(stepper_begin(shape), shape);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * view. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class T, class S, class I>
    template <class ST>
    inline auto xindexview<T, S, I>::xend(const ST& shape) -> xiterator<xindexview_stepper<false, self_type, ST>, ST>
    {
        return xiterator<stepper, ST>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class T, class S, class I>
    template <class ST>
    inline auto xindexview<T, S, I>::xbegin(const ST& shape) const -> xiterator<xindexview_stepper<true, self_type, ST>, ST>
    {
        return xiterator<xindexview_stepper<true, self_type, ST>, ST>(stepper_begin(shape), shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * view. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class T, class S, class I>
    template <class ST>
    inline auto xindexview<T, S, I>::xend(const ST& shape) const -> xiterator<xindexview_stepper<true, self_type, ST>, ST>
    {
        return xiterator<xindexview_stepper<true, self_type, ST>, ST>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class T, class S, class I>
    template <class ST>
    inline auto xindexview<T, S, I>::cxbegin(const ST& shape) const -> xiterator<xindexview_stepper<true, self_type, ST>, ST>
    {
        return xbegin(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * container. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class T, class S, class I>
    template <class ST>
    inline auto xindexview<T, S, I>::cxend(const ST& shape) const -> xiterator<xindexview_stepper<true, self_type, ST>, ST>
    {
        return xend(shape);
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class T, class S, class I>
    template <class ST>
    inline auto xindexview<T, S, I>::stepper_begin(const ST& shape) -> xindexview_stepper<false, self_type, ST>
    {
        return stepper(this, shape);
    }

    template <class T, class S, class I>
    template <class ST>
    inline auto xindexview<T, S, I>::stepper_end(const ST& shape) -> xindexview_stepper<false, self_type, ST>
    {
        auto s = stepper(this, shape);
        s.to_end();
        return s;
    }

    template <class T, class S, class I>
    template <class ST>
    inline auto xindexview<T, S, I>::stepper_begin(const ST& shape) const -> xindexview_stepper<true, self_type, ST>
    {
        return xindexview_stepper<true, self_type, ST>(this, shape);
    }

    template <class T, class S, class I>
    template <class ST>
    inline auto xindexview<T, S, I>::stepper_end(const ST& shape) const -> xindexview_stepper<true, self_type, ST>
    {
        auto s = xindexview_stepper<true, self_type, ST>(this, shape);
        s.to_end();
        return s;
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
     * the elements of the view.
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::storage_begin() -> storage_iterator
    {
        return begin();
    }

    /**
     * Returns an iterator to the element following the last element of
     * the buffer containing the elements of the view.
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::storage_end() -> storage_iterator
    {
        return end();
    }

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the view.
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::storage_begin() const -> const_storage_iterator
    {
        return cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the view.
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::storage_end() const -> const_storage_iterator
    {
        return cend();
    }
    //@}

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the view.
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::storage_cbegin() const -> const_storage_iterator
    {
        return cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the view.
     */
    template <class T, class S, class I>
    inline auto xindexview<T, S, I>::storage_cend() const -> const_storage_iterator
    {
        return cend();
    }
    //@}

    /*************************************
     * xindexview_stepper implementation *
     *************************************/

    template <bool is_const, class V, class S>
    inline xindexview_stepper<is_const, V, S>::xindexview_stepper(view_type* view, const shape_type& shape) noexcept
        : p_view(view), m_shape(shape), m_index(make_sequence<index_type>(shape.size(), size_type(0)))
    {
    }

    template <bool is_const, class V, class S>
    inline void xindexview_stepper<is_const, V, S>::step(size_type dim, size_type n)
    {
        m_index[dim] += n;
    }

    template <bool is_const, class V, class S>
    inline void xindexview_stepper<is_const, V, S>::step_back(size_type dim, size_type n)
    {
        m_index[dim] -= 1;
    }

    template <bool is_const, class V, class S>
    inline void xindexview_stepper<is_const, V, S>::reset(size_type dim)
    {
        m_index[dim] = 0;
    }

    template <bool is_const, class V, class S>
    inline void xindexview_stepper<is_const, V, S>::to_end()
    {
        m_index = m_shape;
    }

    template <bool is_const, class V, class S>
    inline bool xindexview_stepper<is_const, V, S>::equal(const self_type& rhs) const
    {
        return p_view == rhs.p_view && std::equal(m_index.begin(), m_index.end(), rhs.m_index.begin());
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

    template <class T, class I = std::vector<xindex>>
    auto inline make_xindexview(T& arr, const I& indices) noexcept
    {
        return xt::xindexview<T, std::array<std::size_t, 1>, I>(arr, std::move(indices));
    }

    template <class T, std::size_t L>
    auto inline make_xindexview(T& arr, const xindex(&indices)[L]) noexcept
    {
        return xt::xindexview<T, std::array<std::size_t, 1>, std::array<xindex, L>>(arr, to_array(indices));
    }

    template <class T, class O>
    auto inline make_xboolview(T& arr, O&& bool_arr) noexcept
    {
        auto indices = where(std::forward<O>(bool_arr));
        return xt::xindexview<T, std::vector<std::size_t>, decltype(indices)>(arr, std::move(indices));
    }

}

#endif