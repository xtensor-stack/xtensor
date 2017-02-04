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

    template <class E, class S, class I>
    class xindexview;

    template <class E, class S, class I>
    struct xcontainer_inner_types<xindexview<E, S, I>>
    {
        using temporary_type = xarray<typename E::value_type>;
    };

    /**************
     * xindexview *
     **************/

    /**
     * @class xindexview
     * @brief View into xexpression from vector of indices.
     *
     * Th xindexview class implements a flat (1D) view into a multidimensional
     * xexpression yielding the values at the indices of the index array.
     *
     * @tparam E the xexpression type underlying this view.
     * @tparam S the shape type of the view
     * @tparam I the index array type of the view
     */
    template <class E, class S, class I>
    class xindexview : public xview_semantic<xindexview<E, S, I>>
    {

    public:

        using self_type = xindexview<E, S, I>;
        using expression_type = E;
        using semantic_base = xview_semantic<self_type>;

        using value_type = typename E::value_type;
        using reference = typename E::reference;
        using const_reference = typename E::const_reference;
        using pointer = typename E::pointer;
        using const_pointer = typename E::const_pointer;
        using size_type = typename E::size_type;
        using difference_type = typename E::difference_type;

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

        xindexview(E& f, indices_type&& indices) noexcept;

        template <class OE>
        self_type& operator=(const xexpression<OE>& e);

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

        expression_type& m_e;
        const indices_type m_indices;
        const shape_type m_shape;
        
        void assign_temporary_impl(temporary_type& tmp);

        friend class xview_semantic<xindexview<E, S, I>>;
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
     * Constructs an xindexview, selecting the indices specified by \ref indices.
     * The resulting xexpression has a 1D shape with a length of n for n indices.
     * 
     * @param e the underlying xepxression for this view
     * @param indices the indices to select
     */
    template <class E, class S, class I>
    inline xindexview<E, S, I>::xindexview(E& e, indices_type&& indices) noexcept
        : m_e(e), m_indices(std::forward<I>(indices)), m_shape({m_indices.size()})
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
    template <class E, class S, class I>
    template <class OE>
    inline auto xindexview<E, S, I>::operator=(const xexpression<OE>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}
    
    template <class E, class S, class I>
    inline void xindexview<E, S, I>::assign_temporary_impl(temporary_type& tmp)
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
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::dimension() const noexcept -> size_type
    {
        return 1;
    }

    /**
     * Returns the shape of the xindexview.
     */
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::shape() const -> const shape_type&
    {
        return m_shape;
    }
    //@}

    /**
     * @name Data
     */
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::operator()() -> reference
    {
        return m_e();
    }

    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::operator()() const -> const_reference
    {
        return m_e();
    }

    template <class E, class S, class I>
    template <class... Args>
    inline auto xindexview<E, S, I>::operator()(std::size_t idx, Args... /*args*/) -> reference
    {
        return m_e[m_indices[idx]];
    }

    /**
     * Returns the element at the specified position in the xindexview. 
     * 
     * @param idx the position in the view
     */
    template <class E, class S, class I>
    template <class... Args>
    inline auto xindexview<E, S, I>::operator()(std::size_t idx, Args... /*args*/) const -> const_reference
    {
        return m_e[m_indices[idx]];
    }

    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::operator[](const xindex& index) -> reference
    {
        return m_e[m_indices[index[0]]];
    }

    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::operator[](const xindex& index) const -> const_reference
    {
        return m_e[m_indices[index[0]]];
    }

    /**
     * Returns a reference to the element at the specified position in the xindexview.
     * @param first iterator starting the sequence of indices
     * @param second iterator ending the sequence of indices
     * The number of indices in the squence should be equal to or greater 1.
     */
    template <class E, class S, class I>
    template <class It>
    inline auto xindexview<E, S, I>::element(It first, It /*last*/) -> reference
    {
        return m_e[m_indices[(*first)]];
    }

    template <class E, class S, class I>
    template <class It>
    inline auto xindexview<E, S, I>::element(It first, It /*last*/) const -> const_reference
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
    template <class E, class S, class I>
    template <class O>
    inline bool xindexview<E, S, I>::broadcast_shape(O& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class E, class S, class I>
    template <class O>
    inline bool xindexview<E, S, I>::is_trivial_broadcast(const O& /*strides*/) const noexcept
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
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::begin() -> iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns an iterator to the element following the last element
     * of the view.
     */
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::end() -> iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::begin() const -> const_iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the view.
     */
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::end() const -> const_iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::cbegin() const -> const_iterator
    {
        return begin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the view.
     */
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::cend() const -> const_iterator
    {
        return end();
    }

    /**
     * Returns an iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class E, class S, class I>
    template <class ST>
    inline auto xindexview<E, S, I>::xbegin(const ST& shape) -> xiterator<xindexview_stepper<false, self_type, ST>, ST>
    {
        return xiterator<stepper, ST>(stepper_begin(shape), shape);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * view. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class E, class S, class I>
    template <class ST>
    inline auto xindexview<E, S, I>::xend(const ST& shape) -> xiterator<xindexview_stepper<false, self_type, ST>, ST>
    {
        return xiterator<stepper, ST>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class E, class S, class I>
    template <class ST>
    inline auto xindexview<E, S, I>::xbegin(const ST& shape) const -> xiterator<xindexview_stepper<true, self_type, ST>, ST>
    {
        return xiterator<xindexview_stepper<true, self_type, ST>, ST>(stepper_begin(shape), shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * view. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class E, class S, class I>
    template <class ST>
    inline auto xindexview<E, S, I>::xend(const ST& shape) const -> xiterator<xindexview_stepper<true, self_type, ST>, ST>
    {
        return xiterator<xindexview_stepper<true, self_type, ST>, ST>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class E, class S, class I>
    template <class ST>
    inline auto xindexview<E, S, I>::cxbegin(const ST& shape) const -> xiterator<xindexview_stepper<true, self_type, ST>, ST>
    {
        return xbegin(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * container. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class E, class S, class I>
    template <class ST>
    inline auto xindexview<E, S, I>::cxend(const ST& shape) const -> xiterator<xindexview_stepper<true, self_type, ST>, ST>
    {
        return xend(shape);
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class E, class S, class I>
    template <class ST>
    inline auto xindexview<E, S, I>::stepper_begin(const ST& shape) -> xindexview_stepper<false, self_type, ST>
    {
        return stepper(this, shape);
    }

    template <class E, class S, class I>
    template <class ST>
    inline auto xindexview<E, S, I>::stepper_end(const ST& shape) -> xindexview_stepper<false, self_type, ST>
    {
        auto s = stepper(this, shape);
        s.to_end();
        return s;
    }

    template <class E, class S, class I>
    template <class ST>
    inline auto xindexview<E, S, I>::stepper_begin(const ST& shape) const -> xindexview_stepper<true, self_type, ST>
    {
        return xindexview_stepper<true, self_type, ST>(this, shape);
    }

    template <class E, class S, class I>
    template <class ST>
    inline auto xindexview<E, S, I>::stepper_end(const ST& shape) const -> xindexview_stepper<true, self_type, ST>
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
     * Returns an iterator to the first element of the view.
     */
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::storage_begin() -> storage_iterator
    {
        return begin();
    }

    /**
     * Returns an iterator to the element following the last element of
     * the view.
     */
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::storage_end() -> storage_iterator
    {
        return end();
    }

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::storage_begin() const -> const_storage_iterator
    {
        return cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the view.
     */
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::storage_end() const -> const_storage_iterator
    {
        return cend();
    }
    //@}

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::storage_cbegin() const -> const_storage_iterator
    {
        return cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the view.
     */
    template <class E, class S, class I>
    inline auto xindexview<E, S, I>::storage_cend() const -> const_storage_iterator
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
        m_index[dim] -= n;
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

    /**
     * @brief create an indexview from a container of indices.
     *        
     * Returns a 1D view with the elements at \ref indices selected.
     *
     * @param e the underlying xexpression
     * @param indices the indices to select
     * 
     * \code{.cpp}
     * xarray<double> a = {{1,5,3}, {4,5,6}};
     * b = make_xindexview(a, {{0, 0}, {1, 0}, {1, 1}});
     * std::cout << b << std::endl; // {1, 4, 5}
     * b += 100;
     * std::cout << a << std::endl; // {{101, 5, 3}, {104, 105, 6}}
     * \endcode
     */
    template <class E, class I = std::vector<xindex>>
    auto inline make_xindexview(E& e, I&& indices) noexcept
    {
        return xindexview<E, std::array<std::size_t, 1>, I>(e, std::forward<I>(indices));
    }

    /**
     * @brief create an indexview from a initializer list or a static array of indices.
     *        
     * Returns a 1D view with the elements at \ref indices selected.
     *
     * @param e the underlying xexpression
     * @param indices the indices to select
     */
    template <class E, std::size_t L>
    auto inline make_xindexview(E& e, const xindex(&indices)[L]) noexcept
    {
        return xindexview<E, std::array<std::size_t, 1>, std::array<xindex, L>>(e, to_array(indices));
    }

    /**
     * @brief create a view into \ref e filtered by \ref condition.
     *        
     * Returns a 1D view with the elements selected where \ref condition evaluates to \em true.
     * This is equivalent to \verbatim{make_xindexview(e, where(condition));}
     * 
     * @param e the underlying xexpression
     * @param condition xexpression with shape of \ref e which selects indices
     *
     * \code{.cpp}
     * xarray<double> a = {{1,5,3}, {4,5,6}};
     * b = make_xfilter(a, a >= 5);
     * std::cout << b << std::endl; // {5, 5, 6}
     * \endcode
     */
    template <class E, class O>
    auto inline make_xfilter(E& e, O&& condition) noexcept
    {
        auto indices = where(std::forward<O>(condition));
        return xindexview<E, std::vector<std::size_t>, decltype(indices)>(e, std::move(indices));
    }

}

#endif