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

    template <bool is_const, class T, class S>
    class xindexview_stepper;

    template <class T, class S>
    class xindexview;

    template <class T, class S>
    struct xcontainer_inner_types<xindexview<T, S>>
    {
        using temporary_type = xarray<typename T::value_type>;
    };

    /**************
     * xindexview *
     **************/

    /**
     * @class xindexview
     * @brief Multidimensional function operating on indices.
     *
     * Th xindexview class implements a multidimensional function,
     * generating a value from the supplied indices.
     *
     * @tparam T the function type
     * @tparam R the return type of the function
     * @tparam S the shape type of the generator
     */
    template <class T, class S>
    class xindexview : public xview_semantic<xindexview<T, S>>
    {

    public:

        using self_type = xindexview<T, S>;
        using expression_type = T;
        using semantic_base = xview_semantic<self_type>;

        using indices_type = std::vector<xindex>;

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

        using stepper = xindexview_stepper<false, T, shape_type>;
        using iterator = xiterator<stepper, shape_type>;
        using storage_iterator = iterator;

        using const_stepper = xindexview_stepper<true, T, shape_type>;
        using const_iterator = xiterator<const_stepper, shape_type>;
        using const_storage_iterator = const_iterator;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        using base_index_type = get_index_type<shape_type>;

        xindexview(T& f, const indices_type&& indices) noexcept;

        template <class OE>
        self_type& operator=(const xexpression<OE>& e);

        size_type dimension() const noexcept;
        const shape_type& shape() const;

        template <class... Args>
        reference operator()(std::size_t idx, Args... args);
        reference operator()();
        reference operator[](const xindex& index);

        template <class It>
        reference element(const It& first, const It& last);

        template <class... Args>
        const_reference operator()(std::size_t idx, Args... args) const;
        const_reference operator()() const;
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
        xiterator<stepper, ST> xbegin(const ST& shape);
        template <class ST>
        xiterator<stepper, ST> xend(const ST& shape);

        template <class ST>
        xiterator<const_stepper, ST> xbegin(const ST& shape) const;
        template <class ST>
        xiterator<const_stepper, ST> xend(const ST& shape) const;
        template <class ST>
        xiterator<const_stepper, ST> cxbegin(const ST& shape) const;
        template <class ST>
        xiterator<const_stepper, ST> cxend(const ST& shape) const;

        template <class ST>
        stepper stepper_begin(const ST& shape);
        template <class ST>
        stepper stepper_end(const ST& shape);

        template <class ST>
        const_stepper stepper_begin(const ST& shape) const;
        template <class ST>
        const_stepper stepper_end(const ST& shape) const;

        storage_iterator storage_begin();
        storage_iterator storage_end();

        const_storage_iterator storage_begin() const;
        const_storage_iterator storage_end() const;

        const_storage_iterator storage_cbegin() const;
        const_storage_iterator storage_cend() const;

    private:

        T& m_e;
        indices_type m_indices;
        shape_type m_shape;
        
        void assign_temporary_impl(temporary_type& tmp);

        friend class xview_semantic<xindexview<T, S>>;
    };

    /***************************
     * xindexview_stepper      *
     ***************************/

    template <bool is_const, class T, class S>
    class xindexview_stepper
    {

    public:

        using view_type = std::conditional_t<is_const,
                                             const xindexview<T, S>,
                                             xindexview<T, S>>;

        using self_type = xindexview_stepper<is_const, T, S>;

        using value_type = typename view_type::value_type;

        using reference = std::conditional_t<is_const,
                                             typename view_type::const_reference,
                                             typename view_type::reference>;

        using pointer = typename view_type::pointer;
        using size_type = typename view_type::size_type;
        using difference_type = typename view_type::difference_type;
        using iterator_category = std::input_iterator_tag;

        using shape_type = typename view_type::shape_type;
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

    template <bool is_const, class T, class S>
    bool operator==(const xindexview_stepper<is_const, T, S>& it1,
                    const xindexview_stepper<is_const, T, S>& it2);

    template <bool is_const, class T, class S>
    bool operator!=(const xindexview_stepper<is_const, T, S>& it1,
                    const xindexview_stepper<is_const, T, S>& it2);

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
    template <class T, class S>
    inline xindexview<T, S>::xindexview(T& e, const indices_type&& indices) noexcept
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
    template <class T, class S>
    template <class OE>
    inline auto xindexview<T, S>::operator=(const xexpression<OE>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}
    
    template <class T, class S>
    inline void xindexview<T, S>::assign_temporary_impl(temporary_type& tmp)
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
    template <class T, class S>
    inline auto xindexview<T, S>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    /**
     * Returns the shape of the xindexview.
     */
    template <class T, class S>
    inline auto xindexview<T, S>::shape() const -> const shape_type&
    {
        return m_shape;
    }
    //@}

    /**
     * @name Data
     */
    template <class T, class S>
    inline auto xindexview<T, S>::operator()() const -> const_reference
    {
        return m_e();
    }

    template <class T, class S>
    inline auto xindexview<T, S>::operator()() -> reference
    {
        return m_e();
    }

    template <class T, class S>
    template <class... Args>
    inline auto xindexview<T, S>::operator()(std::size_t idx, Args... args) -> reference
    {
        return m_e[m_indices[idx]];
    }

    /**
     * Returns the evaluated element at the specified position in the function.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the function.
     */
    template <class T, class S>
    template <class... Args>
    inline auto xindexview<T, S>::operator()(std::size_t idx, Args... args) const -> const_reference
    {
        return m_e[m_indices[idx]];
    }

    template <class T, class S>
    inline auto xindexview<T, S>::operator[](const xindex& index) -> reference
    {
        return m_e[m_indices[index[0]]];
    }

    template <class T, class S>
    inline auto xindexview<T, S>::operator[](const xindex& index) const -> const_reference
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
    template <class T, class S>
    template <class It>
    inline auto xindexview<T, S>::element(const It& first, const It& /*last*/) -> reference
    {
        return m_e[m_indices[(*first)]];
    }

    template <class T, class S>
    template <class It>
    inline auto xindexview<T, S>::element(const It& first, const It& /*last*/) const -> const_reference
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
    template <class T, class S>
    template <class O>
    inline bool xindexview<T, S>::broadcast_shape(O& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class T, class S>
    template <class O>
    inline bool xindexview<T, S>::is_trivial_broadcast(const O& /*strides*/) const noexcept
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
    template <class T, class S>
    inline auto xindexview<T, S>::begin() -> iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns an iterator to the element following the last element
     * of the view.
     */
    template <class T, class S>
    inline auto xindexview<T, S>::end() -> iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class T, class S>
    inline auto xindexview<T, S>::begin() const -> const_iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the view.
     */
    template <class T, class S>
    inline auto xindexview<T, S>::end() const -> const_iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the view.
     */
    template <class T, class S>
    inline auto xindexview<T, S>::cbegin() const -> const_iterator
    {
        return begin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the view.
     */
    template <class T, class S>
    inline auto xindexview<T, S>::cend() const -> const_iterator
    {
        return end();
    }

    /**
     * Returns an iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class T, class S>
    template <class ST>
    inline auto xindexview<T, S>::xbegin(const ST& shape) -> xiterator<stepper, ST>
    {
        return xiterator<stepper, ST>(stepper_begin(shape), shape);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * view. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class T, class S>
    template <class ST>
    inline auto xindexview<T, S>::xend(const ST& shape) -> xiterator<stepper, ST>
    {
        return xiterator<stepper, ST>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class T, class S>
    template <class ST>
    inline auto xindexview<T, S>::xbegin(const ST& shape) const -> xiterator<const_stepper, ST>
    {
        return xiterator<const_stepper, ST>(stepper_begin(shape), shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * view. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class T, class S>
    template <class ST>
    inline auto xindexview<T, S>::xend(const ST& shape) const -> xiterator<const_stepper, ST>
    {
        return xiterator<const_stepper, ST>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the view. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class T, class S>
    template <class ST>
    inline auto xindexview<T, S>::cxbegin(const ST& shape) const -> xiterator<const_stepper, ST>
    {
        return xbegin(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * container. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class T, class S>
    template <class ST>
    inline auto xindexview<T, S>::cxend(const ST& shape) const -> xiterator<const_stepper, ST>
    {
        return xend(shape);
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class T, class S>
    template <class ST>
    inline auto xindexview<T, S>::stepper_begin(const ST& shape) -> stepper
    {
        return stepper(this, shape);
    }

    template <class T, class S>
    template <class ST>
    inline auto xindexview<T, S>::stepper_end(const ST& shape) -> stepper
    {
        auto s = stepper(this, shape);
        s.to_end();
        return s;
    }

    template <class T, class S>
    template <class ST>
    inline auto xindexview<T, S>::stepper_begin(const ST& shape) const -> const_stepper
    {
        return const_stepper(this, shape);
    }

    template <class T, class S>
    template <class ST>
    inline auto xindexview<T, S>::stepper_end(const ST& shape) const -> const_stepper
    {
        auto s = const_stepper(this, shape);
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
    template <class T, class S>
    inline auto xindexview<T, S>::storage_begin() -> storage_iterator
    {
        return begin();
    }

    /**
     * Returns an iterator to the element following the last element of
     * the buffer containing the elements of the view.
     */
    template <class T, class S>
    inline auto xindexview<T, S>::storage_end() -> storage_iterator
    {
        return end();
    }

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the view.
     */
    template <class T, class S>
    inline auto xindexview<T, S>::storage_begin() const -> const_storage_iterator
    {
        return cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the view.
     */
    template <class T, class S>
    inline auto xindexview<T, S>::storage_end() const -> const_storage_iterator
    {
        return cend();
    }
    //@}

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the view.
     */
    template <class T, class S>
    inline auto xindexview<T, S>::storage_cbegin() const -> const_storage_iterator
    {
        return cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the view.
     */
    template <class T, class S>
    inline auto xindexview<T, S>::storage_cend() const -> const_storage_iterator
    {
        return cend();
    }
    //@}

    /*************************************
     * xindexview_stepper implementation *
     *************************************/

    template <bool is_const, class T, class S>
    inline xindexview_stepper<is_const, T, S>::xindexview_stepper(view_type* view, const shape_type& shape) noexcept
        : p_view(view), m_shape(shape), m_index(make_sequence<index_type>(shape.size(), size_type(0)))
    {
    }

    template <bool is_const, class T, class S>
    inline void xindexview_stepper<is_const, T, S>::step(size_type dim, size_type n)
    {
        m_index[dim] += n;
    }

    template <bool is_const, class T, class S>
    inline void xindexview_stepper<is_const, T, S>::step_back(size_type dim, size_type n)
    {
        m_index[dim] -= 1;
    }

    template <bool is_const, class T, class S>
    inline void xindexview_stepper<is_const, T, S>::reset(size_type dim)
    {
        m_index[dim] = 0;
    }

    template <bool is_const, class T, class S>
    inline void xindexview_stepper<is_const, T, S>::to_end()
    {
        m_index = m_shape;
    }

    template <bool is_const, class T, class S>
    inline bool xindexview_stepper<is_const, T, S>::equal(const self_type& rhs) const
    {
        return p_view == rhs.p_view && std::equal(m_index.begin(), m_index.end(), rhs.m_index.begin());
    }

    template <bool is_const, class T, class S>
    inline auto xindexview_stepper<is_const, T, S>::operator*() const -> reference
    {
        return (*p_view)[m_index];
    }

    template <bool is_const, class T, class S>
    inline bool operator==(const xindexview_stepper<is_const, T, S>& it1,
                           const xindexview_stepper<is_const, T, S>& it2)
    {
        return it1.equal(it2);
    }

    template <bool is_const, class T, class S>
    inline bool operator!=(const xindexview_stepper<is_const, T, S>& it1,
                           const xindexview_stepper<is_const, T, S>& it2)
    {
        return !(it1.equal(it2));
    }

    template <class T>
    auto inline make_xindexview(T& arr, const std::vector<xindex>&& indices)
    {
        return xt::xindexview<T, std::vector<std::size_t>>(arr, std::move(indices));
    }

    template <class T, class O>
    auto inline make_xboolview(T& arr, const O& bool_arr)
    {
        std::vector<xindex> indices;

        auto shape = arr.shape();
        xindex idx = xindex(arr.dimension());

        auto next_idx = [&shape](xindex& idx) {
            for (int i = shape.size() - 1; i >= 0; i--)
                if (idx[i] >= shape[i] - 1)
                    idx[i] = 0;
                else
                    return idx[i]++;
        };

        std::size_t total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        for (std::size_t i = 0; i < total_size; i++) {
            if (bool_arr[idx])
                indices.push_back(idx);
            next_idx(idx);
        }

        return xt::xindexview<T, std::vector<std::size_t>>(arr, std::move(indices));
    }

}

#endif