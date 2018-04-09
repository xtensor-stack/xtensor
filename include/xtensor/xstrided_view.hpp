/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_STRIDED_VIEW_HPP
#define XTENSOR_STRIDED_VIEW_HPP

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <xtl/xsequence.hpp>
#include <xtl/xvariant.hpp>

#include "xarray.hpp"
#include "xexpression.hpp"
#include "xiterable.hpp"
#include "xstrides.hpp"
#include "xutils.hpp"
#include "xview.hpp"

namespace xt
{
    template <class CT, class S, class FS>
    class xstrided_view;

    template <class CT, class S, class FS>
    struct xcontainer_inner_types<xstrided_view<CT, S, FS>>
    {
        using xexpression_type = std::decay_t<CT>;
        using temporary_type = xarray<std::decay_t<typename xexpression_type::value_type>>;
    };

    namespace detail
    {
        template <class T>
        struct is_indexed_stepper
        {
            static const bool value = false;
        };

        template <class T, bool B>
        struct is_indexed_stepper<xindexed_stepper<T, B>>
        {
            static const bool value = true;
        };

        template <class CT>
        class flat_expression_adaptor;

        template <class CT, class T = void>
        struct flat_storage_type;

        template <typename CT>
        struct flat_storage_type<CT, typename std::enable_if_t<has_raw_data_interface<std::decay_t<CT>>::value>>
        {
            // Note: we could also use the container_type typedef.
            // using type = std::conditional_t<
            //    std::is_const<std::remove_reference_t<CT>>::value,
            //    const typename std::decay_t<CT>::container_type&,
            //    typename std::decay_t<CT>::container_type&>;
            using type = decltype(std::declval<CT>().data());
        };

        template <typename CT>
        struct flat_storage_type<CT, typename std::enable_if_t<!has_raw_data_interface<std::decay_t<CT>>::value>>
        {
            using type = flat_expression_adaptor<CT>;
        };

        // with raw_data_interface
        template <class E, std::enable_if_t<has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline decltype(auto) get_flat_storage(E& e)
        {
            return e.data();
        }

        template <class E, std::enable_if_t<has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline std::size_t get_offset(E&& e)
        {
            return e.raw_data_offset();
        }

        template <class E, std::enable_if_t<has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline decltype(auto) get_strides(E&& e)
        {
            return e.strides();
        }

        // without raw_data_interface
        template <class E, std::enable_if_t<!has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline auto get_flat_storage(E& e) -> flat_expression_adaptor<E>
        {
            return flat_expression_adaptor<E>(e);
        }

        template <class E, class S>
        inline auto get_flat_storage(E& e, S&& s, layout_type l) -> flat_expression_adaptor<E>
        {
            return flat_expression_adaptor<E>(e, std::forward<S>(s), l);
        }

        template <class E, std::enable_if_t<!has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline std::size_t get_offset(E&& /*e*/)
        {
            return std::size_t(0);
        }

        template <class E, std::enable_if_t<!has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline auto get_strides(E&& e)
        {
            dynamic_shape<std::size_t> strides;
            strides.resize(e.shape().size());
            compute_strides(e.shape(), DEFAULT_LAYOUT, strides);
            return strides;
        }
    }

    template <class CT, class S, class FS>
    struct xiterable_inner_types<xstrided_view<CT, S, FS>>
    {
        using inner_shape_type = S;
        using inner_strides_type = inner_shape_type;
        using inner_backstrides_type_type = inner_shape_type;

        using const_stepper = std::conditional_t<
            detail::is_indexed_stepper<typename std::decay_t<CT>::stepper>::value,
            xindexed_stepper<const xstrided_view<CT, S, FS>, true>,
            xstepper<const xstrided_view<CT, S, FS>>>;

        using stepper = std::conditional_t<
            detail::is_indexed_stepper<typename std::decay_t<CT>::stepper>::value,
            xindexed_stepper<xstrided_view<CT, S, FS>, false>,
            xstepper<xstrided_view<CT, S, FS>>>;
    };

    /*****************
     * xstrided_view *
     *****************/

    /**
     * @class xstrided_view
     * @brief View of an xexpression using strides
     *
     * The xstrided_view class implements a view utilizing an initial offset
     * and strides.
     *.
     * @tparam CT the closure type of the \ref xexpression type underlying this view
     * @tparam S the strides type of the strided view
     * @tparam FS the flat storage type used for the strided view.
     *
     * @sa strided_view, transpose
     */
    template <class CT, class S, class FS = typename detail::flat_storage_type<CT>::type>
    class xstrided_view : public xview_semantic<xstrided_view<CT, S, FS>>,
                          public xiterable<xstrided_view<CT, S, FS>>
    {
    public:

        using self_type = xstrided_view<CT, S, FS>;
        using xexpression_type = std::decay_t<CT>;
        using semantic_base = xview_semantic<self_type>;

        static constexpr bool is_const = std::is_const<std::remove_reference_t<CT>>::value;

        using value_type = typename xexpression_type::value_type;
        using reference = std::conditional_t<is_const,
                                             typename xexpression_type::const_reference,
                                             typename xexpression_type::reference>;
        using const_reference = typename xexpression_type::const_reference;
        using pointer = std::conditional_t<is_const,
                                           typename xexpression_type::const_pointer,
                                           typename xexpression_type::pointer>;
        using const_pointer = typename xexpression_type::const_pointer;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using inner_container_type = FS;
        using container_type = std::decay_t<inner_container_type>;

        using iterable_base = xiterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;
        using strides_type = shape_type;
        using backstrides_type = shape_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        static constexpr layout_type static_layout = layout_type::dynamic;
        static constexpr bool contiguous_layout = false;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        using base_index_type = xindex_type_t<shape_type>;

        template <class CTA>
        xstrided_view(CTA&& e, S&& shape, S&& strides, std::size_t offset, layout_type layout) noexcept;

        template <class CTA, class FST>
        xstrided_view(CTA&& e, S&& shape, S&& strides, std::size_t offset, layout_type layout, FST&& flatten_strides, layout_type flatten_layout) noexcept;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const shape_type& shape() const noexcept;
        const strides_type& strides() const noexcept;
        const backstrides_type& backstrides() const noexcept;
        layout_type layout() const noexcept;

        reference operator()();
        template <class... Args>
        reference operator()(Args... args);
        template <class... Args>
        reference at(Args... args);
        template <class OS>
        disable_integral_t<OS, reference> operator[](const OS& index);
        template <class I>
        reference operator[](std::initializer_list<I> index);
        reference operator[](size_type i);

        template <class It>
        reference element(It first, It last);

        const_reference operator()() const;
        template <class... Args>
        const_reference operator()(Args... args) const;
        template <class... Args>
        const_reference at(Args... args) const;
        template <class OS>
        disable_integral_t<OS, const_reference> operator[](const OS& index) const;
        template <class I>
        const_reference operator[](std::initializer_list<I> index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class O>
        bool broadcast_shape(O& shape, bool reuse_cache = false) const;

        template <class O>
        bool is_trivial_broadcast(const O& strides) const noexcept;

        template <class ST>
        stepper stepper_begin(const ST& shape);
        template <class ST>
        stepper stepper_end(const ST& shape, layout_type l);

        template <class ST, class STEP = const_stepper>
        std::enable_if_t<!detail::is_indexed_stepper<STEP>::value, STEP>
        stepper_begin(const ST& shape) const;
        template <class ST, class STEP = const_stepper>
        std::enable_if_t<!detail::is_indexed_stepper<STEP>::value, STEP>
        stepper_end(const ST& shape, layout_type l) const;

        template <class ST, class STEP = const_stepper>
        std::enable_if_t<detail::is_indexed_stepper<STEP>::value, STEP>
        stepper_begin(const ST& shape) const;
        template <class ST, class STEP = const_stepper>
        std::enable_if_t<detail::is_indexed_stepper<STEP>::value, STEP>
        stepper_end(const ST& shape, layout_type l) const;

        using container_iterator = std::conditional_t<is_const,
                                                      typename container_type::const_iterator,
                                                      typename container_type::iterator>;
        using const_container_iterator = typename container_type::const_iterator;

        container_type& data() noexcept;
        const container_type& data() const noexcept;

        template <class E = std::decay_t<CT>>
        std::enable_if_t<has_raw_data_interface<std::decay_t<E>>::value, value_type*>
        raw_data() noexcept;
        template <class E = std::decay_t<CT>>
        std::enable_if_t<has_raw_data_interface<std::decay_t<E>>::value, const value_type*>
        raw_data() const noexcept;

        size_type raw_data_offset() const noexcept;

    protected:

        container_iterator data_xbegin() noexcept;
        const_container_iterator data_xbegin() const noexcept;
        container_iterator data_xend(layout_type l) noexcept;
        const_container_iterator data_xend(layout_type l) const noexcept;

    private:

        template <class C>
        friend class xstepper;

        template <class It>
        It data_xbegin_impl(It begin) const noexcept;

        template <class It>
        It data_xend_impl(It end, layout_type l) const noexcept;

        void assign_temporary_impl(temporary_type&& tmp);

        CT m_e;
        inner_container_type m_data;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
        std::size_t m_offset;
        layout_type m_layout;

        friend class xview_semantic<xstrided_view<CT, S, FS>>;
    };

    /********************************
     * xstrided_view implementation *
     ********************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xstrided_view
     *
     * @param e the underlying xexpression for this view
     * @param shape the shape of the view
     * @param strides the strides of the view
     * @param offset the offset of the first element in the underlying container
     * @param layout the layout of the view
     */
    template <class CT, class S, class FS>
    template <class CTA>
    inline xstrided_view<CT, S, FS>::xstrided_view(CTA&& e, S&& shape, S&& strides, std::size_t offset, layout_type layout) noexcept
        : m_e(std::forward<CTA>(e)),
          m_data(detail::get_flat_storage<CT>(m_e)),
          m_shape(std::move(shape)),
          m_strides(std::move(strides)),
          m_offset(offset),
          m_layout(layout)
    {
        m_backstrides = xtl::make_sequence<backstrides_type>(m_shape.size(), 0);
        adapt_strides(m_shape, m_strides, m_backstrides);
    }

    template <class CT, class S, class FS>
    template <class CTA, class FST>
    inline xstrided_view<CT, S, FS>::xstrided_view(CTA&& e, S&& shape, S&& strides, std::size_t offset, layout_type layout, FST&& flatten_strides, layout_type flatten_layout) noexcept
        : m_e(std::forward<CTA>(e)),
          m_data(detail::get_flat_storage<CT>(m_e, std::forward<FST>(flatten_strides), flatten_layout)),
          m_shape(std::move(shape)),
          m_strides(std::move(strides)),
          m_offset(offset),
          m_layout(layout)
    {
        m_backstrides = xtl::make_sequence<backstrides_type>(m_shape.size(), 0);
        adapt_strides(m_shape, m_strides, m_backstrides);
    }
    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class CT, class S, class FS>
    template <class E>
    inline auto xstrided_view<CT, S, FS>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }
    //@}

    template <class CT, class S, class FS>
    template <class E>
    inline auto xstrided_view<CT, S, FS>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(this->begin(), this->end(), e);
        return *this;
    }

    template <class CT, class S, class FS>
    inline void xstrided_view<CT, S, FS>::assign_temporary_impl(temporary_type&& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), this->begin());
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the size of the xstrided_view.
     */
    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    /**
     * Returns the number of dimensions of the xstrided_view.
     */
    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    /**
     * Returns the shape of the xstrided_view.
     */
    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::strides() const noexcept -> const strides_type&
    {
        return m_strides;
    }

    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::backstrides() const noexcept -> const backstrides_type&
    {
        return m_backstrides;
    }

    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::layout() const noexcept -> layout_type
    {
        return m_layout;
    }

    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::data() noexcept -> container_type&
    {
        return m_data;
    }

    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::data() const noexcept -> const container_type&
    {
        return m_data;
    }

    template <class CT, class S, class FS>
    template <class E>
    inline auto xstrided_view<CT, S, FS>::raw_data() noexcept ->
        std::enable_if_t<has_raw_data_interface<std::decay_t<E>>::value, value_type*>
    {
        return m_e.raw_data();
    }

    template <class CT, class S, class FS>
    template <class E>
    inline auto xstrided_view<CT, S, FS>::raw_data() const noexcept ->
        std::enable_if_t<has_raw_data_interface<std::decay_t<E>>::value, const value_type*>
    {
        return m_e.raw_data();
    }

    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::raw_data_offset() const noexcept -> size_type
    {
        return m_offset;
    }
    //@}

    /**
     * @name Data
     */
    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::operator()() -> reference
    {
        return m_data[m_offset];
    }

    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::operator()() const -> const_reference
    {
        return m_data[m_offset];
    }

    template <class CT, class S, class FS>
    template <class... Args>
    inline auto xstrided_view<CT, S, FS>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        size_type index = m_offset + data_offset<size_type>(strides(), static_cast<size_type>(args)...);
        return m_data[index];
    }

    /**
     * Returns the element at the specified position in the xstrided_view.
     *
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the view.
     */
    template <class CT, class S, class FS>
    template <class... Args>
    inline auto xstrided_view<CT, S, FS>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        size_type index = m_offset + data_offset<size_type>(strides(), static_cast<size_type>(args)...);
        return m_data[index];
    }

    /**
     * Returns a reference to the element at the specified position in the expression,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the expression.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class CT, class S, class FS>
    template <class... Args>
    inline auto xstrided_view<CT, S, FS>::at(Args... args) -> reference
    {
        check_access(shape(), static_cast<size_type>(args)...);
        return this->operator()(args...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the expression.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class CT, class S, class FS>
    template <class... Args>
    inline auto xstrided_view<CT, S, FS>::at(Args... args) const -> const_reference
    {
        check_access(shape(), static_cast<size_type>(args)...);
        return this->operator()(args...);
    }

    template <class CT, class S, class FS>
    template <class OS>
    inline auto xstrided_view<CT, S, FS>::operator[](const OS& index)
        -> disable_integral_t<OS, reference>
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT, class S, class FS>
    template <class I>
    inline auto xstrided_view<CT, S, FS>::operator[](std::initializer_list<I> index)
        -> reference
    {
        return element(index.begin(), index.end());
    }

    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::operator[](size_type i) -> reference
    {
        return operator()(i);
    }

    template <class CT, class S, class FS>
    template <class OS>
    inline auto xstrided_view<CT, S, FS>::operator[](const OS& index) const
        -> disable_integral_t<OS, const_reference>
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT, class S, class FS>
    template <class I>
    inline auto xstrided_view<CT, S, FS>::operator[](std::initializer_list<I> index) const
        -> const_reference
    {
        return element(index.begin(), index.end());
    }

    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a reference to the element at the specified position in the xstrided_view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater than the the number
     * of dimensions of the container..
     */
    template <class CT, class S, class FS>
    template <class It>
    inline auto xstrided_view<CT, S, FS>::element(It first, It last) -> reference
    {
        return m_data[m_offset + element_offset<size_type>(strides(), first, last)];
    }

    template <class CT, class S, class FS>
    template <class It>
    inline auto xstrided_view<CT, S, FS>::element(It first, It last) const -> const_reference
    {
        return m_data[m_offset + element_offset<size_type>(strides(), first, last)];
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
    template <class CT, class S, class FS>
    template <class O>
    inline bool xstrided_view<CT, S, FS>::broadcast_shape(O& shape, bool) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class S, class FS>
    template <class O>
    inline bool xstrided_view<CT, S, FS>::is_trivial_broadcast(const O& str) const noexcept
    {
        return str.size() == strides().size() &&
            std::equal(str.cbegin(), str.cend(), strides().begin());
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class CT, class S, class FS>
    template <class ST>
    inline auto xstrided_view<CT, S, FS>::stepper_begin(const ST& shape) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, data_xbegin(), offset);
    }

    template <class CT, class S, class FS>
    template <class ST>
    inline auto xstrided_view<CT, S, FS>::stepper_end(const ST& shape, layout_type l) -> stepper
    {
        size_type offset = shape.size() - dimension();
        return stepper(this, data_xend(l), offset);
    }

    template <class CT, class S, class FS>
    template <class ST, class STEP>
    inline auto xstrided_view<CT, S, FS>::stepper_begin(const ST& shape) const -> std::enable_if_t<!detail::is_indexed_stepper<STEP>::value, STEP>
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, data_xbegin(), offset);
    }

    template <class CT, class S, class FS>
    template <class ST, class STEP>
    inline auto xstrided_view<CT, S, FS>::stepper_end(const ST& shape, layout_type l) const -> std::enable_if_t<!detail::is_indexed_stepper<STEP>::value, STEP>
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, data_xend(l), offset);
    }

    template <class CT, class S, class FS>
    template <class ST, class STEP>
    inline auto xstrided_view<CT, S, FS>::stepper_begin(const ST& shape) const -> std::enable_if_t<detail::is_indexed_stepper<STEP>::value, STEP>
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset);
    }

    template <class CT, class S, class FS>
    template <class ST, class STEP>
    inline auto xstrided_view<CT, S, FS>::stepper_end(const ST& shape, layout_type /*l*/) const -> std::enable_if_t<detail::is_indexed_stepper<STEP>::value, STEP>
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset, true);
    }

    template <class CT, class S, class FS>
    template <class It>
    inline It xstrided_view<CT, S, FS>::data_xbegin_impl(It begin) const noexcept
    {
        return begin + m_offset;
    }

    template <class CT, class S, class FS>
    template <class It>
    inline It xstrided_view<CT, S, FS>::data_xend_impl(It end, layout_type l) const noexcept
    {
        return strided_data_end(*this, end, l);
    }

    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::data_xbegin() noexcept -> container_iterator
    {
        return data_xbegin_impl(m_data.begin());
    }

    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::data_xbegin() const noexcept -> const_container_iterator
    {
        return data_xbegin_impl(m_data.cbegin());
    }

    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::data_xend(layout_type l) noexcept -> container_iterator
    {
        return data_xend_impl(m_data.end(), l);
    }

    template <class CT, class S, class FS>
    inline auto xstrided_view<CT, S, FS>::data_xend(layout_type l) const noexcept -> const_container_iterator
    {
        return data_xend_impl(m_data.cend(), l);
    }

    /**
     * Construct a strided view from an xexpression, shape, strides and offset.
     *
     * @param e xexpression
     * @param shape the shape of the view
     * @param strides the new strides of the view
     * @param offset the offset of the first element in the underlying container
     * @param layout the new layout of the expression
     *
     * @tparam E type of xexpression
     * @tparam I shape and strides type
     *
     * @return the view
     */
    template <class E, class I>
    inline auto strided_view(E&& e, I&& shape, I&& strides, std::size_t offset = 0, layout_type layout = layout_type::dynamic) noexcept
    {
        using view_type = xstrided_view<xclosure_t<E>, I>;
        return view_type(std::forward<E>(e), std::forward<I>(shape), std::forward<I>(strides), offset, layout);
    }

    namespace detail
    {
        template <class CT>
        class flat_expression_adaptor
        {
        public:

            using xexpression_type = std::decay_t<CT>;
            using shape_type = typename xexpression_type::shape_type;
            using index_type = xindex_type_t<shape_type>;
            using size_type = typename xexpression_type::size_type;
            using value_type = typename xexpression_type::value_type;
            using reference = typename xexpression_type::reference;
            using const_reference = typename xexpression_type::const_reference;

            using iterator = typename xexpression_type::iterator;
            using const_iterator = typename xexpression_type::const_iterator;

            flat_expression_adaptor(CT& e)
                : m_e(e)
            {
                resize_container(m_index, m_e.dimension());
                resize_container(m_strides, m_e.dimension());
                m_size = compute_size(m_e.shape());
                // Fallback to DEFAULT_LAYOUT when the underlying layout is not
                // row-major or column major.
                m_layout = default_assignable_layout(m_e.layout());
                compute_strides(m_e.shape(), m_layout, m_strides);
            }

            template <class FS>
            flat_expression_adaptor(CT& e, FS&& strides, layout_type layout)
                : m_e(e), m_strides(xtl::forward_sequence<shape_type>(strides)), m_layout(layout)
            {
                resize_container(m_index, m_e.dimension());
                m_size = e.size();
            }

            reference operator[](std::size_t idx)
            {
                m_index = detail::unravel_noexcept(idx, m_strides, m_layout);
                return m_e.element(m_index.cbegin(), m_index.cend());
            }

            const_reference operator[](std::size_t idx) const
            {
                m_index = detail::unravel_noexcept(idx, m_strides, m_layout);
                return m_e.element(m_index.cbegin(), m_index.cend());
            }

            size_type size() const
            {
                return m_size;
            }

            iterator begin()
            {
                return m_e.begin();
            }

            iterator end()
            {
                return m_e.end();
            }

            const_iterator begin() const
            {
                return m_e.cbegin();
            }

            const_iterator end() const
            {
                return m_e.cend();
            }

            const_iterator cbegin() const
            {
                return m_e.cbegin();
            }

            const_iterator cend() const
            {
                return m_e.cend();
            }

        private:

            CT& m_e;
            shape_type m_strides;
            mutable index_type m_index;
            size_type m_size;
            layout_type m_layout;
        };

        template <class E>
        struct slice_getter_impl
        {
            E& m_expr;
            mutable std::size_t idx;

            slice_getter_impl(E& expr)
                : m_expr(expr), idx(0)
            {
            }

            template <class T>
            std::array<std::ptrdiff_t, 3> operator()(const T& t) const
            {
                auto sl = get_slice_implementation(m_expr, t, idx);
                return std::array<std::ptrdiff_t, 3>({std::ptrdiff_t(sl(0)), std::ptrdiff_t(sl.size()), std::ptrdiff_t(sl.step_size())});
            }

            std::array<std::ptrdiff_t, 3> operator()(const std::ptrdiff_t& /*t*/) const
            {
                return std::array<std::ptrdiff_t, 3>({0, 0, 0});
            }
        };
    }

    template <class T>
    using slice_variant = xtl::variant<
        T,

        xrange_adaptor<placeholders::xtuph, T, T>,
        xrange_adaptor<T, placeholders::xtuph, T>,
        xrange_adaptor<T, T, placeholders::xtuph>,

        xrange_adaptor<T, placeholders::xtuph, placeholders::xtuph>,
        xrange_adaptor<placeholders::xtuph, T, placeholders::xtuph>,
        xrange_adaptor<placeholders::xtuph, placeholders::xtuph, T>,

        xrange_adaptor<T, T, T>,
        xrange_adaptor<placeholders::xtuph, placeholders::xtuph, placeholders::xtuph>,

        xall_tag,
        xellipsis_tag,
        xnewaxis_tag
    >;

    /**
     * @typedef slice_vector
     * @brief vector of slices used to build a `xstrided_view`
     */
    using slice_vector = std::vector<slice_variant<std::ptrdiff_t>>;

    /**
     * Function to create a dynamic view from
     * a xexpression and a slice_vector.
     *
     * @param e xexpression
     * @param slices the slice vector
     *
     * @return initialized strided_view according to slices
     *
     * \code{.cpp}
     * xt::xarray<double> a = {{1, 2, 3}, {4, 5, 6}};
     * xt::slice_vector sv({xt::range(0, 1)});
     * sv.push_back(xt::range(0, 3, 2));
     * auto v = xt::dynamic_view(a, sv);
     * // ==> {{1, 3}}
     * \endcode
     *
     * You can also achieve the same with the following short-hand syntax:
     *
     * \code{.cpp}
     * xt::xarray<double> a = {{1, 2, 3}, {4, 5, 6}};
     * auto v = xt::dynamic_view(a, {xt::range(0, 1), xt::range(0, 3, 2)});
     * // ==> {{1, 3}}
     * \endcode
     */
    template <class E>
    inline auto dynamic_view(E&& e, const slice_vector& slices)
    {
        // Compute dimension
        std::size_t dimension = e.dimension(), n_newaxis = 0, n_add_all = 0;
        std::ptrdiff_t dimension_check = static_cast<std::ptrdiff_t>(e.dimension());

        bool has_ellipsis = false;
        for (const auto& el : slices)
        {
            if (xtl::get_if<xt::xnewaxis_tag>(&el) != nullptr)
            {
                ++dimension;
                ++n_newaxis;
            }
            else if (xtl::get_if<std::ptrdiff_t>(&el) != nullptr)
            {
                --dimension;
                --dimension_check;
            }
            else if (xtl::get_if<xt::xellipsis_tag>(&el) != nullptr)
            {
                if (has_ellipsis == true)
                {
                    throw std::runtime_error("Ellipsis can only appear once.");
                }
                has_ellipsis = true;
            }
            else
            {
                --dimension_check;
            }
        }

        if (dimension_check < 0)
        {
            throw std::runtime_error("Too many slices for view.");
        }

        if (has_ellipsis)
        {
            // replace ellipsis with N * xt::all
            // remove -1 because of the ellipsis slize itself
            n_add_all = e.dimension() - (slices.size() - 1 - n_newaxis);
        }

        // Compute strided view
        std::size_t offset = detail::get_offset(e);
        using shape_type = dynamic_shape<std::size_t>;

        shape_type new_shape(dimension);
        shape_type new_strides(dimension);

        auto old_shape = e.shape();
        auto&& old_strides = detail::get_strides(e);

#define MS(v) static_cast<std::ptrdiff_t>(v)
#define MU(v) static_cast<std::size_t>(v)

        std::ptrdiff_t i = 0, axis_skip = 0;
        std::size_t idx = 0;

        auto slice_getter = detail::slice_getter_impl<E>(e);

        for (; i < MS(slices.size()); ++i)
        {
            auto ptr = xtl::get_if<std::ptrdiff_t>(&slices[MU(i)]);
            if (ptr != nullptr)
            {
                std::size_t slice0 = static_cast<std::size_t>(*ptr);
                offset += slice0 * old_strides[MU(i - axis_skip)];
            }
            else if (xtl::get_if<xt::xnewaxis_tag>(&slices[MU(i)]) != nullptr)
            {
                new_shape[idx] = 1;
                new_strides[idx] = 0;
                ++axis_skip;
                ++idx;
            }
            else if (xtl::get_if<xt::xellipsis_tag>(&slices[MU(i)]) != nullptr)
            {
                for (std::size_t j = 0; j < n_add_all; ++j)
                {
                    new_shape[idx] = old_shape[MU(i - axis_skip)];
                    new_strides[idx] = old_strides[MU(i - axis_skip)];
                    ++idx, --axis_skip;
                }
                ++axis_skip;  // because i++
            }
            else if (xtl::get_if<xt::xall_tag>(&slices[MU(i)]) != nullptr)
            {
                new_shape[idx] = old_shape[MU(i - axis_skip)];
                new_strides[idx] = old_strides[MU(i - axis_skip)];
                ++idx;
            }
            else
            {
                slice_getter.idx = MU(i - axis_skip);
                std::array<std::ptrdiff_t, 3> info = xtl::visit(slice_getter, slices[MU(i)]);
                offset += std::size_t(info[0]) * old_strides[MU(i - axis_skip)];
                new_shape[idx] = std::size_t(info[1]);
                new_strides[idx] = std::size_t(info[2]) * old_strides[MU(i - axis_skip)];
                ++idx;
            }
        }

        for (; MU(i - axis_skip) < old_shape.size(); ++i)
        {
            new_shape[idx] = old_shape[MU(i - axis_skip)];
            new_strides[idx] = old_strides[MU(i - axis_skip)];
            ++idx;
        }

        using view_type = xstrided_view<xclosure_t<E>, shape_type>;

        // TODO change layout type?
        return view_type(std::forward<E>(e), std::move(new_shape), std::move(new_strides), offset, layout_type::dynamic);

#undef MU
#undef MS
    }

    /****************************
     * transpose implementation *
     ****************************/

    namespace detail
    {
        inline layout_type transpose_layout_noexcept(layout_type l) noexcept
        {
            layout_type result = l;
            if (l == layout_type::row_major)
            {
                result = layout_type::column_major;
            }
            else if (l == layout_type::column_major)
            {
                result = layout_type::row_major;
            }
            return result;
        }

        inline layout_type transpose_layout(layout_type l)
        {
            if (l != layout_type::row_major && l != layout_type::column_major)
            {
                throw transpose_error("cannot compute transposed layout of dynamic layout");
            }
            return transpose_layout_noexcept(l);
        }

        template <class E, class S>
        inline auto transpose_impl(E&& e, S&& permutation, check_policy::none)
        {
            if (sequence_size(permutation) != e.dimension())
            {
                throw transpose_error("Permutation does not have the same size as shape");
            }

            // permute stride and shape
            using strides_type = typename std::decay_t<E>::strides_type;
            strides_type temp_strides;
            resize_container(temp_strides, e.strides().size());

            using shape_type = typename std::decay_t<E>::shape_type;
            shape_type temp_shape;
            resize_container(temp_shape, e.shape().size());

            using size_type = typename std::decay_t<E>::size_type;
            for (std::size_t i = 0; i < e.shape().size(); ++i)
            {
                if (std::size_t(permutation[i]) >= e.dimension())
                {
                    throw transpose_error("Permutation contains wrong axis");
                }
                size_type perm = static_cast<size_type>(permutation[i]);
                temp_shape[i] = e.shape()[perm];
                temp_strides[i] = e.strides()[perm];
            }

            layout_type new_layout = layout_type::dynamic;
            if (std::is_sorted(std::begin(permutation), std::end(permutation)))
            {
                // keep old layout
                new_layout = e.layout();
            }
            else if (std::is_sorted(std::begin(permutation), std::end(permutation), std::greater<>()))
            {
                new_layout = transpose_layout_noexcept(e.layout());
            }

            using view_type = xstrided_view<xclosure_t<E>, shape_type>;
            return view_type(std::forward<E>(e), std::move(temp_shape), std::move(temp_strides), 0, new_layout);
        }

        template <class E, class S>
        inline auto transpose_impl(E&& e, S&& permutation, check_policy::full)
        {
            // check if axis appears twice in permutation
            for (std::size_t i = 0; i < sequence_size(permutation); ++i)
            {
                for (std::size_t j = i + 1; j < sequence_size(permutation); ++j)
                {
                    if (permutation[i] == permutation[j])
                    {
                        throw transpose_error("Permutation contains axis more than once");
                    }
                }
            }
            return transpose_impl(std::forward<E>(e), std::forward<S>(permutation), check_policy::none());
        }

        template <class E, class S, std::enable_if_t<has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline void compute_transposed_strides(E&& e, const S&, S& strides)
        {
            std::copy(e.strides().crbegin(), e.strides().crend(), strides.begin());
        }

        template <class E, class S, std::enable_if_t<!has_raw_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline void compute_transposed_strides(E&&, const S& shape, S& strides)
        {
            layout_type l = transpose_layout(std::decay_t<E>::static_layout);
            compute_strides(shape, l, strides);
        }
    }

    /**
     * Returns a transpose view by reversing the dimensions of xexpression e
     * @param e the input expression
     */
    template <class E>
    inline auto transpose(E&& e) noexcept
    {
        using shape_type = typename std::decay_t<E>::shape_type;
        shape_type shape;
        resize_container(shape, e.shape().size());
        std::copy(e.shape().crbegin(), e.shape().crend(), shape.begin());

        shape_type strides;
        resize_container(strides, e.shape().size());
        detail::compute_transposed_strides(e, shape, strides);

        layout_type new_layout = detail::transpose_layout_noexcept(e.layout());

        using view_type = xstrided_view<xclosure_t<E>, shape_type>;
        return view_type(std::forward<E>(e), std::move(shape), std::move(strides), 0, new_layout);
    }

    /**
     * Returns a transpose view by permuting the xexpression e with @p permutation.
     * @param e the input expression
     * @param permutation the sequence containing permutation
     * @param check_policy the check level (check_policy::full() or check_policy::none())
     * @tparam Tag selects the level of error checking on permutation vector defaults to check_policy::none.
     */
    template <class E, class S, class Tag = check_policy::none>
    inline auto transpose(E&& e, S&& permutation, Tag check_policy = Tag())
    {
        return detail::transpose_impl(std::forward<E>(e), std::forward<S>(permutation), check_policy);
    }

    /// @cond DOXYGEN_INCLUDE_SFINAE
#ifdef X_OLD_CLANG
    template <class E, class I, class Tag = check_policy::none>
    inline auto transpose(E&& e, std::initializer_list<I> permutation, Tag check_policy = Tag())
    {
        dynamic_shape<I> perm(permutation);
        return detail::transpose_impl(std::forward<E>(e), std::move(perm), check_policy);
    }
#else
    template <class E, class I, std::size_t N, class Tag = check_policy::none>
    inline auto transpose(E&& e, const I(&permutation)[N], Tag check_policy = Tag())
    {
        return detail::transpose_impl(std::forward<E>(e), permutation, check_policy);
    }
#endif
    /// @endcond

    /***************************
     * ravel and flatten views *
     ***************************/

    namespace detail
    {
        template <class E>
        inline auto build_ravel_view(E&& e)
        {
            using shape_type = static_shape<std::size_t, 1>;
            using view_type = xstrided_view<xclosure_t<E>, shape_type>;

            shape_type new_shape, new_strides;
            new_shape[0] = e.size();
            new_strides[0] = std::size_t(1);
            std::size_t offset = detail::get_offset(e);

            return view_type(std::forward<E>(e),
                             std::move(new_shape),
                             std::move(new_strides),
                             offset,
                             layout_type::dynamic);
        }

        template <class E, class S>
        inline auto build_ravel_view(E&& e, S&& flatten_strides, layout_type l)
        {
            using shape_type = static_shape<std::size_t, 1>;
            using view_type = xstrided_view<xclosure_t<E>, shape_type, detail::flat_expression_adaptor<xclosure_t<E>>>;

            shape_type new_shape, new_strides;
            new_shape[0] = e.size();
            new_strides[0] = std::size_t(1);
            std::size_t offset = detail::get_offset(e);

            return view_type(std::forward<E>(e),
                             std::move(new_shape),
                             std::move(new_strides),
                             offset,
                             layout_type::dynamic,
                             std::move(flatten_strides),
                             l);
        }

        template <bool same_layout>
        struct ravel_impl
        {
            template <class E>
            inline static auto run(E&& e)
            {
                return build_ravel_view(std::forward<E>(e));
            }
        };

        template <>
        struct ravel_impl<false>
        {
            template <class E>
            inline static auto run(E&& e)
            {
                // Case where the static layout is either row_major or column major.
                using flat_proxy_type = flat_expression_adaptor<xclosure_t<E>>;
                using shape_type = typename std::decay_t<E>::shape_type;
                shape_type strides;
                resize_container(strides, e.shape().size());
                layout_type l = detail::transpose_layout(e.layout());
                compute_strides(e.shape(), l, strides);
                return build_ravel_view(std::forward<E>(e), std::move(strides), l);
            }
        };
    }

    /**
     * Returns a flatten view of the given expression. No copy is made.
     * @param e the input expression
     * @tparam L the layout used to read the elements of e
     * @tparam E the type of the expression
     */
    template <layout_type L, class E>
    inline auto ravel(E&& e)
    {
        return detail::ravel_impl<std::decay_t<E>::static_layout == L>::run(std::forward<E>(e));
    }

    /**
     * Returns a flatten view of the given expression. No copy is made.
     * The layout used to read the elements is the one of e.
     * @param e the input expression
     * @tparam E the type of the expression
     */
    template <class E>
    inline auto flatten(E&& e)
    {
        return ravel<std::decay_t<E>::static_layout>(std::forward<E>(e));
    }

    /**
     * Trim zeros at beginning, end or both of 1D sequence.
     *
     * @param e input xexpression
     * @param direction string of either 'f' for trim from beginning, 'b' for trim from end
     *                  or 'fb' (default) for both.
     * @return returns a view without zeros at the beginning and end
     */
    template <class E>
    inline auto trim_zeros(E&& e, const std::string& direction = "fb")
    {
        XTENSOR_ASSERT_MSG(e.dimension() == 1, "Dimension for trim_zeros has to be 1.");

        std::ptrdiff_t begin = 0, end = static_cast<std::ptrdiff_t>(e.size());

        auto find_fun = [](const auto& i) {
            return i != 0;
        };

        if (direction.find("f") != std::string::npos)
        {
            begin = std::find_if(e.cbegin(), e.cend(), find_fun) - e.cbegin();
        }

        if (direction.find("b") != std::string::npos && begin != end)
        {
            end -= std::find_if(e.crbegin(), e.crend(), find_fun) - e.crbegin();
        }

        return dynamic_view(std::forward<E>(e), {range(static_cast<int>(begin), static_cast<int>(end))});
    }

    /**
     * Returns a squeeze view of the given expression. No copy is made.
     * Squeezing an expression removes dimensions of extent 1.
     *
     * @param e the input expression
     * @tparam E the type of the expression
     */
    template <class E>
    inline auto squeeze(E&& e)
    {
        xt::dynamic_shape<std::size_t> new_shape, new_strides;
        std::copy_if(e.shape().cbegin(), e.shape().cend(), std::back_inserter(new_shape),
                     [](std::size_t i) { return i != 1; });
        auto&& old_strides = detail::get_strides(e);
        std::copy_if(old_strides.cbegin(), old_strides.cend(), std::back_inserter(new_strides),
                     [](std::size_t i) { return i != 0; });

        using view_type = xstrided_view<xclosure_t<E>, xt::dynamic_shape<std::size_t>>;
        return view_type(std::forward<E>(e), std::move(new_shape), std::move(new_strides), 0, e.layout());
    }

    namespace detail
    {
        template <class E, class S>
        inline auto squeeze_impl(E&& e, S&& axis, check_policy::none)
        {
            std::size_t new_dim = e.dimension() - axis.size();
            xt::dynamic_shape<std::size_t> new_shape(new_dim), new_strides(new_dim);

            decltype(auto) old_strides = detail::get_strides(e);

            for (std::size_t i = 0, ix = 0; i < e.dimension(); ++i)
            {
                if (axis.cend() == std::find(axis.cbegin(), axis.cend(), i))
                {
                    new_shape[ix] = e.shape()[i];
                    new_strides[ix++] = old_strides[i];
                }
            }

            using view_type = xstrided_view<xclosure_t<E>, xt::dynamic_shape<std::size_t>>;
            return view_type(std::forward<E>(e), std::move(new_shape), std::move(new_strides), 0, e.layout());
        }

        template <class E, class S>
        inline auto squeeze_impl(E&& e, S&& axis, check_policy::full)
        {
            for (auto ix : axis)
            {
                if (static_cast<std::size_t>(ix) > e.dimension())
                {
                    throw std::runtime_error("Axis argument to squeeze > dimension of expression");
                }
                if (e.shape()[static_cast<std::size_t>(ix)] != 1)
                {
                    throw std::runtime_error("Trying to squeeze axis != 1");
                }
            }
            return squeeze_impl(std::forward<E>(e), std::forward<S>(axis), check_policy::none());
        }
    }

    /**
     * @brief Remove single-dimensional entries from the shape of an xexpression
     *
     * @param e input xexpression
     * @param axis integer or container of integers, select a subset of single-dimensional
     *        entries of the shape.
     * @param check_policy select check_policy. With check_policy::full(), selecting an axis
     *        which is greater than one will throw a runtime_error.
     */
    template <class E, class S, class Tag = check_policy::none, std::enable_if_t<!std::is_integral<S>::value, int> = 0>
    inline auto squeeze(E&& e, S&& axis, Tag check_policy = Tag())
    {
        return detail::squeeze_impl(std::forward<E>(e), std::forward<S>(axis), check_policy);
    }

    /// @cond DOXYGEN_INCLUDE_SFINAE
#ifdef X_OLD_CLANG
    template <class E, class I, class Tag = check_policy::none>
    inline auto squeeze(E&& e, std::initializer_list<I> axis, Tag check_policy = Tag())
    {
        dynamic_shape<I> ax(axis);
        return detail::squeeze_impl(std::forward<E>(e), std::move(ax), check_policy);
    }
#else
    template <class E, class I, std::size_t N, class Tag = check_policy::none>
    inline auto squeeze(E&& e, const I(&axis)[N], Tag check_policy = Tag())
    {
        using arr_t = std::array<I, N>;
        return detail::squeeze_impl(std::forward<E>(e), xtl::forward_sequence<arr_t>(axis), check_policy);
    }
#endif

    template <class E, class Tag = check_policy::none>
    inline auto squeeze(E&& e, std::size_t axis, Tag check_policy = Tag())
    {
        return squeeze(std::forward<E>(e), std::array<std::size_t, 1>({axis}), check_policy);
    }
    /// @endcond

    /**
     * @brief Expand the shape of an xexpression.
     *
     * Insert a new axis that will appear at the axis position in the expanded array shape.
     * This will return a ``dynamic_view`` with a ``xt::newaxis()`` at the indicated axis.
     *
     * @param e input xexpression
     * @param axis axis to expand
     * @return returns a ``dynamic_view`` with expanded dimension
     */
    template <class E>
    auto expand_dims(E&& e, std::size_t axis)
    {
        slice_vector sv(e.dimension() + 1, xt::all());
        sv[axis] = xt::newaxis();
        return dynamic_view(std::forward<E>(e), std::move(sv));
    }

    /**
     * Expand dimensions of xexpression to at least `N`
     *
     * This adds ``newaxis()`` slices to a ``dynamic_view`` until
     * the dimension of the view reaches at least `N`.
     * Note: dimensions are added equally at the beginning and the end.
     * For example, a 1-D array of shape (N,) becomes a view of shape (1, N, 1).
     *
     * @param e input xexpression
     * @tparam N the number of requested dimensions
     * @return ``dynamic_view`` with expanded dimensions
     */
    template <std::size_t N, class E>
    auto atleast_Nd(E&& e)
    {
        slice_vector sv(std::max(e.dimension(), N), xt::all());
        if (e.dimension() < N)
        {
            std::size_t i = 0;
            for (; i < std::round(double((N - e.dimension()) / double(N))); ++i)
            {
                sv[i] = xt::newaxis();
            }
            i += e.dimension();
            for (; i < N; ++i)
            {
                sv[i] = xt::newaxis();
            }
        }
        return dynamic_view(std::forward<E>(e), std::move(sv));
    }

    /**
     * Expand to at least 1D
     * @sa atleast_Nd
     */
    template <class E>
    auto atleast_1d(E&& e)
    {
        return atleast_Nd<1>(std::forward<E>(e));
    }

    /**
     * Expand to at least 2D
     * @sa atleast_Nd
     */
    template <class E>
    auto atleast_2d(E&& e)
    {
        return atleast_Nd<2>(std::forward<E>(e));
    }

    /**
     * Expand to at least 3D
     * @sa atleast_Nd
     */
    template <class E>
    auto atleast_3d(E&& e)
    {
        return atleast_Nd<3>(std::forward<E>(e));
    }

    /**
     * @brief Split xexpression along axis into subexpressions
     *
     * This splits an xexpression along the axis in `n` equal parts and
     * returns a vector of ``dynamic_view``.
     * Calling split with axis > dimension of e or a `n` that does not result in
     * an equal division of the xexpression will throw a runtime_error.
     *
     * @param e input xexpression
     * @param n number of elements to return
     * @param axis axis along which to split the expression
     */
    template <class E>
    auto split(E& e, std::size_t n, std::size_t axis = 0)
    {
        if (axis >= e.dimension())
        {
            throw std::runtime_error("Split along axis > dimension.");
        }

        std::size_t ax_sz = e.shape()[axis];
        slice_vector sv(e.dimension(), xt::all());
        std::size_t step = ax_sz / n;
        std::size_t rest = ax_sz % n;

        if (rest)
        {
            throw std::runtime_error("Split does not result in equal division.");
        }

        std::vector<decltype(dynamic_view(e, sv))> result;
        for (std::size_t i = 0; i < n; ++i)
        {
            sv[axis] = range(static_cast<int>(i * step), static_cast<int>((i + 1) * step));
            result.emplace_back(dynamic_view(e, sv));
        }
        return result;
    }
}

#endif
