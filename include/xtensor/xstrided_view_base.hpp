/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_STRIDED_VIEW_BASE_HPP
#define XTENSOR_STRIDED_VIEW_BASE_HPP

#include <xtl/xsequence.hpp>

#include "xtensor_forward.hpp"
#include "xslice.hpp"
#include "xstrides.hpp"
#include "xutils.hpp"

namespace xt
{
    template <class CT, class S, layout_type L, class FST>
    class xstrided_view_base
    {
    public:

        using xexpression_type = std::decay_t<CT>;
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

        using inner_storage_type = FST;
        using storage_type = std::remove_reference_t<inner_storage_type>;

        using shape_type = S;
        using strides_type = shape_type;
        using backstrides_type = shape_type;

        static constexpr layout_type static_layout = L;
        static constexpr bool contiguous_layout = static_layout != layout_type::dynamic;

        template <class CTA>
        xstrided_view_base(CTA&& e, S&& shape, S&& strides, std::size_t offset, layout_type layout) noexcept;

        template <class CTA, class FLS>
        xstrided_view_base(CTA&& e, S&& shape, S&& strides, std::size_t offset,
                           layout_type layout, FLS&& flatten_strides, layout_type flatten_layout) noexcept;

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const shape_type& shape() const noexcept;
        const strides_type& strides() const noexcept;
        const backstrides_type& backstrides() const noexcept;
        layout_type layout() const noexcept;

        reference operator()();
        const_reference operator()() const;

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class... Args>
        reference at(Args... args);

        template <class... Args>
        const_reference at(Args... args) const;

        template <class... Args>
        reference unchecked(Args... args);

        template <class... Args>
        const_reference unchecked(Args... args) const;

        template <class OS>
        disable_integral_t<OS, reference> operator[](const OS& index);
        template <class I>
        reference operator[](std::initializer_list<I> index);
        reference operator[](size_type i);

        template <class OS>
        disable_integral_t<OS, const_reference> operator[](const OS& index) const;
        template <class I>
        const_reference operator[](std::initializer_list<I> index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        storage_type& storage() noexcept;
        const storage_type& storage() const noexcept;

        template <class E = std::decay_t<CT>>
        std::enable_if_t<has_data_interface<std::decay_t<E>>::value, value_type*>
        data() noexcept;
        template <class E = std::decay_t<CT>>
        std::enable_if_t<has_data_interface<std::decay_t<E>>::value, const value_type*>
        data() const noexcept;
        size_type data_offset() const noexcept;

        xexpression_type& expression() noexcept;
        const xexpression_type& expression() const noexcept;

        template <class O>
        bool broadcast_shape(O& shape, bool reuse_cache = false) const;

        template <class O>
        bool is_trivial_broadcast(const O& strides) const noexcept;

    private:

        CT m_e;
        inner_storage_type m_storage;
        shape_type m_shape;
        strides_type m_strides;
        backstrides_type m_backstrides;
        std::size_t m_offset;
        layout_type m_layout;
    };

    /***************************
     * flat_expression_adaptor *
     ***************************/

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

            flat_expression_adaptor(CT& e);

            template <class FST>
            flat_expression_adaptor(CT& e, FST&& strides, layout_type layout);

            size_type size() const;
            reference operator[](std::size_t idx);
            const_reference operator[](std::size_t idx) const;

            iterator begin();
            iterator end();
            const_iterator begin() const;
            const_iterator end() const;
            const_iterator cbegin() const;
            const_iterator cend() const;

        private:

            CT& m_e;
            shape_type m_strides;
            mutable index_type m_index;
            size_type m_size;
            layout_type m_layout;
        };

        template <class CT, class T = void>
        struct flat_storage_type;

        template <class CT>
        struct flat_storage_type<CT, typename std::enable_if_t<has_data_interface<std::decay_t<CT>>::value>>
        {
            // Note: we could also use the storage_type typedef.
            // using type = std::conditional_t<
            //    std::is_const<std::remove_reference_t<CT>>::value,
            //    const typename std::decay_t<CT>::storage_type&,
            //    typename std::decay_t<CT>::storage_type&>;
            using type = decltype(std::declval<CT>().storage());
        };

        template <class CT>
        struct flat_storage_type<CT, typename std::enable_if_t<!has_data_interface<std::decay_t<CT>>::value>>
        {
            using type = flat_expression_adaptor<CT>;
        };

        template <class CT>
        using flat_storage_type_t = typename flat_storage_type<CT>::type;

        // with data_interface
        template <class E, std::enable_if_t<has_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline decltype(auto) get_flat_storage(E& e)
        {
            return e.storage();
        }

        template <class E, std::enable_if_t<has_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline std::size_t get_offset(E&& e)
        {
            return e.data_offset();
        }

        template <class E, std::enable_if_t<has_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline decltype(auto) get_strides(E&& e)
        {
            return e.strides();
        }

        // without data_interface
        template <class E, std::enable_if_t<!has_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline auto get_flat_storage(E& e) -> flat_expression_adaptor<E>
        {
            return flat_expression_adaptor<E>(e);
        }

        template <class E, class S>
        inline auto get_flat_storage(E& e, S&& s, layout_type l) -> flat_expression_adaptor<E>
        {
            return flat_expression_adaptor<E>(e, std::forward<S>(s), l);
        }

        template <class E, std::enable_if_t<!has_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline std::size_t get_offset(E&& /*e*/)
        {
            return std::size_t(0);
        }

        template <class E, std::enable_if_t<!has_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline auto get_strides(E&& e)
        {
            dynamic_shape<std::size_t> strides;
            strides.resize(e.shape().size());
            compute_strides(e.shape(), XTENSOR_DEFAULT_LAYOUT, strides);
            return strides;
        }
    }

    /*************************************
     * xstrided_view_base implementation *
     *************************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xstrided_view_base
     *
     * @param e the underlying xexpression for this view
     * @param shape the shape of the view
     * @param strides the strides of the view
     * @param offset the offset of the first element in the underlying container
     * @param layout the layout of the view
     */
    template <class CT, class S, layout_type L, class FST>
    template <class CTA>
    inline xstrided_view_base<CT, S, L, FST>::xstrided_view_base(CTA&& e, S&& shape, S&& strides, std::size_t offset, layout_type layout) noexcept
        : m_e(std::forward<CTA>(e)),
          m_storage(detail::get_flat_storage<CT>(m_e)),
          m_shape(std::move(shape)),
          m_strides(std::move(strides)),
          m_offset(offset),
          m_layout(layout)
    {
        m_backstrides = xtl::make_sequence<backstrides_type>(m_shape.size(), 0);
        adapt_strides(m_shape, m_strides, m_backstrides);
    }

    template <class CT, class S, layout_type L, class FST>
    template <class CTA, class FLS>
    inline xstrided_view_base<CT, S, L, FST>::xstrided_view_base(CTA&& e, S&& shape, S&& strides,
                                                                 std::size_t offset, layout_type layout,
                                                                 FLS&& flatten_strides, layout_type flatten_layout) noexcept
        : m_e(std::forward<CTA>(e)),
          m_storage(detail::get_flat_storage<CT>(m_e, std::forward<FLS>(flatten_strides), flatten_layout)),
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
     * @name Size and shape
     */
    //@{
    /**
     * Returns the size of the xtrided_view_base.
     */
    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    /**
     * Returns the number of dimensions of the xtrided_view_base.
     */
    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    /**
     * Returns the shape of the xtrided_view_base.
     */
    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    /**
     * Returns the strides of the xtrided_view_base.
     */
    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::strides() const noexcept -> const strides_type&
    {
        return m_strides;
    }

    /**
     * Returns the backstrides of the xtrided_view_base.
     */
    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::backstrides() const noexcept -> const backstrides_type&
    {
        return m_backstrides;
    }

    /**
     * Returns the layout of the xtrided_view_base.
     */
    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::layout() const noexcept -> layout_type
    {
        return m_layout;
    }
    //@}

    /**
     * @name Data
     */
    //@{

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::operator()() -> reference
    {
        return m_storage[m_offset];
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::operator()() const -> const_reference
    {
        return m_storage[m_offset];
    }

    /**
     * Returns a reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the view.
     */
    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xstrided_view_base<CT, S, L, FST>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        using offset_type = typename strides_type::value_type;
        offset_type index = static_cast<offset_type>(m_offset) + xt::data_offset<offset_type>(strides(), static_cast<offset_type>(args)...);
        return m_storage[static_cast<size_type>(index)];
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the view.
     */
    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xstrided_view_base<CT, S, L, FST>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        using offset_type = typename strides_type::value_type;
        offset_type index = static_cast<offset_type>(m_offset) + xt::data_offset<offset_type>(strides(), static_cast<offset_type>(args)...);
        return m_storage[static_cast<size_type>(index)];
    }

    /**
     * Returns a reference to the element at the specified position in the view,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the view.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xstrided_view_base<CT, S, L, FST>::at(Args... args) -> reference
    {
        check_access(shape(), static_cast<size_type>(args)...);
        return this->operator()(args...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the view,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the view.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xstrided_view_base<CT, S, L, FST>::at(Args... args) const -> const_reference
    {
        check_access(shape(), static_cast<size_type>(args)...);
        return this->operator()(args...);
    }

    /**
     * Returns a reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the view, else the behavior is undefined.
     *
     * @warning This method is meant for performance, for expressions with a dynamic
     * number of dimensions (i.e. not known at compile time). Since it may have
     * undefined behavior (see parameters), operator() should be prefered whenever
     * it is possible.
     * @warning This method is NOT compatible with broadcasting, meaning the following
     * code has undefined behavior:
     * \code{.cpp}
     * xt::xarray<double> a = {{0, 1}, {2, 3}};
     * xt::xarray<double> b = {0, 1};
     * auto fd = a + b;
     * double res = fd.uncheked(0, 1);
     * \endcode
     */
    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xstrided_view_base<CT, S, L, FST>::unchecked(Args... args) -> reference
    {
        using offset_type = typename strides_type::value_type;
        offset_type index = static_cast<offset_type>(m_offset) + xt::unchecked_data_offset<offset_type>(strides(), static_cast<offset_type>(args)...);
        return m_storage[static_cast<size_type>(index)];
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the view, else the behavior is undefined.
     *
     * @warning This method is meant for performance, for expressions with a dynamic
     * number of dimensions (i.e. not known at compile time). Since it may have
     * undefined behavior (see parameters), operator() should be prefered whenever
     * it is possible.
     * @warning This method is NOT compatible with broadcasting, meaning the following
     * code has undefined behavior:
     * \code{.cpp}
     * xt::xarray<double> a = {{0, 1}, {2, 3}};
     * xt::xarray<double> b = {0, 1};
     * auto fd = a + b;
     * double res = fd.uncheked(0, 1);
     * \endcode
     */
    template <class CT, class S, layout_type L, class FST>
    template <class... Args>
    inline auto xstrided_view_base<CT, S, L, FST>::unchecked(Args... args) const -> const_reference
    {
        size_type index = m_offset + xt::unchecked_data_offset<size_type>(strides(), static_cast<size_type>(args)...);
        return m_storage[index];
    }

    /**
     * Returns a reference to the element at the specified position in the view.
     * @param index a sequence of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the view.
     */
    template <class CT, class S, layout_type L, class FST>
    template <class OS>
    inline auto xstrided_view_base<CT, S, L, FST>::operator[](const OS& index)
        -> disable_integral_t<OS, reference>
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT, class S, layout_type L, class FST>
    template <class I>
    inline auto xstrided_view_base<CT, S, L, FST>::operator[](std::initializer_list<I> index)
        -> reference
    {
        return element(index.begin(), index.end());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::operator[](size_type i) -> reference
    {
        return operator()(i);
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param index a sequence of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the view.
     */
    template <class CT, class S, layout_type L, class FST>
    template <class OS>
    inline auto xstrided_view_base<CT, S, L, FST>::operator[](const OS& index) const
        -> disable_integral_t<OS, const_reference>
    {
        return element(index.cbegin(), index.cend());
    }

    template <class CT, class S, layout_type L, class FST>
    template <class I>
    inline auto xstrided_view_base<CT, S, L, FST>::operator[](std::initializer_list<I> index) const
        -> const_reference
    {
        return element(index.begin(), index.end());
    }

    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a reference to the element at the specified position in the view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater than the the number
     * of dimensions of the view..
     */
    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline auto xstrided_view_base<CT, S, L, FST>::element(It first, It last) -> reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return m_storage[m_offset + element_offset<size_type>(strides(), first, last)];
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater than the the number
     * of dimensions of the view..
     */
    template <class CT, class S, layout_type L, class FST>
    template <class It>
    inline auto xstrided_view_base<CT, S, L, FST>::element(It first, It last) const -> const_reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return m_storage[m_offset + element_offset<size_type>(strides(), first, last)];
    }

    /**
     * Returns a reference to the buffer containing the elements of the view.
     */
    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::storage() noexcept -> storage_type&
    {
        return m_storage;
    }

    /**
     * Returns a constant reference to the buffer containing the elements of the view.
     */
    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::storage() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    /**
     * Returns a pointer to the underlying array serving as element storage.
     * The first element of the view is at data() + data_offset().
     */
    template <class CT, class S, layout_type L, class FST>
    template <class E>
    inline auto xstrided_view_base<CT, S, L, FST>::data() noexcept ->
        std::enable_if_t<has_data_interface<std::decay_t<E>>::value, value_type*>
    {
        return m_e.data();
    }

    /**
     * Returns a constant pointer to the underlying array serving as element storage.
     * The first element of the view is at data() + data_offset().
     */
    template <class CT, class S, layout_type L, class FST>
    template <class E>
    inline auto xstrided_view_base<CT, S, L, FST>::data() const noexcept ->
        std::enable_if_t<has_data_interface<std::decay_t<E>>::value, const value_type*>
    {
        return m_e.data();
    }

    /**
     * Returns the offset to the first element in the view.
     */
    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::data_offset() const noexcept -> size_type
    {
        return m_offset;
    }

    /**
     * Returns a reference to the underlying reference of the view.
     */
    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::expression() noexcept -> xexpression_type&
    {
        return m_e;
    }

    /**
     * Returns a constant reference to the underlying reference of the view.
     */
    template <class CT, class S, layout_type L, class FST>
    inline auto xstrided_view_base<CT, S, L, FST>::expression() const noexcept -> const xexpression_type&
    {
        return m_e;
    }
    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the view to the specified parameter.
     * @param shape the result shape
     * @param reuse_cache parameter for internal optimization
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class S, layout_type L, class FST>
    template <class O>
    inline bool xstrided_view_base<CT, S, L, FST>::broadcast_shape(O& shape, bool) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the view to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class S, layout_type L, class FST>
    template <class O>
    inline bool xstrided_view_base<CT, S, L, FST>::is_trivial_broadcast(const O& str) const noexcept
    {
        return str.size() == strides().size() &&
            std::equal(str.cbegin(), str.cend(), strides().begin());
    }
    //@}

    /******************************************
     * flat_expression_adaptor implementation *
     ******************************************/

        namespace detail
    {
        template <class CT>
        inline flat_expression_adaptor<CT>::flat_expression_adaptor(CT& e)
            : m_e(e)
        {
            resize_container(m_index, m_e.dimension());
            resize_container(m_strides, m_e.dimension());
            m_size = compute_size(m_e.shape());
            // Fallback to XTENSOR_DEFAULT_LAYOUT when the underlying layout is not
            // row-major or column major.
            m_layout = default_assignable_layout(m_e.layout());
            compute_strides(m_e.shape(), m_layout, m_strides);
        }

        template <class CT>
        template <class FST>
        inline flat_expression_adaptor<CT>::flat_expression_adaptor(CT& e, FST&& strides, layout_type layout)
            : m_e(e), m_strides(xtl::forward_sequence<shape_type>(strides)), m_layout(layout)
        {
            resize_container(m_index, m_e.dimension());
            m_size = e.size();
        }

        template <class CT>
        inline auto flat_expression_adaptor<CT>::size() const -> size_type
        {
            return m_size;
        }

        template <class CT>
        inline auto flat_expression_adaptor<CT>::operator[](std::size_t idx) -> reference
        {
            m_index = detail::unravel_noexcept(idx, m_strides, m_layout);
            return m_e.element(m_index.cbegin(), m_index.cend());
        }

        template <class CT>
        inline auto flat_expression_adaptor<CT>::operator[](std::size_t idx) const -> const_reference
        {
            m_index = detail::unravel_noexcept(idx, m_strides, m_layout);
            return m_e.element(m_index.cbegin(), m_index.cend());
        }

        template <class CT>
        inline auto flat_expression_adaptor<CT>::begin() -> iterator
        {
            return m_e.begin();
        }

        template <class CT>
        inline auto flat_expression_adaptor<CT>::end() -> iterator
        {
            return m_e.end();
        }

        template <class CT>
        inline auto flat_expression_adaptor<CT>::begin() const -> const_iterator
        {
            return m_e.cbegin();
        }

        template <class CT>
        inline auto flat_expression_adaptor<CT>::end() const -> const_iterator
        {
            return m_e.cend();
        }

        template <class CT>
        inline auto flat_expression_adaptor<CT>::cbegin() const -> const_iterator
        {
            return m_e.cbegin();
        }

        template <class CT>
        inline auto flat_expression_adaptor<CT>::cend() const ->const_iterator
        {
            return m_e.cend();
        }
    }
}

#endif
