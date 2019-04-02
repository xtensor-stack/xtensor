/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_STRIDED_VIEW_BASE_HPP
#define XTENSOR_STRIDED_VIEW_BASE_HPP

#include <type_traits>

#include <xtl/xsequence.hpp>

#include "xaccessible.hpp"
#include "xtensor_forward.hpp"
#include "xslice.hpp"
#include "xstrides.hpp"
#include "xutils.hpp"

namespace xt
{
    template <class D>
    class xstrided_view_base : public xaccessible<D>
    {
    public:

        using base_type = xaccessible<D>;
        using inner_types = xcontainer_inner_types<D>;
        using xexpression_type = typename inner_types::xexpression_type;
        using inner_closure_type = typename inner_types::inner_closure_type;
        static constexpr bool is_const = std::is_const<std::remove_reference_t<inner_closure_type>>::value;

        using value_type = typename xexpression_type::value_type;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using pointer = std::conditional_t<is_const,
                                           typename xexpression_type::const_pointer,
                                           typename xexpression_type::pointer>;
        using const_pointer = typename xexpression_type::const_pointer;
        using size_type = typename inner_types::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using inner_storage_type = typename inner_types::inner_storage_type;
        using storage_type = std::remove_reference_t<inner_storage_type>;

        using shape_type = typename inner_types::shape_type;
        using strides_type = get_strides_t<shape_type>;
        using backstrides_type = strides_type;

        using inner_shape_type = shape_type;
        using inner_strides_type = strides_type;
        using inner_backstrides_type = backstrides_type;

        using undecay_shape = typename inner_types::undecay_shape;

        static constexpr layout_type static_layout = inner_types::layout;
        static constexpr bool contiguous_layout = static_layout != layout_type::dynamic && xexpression_type::contiguous_layout;

        template <class CTA>
        xstrided_view_base(CTA&& e, undecay_shape&& shape, strides_type&& strides, size_type offset, layout_type layout) noexcept;

        xstrided_view_base(xstrided_view_base&& rhs);

        xstrided_view_base(const xstrided_view_base& rhs);

        const inner_shape_type& shape() const noexcept;
        const inner_strides_type& strides() const noexcept;
        const inner_backstrides_type& backstrides() const noexcept;
        layout_type layout() const noexcept;

        reference operator()();
        const_reference operator()() const;

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class... Args>
        reference unchecked(Args... args);

        template <class... Args>
        const_reference unchecked(Args... args) const;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        storage_type& storage() noexcept;
        const storage_type& storage() const noexcept;

        template <class E = xexpression_type>
        std::enable_if_t<has_data_interface<std::decay_t<E>>::value, value_type*>
        data() noexcept;
        template <class E = xexpression_type>
        std::enable_if_t<has_data_interface<std::decay_t<E>>::value, const value_type*>
        data() const noexcept;
        size_type data_offset() const noexcept;

        xexpression_type& expression() noexcept;
        const xexpression_type& expression() const noexcept;

        template <class O>
        bool broadcast_shape(O& shape, bool reuse_cache = false) const;

        template <class O>
        bool has_linear_assign(const O& strides) const noexcept;

    protected:

        using offset_type = typename strides_type::value_type;

        template <class... Args>
        offset_type compute_index(Args... args) const;

        template <class... Args>
        offset_type compute_unchecked_index(Args... args) const;

        template <class It>
        offset_type compute_element_index(It first, It last) const;

    private:

        using flat_storage = typename inner_types::flat_storage;

        inner_closure_type m_e;
        inner_storage_type m_storage;
        inner_shape_type m_shape;
        inner_strides_type m_strides;
        inner_backstrides_type m_backstrides;
        size_type m_offset;
        layout_type m_layout;
    };

    /****************
     * flat_storage *
     ****************/

    template <class CT, class S, layout_type L, class FST>
    class xstrided_view;

    namespace detail
    {
        template <class I, class CI>
        class flat_expression_adaptor
        {
        public:

            using value_type = typename std::iterator_traits<I>::value_type;
            using reference = typename std::iterator_traits<I>::reference;
            using const_reference = typename std::iterator_traits<CI>::reference;
            using size_type = std::size_t;
            using difference_type = typename std::iterator_traits<I>::difference_type;
            using iterator = I;
            using const_iterator = CI;

            inline flat_expression_adaptor(I it, CI cit, size_type size)
                : m_it(it), m_cit(cit), m_size(size)
            {
            }

            inline size_type size() const
            {
                return m_size;
            }

            inline reference operator[](size_type i)
            {
                return m_it[static_cast<difference_type>(i)];
            }

            inline const_reference operator[](size_type i) const
            {
                return m_cit[static_cast<difference_type>(i)];
            }

            inline iterator begin() { return m_it; }
            inline iterator end() { return m_it + static_cast<difference_type>(m_size); }

            inline const_iterator begin() const { return cbegin(); }
            inline const_iterator end() const { return cend(); }

            inline const_iterator cbegin() const { return m_cit; }
            inline const_iterator cend() const { return m_cit + static_cast<difference_type>(m_size); }

        private:

            I m_it;
            CI m_cit;
            size_type m_size;
        };

        template <class E, layout_type L>
        struct flat_storage_base
        {
            static const layout_type layout = L;
            using iterator = decltype(std::declval<E>().template begin<L>());
            using const_iterator = decltype(std::declval<E>().template cbegin<L>());
            using value_type = typename std::iterator_traits<iterator>::value_type;
            //using type = xbuffer_adaptor<iterator, no_ownership, std::allocator<value_type>>;
            using type = flat_expression_adaptor<iterator, const_iterator>;
        };

        
        template <class E>
        struct flat_storage_base<E, layout_type::dynamic>
            : flat_storage_base<E, default_assignable_layout(std::decay_t<E>::static_layout)>
        {
        };

        template <class E>
        struct flat_storage_base<E, layout_type::any>
            : flat_storage_base<E, layout_type::dynamic>
        {
        };

        template <class E, class = void>
        struct has_inner_closure_type : std::false_type
        {
        };

        template <class E>
        struct has_inner_closure_type<E, void_t<typename E::inner_closure_type>>
            : std::true_type
        {
        };

        template <class E>
        struct has_recursive_flat_storage
            : xtl::conjunction<has_data_interface<E>, has_inner_closure_type<E>>
        {
        };

        template <class E>
        struct select_inner_closure_type
        {
            using closure_type = typename std::decay_t<E>::inner_closure_type;
            static constexpr bool is_const = std::is_const<std::remove_reference_t<E>>::value;
            using type = std::conditional_t<is_const, xtl::constify_t<closure_type>, closure_type>;
        };

        template <class E>
        using select_inner_closure_type_t = typename select_inner_closure_type<E>::type;

        template <class E, layout_type L = std::decay_t<E>::static_layout,
                  bool = has_recursive_flat_storage<std::decay_t<E>>::value>
        struct flat_storage : flat_storage_base<E, L>
        {
        };

        template <class E, layout_type L>
        struct flat_storage<E, L, true> : flat_storage<select_inner_closure_type_t<E>, L>
        {
        };

        template <class FST, class E, std::enable_if_t<!has_recursive_flat_storage<std::decay_t<E>>::value>* = nullptr>
        inline auto get_flat_storage(E& e) -> typename FST::type
        {
            using type = typename FST::type;
            return type(e.template begin<FST::layout>(), e.template cbegin<FST::layout>(), e.size());
        }

        template <class FST, class E, std::enable_if_t<has_recursive_flat_storage<std::decay_t<E>>::value>* = nullptr>
        inline auto get_flat_storage(E& e) -> typename FST::type
        {
            return get_flat_storage<FST>(e.expression());
        }

        // with data_interface
        template <class E, std::enable_if_t<has_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline auto get_offset(E&& e)
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
        inline auto get_offset(E&& /*e*/)
        {
            return typename std::decay_t<E>::size_type(0);
        }

        template <class E, std::enable_if_t<!has_data_interface<std::decay_t<E>>::value>* = nullptr>
        inline auto get_strides(E&& e)
        {
            dynamic_shape<std::ptrdiff_t> strides;
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
    template <class D>
    template <class CTA>
    inline xstrided_view_base<D>::xstrided_view_base(CTA&& e, undecay_shape&& shape, strides_type&& strides, size_type offset, layout_type layout) noexcept
        : m_e(std::forward<CTA>(e)),
          m_storage(detail::get_flat_storage<flat_storage>(m_e)),
          m_shape(std::move(shape)),
          m_strides(std::move(strides)),
          m_offset(offset),
          m_layout(layout)
    {
        m_backstrides = xtl::make_sequence<backstrides_type>(m_shape.size(), 0);
        adapt_strides(m_shape, m_strides, m_backstrides);
    }

    template <class D>
    inline xstrided_view_base<D>::xstrided_view_base(xstrided_view_base&& rhs)
        : base_type(std::move(rhs)),
          m_e(std::forward<inner_closure_type>(rhs.m_e)),
          m_storage(detail::get_flat_storage<flat_storage>(m_e)),
          m_shape(std::move(rhs.m_shape)),
          m_strides(std::move(rhs.m_strides)),
          m_backstrides(std::move(rhs.m_backstrides)),
          m_offset(std::move(rhs.m_offset)),
          m_layout(std::move(rhs.m_layout))
    {
    }

    template <class D>
    inline xstrided_view_base<D>::xstrided_view_base(const xstrided_view_base& rhs)
        : base_type(rhs),
          m_e(rhs.m_e),
          m_storage(detail::get_flat_storage<flat_storage>(m_e)),
          m_shape(rhs.m_shape),
          m_strides(rhs.m_strides),
          m_backstrides(rhs.m_backstrides),
          m_offset(rhs.m_offset),
          m_layout(rhs.m_layout)
    {
    }
    //@}

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the shape of the xtrided_view_base.
     */
    template <class D>
    inline auto xstrided_view_base<D>::shape() const noexcept -> const inner_shape_type&
    {
        return m_shape;
    }

    /**
     * Returns the strides of the xtrided_view_base.
     */
    template <class D>
    inline auto xstrided_view_base<D>::strides() const noexcept -> const inner_strides_type&
    {
        return m_strides;
    }

    /**
     * Returns the backstrides of the xtrided_view_base.
     */
    template <class D>
    inline auto xstrided_view_base<D>::backstrides() const noexcept -> const inner_backstrides_type&
    {
        return m_backstrides;
    }

    /**
     * Returns the layout of the xtrided_view_base.
     */
    template <class D>
    inline auto xstrided_view_base<D>::layout() const noexcept -> layout_type
    {
        return m_layout;
    }
    //@}

    /**
     * @name Data
     */
    //@{
    template <class D>
    inline auto xstrided_view_base<D>::operator()() -> reference
    {
        return m_storage[static_cast<size_type>(m_offset)];
    }

    template <class D>
    inline auto xstrided_view_base<D>::operator()() const -> const_reference
    {
        return m_storage[static_cast<size_type>(m_offset)];
    }

    /**
     * Returns a reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the view.
     */
    template <class D>
    template <class... Args>
    inline auto xstrided_view_base<D>::operator()(Args... args) -> reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        offset_type index = compute_index(args...);
        std::cout << "base::operator[]: " << index << std::endl;
        return m_storage[static_cast<size_type>(index)];
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the view.
     */
    template <class D>
    template <class... Args>
    inline auto xstrided_view_base<D>::operator()(Args... args) const -> const_reference
    {
        XTENSOR_TRY(check_index(shape(), args...));
        XTENSOR_CHECK_DIMENSION(shape(), args...);
        offset_type index = compute_index(args...);
        std::cout << "base::operator[]: " << index << std::endl;
        return m_storage[static_cast<size_type>(index)];
    }

    /**
     * Returns a reference to the element at the specified position in the view.
     * @param args a list of indices specifying the position in the view. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the view, else the behavior is undefined.
     *
     * @warning This method is meant for performance, for expressions with a dynamic
     * number of dimensions (i.e. not known at compile time). Since it may have
     * undefined behavior (see parameters), operator() should be preferred whenever
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
    template <class D>
    template <class... Args>
    inline auto xstrided_view_base<D>::unchecked(Args... args) -> reference
    {
        offset_type index = compute_unchecked_index(args...);
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
     * undefined behavior (see parameters), operator() should be preferred whenever
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
    template <class D>
    template <class... Args>
    inline auto xstrided_view_base<D>::unchecked(Args... args) const -> const_reference
    {
        offset_type index = compute_unchecked_index(args...);
        return m_storage[static_cast<size_type>(index)];
    }

    /**
     * Returns a reference to the element at the specified position in the view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater than the the number
     * of dimensions of the view..
     */
    template <class D>
    template <class It>
    inline auto xstrided_view_base<D>::element(It first, It last) -> reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return m_storage[static_cast<size_type>(compute_element_index(first, last))];
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater than the the number
     * of dimensions of the view..
     */
    template <class D>
    template <class It>
    inline auto xstrided_view_base<D>::element(It first, It last) const -> const_reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
        return m_storage[static_cast<size_type>(compute_element_index(first, last))];
    }

    /**
     * Returns a reference to the buffer containing the elements of the view.
     */
    template <class D>
    inline auto xstrided_view_base<D>::storage() noexcept -> storage_type&
    {
        return m_storage;
    }

    /**
     * Returns a constant reference to the buffer containing the elements of the view.
     */
    template <class D>
    inline auto xstrided_view_base<D>::storage() const noexcept -> const storage_type&
    {
        return m_storage;
    }

    /**
     * Returns a pointer to the underlying array serving as element storage.
     * The first element of the view is at data() + data_offset().
     */
    template <class D>
    template <class E>
    inline auto xstrided_view_base<D>::data() noexcept ->
        std::enable_if_t<has_data_interface<std::decay_t<E>>::value, value_type*>
    {
        return m_e.data();
    }

    /**
     * Returns a constant pointer to the underlying array serving as element storage.
     * The first element of the view is at data() + data_offset().
     */
    template <class D>
    template <class E>
    inline auto xstrided_view_base<D>::data() const noexcept ->
        std::enable_if_t<has_data_interface<std::decay_t<E>>::value, const value_type*>
    {
        return m_e.data();
    }

    /**
     * Returns the offset to the first element in the view.
     */
    template <class D>
    inline auto xstrided_view_base<D>::data_offset() const noexcept -> size_type
    {
        return m_offset;
    }

    /**
     * Returns a reference to the underlying expression of the view.
     */
    template <class D>
    inline auto xstrided_view_base<D>::expression() noexcept -> xexpression_type&
    {
        return m_e;
    }

    /**
     * Returns a constant reference to the underlying expression of the view.
     */
    template <class D>
    inline auto xstrided_view_base<D>::expression() const noexcept -> const xexpression_type&
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
    template <class D>
    template <class O>
    inline bool xstrided_view_base<D>::broadcast_shape(O& shape, bool) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Checks whether the xstrided_view_base can be linearly assigned to an expression
     * with the specified strides.
     * @return a boolean indicating whether a linear assign is possible
     */
    template <class D>
    template <class O>
    inline bool xstrided_view_base<D>::has_linear_assign(const O& str) const noexcept
    {
        return has_data_interface<xexpression_type>::value && str.size() == strides().size() &&
            std::equal(str.cbegin(), str.cend(), strides().begin());
    }
    //@}

    template <class D>
    template <class... Args>
    inline auto xstrided_view_base<D>::compute_index(Args... args) const -> offset_type
    {
        return static_cast<offset_type>(m_offset) + xt::data_offset<offset_type>(strides(), static_cast<offset_type>(args)...);
    }

    template <class D>
    template <class... Args>
    inline auto xstrided_view_base<D>::compute_unchecked_index(Args... args) const -> offset_type
    {
        return static_cast<offset_type>(m_offset) + xt::unchecked_data_offset<offset_type>(strides(), static_cast<offset_type>(args)...);
    }

    template <class D>
    template <class It>
    inline auto xstrided_view_base<D>::compute_element_index(It first, It last) const -> offset_type
    {
        return static_cast<offset_type>(m_offset) + xt::element_offset<offset_type>(strides(), first, last);
    }

    /**********************************
     * Builder helpers implementation *
     **********************************/

    namespace detail
    {
        template <class S>
        struct slice_getter_impl
        {
            const S& m_shape;
            mutable std::size_t idx;
            using array_type = std::array<std::ptrdiff_t, 3>;

            explicit slice_getter_impl(const S& shape)
                : m_shape(shape), idx(0)
            {
            }

            template <class T>
            array_type operator()(const T& /*t*/) const
            {
                return array_type{{0, 0, 0}};
            }

            template <class A, class B, class C>
            array_type operator()(const xrange_adaptor<A, B, C>& range) const
            {
                auto sl = range.get(static_cast<std::size_t>(m_shape[idx]));
                return array_type({sl(0), sl.size(), sl.step_size()});
            }

            template <class T>
            array_type operator()(const xrange<T>& range) const
            {
                return array_type({ range(T(0)), range.size(), T(1) });
            }

            template <class T>
            array_type operator()(const xstepped_range<T>& range) const
            {
                return array_type({ range(T(0)), range.size(), range.step_size(T(0)) });
            }
        };

        template <class adj_strides_policy>
        struct strided_view_args : adj_strides_policy
        {
            using base_type = adj_strides_policy;

            template <class S, class ST, class V>
            void fill_args(const S& shape, ST&& old_strides, std::size_t base_offset, layout_type layout, const V& slices)
            {
                // Compute dimension
                std::size_t dimension = shape.size(), n_newaxis = 0, n_add_all = 0;
                std::ptrdiff_t dimension_check = static_cast<std::ptrdiff_t>(shape.size());

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
                    n_add_all = shape.size() - (slices.size() - 1 - n_newaxis);
                }

                // Compute strided view
                new_offset = base_offset;
                new_shape.resize(dimension);
                new_strides.resize(dimension);
                base_type::resize(dimension);

                auto old_shape = shape;
                using old_strides_value_type = typename std::decay_t<ST>::value_type;

                std::ptrdiff_t axis_skip = 0;
                std::size_t idx = 0, i = 0, i_ax = 0;

                auto slice_getter = detail::slice_getter_impl<S>(shape);

                for (; i < slices.size(); ++i)
                {
                    i_ax = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(i) - axis_skip);
                    auto ptr = xtl::get_if<std::ptrdiff_t>(&slices[i]);
                    if (ptr != nullptr)
                    {
                        auto slice0 = static_cast<old_strides_value_type>(*ptr);
                        new_offset += static_cast<std::size_t>(slice0 * old_strides[i_ax]);
                    }
                    else if (xtl::get_if<xt::xnewaxis_tag>(&slices[i]) != nullptr)
                    {
                        new_shape[idx] = 1;
                        base_type::set_fake_slice(idx);
                        ++axis_skip, ++idx;
                    }
                    else if (xtl::get_if<xt::xellipsis_tag>(&slices[i]) != nullptr)
                    {
                        for (std::size_t j = 0; j < n_add_all; ++j)
                        {
                            new_shape[idx] = old_shape[i_ax];
                            new_strides[idx] = old_strides[i_ax];
                            base_type::set_fake_slice(idx);
                            ++idx, ++i_ax;
                        }
                        axis_skip = axis_skip - static_cast<std::ptrdiff_t>(n_add_all) + 1;
                    }
                    else if (xtl::get_if<xt::xall_tag>(&slices[i]) != nullptr)
                    {
                        new_shape[idx] = old_shape[i_ax];
                        new_strides[idx] = old_strides[i_ax];
                        base_type::set_fake_slice(idx);
                        ++idx;
                    }
                    else if (base_type::fill_args(slices, i, idx,
                                                  old_shape[i_ax],
                                                  old_strides[i_ax],
                                                  new_shape, new_strides))
                    {
                        ++idx;
                    }
                    else
                    {
                        slice_getter.idx = i_ax;
                        auto info = xtl::visit(slice_getter, slices[i]);
                        new_offset += static_cast<std::size_t>(info[0] * old_strides[i_ax]);
                        new_shape[idx] = static_cast<std::size_t>(info[1]);
                        new_strides[idx] = info[2] * old_strides[i_ax];
                        base_type::set_fake_slice(idx);
                        ++idx;
                    }
                }

                i_ax = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(i) - axis_skip);
                for (; i_ax < old_shape.size(); ++i_ax, ++idx)
                {
                    new_shape[idx] = old_shape[i_ax];
                    new_strides[idx] = old_strides[i_ax];
                    base_type::set_fake_slice(idx);
                }

                new_layout = do_strides_match(new_shape, new_strides, layout, true) ? layout : layout_type::dynamic;
            }

            using shape_type = dynamic_shape<std::size_t>;
            shape_type new_shape;
            using strides_type = dynamic_shape<std::ptrdiff_t>;
            strides_type new_strides;
            std::size_t new_offset;
            layout_type new_layout;
        };
    }
}

#endif
