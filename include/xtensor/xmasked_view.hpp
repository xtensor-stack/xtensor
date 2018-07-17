/***************************************************************************
* Copyright (c) 2017, Johan Mabille, Sylvain Corlay, Wolf Vollprecht and   *
* Martin Renou                                                             *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_XMASKED_VIEW_HPP
#define XTENSOR_XMASKED_VIEW_HPP

#include "xtensor/xoptional_assembly_base.hpp"
#include "xtensor/xoptional_assembly_storage.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xiterable.hpp"

namespace xt
{
    /****************************
    * xmasked_view declaration  *
    *****************************/

    template <class CTD, class CTM>
    class xmasked_view;

    template <class CTD, class CTM>
    struct xiterable_inner_types<xmasked_view<CTD, CTM>>
    {
        using assembly_type = xmasked_view<CTD, CTM>;
        using inner_shape_type = typename std::decay_t<CTD>::inner_shape_type;
        using stepper = xoptional_assembly_stepper<assembly_type, false>;
        using const_stepper = xoptional_assembly_stepper<assembly_type, true>;
    };

    /**
     * @class xmasked_view
     * @brief View on an xoptional_assembly or xoptional_assembly_adaptor
     * hiding values depending on a given mask.
     *
     * The xmasked_view class implements a view on an xoptional_assembly or
     * xoptional_assembly_adaptor, it takes this xoptional_assembly and a
     * mask as input. The mask is an xexpression containing boolean values,
     * whenever the value of the mask is false, the optional value of
     * xmasked_view is considered missing, otherwise it depends on the
     * underlying xoptional_assembly.
     *
     * @tparam CTD The type of expression holding the values.
     * @tparam CTM The type of expression holding the mask.
     */
    template <class CTD, class CTM>
    class xmasked_view : public xexpression<xmasked_view<CTD, CTM>>, // Will be replaced by xview_semantic
                         private xiterable<xmasked_view<CTD, CTM>>
    {
    public:

        using self_type = xmasked_view<CTD, CTM>;
        using assembly_type = self_type;
        using data_closure_type = CTD;
        using mask_closure_type = CTM;

        static constexpr bool is_data_const = std::is_const<std::remove_reference_t<data_closure_type>>::value;

        using data_type = std::decay_t<data_closure_type>;
        using value_expression = typename data_type::value_expression;
        using flag_expression = typename data_type::flag_expression;

        using value_type = typename data_type::value_type;
        using inner_value_type = typename value_type::value_type;
        using reference = std::conditional_t<is_data_const,
                                             typename data_type::const_reference,
                                             typename data_type::reference>;
        using const_reference = typename data_type::const_reference;
        using pointer = std::conditional_t<is_data_const,
                                           typename data_type::const_pointer,
                                           typename data_type::pointer>;
        using const_pointer = typename data_type::const_pointer;
        using size_type = typename data_type::size_type;
        using difference_type = typename data_type::difference_type;

        using shape_type = typename data_type::shape_type;
        using strides_type = typename data_type::strides_type;
        using storage_type = typename data_type::storage_type;

        static constexpr layout_type static_layout = value_expression::static_layout;

        using inner_shape_type = typename value_expression::inner_shape_type;
        using inner_strides_type = typename value_expression::inner_strides_type;
        using inner_backstrides_type = typename value_expression::inner_backstrides_type;

        using expression_tag = xoptional_expression_tag;

        using iterable_base = xiterable<xmasked_view<CTD, CTM>>;
        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        template <layout_type L>
        using layout_iterator = typename iterable_base::template layout_iterator<L>;
        template <layout_type L>
        using const_layout_iterator = typename iterable_base::template const_layout_iterator<L>;
        template <layout_type L>
        using reverse_layout_iterator = typename iterable_base::template reverse_layout_iterator<L>;
        template <layout_type L>
        using const_reverse_layout_iterator = typename iterable_base::template const_reverse_layout_iterator<L>;

        template <class S, layout_type L>
        using broadcast_iterator = typename iterable_base::template broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_broadcast_iterator = typename iterable_base::template const_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using reverse_broadcast_iterator = typename iterable_base::template reverse_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_reverse_broadcast_iterator = typename iterable_base::template const_reverse_broadcast_iterator<S, L>;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;
        using reverse_iterator = typename iterable_base::reverse_iterator;
        using const_reverse_iterator = typename iterable_base::const_reverse_iterator;

        template <class D, class M>
        xmasked_view(D&& data, M&& mask);

        size_type size() const noexcept;
        constexpr size_type dimension() const noexcept;

        const inner_shape_type& shape() const noexcept;
        const inner_strides_type& strides() const noexcept;
        const inner_backstrides_type& backstrides() const noexcept;

        layout_type layout() const noexcept;

        template <class T>
        void fill(const T& value);

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

        template <class S>
        disable_integral_t<S, reference> operator[](const S& index);
        template <class I>
        reference operator[](std::initializer_list<I> index);
        reference operator[](size_type i);

        template <class S>
        disable_integral_t<S, const_reference> operator[](const S& index) const;
        template <class I>
        const_reference operator[](std::initializer_list<I> index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        storage_type& storage() noexcept;
        const storage_type& storage() const noexcept;

        value_expression& value() noexcept;
        const value_expression& value() const noexcept;

        flag_expression& has_value() noexcept;
        const flag_expression& has_value() const noexcept;

        value_type* data() noexcept;
        const value_type* data() const noexcept;
        const size_type data_offset() const noexcept;

        using iterable_base::begin;
        using iterable_base::end;
        using iterable_base::cbegin;
        using iterable_base::cend;
        using iterable_base::rbegin;
        using iterable_base::rend;
        using iterable_base::crbegin;
        using iterable_base::crend;

        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape, layout_type l) noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

    private:

        template <class T, class... Args>
        T call_operator_impl(T&& missing_val, Args... args) const;

        template <class T, class... Args>
        T unchecked_impl(T&& missing_val, Args... args) const;

        template <class T, class S>
        disable_integral_t<S, T> access_operator_impl(T&& missing_val, const S& index) const;

        template <class T, class I>
        T access_operator_impl(T&& missing_val, std::initializer_list<I> index) const;

        template <class T>
        T access_operator_impl(T&& missing_val, size_type i) const;

        template <class T, class It>
        T element_impl(T&& missing_val, It first, It last) const;

        const_reference missing() const;
        reference missing();

        void evaluate_cache() const;

        CTD m_data;
        CTM m_mask;
        mutable flag_expression m_flag_container_cache;
        mutable storage_type m_storage_cache;
        mutable bool m_data_cached;
        thread_local static value_type m_missing_ref;

        friend class xiterable<xmasked_view<CTD, CTM>>;
        friend class xconst_iterable<xmasked_view<CTD, CTM>>;
    };

    /*******************************
     * xmasked_view implementation *
     *******************************/

    template <class CTD, class CTM>
    thread_local typename xmasked_view<CTD, CTM>::value_type xmasked_view<CTD, CTM>::m_missing_ref = value_type(inner_value_type(0), false);

    /**
     * @name Constructors
     */
    //@{
    /**
     * Creates an xmasked_view, given the xoptional_assembly or
     * xoptional_assembly_adaptor and the mask
     *
     * @param data the underlying xoptional_assembly or xoptional_assembly_adaptor
     * @param mask the mask.
     */
    template <class CTD, class CTM>
    template <class D, class M>
    inline xmasked_view<CTD, CTM>::xmasked_view(D&& data, M&& mask)
        : m_data(std::forward<D>(data)),
          m_mask(std::forward<M>(mask)),
          m_flag_container_cache(m_data.has_value().shape(), m_data.has_value().layout()),
          m_storage_cache(m_data.storage().value(), m_flag_container_cache.storage()),
          m_data_cached(false)
    {
    }

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of dimensions of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline constexpr auto xmasked_view<CTD, CTM>::dimension() const noexcept -> size_type
    {
        return m_data.dimension();
    }

    /**
     * Returns the number of elements in the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::size() const noexcept -> size_type
    {
        return m_data.size();
    }

    /**
     * Returns the shape of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::shape() const noexcept -> const inner_shape_type&
    {
        return m_data.shape();
    }

    /**
     * Returns the strides of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::strides() const noexcept -> const inner_strides_type&
    {
        return m_data.strides();
    }

    /**
     * Returns the backstrides of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::backstrides() const noexcept -> const inner_backstrides_type&
    {
        return m_data.backstrides();
    }
    //@}

    /**
     * Return the layout_type of the xmasked_view
     * @return layout_type of the xmasked_view
     */
    template <class CTD, class CTM>
    inline layout_type xmasked_view<CTD, CTM>::layout() const noexcept
    {
        return m_data.layout();
    }

    /**
     * Fills the data with the given value.
     * @param value the value to fill the data with.
     */
    template <class CTD, class CTM>
    template <class T>
    inline void xmasked_view<CTD, CTM>::fill(const T& value)
    {
        m_data.fill(value);
        m_data_cached = false;
    }

    /**
     * @name Data
     */
    //@{
    /**
     * Returns a reference to the element at the specified position in the xmasked_view.
     * @param args a list of indices specifying the position in the xmasked_view. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the xmasked_view.
     */
    template <class CTD, class CTM>
    template <class... Args>
    inline auto xmasked_view<CTD, CTM>::operator()(Args... args) -> reference
    {
        return call_operator_impl(missing(), args...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the xmasked_view.
     * @param args a list of indices specifying the position in the xmasked_view. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the xmasked_view.
     */
    template <class CTD, class CTM>
    template <class... Args>
    inline auto xmasked_view<CTD, CTM>::operator()(Args... args) const -> const_reference
    {
        return call_operator_impl(missing(), args...);
    }

    /**
     * Returns a reference to the element at the specified position in the xmasked_view,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the xmasked_view. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the xmasked_view.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class CTD, class CTM>
    template <class... Args>
    inline auto xmasked_view<CTD, CTM>::at(Args... args) -> reference
    {
        check_access(shape(), static_cast<size_type>(args)...);
        return call_operator_impl(missing(), args...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the xmasked_view,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the xmasked_view. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the xmasked_view.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class CTD, class CTM>
    template <class... Args>
    inline auto xmasked_view<CTD, CTM>::at(Args... args) const -> const_reference
    {
        check_access(shape(), static_cast<size_type>(args)...);
        return call_operator_impl(missing(), args...);
    }

    /**
     * Returns a reference to the element at the specified position in the  xmasked_view.
     * @param args a list of indices specifying the position in the  xmasked_view. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the  xmasked_view, else the behavior is undefined.
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
    template <class CTD, class CTM>
    template <class... Args>
    inline auto xmasked_view<CTD, CTM>::unchecked(Args... args) -> reference
    {
        return unchecked_impl(missing(), args...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the xmasked_view.
     * @param args a list of indices specifying the position in the  xmasked_view. Indices
     * must be unsigned integers, the number of indices must be equal to the number of
     * dimensions of the  xmasked_view, else the behavior is undefined.
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
    template <class CTD, class CTM>
    template <class... Args>
    inline auto xmasked_view<CTD, CTM>::unchecked(Args... args) const -> const_reference
    {
        return unchecked_impl(missing(), args...);
    }

    /**
     * Returns a reference to the element at the specified position in the xmasked_view.
     * @param index a sequence of indices specifying the position in the xmasked_view. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the xmasked_view.
     */
    template <class CTD, class CTM>
    template <class S>
    inline auto xmasked_view<CTD, CTM>::operator[](const S& index) -> disable_integral_t<S, reference>
    {
        return access_operator_impl(missing(), index);
    }

    template <class CTD, class CTM>
    template <class I>
    inline auto xmasked_view<CTD, CTM>::operator[](std::initializer_list<I> index) -> reference
    {
        return access_operator_impl(missing(), index);
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::operator[](size_type i) -> reference
    {
        return access_operator_impl(missing(), i);
    }

    /**
     * Returns a constant reference to the element at the specified position in the xmasked_view.
     * @param index a sequence of indices specifying the position in the xmasked_view. Indices
     * must be unsigned integers, the number of indices in the list should be equal or greater
     * than the number of dimensions of the xmasked_view.
     */
    template <class CTD, class CTM>
    template <class S>
    inline auto xmasked_view<CTD, CTM>::operator[](const S& index) const -> disable_integral_t<S, const_reference>
    {
        return access_operator_impl(missing(), index);
    }

    template <class CTD, class CTM>
    template <class I>
    inline auto xmasked_view<CTD, CTM>::operator[](std::initializer_list<I> index) const -> const_reference
    {
        return access_operator_impl(missing(), index);
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::operator[](size_type i) const -> const_reference
    {
        return access_operator_impl(missing(), i);
    }

    /**
     * Returns a reference to the element at the specified position in the xmasked_view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the xmasked_view.
     */
    template <class CTD, class CTM>
    template <class It>
    inline auto xmasked_view<CTD, CTM>::element(It first, It last) -> reference
    {
        return element_impl(missing(), first, last);
    }

    /**
     * Returns a constant reference to the element at the specified position in the xmasked_view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the xmasked_view.
     */
    template <class CTD, class CTM>
    template <class It>
    inline auto xmasked_view<CTD, CTM>::element(It first, It last) const -> const_reference
    {
        return element_impl(missing(), first, last);
    }
    //@}

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::storage() noexcept -> storage_type&
    {
        evaluate_cache();
        return m_storage_cache;
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::storage() const noexcept -> const storage_type&
    {
        evaluate_cache();
        return m_storage_cache;
    }

    /**
     * Return an expression for the values of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::value() noexcept -> value_expression&
    {
        return m_data.value();
    }

    /**
     * Return a constant expression for the values of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::value() const noexcept -> const value_expression&
    {
        return m_data.value();
    }

    /**
     * Return an expression for the missing mask of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::has_value() noexcept -> flag_expression&
    {
        evaluate_cache();
        return m_flag_container_cache;
    }

    /**
     * Return a constant expression for the missing mask of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::has_value() const noexcept -> const flag_expression&
    {
        evaluate_cache();
        return m_flag_container_cache;
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::data() noexcept -> value_type*
    {
        evaluate_cache();
        return m_storage_cache.data();
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::data() const noexcept -> const value_type*
    {
        evaluate_cache();
        return m_storage_cache.data();
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::data_offset() const noexcept -> const size_type
    {
        return size_type(0);
    }

    template <class CTD, class CTM>
    template <class S>
    inline auto xmasked_view<CTD, CTM>::stepper_begin(const S& shape) noexcept -> stepper
    {
        return stepper(value().stepper_begin(shape), has_value().stepper_begin(shape));
    }

    template <class CTD, class CTM>
    template <class S>
    inline auto xmasked_view<CTD, CTM>::stepper_end(const S& shape, layout_type l) noexcept -> stepper
    {
        return stepper(value().stepper_end(shape, l), has_value().stepper_end(shape, l));
    }

    template <class CTD, class CTM>
    template <class S>
    inline auto xmasked_view<CTD, CTM>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        return const_stepper(value().stepper_begin(shape), has_value().stepper_begin(shape));
    }

    template <class CTD, class CTM>
    template <class S>
    inline auto xmasked_view<CTD, CTM>::stepper_end(const S& shape, layout_type l) const noexcept -> const_stepper
    {
        return const_stepper(value().stepper_end(shape, l), has_value().stepper_end(shape, l));
    }

    template <class CTD, class CTM>
    template <class T, class... Args>
    inline auto xmasked_view<CTD, CTM>::call_operator_impl(T&& missing_val, Args... args) const -> T
    {
        if (!m_mask(args...))
        {
            return missing_val;
        }

        return m_data(args...);
    }

    template <class CTD, class CTM>
    template <class T, class... Args>
    inline auto xmasked_view<CTD, CTM>::unchecked_impl(T&& missing_val, Args... args) const -> T
    {
        if (!m_mask(args...))
        {
            return missing_val;
        }

        return m_data.unchecked(args...);
    }

    template <class CTD, class CTM>
    template <class T, class S>
    inline auto xmasked_view<CTD, CTM>::access_operator_impl(T&& missing_val, const S& index) const -> disable_integral_t<S, T>
    {
        if (!m_mask[index])
        {
            return missing_val;
        }

        return m_data[index];
    }

    template <class CTD, class CTM>
    template <class T, class I>
    inline auto xmasked_view<CTD, CTM>::access_operator_impl(T&& missing_val, std::initializer_list<I> index) const -> T
    {
        if (!m_mask[index])
        {
            return missing_val;
        }

        return m_data[index];
    }

    template <class CTD, class CTM>
    template <class T>
    inline auto xmasked_view<CTD, CTM>::access_operator_impl(T&& missing_val, size_type i) const -> T
    {
        if (!m_mask[i])
        {
            return missing_val;
        }

        return m_data[i];
    }

    template <class CTD, class CTM>
    template <class T, class It>
    inline auto xmasked_view<CTD, CTM>::element_impl(T&& missing_val, It first, It last) const -> T
    {
        if (!m_mask.element(first, last))
        {
            return missing_val;
        }

        return m_data.element(first, last);
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::missing() const -> const_reference
    {
        return const_reference(inner_value_type(0), false);
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::missing() -> reference
    {
        m_missing_ref.has_value() = false;
        return reference(m_missing_ref.value(), m_missing_ref.has_value());
    }

    template <class CTD, class CTM>
    inline void xmasked_view<CTD, CTM>::evaluate_cache() const
    {
        if (m_data_cached)
        {
            return;
        }

        m_flag_container_cache = m_data.has_value() && m_mask;

        m_data_cached = true;
    }

    template <class CTD, class CTM>
    inline xmasked_view<CTD, CTM> masked_view(CTD&& data, CTM&& mask)
    {
        return xmasked_view<CTD, CTM>(std::forward<CTD>(data), std::forward<CTM>(mask));
    }
}

#endif
