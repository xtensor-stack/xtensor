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

#include "xtl/xoptional.hpp"

#include "xoptional_assembly_base.hpp"
#include "xoptional_assembly_storage.hpp"
#include "xexpression.hpp"
#include "xiterable.hpp"

namespace xt
{
    /****************************
    * xmasked_view declaration  *
    *****************************/

    template <class CTD, class CTM>
    class xmasked_view;

    template <class T, class IO = void>
    struct xcontainer_inner_types_impl;

    template <class CTD, class CTM>
    struct xcontainer_inner_types_impl<xmasked_view<CTD, CTM>, std::enable_if_t<is_xoptional_expression<std::decay_t<CTD>>::value>>
    {
        using temporary_type = CTD;
    };

    template <class CTD, class CTM>
    struct xcontainer_inner_types_impl<xmasked_view<CTD, CTM>, std::enable_if_t<!is_xoptional_expression<std::decay_t<CTD>>::value>>
    {
        using temporary_type = xoptional_assembly<CTD, CTM>;
    };

    template <class CTD, class CTM>
    struct xcontainer_inner_types<xmasked_view<CTD, CTM>>
    {
        using temporary_type = typename xcontainer_inner_types_impl<xmasked_view<CTD, CTM>>::temporary_type;
    };

    template <class CTD, class CTM>
    struct xiterable_inner_types<xmasked_view<CTD, CTM>>
    {
        using assembly_type = xmasked_view<CTD, CTM>;
        using inner_shape_type = typename std::decay_t<CTD>::inner_shape_type;
        using stepper = xoptional_assembly_stepper<assembly_type, false>;
        using const_stepper = xoptional_assembly_stepper<assembly_type, true>;
    };

    template <class T, class IO = void>
    struct xmasked_view_inner_types;

    template <class CTD, class CTM>
    struct xmasked_view_inner_types<xmasked_view<CTD, CTM>, std::enable_if_t<is_xoptional_expression<std::decay_t<CTD>>::value>>
    {
        using data_closure_type = CTD;
        using mask_closure_type = CTM;

        using data_type = std::decay_t<data_closure_type>;
        using value_expression = typename data_type::value_expression;
        using flag_expression = typename data_type::flag_expression;
        using storage_type = typename data_type::storage_type;
    };

    template <class CTD, class CTM>
    struct xmasked_view_inner_types<xmasked_view<CTD, CTM>, std::enable_if_t<!is_xoptional_expression<std::decay_t<CTD>>::value>>
    {
        using data_closure_type = CTD;
        using mask_closure_type = CTM;

        using value_expression = std::decay_t<data_closure_type>;
        using flag_expression = std::decay_t<mask_closure_type>;

        using data_storage_type = typename value_expression::storage_type&;
        using mask_storage_type = typename flag_expression::storage_type&;

        using storage_type = xoptional_assembly_storage<data_storage_type, mask_storage_type>;
    };

    namespace detail
    {
        template <class CTD, class CTM, class IO = void>
        struct xmasked_view_storage;

        template <class CTD, class CTM>
        struct xmasked_view_storage<CTD, CTM, std::enable_if_t<is_xoptional_expression<std::decay_t<CTD>>::value>>
        {
            using data_type = std::decay_t<CTD>;

            using flag_expression = typename data_type::flag_expression;
            using storage_type = typename data_type::storage_type;
            using size_type = typename data_type::size_type;

            flag_expression get_flag_container_cache(CTD data, CTM /*mask*/)
            {
                return flag_expression(data.has_value().shape(), data.has_value().layout());
            }

            storage_type get_storage_cache(CTD data, CTM /*mask*/, flag_expression& flag_container_cache)
            {
                return storage_type(data.storage().value(), flag_container_cache.storage());
            }

            template <class T, class D, class M, class... Args>
            inline T call_operator_impl(T&& missing_val, D&& data, M&& mask, Args... args)
            {
                if (!mask(args...))
                {
                    return missing_val;
                }

                return data(args...);
            }

            template <class T, class D, class M, class... Args>
            inline T unchecked_impl(T&& missing_val, D&& data, M&& mask, Args... args)
            {
                if (!mask.unchecked(args...))
                {
                    return missing_val;
                }

                return data.unchecked(args...);
            }

            template <class T, class D, class M, class S>
            inline disable_integral_t<S, T> access_operator_impl(T&& missing_val, D&& data, M&& mask, const S& index)
            {
                if (!mask[index])
                {
                    return missing_val;
                }

                return data[index];
            }

            template <class T, class D, class M, class I>
            inline T access_operator_impl(T&& missing_val, D&& data, M&& mask, std::initializer_list<I> index)
            {
                if (!mask[index])
                {
                    return missing_val;
                }

                return data[index];
            }

            template <class T, class D, class M>
            inline T access_operator_impl(T&& missing_val, D&& data, M&& mask, size_type i)
            {
                if (!mask[i])
                {
                    return missing_val;
                }

                return data[i];
            }

            template <class T, class D, class M, class It>
            inline T element_impl(T&& missing_val, D&& data, M&& mask, It first, It last)
            {
                if (!mask.element(first, last))
                {
                    return missing_val;
                }

                return data.element(first, last);
            }

            template <class D>
            inline auto value_impl(D&& data) -> decltype(data.value())
            {
                return data.value();
            }
        };

        template <class CTD, class CTM>
        struct xmasked_view_storage<CTD, CTM, std::enable_if_t<!is_xoptional_expression<std::decay_t<CTD>>::value>>
        {
            using data_type = std::decay_t<CTD>;

            using flag_expression = CTM;
            using storage_type = xoptional_assembly_storage<typename std::decay_t<CTD>::storage_type&, typename std::decay_t<CTM>::storage_type&>;
            using size_type = typename data_type::size_type;

            flag_expression get_flag_container_cache(CTD /*data*/, CTM mask)
            {
                return mask;
            }

            storage_type get_storage_cache(CTD data, CTM mask, flag_expression& /*flag_container_cache*/)
            {
                return storage_type(data.storage(), mask.storage());
            }

            template <class T, class D, class M, class... Args>
            inline T call_operator_impl(T&& /*missing_val*/, D&& data, M&& mask, Args... args)
            {
                return T(data(args...), mask(args...));
            }

            template <class T, class D, class M, class... Args>
            inline T unchecked_impl(T&& /*missing_val*/, D&& data, M&& mask, Args... args)
            {
                return T(data.unchecked(args...), mask.unchecked(args...));
            }

            template <class T, class D, class M, class S>
            inline disable_integral_t<S, T> access_operator_impl(T&& /*missing_val*/, D&& data, M&& mask, const S& index)
            {
                return disable_integral_t<S, T>(data[index], mask[index]);
            }

            template <class T, class D, class M, class I>
            inline T access_operator_impl(T&& /*missing_val*/, D&& data, M&& mask, std::initializer_list<I> index)
            {
                return T(data[index], mask[index]);
            }

            template <class T, class D, class M>
            inline T access_operator_impl(T&& /*missing_val*/, D&& data, M&& mask, size_type i)
            {
                return T(data[i], mask[i]);
            }

            template <class T, class D, class M, class It>
            inline T element_impl(T&& /*missing_val*/, D&& data, M&& mask, It first, It last)
            {
                return T(data.element(first, last), mask.element(first, last));
            }

            template <class D>
            inline auto value_impl(D&& data) -> decltype(data)
            {
                return data;
            }
        };
    }

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
    class xmasked_view : public xview_semantic<xmasked_view<CTD, CTM>>,
                         private xiterable<xmasked_view<CTD, CTM>>
    {
    public:

        using self_type = xmasked_view<CTD, CTM>;
        using semantic_base = xview_semantic<xmasked_view<CTD, CTM>>;
        using assembly_type = self_type;
        using data_closure_type = CTD;
        using mask_closure_type = CTM;

        static constexpr bool is_data_const = std::is_const<std::remove_reference_t<data_closure_type>>::value;

        using data_type = std::decay_t<data_closure_type>;
        using masked_view_inner_types = xmasked_view_inner_types<self_type>;

        using value_expression = typename masked_view_inner_types::value_expression;
        using flag_expression = typename masked_view_inner_types::flag_expression;
        using storage_type = typename masked_view_inner_types::storage_type;

        using value_type = typename storage_type::value_type;
        using inner_value_type = typename value_type::value_type;
        using reference = std::conditional_t<is_data_const,
                                             typename storage_type::const_reference,
                                             typename storage_type::reference>;
        using const_reference = typename storage_type::const_reference;
        using pointer = std::conditional_t<is_data_const,
                                           typename storage_type::const_pointer,
                                           typename storage_type::pointer>;
        using const_pointer = typename storage_type::const_pointer;
        using size_type = typename data_type::size_type;
        using difference_type = typename data_type::difference_type;

        using shape_type = typename data_type::shape_type;
        using strides_type = typename data_type::strides_type;

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

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

    private:

        template <class E>
        std::enable_if_t<is_xoptional_expression<E>::value> evaluate_cache() const;

        template <class E>
        std::enable_if_t<!is_xoptional_expression<E>::value> evaluate_cache() const;

        const_reference missing() const;
        reference missing();

        CTD m_data;
        CTM m_mask;
        detail::xmasked_view_storage<CTD, CTM> m_masked_view_storage;
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
          m_masked_view_storage(),
          m_flag_container_cache(m_masked_view_storage.get_flag_container_cache(m_data, m_mask)),
          m_storage_cache(m_masked_view_storage.get_storage_cache(m_data, m_mask, m_flag_container_cache)),
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
        return m_masked_view_storage.call_operator_impl(missing(), m_data, m_mask, args...);
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
        return m_masked_view_storage.call_operator_impl(missing(), m_data, m_mask, args...);
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
        return m_masked_view_storage.call_operator_impl(missing(), m_data, m_mask, args...);
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
        return m_masked_view_storage.call_operator_impl(missing(), m_data, m_mask, args...);
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
        return m_masked_view_storage.unchecked_impl(missing(), m_data, m_mask, args...);
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
        return m_masked_view_storage.unchecked_impl(missing(), m_data, m_mask, args...);
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
        return m_masked_view_storage.access_operator_impl(missing(), m_data, m_mask, index);
    }

    template <class CTD, class CTM>
    template <class I>
    inline auto xmasked_view<CTD, CTM>::operator[](std::initializer_list<I> index) -> reference
    {
        return m_masked_view_storage.access_operator_impl(missing(), m_data, m_mask, index);
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::operator[](size_type i) -> reference
    {
        return m_masked_view_storage.access_operator_impl(missing(), m_data, m_mask, i);
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
        return m_masked_view_storage.access_operator_impl(missing(), m_data, m_mask, index);
    }

    template <class CTD, class CTM>
    template <class I>
    inline auto xmasked_view<CTD, CTM>::operator[](std::initializer_list<I> index) const -> const_reference
    {
        return m_masked_view_storage.access_operator_impl(missing(), m_data, m_mask, index);
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::operator[](size_type i) const -> const_reference
    {
        return m_masked_view_storage.access_operator_impl(missing(), m_data, m_mask, i);
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
        return m_masked_view_storage.element_impl(missing(), m_data, m_mask, first, last);
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
        return m_masked_view_storage.element_impl(missing(), m_data, m_mask, first, last);
    }
    //@}

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::storage() noexcept -> storage_type&
    {
        evaluate_cache<CTD>();
        return m_storage_cache;
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::storage() const noexcept -> const storage_type&
    {
        evaluate_cache<CTD>();
        return m_storage_cache;
    }

    /**
     * Return an expression for the values of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::value() noexcept -> value_expression&
    {
        return m_masked_view_storage.value_impl(m_data);
    }

    /**
     * Return a constant expression for the values of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::value() const noexcept -> const value_expression&
    {
        return m_masked_view_storage.value_impl(m_data);
    }

    /**
     * Return an expression for the missing mask of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::has_value() noexcept -> flag_expression&
    {
        evaluate_cache<CTD>();
        return m_flag_container_cache;
    }

    /**
     * Return a constant expression for the missing mask of the xmasked_view.
     */
    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::has_value() const noexcept -> const flag_expression&
    {
        evaluate_cache<CTD>();
        return m_flag_container_cache;
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::data() noexcept -> value_type*
    {
        evaluate_cache<CTD>();
        return m_storage_cache.data();
    }

    template <class CTD, class CTM>
    inline auto xmasked_view<CTD, CTM>::data() const noexcept -> const value_type*
    {
        evaluate_cache<CTD>();
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
    template <class E>
    inline auto xmasked_view<CTD, CTM>::operator=(const xexpression<E>& e) -> self_type&
    {
        return semantic_base::operator=(e);
    }

    template <class CTD, class CTM>
    template <class E>
    inline auto xmasked_view<CTD, CTM>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(this->begin(), this->end(), e);
        return *this;
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
    template <class E>
    inline std::enable_if_t<is_xoptional_expression<E>::value> xmasked_view<CTD, CTM>::evaluate_cache() const
    {
        if (m_data_cached)
        {
            return;
        }

        m_flag_container_cache = m_data.has_value() && m_mask;

        m_data_cached = true;
    }

    template <class CTD, class CTM>
    template <class E>
    inline std::enable_if_t<!is_xoptional_expression<E>::value> xmasked_view<CTD, CTM>::evaluate_cache() const
    {
    }

    template <class CTD, class CTM>
    inline xmasked_view<CTD, CTM> masked_view(CTD&& data, CTM&& mask)
    {
        return xmasked_view<CTD, CTM>(std::forward<CTD>(data), std::forward<CTM>(mask));
    }
}

#endif
