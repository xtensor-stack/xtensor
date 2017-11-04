/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_OPTIONAL_HPP
#define XTENSOR_OPTIONAL_HPP

#include <type_traits>
#include <utility>

#include "xtl/xoptional.hpp"
#include "xtl/xoptional_sequence.hpp"

#include "xarray.hpp"
#include "xtensor.hpp"

namespace xt
{

    using xtl::xoptional;
    using xtl::missing;
    using xtl::disable_xoptional;
    using xtl::enable_xoptional;

    /****************************************************
     * Metafunction for splitting xoptional expressions *
     ****************************************************/

    namespace detail
    {
        template <class CT, class CB>
        struct functor_return_type<xoptional<CT, CB>, bool>
        {
            using type = ::xtl::xoptional<bool>;
            using simd_type = ::xtl::xoptional<bool>;
        };

        template <class T, class Tag>
        struct split_optional_expression_impl
        {
            using value_expression = T;
            using flag_expression = decltype(ones<bool>(std::declval<T>().shape()));

            template <class U>
            static inline U&& value(U&& arg)
            {
                return std::forward<U>(arg);
            }

            template <class U>
            static inline flag_expression has_value(U&& arg)
            {
                return ones<bool>(arg.shape());
            }
        };

        template <class T>
        struct split_optional_expression_impl<T, xoptional_expression_tag>
        {
            using value_expression = decltype(std::declval<T>().value());
            using flag_expression = decltype(std::declval<T>().has_value());

            template <class U>
            static inline value_expression value(U&& arg)
            {
                return arg.value();
            }

            template <class U>
            static inline flag_expression has_value(U&& arg)
            {
                return arg.has_value();
            }
        };

        template <class T>
        struct split_optional_expression
            : split_optional_expression_impl<T, xexpression_tag_t<std::decay_t<T>>>
        {
        };

        template <class E>
        struct optional_containers
        {
            using optional_expression = std::remove_const_t<E>;
            using optional_container = typename optional_expression::container_type;
            using tmp_value_container = typename optional_container::base_container_type;
            using tmp_flag_container = typename optional_container::flag_container_type;
            using value_container = std::conditional_t<std::is_const<E>::value, const tmp_value_container, tmp_value_container>;
            using flag_container = std::conditional_t<std::is_const<E>::value, const tmp_flag_container, tmp_flag_container>;
        };

        template <class OA, layout_type L>
        struct split_optional_array
        {
            using optional_array = OA;
            using value_container = typename optional_containers<optional_array>::value_container;
            using flag_container = typename optional_containers<optional_array>::flag_container;
            using value_expression = xarray_container<value_container, L>;
            using flag_expression = xarray_container<flag_container, L>;

            static inline value_expression value(OA arg)
            {
                return value_expression(std::move(arg.data().value()), arg.shape());
            }

            static inline flag_expression has_value(OA arg)
            {
                return flag_expression(std::move(arg.data().has_value()), arg.shape());
            }
        };

        template <class OA, layout_type L>
        struct split_optional_array_ref
        {
            using optional_array = OA;
            using value_container = typename optional_containers<optional_array>::value_container;
            using flag_container = typename optional_containers<optional_array>::flag_container;
            using value_expression = xarray_adaptor<typename optional_containers<optional_array>::value_container, L>;
            using flag_expression = xarray_adaptor<typename optional_containers<optional_array>::flag_container, L>;

            static inline value_expression value(OA& arg)
            {
                return value_expression(arg.data().value(), arg.shape());
            }

            static inline flag_expression has_value(OA& arg)
            {
                return flag_expression(arg.data().has_value(), arg.shape());
            }
        };

        template <class T, layout_type L, class A, class BC, class SA>
        struct split_optional_expression<xarray_optional<T, L, A, BC, SA>>
            : split_optional_array<xarray_optional<T, L, A, BC, SA>, L>
        {
        };

        template <class T, layout_type L, class A, class BC, class SA>
        struct split_optional_expression<xarray_optional<T, L, A, BC, SA>&>
            : split_optional_array_ref<xarray_optional<T, L, A, BC, SA>, L>
        {
        };

        template <class T, layout_type L, class A, class BC, class SA>
        struct split_optional_expression<const xarray_optional<T, L, A, BC, SA>&>
            : split_optional_array_ref<const xarray_optional<T, L, A, BC, SA>, L>
        {
        };

        template <class OT, std::size_t N, layout_type L>
        struct split_optional_tensor
        {
            using optional_tensor = OT;
            using value_container = typename optional_containers<optional_tensor>::value_container;
            using flag_container = typename optional_containers<optional_tensor>::flag_container;
            using value_expression = xtensor_container<value_container, N, L>;
            using flag_expression = xtensor_container<flag_container, N, L>;

            static inline value_expression value(OT arg)
            {
                return value_expression(std::move(arg.data().value()), arg.shape());
            }

            static inline flag_expression has_value(OT arg)
            {
                return flag_expression(std::move(arg.data().has_value()), arg.shape());
            }
        };

        template <class OT, std::size_t N, layout_type L>
        struct split_optional_tensor_ref
        {
            using optional_tensor = OT;
            using value_container = typename optional_containers<optional_tensor>::value_container;
            using flag_container = typename optional_containers<optional_tensor>::flag_container;
            using value_expression = xtensor_adaptor<value_container&, N, L>;
            using flag_expression = xtensor_adaptor<flag_container&, N, L>;

            static inline value_expression value(OT& arg)
            {
                return value_expression(arg.data().value(), arg.shape());
            }

            static inline flag_expression has_value(OT& arg)
            {
                return flag_expression(arg.data().has_value(), arg.shape());
            }
        };

        template <class T, std::size_t N, layout_type L, class A, class BC>
        struct split_optional_expression<xtensor_optional<T, N, L, A, BC>>
            : split_optional_tensor<xtensor_optional<T, N, L, A, BC>, N, L>
        {
        };

        template <class T, std::size_t N, layout_type L, class A, class BC>
        struct split_optional_expression<xtensor_optional<T, N, L, A, BC>&>
            : split_optional_tensor_ref<xtensor_optional<T, N, L, A, BC>, N, L>
        {
        };

        template <class T, std::size_t N, layout_type L, class A, class BC>
        struct split_optional_expression<const xtensor_optional<T, N, L, A, BC>&>
            : split_optional_tensor_ref<const xtensor_optional<T, N, L, A, BC>, N, L>
        {
        };

        template <class T>
        using value_expression_t = typename split_optional_expression<T>::value_expression;

        template <class T>
        using flag_expression_t = typename split_optional_expression<T>::flag_expression;

        template <class T = bool>
        struct optional_bitwise
        {
            using return_type = T;
            using first_argument_type = T;
            using second_argument_type = T;
            using result_type = T;
            using simd_value_type = bool;
            using simd_result_type = bool;
            template <class T1, class T2>
            constexpr result_type operator()(const T1& arg1, const T2& arg2) const
            {
                return (arg1 & arg2);
            }
            constexpr simd_result_type simd_apply(const simd_value_type& arg1,
                                                  const simd_value_type& arg2) const
            {
                return (arg1 & arg2);
            }
        };
    }

    /**********************
     * optional functions *
     **********************/

    template <class T, class B>
    auto sign(const xoptional<T, B>& e);

    template <class E>
    detail::value_expression_t<E> value(E&&);

    template <class E>
    detail::flag_expression_t<E> has_value(E&&);

    template <>
    class xexpression_assigner_base<xoptional_expression_tag>
    {
    public:

        template <class E1, class E2>
        static void assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial);
    };

    /**********************
     * xoptional_function *
     **********************/

#define DL DEFAULT_LAYOUT

    template <class F, class R, class... CT>
    class xoptional_function : public xexpression<xoptional_function<F, R, CT...>>,
                               private xconst_iterable<xfunction<F, R, CT...>>
    {
    public:

        using self_type = xoptional_function<F, R, CT...>;
        using implementation_type = xfunction<F, R, CT...>;

        // Delegating to xfunction. Private inheritance and using declarations
        // lead to ambiguous call when the function parameter is xexpression<E>
        // on gcc: the compiler can't choose between xexpression<xoptional_function>
        // and xexpression<xfunction>.

        using only_scalar = typename implementation_type::only_scalar;
        using functor_type = typename implementation_type::functor_type;
        using value_type = typename implementation_type::value_type;
        using reference = typename implementation_type::reference;
        using const_reference = typename implementation_type::const_reference;
        using pointer = typename implementation_type::pointer;
        using const_pointer = typename implementation_type::const_pointer;
        using size_type = typename implementation_type::size_type;
        using difference_type = typename implementation_type::difference_type;
        using simd_value_type = typename implementation_type::simd_value_type;
        using iterable_base = xconst_iterable<xfunction<F, R, CT...>>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        static constexpr layout_type static_layout = implementation_type::static_layout;
        static constexpr bool contiguous_layout = implementation_type::contiguous_layout;

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

        using storage_iterator = typename implementation_type::storage_iterator;
        using const_storage_iterator = typename implementation_type::const_storage_iterator;
        using reverse_storage_iterator = typename implementation_type::reverse_storage_iterator;
        using const_reverse_storage_iterator = typename implementation_type::const_reverse_storage_iterator;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;
        using reverse_iterator = typename iterable_base::reverse_iterator;
        using const_reverse_iterator = typename iterable_base::const_reverse_iterator;

        template <class Func>
        xoptional_function(Func&& func, CT... e) noexcept;

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const shape_type& shape() const;
        layout_type layout() const noexcept;

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class... Args>
        const_reference at(Args... args) const;

        const_reference operator[](const xindex& index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool broadcast_shape(S& shape) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const noexcept;

        using iterable_base::begin;
        using iterable_base::end;
        using iterable_base::cbegin;
        using iterable_base::cend;
        using iterable_base::rbegin;
        using iterable_base::rend;
        using iterable_base::crbegin;
        using iterable_base::crend;

        template <layout_type L = DL>
        const_storage_iterator storage_begin() const noexcept;
        template <layout_type L = DL>
        const_storage_iterator storage_end() const noexcept;
        template <layout_type L = DL>
        const_storage_iterator storage_cbegin() const noexcept;
        template <layout_type L = DL>
        const_storage_iterator storage_cend() const noexcept;

        template <layout_type L = DL>
        const_reverse_storage_iterator storage_rbegin() const noexcept;
        template <layout_type L = DL>
        const_reverse_storage_iterator storage_rend() const noexcept;
        template <layout_type L = DL>
        const_reverse_storage_iterator storage_crbegin() const noexcept;
        template <layout_type L = DL>
        const_reverse_storage_iterator storage_crend() const noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

        const_reference data_element(size_type i) const;

        template <class UT = self_type, class = typename std::enable_if<UT::only_scalar::value>::type>
        operator value_type() const;

        template <class align, class simd = simd_value_type>
        detail::simd_return_type_t<functor_type, simd> load_simd(size_type i) const;

        // Specific to xoptional_function

        using expression_tag = xoptional_expression_tag;
        using value_functor = typename F::template rebind<typename R::value_type>::type;
        using flag_functor = detail::optional_bitwise<bool>;

        using value_expression = xfunction<value_functor,
                                           typename R::value_type,
                                           detail::value_expression_t<CT>...>;

        using flag_expression = xfunction<flag_functor,
                                          bool,
                                          detail::flag_expression_t<CT>...>;

        value_expression value() const;
        flag_expression has_value() const;

    private:

        implementation_type m_func;

        template <std::size_t... I>
        value_expression value_impl(std::index_sequence<I...>) const;

        template <std::size_t... I>
        flag_expression has_value_impl(std::index_sequence<I...>) const;
    };

#undef DL

    /*************************************
     * xoptional_function implementation *
     *************************************/

    template <class F, class R, class... CT>
    template <class Func>
    inline xoptional_function<F, R, CT...>::xoptional_function(Func&& func, CT... e) noexcept
        : m_func(std::forward<Func>(func), e...)
    {
    }

    template <class F, class R, class... CT>
    inline auto xoptional_function<F, R, CT...>::size() const noexcept -> size_type
    {
        return m_func.size();
    }

    template <class F, class R, class... CT>
    inline auto xoptional_function<F, R, CT...>::dimension() const noexcept -> size_type
    {
        return m_func.dimension();
    }

    template <class F, class R, class... CT>
    inline auto xoptional_function<F, R, CT...>::shape() const -> const shape_type&
    {
        return m_func.shape();
    }

    template <class F, class R, class... CT>
    inline auto xoptional_function<F, R, CT...>::layout() const noexcept -> layout_type
    {
        return m_func.layout();
    }

    template <class F, class R, class... CT>
    template <class... Args>
    inline auto xoptional_function<F, R, CT...>::operator()(Args... args) const -> const_reference
    {
        return m_func(args...);
    }

    template <class F, class R, class... CT>
    template <class... Args>
    inline auto xoptional_function<F, R, CT...>::at(Args... args) const -> const_reference
    {
        return m_func.at(args...);
    }

    template <class F, class R, class... CT>
    inline auto xoptional_function<F, R, CT...>::operator[](const xindex& index) const -> const_reference
    {
        return m_func[index];
    }

    template <class F, class R, class... CT>
    inline auto xoptional_function<F, R, CT...>::operator[](size_type i) const -> const_reference
    {
        return m_func[i];
    }


    template <class F, class R, class... CT>
    template <class It>
    inline auto xoptional_function<F, R, CT...>::element(It first, It last) const -> const_reference
    {
        return m_func.element(first, last);
    }


    template <class F, class R, class... CT>
    template <class S>
    bool xoptional_function<F, R, CT...>::broadcast_shape(S& shape) const
    {
        return m_func.broadcast_shape(shape);
    }

    template <class F, class R, class... CT>
    template <class S>
    bool xoptional_function<F, R, CT...>::is_trivial_broadcast(const S& strides) const noexcept
    {
        return m_func.is_trivial_broadcast(strides);
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xoptional_function<F, R, CT...>::storage_begin() const noexcept -> const_storage_iterator
    {
        return m_func.template storage_begin<L>();
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xoptional_function<F, R, CT...>::storage_end() const noexcept -> const_storage_iterator
    {
        return m_func.template storage_end<L>();
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xoptional_function<F, R, CT...>::storage_cbegin() const noexcept -> const_storage_iterator
    {
        return m_func.template storage_cbegin<L>();
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xoptional_function<F, R, CT...>::storage_cend() const noexcept -> const_storage_iterator
    {
        return m_func.template storage_cend<L>();
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xoptional_function<F, R, CT...>::storage_rbegin() const noexcept -> const_reverse_storage_iterator
    {
        return m_func.template storage_rbegin<L>();
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xoptional_function<F, R, CT...>::storage_rend() const noexcept -> const_reverse_storage_iterator
    {
        return m_func.template storage_rend<L>();
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xoptional_function<F, R, CT...>::storage_crbegin() const noexcept -> const_reverse_storage_iterator
    {
        return m_func.template storage_crbegin<L>();
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xoptional_function<F, R, CT...>::storage_crend() const noexcept -> const_reverse_storage_iterator
    {
        return m_func.template storage_crend<L>();
    }

    template <class F, class R, class... CT>
    template <class S>
    inline auto xoptional_function<F, R, CT...>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        return m_func.stepper_begin(shape);
    }

    template <class F, class R, class... CT>
    template <class S>
    inline auto xoptional_function<F, R, CT...>::stepper_end(const S& shape, layout_type l) const noexcept -> const_stepper
    {
        return m_func.stepper_end(shape, l);
    }

    template <class F, class R, class... CT>
    inline auto xoptional_function<F, R, CT...>::data_element(size_type i) const -> const_reference
    {
        return m_func.data_element(i);
    }

    template <class F, class R, class... CT>
    template <class UT, class>
    inline xoptional_function<F, R, CT...>::operator value_type() const
    {
        return operator()();
    }

    template <class F, class R, class... CT>
    template <class align, class simd>
    inline auto xoptional_function<F, R, CT...>::load_simd(size_type i) const -> detail::simd_return_type_t<functor_type, simd>
    {
        return m_func.template load_simd<align, simd>(i);
    }

    template <class F, class R, class... CT>
    inline auto xoptional_function<F, R, CT...>::value() const -> value_expression
    {
        return value_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class R, class... CT>
    inline auto xoptional_function<F, R, CT...>::has_value() const -> flag_expression
    {
        return has_value_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class R, class... CT>
    template <std::size_t... I>
    inline auto xoptional_function<F, R, CT...>::value_impl(std::index_sequence<I...>) const -> value_expression
    {
        return value_expression(value_functor(),
            detail::split_optional_expression<CT>::value(std::forward<CT>(std::get<I>(m_func.arguments())))...);
    }

    template <class F, class R, class... CT>
    template <std::size_t... I>
    inline auto xoptional_function<F, R, CT...>::has_value_impl(std::index_sequence<I...>) const -> flag_expression
    {
        return flag_expression(flag_functor(),
            detail::split_optional_expression<CT>::has_value(std::forward<CT>(std::get<I>(m_func.arguments())))...);
    }

    /********************************
     * sign function implementation *
     ********************************/

    namespace math
    {
        template <class T, class B>
        struct sign_fun<xoptional<T, B>>
        {
            using argument_type = xoptional<T, B>;
            using result_type = xoptional<std::decay_t<T>>;

            constexpr result_type operator()(const xoptional<T, B>& x) const
            {
                return x.has_value() ? xoptional<T>(detail::sign_impl(x.value()))
                                     : missing<std::decay_t<T>>();
            }

            template <class U>
            struct rebind
            {
                using type = sign_fun<U>;
            };
        };
    }

    template <class T, class B>
    inline auto sign(const xoptional<T, B>& e)
    {
        return e.has_value() ? math::detail::sign_impl(e.value()) : missing<std::decay_t<T>>();
    }

    /******************************************
     * value() and has_value() implementation *
     ******************************************/

    template <class E>
    inline auto value(E&& e) -> detail::value_expression_t<E>
    {
        return detail::split_optional_expression<E>::value(std::forward<E>(e));
    }

    template <class E>
    inline auto has_value(E&& e) -> detail::flag_expression_t<E>
    {
        return detail::split_optional_expression<E>::has_value(std::forward<E>(e));
    }

    template <class E1, class E2>
    inline void xexpression_assigner_base<xoptional_expression_tag>::assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial)
    {
        E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();

        bool trivial_broadcast = trivial && detail::is_trivial_broadcast(de1, de2);
        if (trivial_broadcast)
        {
            using base_value_type1 = typename std::decay_t<decltype(value(de1))>::value_type;
            using base_value_type2 = typename std::decay_t<decltype(value(de2))>::value_type;
            constexpr bool contiguous_layout = E1::contiguous_layout && E2::contiguous_layout;
            constexpr bool same_type = std::is_same<base_value_type1, base_value_type2>::value;
            constexpr bool simd_size = xsimd::simd_traits<base_value_type1>::size > 1;
            constexpr bool forbid_simd = detail::forbid_simd_assign<E2>::value;
            constexpr bool simd_assign = contiguous_layout && same_type && simd_size && !forbid_simd;
            decltype(auto) bde1 = value(de1);
            decltype(auto) hde1 = has_value(de1);
            trivial_assigner<simd_assign>::run(bde1, value(de2));
            trivial_assigner<false>::run(hde1, has_value(de2));
        }
        else
        {
            data_assigner<E1, E2, default_assignable_layout(E1::static_layout)> assigner(de1, de2);
            assigner.run();
        }
    }
}

#endif
