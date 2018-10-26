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

#include <xtl/xoptional.hpp>
#include <xtl/xoptional_sequence.hpp>

#include "xarray.hpp"
#include "xdynamic_view.hpp"
#include "xscalar.hpp"
#include "xtensor.hpp"

namespace xt
{

    /****************************************************
     * Metafunction for splitting xoptional expressions *
     ****************************************************/

    namespace extension
    {

        /**************************************
         * get_expression_tag specializations *
         **************************************/

        template <class T, class B>
        struct get_expression_tag<xtl::xoptional<T, B>>
        {
            using type = xoptional_expression_tag;
        };
    }

    namespace detail
    {
        /*****************************
         * split_optional_expression *
         *****************************/

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

        template <class T, class Tag>
        struct split_optional_expression_impl<xscalar<T>, Tag>
        {
            using value_expression = xscalar<T>;
            using flag_expression = xscalar<bool>;

            template <class U>
            static inline U&& value(U&& arg)
            {
                return std::forward<U>(arg);
            }

            template <class U>
            static inline flag_expression has_value(U&&)
            {
                return xscalar<bool>(true);
            }
        };

        template <class T>
        struct split_optional_expression_impl_base
        {
            static constexpr bool is_const = std::is_const<std::remove_reference_t<T>>::value;
            using decay_type = std::decay_t<T>;

            using value_expression = std::conditional_t<is_const,
                                                        typename decay_type::const_value_expression,
                                                        typename decay_type::value_expression>;
            using flag_expression = std::conditional_t<is_const,
                                                       typename decay_type::const_flag_expression,
                                                       typename decay_type::flag_expression>;

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
        struct split_optional_expression_impl<T, xoptional_expression_tag>
            : split_optional_expression_impl_base<T>
        {
        };

        template <class T>
        struct split_optional_expression_impl<xscalar<T>, xoptional_expression_tag>
            : split_optional_expression_impl_base<xscalar<T>>
        {
        };

        template <class T>
        struct split_optional_expression
            : split_optional_expression_impl<T, xexpression_tag_t<std::decay_t<T>>>
        {
        };

        template <class T>
        using value_expression_t = typename split_optional_expression<T>::value_expression;

        template <class T>
        using flag_expression_t = typename split_optional_expression<T>::flag_expression;

        /********************
         * optional_bitwise *
         ********************/

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
    auto sign(const xtl::xoptional<T, B>& e);

    template <class E, XTENSOR_REQUIRE<is_xexpression<E>::value>>
    detail::value_expression_t<E> value(E&&);

    template <class E, XTENSOR_REQUIRE<is_xexpression<E>::value>>
    detail::flag_expression_t<E> has_value(E&&);

    template <>
    class xexpression_assigner_base<xoptional_expression_tag>
    {
    public:

        template <class E1, class E2>
        static void assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial);
    };

    /**********************************
     * xscalar extension for optional *
     **********************************/

    namespace extension
    {
       template <class CT>
        struct xscalar_optional_traits
        {
            using closure_type = CT;
            static constexpr bool is_ref = std::is_reference<closure_type>::value;
            using unref_closure_type = std::remove_reference_t<closure_type>;
            static constexpr bool is_const = std::is_const<unref_closure_type>::value;
            using raw_closure_type = std::decay_t<CT>;

            using raw_value_closure = typename raw_closure_type::value_closure;
            using raw_flag_closure = typename raw_closure_type::flag_closure;
            using const_raw_value_closure = std::add_const_t<raw_value_closure>;
            using const_raw_flag_closure = std::add_const_t<raw_flag_closure>;

            using value_closure = std::conditional_t<is_ref,
                                                     std::add_lvalue_reference_t<raw_value_closure>,
                                                     raw_value_closure>;
            using flag_closure = std::conditional_t<is_ref,
                                                    std::add_lvalue_reference_t<raw_flag_closure>,
                                                    raw_flag_closure>;
            using const_value_closure = std::conditional_t<is_ref,
                                                           std::add_lvalue_reference_t<const_raw_value_closure>,
                                                           raw_value_closure>;
            using const_flag_closure = std::conditional_t<is_ref,
                                                          std::add_lvalue_reference_t<const_raw_flag_closure>,
                                                          raw_flag_closure>;

            using value_expression = xscalar<std::conditional_t<is_const, const_value_closure, value_closure>>;
            using flag_expression = xscalar<std::conditional_t<is_const, const_flag_closure, flag_closure>>;
            using const_value_expression = xscalar<const_value_closure>;
            using const_flag_expression = xscalar<const_flag_closure>;
        };

        template <class CT>
        class xscalar_optional_base
        {
        public:

            using traits = xscalar_optional_traits<CT>;
            using value_expression = typename traits::value_expression;
            using flag_expression = typename traits::flag_expression;
            using const_value_expression = typename traits::const_value_expression;
            using const_flag_expression = typename traits::const_flag_expression;
            using expression_tag = xoptional_expression_tag;

            value_expression value();
            const_value_expression value() const;

            flag_expression has_value();
            const_flag_expression has_value() const;

        private:

            using derived_type = xscalar<CT>;

            derived_type& derived_cast() noexcept;
            const derived_type& derived_cast() const noexcept;
        };

        template <class CT>
        struct xscalar_base_impl<xoptional_expression_tag, CT>
        {
            using type = xscalar_optional_base<CT>;
        };
    }

    /*************************************
     * xcontainer extention for optional *
     *************************************/

    namespace extension
    {
        template <class T>
        class xcontainer_optional_base
        {
        public:

            using traits = T;
            using value_expression = typename traits::value_expression;
            using flag_expression = typename traits::flag_expression;
            using const_value_expression = typename traits::const_value_expression;
            using const_flag_expression = typename traits::const_flag_expression;
            using expression_tag = xoptional_expression_tag;

            value_expression value();
            const_value_expression value() const;

            flag_expression has_value();
            const_flag_expression has_value() const;

        private:

            using derived_type = typename traits::derived_type;

            derived_type& derived_cast() noexcept;
            const derived_type& derived_cast() const noexcept;
        };
    }

    /*******************************************
     * xarray_container extension for optional *
     *******************************************/

    namespace extension
    {
        template <class EC, layout_type L, class SC>
        struct xarray_optional_traits
        {
            using value_container = typename EC::base_container_type;
            using flag_container = typename EC::flag_container_type;
            using value_expression = xarray_adaptor<value_container&, L, SC>;
            using flag_expression = xarray_adaptor<flag_container&, L, SC>;
            using const_value_expression = xarray_adaptor<const value_container&, L, SC>;
            using const_flag_expression = xarray_adaptor<const flag_container&, L, SC>;
            using derived_type = xarray_container<EC, L, SC, xoptional_expression_tag>;
        };

        template <class EC, layout_type L, class SC>
        struct xarray_container_base<EC, L, SC, xoptional_expression_tag>
        {
            using traits = xarray_optional_traits<EC, L, SC>;
            using type = xcontainer_optional_base<traits>;
        };
    }

    /********************************************
     * xtensor_container extension for optional *
     ********************************************/

    namespace extension
    {
        template <class EC, std::size_t N, layout_type L>
        struct xtensor_optional_traits
        {
            using value_container = typename EC::base_container_type;
            using flag_container = typename EC::flag_container_type;
            using value_expression = xtensor_adaptor<value_container&, N, L>;
            using flag_expression = xtensor_adaptor<flag_container&, N, L>;
            using const_value_expression = xtensor_adaptor<const value_container&, N, L>;
            using const_flag_expression = xtensor_adaptor<const flag_container&, N, L>;
            using derived_type = xtensor_container<EC, N, L, xoptional_expression_tag>;
        };

        template <class EC, std::size_t N, layout_type L>
        struct xtensor_container_base<EC, N, L, xoptional_expression_tag>
        {
            using traits = xtensor_optional_traits<EC, N, L>;
            using type = xcontainer_optional_base<traits>;
        };
    }

    /************************************************
     * xfunction extension for optional expressions *
     ************************************************/

    namespace extension
    {
        template <class F, class... CT>
        class xfunction_optional_base
        {
        public:

            using expression_tag = xoptional_expression_tag;
            using value_functor = F;
            using flag_functor = detail::optional_bitwise<bool>;
        
            using value_expression = xfunction<value_functor, detail::value_expression_t<CT>...>;
            using flag_expression = xfunction<flag_functor, detail::flag_expression_t<CT>...>;
            using const_value_expression = value_expression;
            using const_flag_expression = flag_expression;

            value_expression value() const;
            flag_expression has_value() const;
            
        private:

            template <std::size_t... I>
            value_expression value_impl(std::index_sequence<I...>) const;

            template <std::size_t... I>
            flag_expression has_value_impl(std::index_sequence<I...>) const;
            
            using derived_type = xfunction<F, CT...>;
            const derived_type& derived_cast() const noexcept;
        };

        template <class F, class... CT>
        struct xfunction_base_impl<xoptional_expression_tag, F, CT...>
        {
            using type = xfunction_optional_base<F, CT...>;
        };
    }

    /****************************************************
     * xdynamic_view extension for optional expressions *
     ****************************************************/

    namespace extension
    {
        template <class CT, class S, layout_type L, class FST>
        class xdynamic_view_optional
        {
        public:

            using expression_tag = xoptional_expression_tag;
            using uvt = typename std::decay_t<CT>::value_expression;
            using uft = typename std::decay_t<CT>::flag_expression;
            using ucvt = typename std::decay_t<CT>::const_value_expression;
            using ucft = typename std::decay_t<CT>::const_flag_expression;
            using value_expression = xdynamic_view<uvt, S, L, detail::flat_storage_type_t<uvt>>;
            using flag_expression = xdynamic_view<uft, S, L, detail::flat_storage_type_t<uft>>;
            using const_value_expression = xdynamic_view<ucvt, S, L, detail::flat_storage_type_t<ucvt>>;
            using const_flag_expression = xdynamic_view<ucft, S, L, detail::flat_storage_type_t<ucft>>;

            value_expression value();
            const_value_expression value() const;

            flag_expression has_value();
            const_flag_expression has_value() const;

        private:

            using derived_type = xdynamic_view<CT, S, L, FST>;

            derived_type& derived_cast() noexcept;
            const derived_type& derived_cast() const noexcept;
        };

        template <class CT, class S, layout_type L, class FST>
        struct xdynamic_view_base_impl<xoptional_expression_tag, CT, S, L, FST>
        {
            using type = xdynamic_view_optional<CT, S, L, FST>;
        };
    }

    /****************************************
     * xscalar_optional_base implementation *
     ****************************************/

    namespace extension
    {
        template <class CT>
        inline auto xscalar_optional_base<CT>::value() -> value_expression
        {
            return derived_cast().expression().value();
        }

        template <class CT>
        inline auto xscalar_optional_base<CT>::value() const -> const_value_expression
        {
            return derived_cast().expression().value();
        }

        template <class CT>
        inline auto xscalar_optional_base<CT>::has_value() -> flag_expression
        {
            return derived_cast().expression().has_value();
        }

        template <class CT>
        inline auto xscalar_optional_base<CT>::has_value() const -> const_flag_expression
        {
            return derived_cast().expression().has_value();
        }

        template <class CT>
        inline auto xscalar_optional_base<CT>::derived_cast() noexcept -> derived_type&
        {
            return *static_cast<derived_type*>(this);
        }

        template <class CT>
        inline auto xscalar_optional_base<CT>::derived_cast() const noexcept -> const derived_type&
        {
            return *static_cast<const derived_type*>(this);
        }
    }

    /*******************************************
     * xcontainer_optional_base implementation *
     *******************************************/

    namespace extension
    {
        template <class T>
        inline auto xcontainer_optional_base<T>::value() -> value_expression
        {
            return value_expression(derived_cast().storage().value(), derived_cast().shape());
        }

        template <class T>
        inline auto xcontainer_optional_base<T>::value() const -> const_value_expression
        {
            return const_value_expression(derived_cast().storage().value(), derived_cast().shape());
        }

        template <class T>
        inline auto xcontainer_optional_base<T>::has_value() -> flag_expression
        {
            return flag_expression(derived_cast().storage().has_value(), derived_cast().shape());
        }

        template <class T>
        inline auto xcontainer_optional_base<T>::has_value() const -> const_flag_expression
        {
            return const_flag_expression(derived_cast().storage().has_value(), derived_cast().shape());
        }

        template <class T>
        inline auto xcontainer_optional_base<T>::derived_cast() noexcept -> derived_type&
        {
            return *static_cast<derived_type*>(this);
        }

        template <class T>
        inline auto xcontainer_optional_base<T>::derived_cast() const noexcept -> const derived_type&
        {
            return *static_cast<const derived_type*>(this);
        }
    }

    /******************************************
     * xfunction_optional_base implementation *
     ******************************************/

    namespace extension
    {
        template <class F, class... CT>
        inline auto xfunction_optional_base<F, CT...>::value() const -> value_expression
        {
            return value_impl(std::make_index_sequence<sizeof...(CT)>());
        }

        template <class F, class... CT>
        inline auto xfunction_optional_base<F, CT...>::has_value() const -> flag_expression
        {
            return has_value_impl(std::make_index_sequence<sizeof...(CT)>());
        }

        template <class F, class... CT>
        template <std::size_t... I>
        inline auto xfunction_optional_base<F, CT...>::value_impl(std::index_sequence<I...>) const -> value_expression
        {
            return value_expression(value_functor(),
                detail::split_optional_expression<CT>::value(std::get<I>(derived_cast().arguments()))...);
        }

        template <class F, class... CT>
        template <std::size_t... I>
        inline auto xfunction_optional_base<F, CT...>::has_value_impl(std::index_sequence<I...>) const -> flag_expression
        {
            return flag_expression(flag_functor(),
                detail::split_optional_expression<CT>::has_value(std::get<I>(derived_cast().arguments()))...);
        }

        template <class F, class... CT>
        inline auto xfunction_optional_base<F, CT...>::derived_cast() const noexcept -> const derived_type&
        {
            return *static_cast<const derived_type*>(this);
        }
    }

    /*****************************************
     * xdynamic_view_optional implementation *
     *****************************************/

    namespace extension
    {
        template <class CT, class S, layout_type L, class FST>
        inline auto xdynamic_view_optional<CT, S, L, FST>::value() -> value_expression
        {
            return derived_cast().build_view(derived_cast().expression().value());
        }

        template <class CT, class S, layout_type L, class FST>
        inline auto xdynamic_view_optional<CT, S, L, FST>::value() const -> const_value_expression 
        {
            return derived_cast().build_view(derived_cast().expression().value());
        }

        template <class CT, class S, layout_type L, class FST>
        inline auto xdynamic_view_optional<CT, S, L, FST>::has_value() -> flag_expression 
        {
            return derived_cast().build_view(derived_cast().expression().has_value());
        }

        template <class CT, class S, layout_type L, class FST>
        inline auto xdynamic_view_optional<CT, S, L, FST>::has_value() const -> const_flag_expression 
        {
            return derived_cast().build_view(derived_cast().expression().has_value());
        }

        template <class CT, class S, layout_type L, class FST>
        inline auto xdynamic_view_optional<CT, S, L, FST>::derived_cast() noexcept -> derived_type&
        {
            return *static_cast<derived_type*>(this);
        }

        template <class CT, class S, layout_type L, class FST>
        inline auto xdynamic_view_optional<CT, S, L, FST>::derived_cast() const noexcept -> const derived_type&
        {
            return *static_cast<const derived_type*>(this);
        }
    }

    /********************************
     * sign function implementation *
     ********************************/

    namespace math
    {
        template <class T, class B>
        struct sign_impl<xtl::xoptional<T, B>>
        {
            static constexpr auto run(const xtl::xoptional<T, B>& x)
            {
                return sign(x); // use overload declared above
            }
        };
    }

    template <class T, class B>
    inline auto sign(const xtl::xoptional<T, B>& e)
    {
        using value_type = std::decay_t<T>;
        return e.has_value() ? math::sign_impl<value_type>::run(e.value()) : xtl::missing<value_type>();
    }

    /******************************************
     * value() and has_value() implementation *
     ******************************************/

    template <class E, class>
    inline auto value(E&& e) -> detail::value_expression_t<E>
    {
        return detail::split_optional_expression<E>::value(std::forward<E>(e));
    }

    template <class E, class>
    inline auto has_value(E&& e) -> detail::flag_expression_t<E>
    {
        return detail::split_optional_expression<E>::has_value(std::forward<E>(e));
    }

    template <class E1, class E2>
    inline void xexpression_assigner_base<xoptional_expression_tag>::assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial)
    {
        E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();

        decltype(auto) bde1 = xt::value(de1);
        decltype(auto) hde1 = xt::has_value(de1);
        xexpression_assigner_base<xtensor_expression_tag>::assign_data(bde1, xt::value(de2), trivial);
        xexpression_assigner_base<xtensor_expression_tag>::assign_data(hde1, xt::has_value(de2), trivial);
    }
}

#endif
