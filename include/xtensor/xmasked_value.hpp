/***************************************************************************
* Copyright (c) 2017, Johan Mabille, Sylvain Corlay, Wolf Vollprecht and   *
* Martin Renou                                                             *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_XMASKED_VALUE_HPP
#define XTENSOR_XMASKED_VALUE_HPP

#include "xtl/xoptional.hpp"

#include "xutils.hpp"

namespace xt
{
    template <class T, class B = bool>
    class xmasked_value;

    template <class T>
    inline xmasked_value<T, bool> missing() noexcept
    {
        return xmasked_value<T, bool>(T(0), false);
    }

    namespace detail
    {
        template <class E>
        struct is_xmasked_value_impl : std::false_type
        {
        };

        template <class T, class B>
        struct is_xmasked_value_impl<xmasked_value<T, B>> : std::true_type
        {
        };

        template <class... Args>
        struct common_masked_value_impl;

        template <class T>
        struct common_masked_value_impl<T>
        {
            using type = std::conditional_t<is_xmasked_value_impl<T>::value , T, xmasked_value<T>>;
        };

        template <class T>
        struct id
        {
            using type = T;
        };

        template <class T>
        struct get_value_type
        {
            using type = typename T::value_type;
        };

        template<class T1, class T2>
        struct common_masked_value_impl<T1, T2>
        {
            using decay_t1 = std::decay_t<T1>;
            using decay_t2 = std::decay_t<T2>;
            using type1 = xtl::mpl::eval_if_t<std::is_fundamental<decay_t1>, id<decay_t1>, get_value_type<decay_t1>>;
            using type2 = xtl::mpl::eval_if_t<std::is_fundamental<decay_t2>, id<decay_t2>, get_value_type<decay_t2>>;
            using type = xmasked_value<std::common_type_t<type1, type2>>;
        };

        template <class T1, class T2, class B2>
        struct common_masked_value_impl<T1, xmasked_value<T2, B2>>
            : common_masked_value_impl<T1, T2>
        {
        };

        template <class T1, class B1, class T2>
        struct common_masked_value_impl<xmasked_value<T1, B1>, T2>
            : common_masked_value_impl<T2, xmasked_value<T1, B1>>
        {
        };

        template <class T1, class B1, class T2, class B2>
        struct common_masked_value_impl<xmasked_value<T1, B1>, xmasked_value<T2, B2>>
            : common_masked_value_impl<T1, T2>
        {
        };

        template <class T1, class T2, class... Args>
        struct common_masked_value_impl<T1, T2, Args...>
        {
            using type = typename common_masked_value_impl<
                             typename common_masked_value_impl<T1, T2>::type,
                             Args...
                         >::type;
        };
    }

    template <class E>
    using is_xmasked_value = detail::is_xmasked_value_impl<E>;

    template <class E, class R>
    using disable_xoptional_like = std::enable_if_t<!xtl::is_xoptional<E>::value && !is_xmasked_value<E>::value, R>;

    template <class E1, class E2, class R = void>
    using disable_both_xoptional_like = std::enable_if_t<
        !xtl::is_xoptional<E1>::value && !is_xmasked_value<E1>::value &&
            !xtl::is_xoptional<E2>::value && !is_xmasked_value<E2>::value,
        R
    >;

    template <class... Args>
    struct common_masked_value : detail::common_masked_value_impl<Args...>
    {
    };

    template <class... Args>
    using common_masked_value_t = typename common_masked_value<Args...>::type;

    /****************
    * xmasked_value *
    *****************/

    template <class T, class B>
    class xmasked_value
    {
    public:

        using self_type = xmasked_value<T, B>;

        using value_type = T;
        using flag_type = B;
        using optional_type = xtl::xoptional<value_type, flag_type>;

        template <class T1, class B1>
        inline constexpr xmasked_value(T1&& value, B1&& flag)
            : m_value(std::forward<T1>(value)), m_flag(std::forward<B1>(flag))
        {
        }

        inline explicit constexpr xmasked_value(const xtl::xoptional<T, B>& opt)
            : m_value(opt.value()), m_flag(opt.has_value())
        {
        }

        inline operator optional_type&() {
            static auto val = optional_type(m_value, m_flag);
            return val;
        }

        inline value_type value()
        {
            return m_value;
        }

        inline flag_type has_value()
        {
            return m_flag;
        }

        inline value_type value() const
        {
            return m_value;
        }

        inline flag_type has_value() const
        {
            return m_flag;
        }

        template <class T1, class B1>
        inline bool equal(const xmasked_value<T1, B1>& rhs) const noexcept
        {
            return (!m_flag && !rhs.m_flag) || (m_value == rhs.m_value && (m_flag && rhs.m_flag));
        }

        template <class T1, class B1>
        inline bool equal(const xtl::xoptional<T1, B1>& rhs) const noexcept
        {
            return (!m_flag && !rhs.has_value()) || (m_value == rhs.value() && (m_flag && rhs.has_value()));
        }

        template <class T1>
        inline auto equal(const T1& rhs) const noexcept -> disable_xoptional_like<T1, bool>
        {
            return m_flag && m_value == rhs;
        }

#define DEFINE_ASSIGN_OPERATOR(OP)                                                            \
        template <class T1>                                                                   \
        inline auto operator OP(const T1& value) -> disable_xoptional_like<T1, xmasked_value&>\
        {                                                                                     \
            if (m_flag)                                                                       \
            {                                                                                 \
                m_value OP value;                                                             \
            }                                                                                 \
            return *this;                                                                     \
        }                                                                                     \
                                                                                              \
        template <class T1, class B1>                                                         \
        inline xmasked_value& operator OP(const xtl::xoptional<T1, B1>& value)                \
        {                                                                                     \
            m_flag = m_flag && value.has_value();                                             \
            if (m_flag)                                                                       \
            {                                                                                 \
                m_value OP value.value();                                                     \
            }                                                                                 \
            return *this;                                                                     \
        }                                                                                     \
                                                                                              \
        template <class T1, class B1>                                                         \
        inline xmasked_value& operator OP(const xmasked_value<T1, B1>& value)                 \
        {                                                                                     \
            m_flag = m_flag && value.has_value();                                             \
            if (m_flag)                                                                       \
            {                                                                                 \
                m_value OP value.value();                                                     \
            }                                                                                 \
            return *this;                                                                     \
        }

        DEFINE_ASSIGN_OPERATOR(=);
        DEFINE_ASSIGN_OPERATOR(+=);
        DEFINE_ASSIGN_OPERATOR(-=);
        DEFINE_ASSIGN_OPERATOR(*=);
        DEFINE_ASSIGN_OPERATOR(/=);
        DEFINE_ASSIGN_OPERATOR(%=);
        DEFINE_ASSIGN_OPERATOR(&=);
        DEFINE_ASSIGN_OPERATOR(|=);
        DEFINE_ASSIGN_OPERATOR(^=);

#undef DEFINE_ASSIGN_OPERATOR

        inline void swap(xmasked_value& other)
        {
            std::swap(m_value, other.m_value);
            std::swap(m_flag, other.m_flag);
        }

    private:

        value_type m_value;
        flag_type m_flag;
    };

    template <class T>
    inline auto masked_value(T&& val) -> disable_xoptional_like<std::decay_t<T>, xmasked_value<T>>
    {
        return xmasked_value<T>(std::forward<T>(val));
    }

    template <class T, class B>
    inline auto masked_value(const xtl::xoptional<T, B>& val)
    {
        return xmasked_value<T, B>(val);
    }

    template <class T, class B>
    inline auto masked_value(T&& val, B&& mask)
    {
        return xmasked_value<T, B>(std::forward<T>(val), std::forward<B>(mask));
    }

    template <class T1, class B1, class T2, class B2>
    inline bool operator==(const xmasked_value<T1, B1>& lhs, const xmasked_value<T2, B2>& rhs) noexcept
    {
        return lhs.equal(rhs);
    }

    template <class T1, class B1, class T2, class B2>
    inline bool operator==(const xmasked_value<T1, B1>& lhs, const xtl::xoptional<T2, B2>& rhs) noexcept
    {
        return lhs.equal(rhs);
    }

    template <class T1, class B1, class T2, class B2>
    inline bool operator==(const xtl::xoptional<T1, B1>& lhs, const xmasked_value<T2, B2>& rhs) noexcept
    {
        return rhs.equal(lhs);
    }

    template <class T1, class T2, class B2>
    inline auto operator==(const T1& lhs, const xmasked_value<T2, B2>& rhs) noexcept -> disable_xoptional_like<T1, bool>
    {
        return rhs.equal(lhs);
    }

    template <class T1, class B1, class T2>
    inline auto operator==(const xmasked_value<T1, B1>& lhs, const T2& rhs) noexcept -> disable_xoptional_like<T2, bool>
    {
        return lhs.equal(rhs);
    }

    template <class T1, class B1, class T2, class B2>
    inline bool operator!=(const xmasked_value<T1, B1>& lhs, const xmasked_value<T2, B2>& rhs) noexcept
    {
        return !lhs.equal(rhs);
    }

    template <class T1, class B1, class T2, class B2>
    inline bool operator!=(xmasked_value<T1, B1> lhs, xtl::xoptional<T2, B2> rhs) noexcept
    {
        return !lhs.equal(rhs);
    }

    template <class T1, class B1, class T2, class B2>
    inline bool operator!=(xtl::xoptional<T1, B1> lhs, xmasked_value<T2, B2> rhs) noexcept
    {
        return !rhs.equal(lhs);
    }

    template <class T1, class T2, class B2>
    inline auto operator!=(const T1& lhs, const xmasked_value<T2, B2>& rhs) noexcept -> disable_xoptional_like<T1, bool>
    {
        return !rhs.equal(lhs);
    }

    template <class T1, class B1, class T2>
    inline auto operator!=(const xmasked_value<T1, B1>& lhs, const T2& rhs) noexcept -> disable_xoptional_like<T2, bool>
    {
        return !lhs.equal(rhs);
    }

    template <class T, class B>
    inline auto operator+(const xmasked_value<T, B>& e) noexcept
        -> xmasked_value<std::decay_t<T>, std::decay_t<B>>
    {
        return xmasked_value<std::decay_t<T>, std::decay_t<B>>(e.value(), e.has_value());
    }

    template <class T, class B>
    inline auto operator-(const xmasked_value<T, B>& e) noexcept
        -> xmasked_value<std::decay_t<T>, std::decay_t<B>>
    {
        return xmasked_value<std::decay_t<T>, std::decay_t<B>>(-e.value(), e.has_value());
    }

    template <class T, class B>
    inline auto operator~(const xmasked_value<T, B>& e) noexcept
        -> xmasked_value<std::decay_t<T>>
    {
        using value_type = std::decay_t<T>;
        return e.has_value() ? masked_value(~e.value()) : missing<value_type>();
    }

    template <class T, class B>
    inline auto operator!(const xmasked_value<T, B>& e) noexcept
        -> xmasked_value<bool>
    {
        return e.has_value() ? masked_value(!e.value()) : missing<bool>();
    }

    template <class T, class B, class OC, class OT>
    inline std::basic_ostream<OC, OT>& operator<<(std::basic_ostream<OC, OT>& out, xmasked_value<T, B> v)
    {
        if (v.has_value())
        {
            out << v.value();
        }
        else
        {
            out << "N/A";
        }
        return out;
    }

#define DEFINE_OPERATOR(OP)                                                                                      \
    template <class T1, class B1, class T2, class B2>                                                            \
    inline auto operator OP(const xmasked_value<T1, B1>& e1, const xtl::xoptional<T2, B2>& e2) noexcept          \
        -> xmasked_value<xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>>                                 \
    {                                                                                                            \
        using value_type = xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>;                               \
        return e1.has_value() && e2.has_value() ? masked_value(e1.value() OP e2.value()) : missing<value_type>();\
    }                                                                                                            \
                                                                                                                 \
    template <class T1, class B1, class T2, class B2>                                                            \
    inline auto operator OP(const xtl::xoptional<T1, B1>& e1, const xmasked_value<T2, B2>& e2) noexcept          \
        -> xmasked_value<xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>>                                 \
    {                                                                                                            \
        using value_type = xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>;                               \
        return e1.has_value() && e2.has_value() ? masked_value(e1.value() OP e2.value()) : missing<value_type>();\
    }                                                                                                            \
                                                                                                                 \
    template <class T1, class B1, class T2, class B2>                                                            \
    inline auto operator OP(const xmasked_value<T1, B1>& e1, const xmasked_value<T2, B2>& e2) noexcept           \
        -> xmasked_value<xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>>                                 \
    {                                                                                                            \
        using value_type = xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>;                               \
        return e1.has_value() && e2.has_value() ? masked_value(e1.value() OP e2.value()) : missing<value_type>();\
    }                                                                                                            \
                                                                                                                 \
    template <class T1, class B1, class T2>                                                                      \
    inline auto operator OP(const xmasked_value<T1, B1>& e1, const T2& e2) noexcept                              \
        -> disable_xoptional_like<T2, xmasked_value<xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>>>     \
    {                                                                                                            \
        using value_type = xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>;                               \
        return e1.has_value() ? masked_value(e1.value() OP e2) : missing<value_type>();                          \
    }                                                                                                            \
                                                                                                                 \
    template <class T1, class T2, class B2>                                                                      \
    inline auto operator OP(const T1& e1, const xmasked_value<T2, B2>& e2) noexcept                              \
        -> disable_xoptional_like<T1, xmasked_value<xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>>>     \
    {                                                                                                            \
        using value_type = xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>;                               \
        return e2.has_value() ? masked_value(e1 OP e2.value()) : missing<value_type>();                          \
    }

#define DEFINE_BOOL_OPERATOR(OP)                                                                           \
    template <class T1, class B1, class T2, class B2>                                                      \
    inline auto operator OP(const xmasked_value<T1, B1>& e1, const xtl::xoptional<T2, B2>& e2) noexcept    \
        -> xmasked_value<bool>                                                                             \
    {                                                                                                      \
        return e1.has_value() && e2.has_value() ? masked_value(e1.value() OP e2.value()) : missing<bool>();\
    }                                                                                                      \
                                                                                                           \
    template <class T1, class B1, class T2, class B2>                                                      \
    inline auto operator OP(const xtl::xoptional<T1, B1>& e1, const xmasked_value<T2, B2>& e2) noexcept    \
        -> xmasked_value<bool>                                                                             \
    {                                                                                                      \
        return e1.has_value() && e2.has_value() ? masked_value(e1.value() OP e2.value()) : missing<bool>();\
    }                                                                                                      \
                                                                                                           \
    template <class T1, class B1, class T2, class B2>                                                      \
    inline auto operator OP(const xmasked_value<T1, B1>& e1, const xmasked_value<T2, B2>& e2) noexcept     \
        -> xmasked_value<bool>                                                                             \
    {                                                                                                      \
        return e1.has_value() && e2.has_value() ? masked_value(e1.value() OP e2.value()) : missing<bool>();\
    }                                                                                                      \
                                                                                                           \
    template <class T1, class B1, class T2>                                                                \
    inline auto operator OP(const xmasked_value<T1, B1>& e1, const T2& e2) noexcept                        \
        -> disable_xoptional_like<T2, xmasked_value<bool>>                                                 \
    {                                                                                                      \
        return e1.has_value() ? masked_value(e1.value() OP e2) : missing<bool>();                          \
    }                                                                                                      \
                                                                                                           \
    template <class T1, class T2, class B2>                                                                \
    inline auto operator OP(const T1& e1, const xmasked_value<T2, B2>& e2) noexcept                        \
        -> disable_xoptional_like<T1, xmasked_value<bool>>                                                 \
    {                                                                                                      \
        return e2.has_value() ? masked_value(e1 OP e2.value()) : missing<bool>();                          \
    }

    DEFINE_OPERATOR(+);
    DEFINE_OPERATOR(-);
    DEFINE_OPERATOR(*);
    DEFINE_OPERATOR(/);
    DEFINE_OPERATOR(%);
    DEFINE_BOOL_OPERATOR(||);
    DEFINE_BOOL_OPERATOR(&&);
    DEFINE_OPERATOR(&);
    DEFINE_OPERATOR(|);
    DEFINE_OPERATOR(^);
    DEFINE_BOOL_OPERATOR(<);
    DEFINE_BOOL_OPERATOR(<=);
    DEFINE_BOOL_OPERATOR(>);
    DEFINE_BOOL_OPERATOR(>=);

#define DEFINE_UNARY_OPERATOR(OP)                                                             \
    template <class T, class B>                                                               \
    inline xmasked_value<std::decay_t<T>> OP(const xmasked_value<T, B>& e)                    \
    {                                                                                         \
        return e.has_value() ? masked_value(std::OP(e.value())) : missing<std::decay_t<T>>(); \
    }

#define DEFINE_UNARY_BOOL_OPERATOR(OP)                                                   \
    template <class T, class B>                                                          \
    inline xmasked_value<bool> OP(const xmasked_value<T, B>& e)                          \
    {                                                                                    \
        return e.has_value() ? masked_value(bool(std::OP(e.value()))) : missing<bool>(); \
    }

#define DEFINE_BINARY_OPERATOR_MM(OP)                                                                                    \
    template <class T1, class B1, class T2, class B2>                                                                    \
    inline auto OP(const xmasked_value<T1, B1>& e1, const xmasked_value<T2, B2>& e2)                                     \
        -> common_masked_value_t<T1, T2>                                                                                 \
    {                                                                                                                    \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;                                       \
        return e1.has_value() && e2.has_value() ? masked_value(std::OP(e1.value(), e2.value())) : missing<value_type>(); \
    }

#define DEFINE_BINARY_OPERATOR_MT(OP)                                                            \
    template <class T1, class B1, class T2>                                                      \
    inline auto OP(const xmasked_value<T1, B1>& e1, const T2& e2)                                \
        -> disable_xoptional_like<T2, common_masked_value_t<T1, T2>>                             \
    {                                                                                            \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;               \
        return e1.has_value() ? masked_value(std::OP(e1.value(), e2)) : missing<value_type>();   \
    }

#define DEFINE_BINARY_OPERATOR_TM(OP)                                                            \
    template <class T1, class T2, class B2>                                                      \
    inline auto OP(const T1& e1, const xmasked_value<T2, B2>& e2)                                \
        -> disable_xoptional_like<T1, common_masked_value_t<T1, T2>>                             \
    {                                                                                            \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;               \
        return e2.has_value() ? masked_value(std::OP(e1, e2.value())) : missing<value_type>();   \
    }

#define DEFINE_BINARY_OPERATOR_MO(OP)                                                                                    \
    template <class T1, class B1, class T2, class B2>                                                                    \
    inline auto OP(const xmasked_value<T1, B1>& e1, const xtl::xoptional<T2, B2>& e2)                                    \
        -> common_masked_value_t<T1, T2>                                                                                 \
    {                                                                                                                    \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;                                       \
        return e1.has_value() && e2.has_value() ? masked_value(std::OP(e1.value(), e2.value())) : missing<value_type>(); \
    }

#define DEFINE_BINARY_OPERATOR_OM(OP)                                                                                    \
    template <class T1, class B1, class T2, class B2>                                                                    \
    inline auto OP(const xtl::xoptional<T1, B1>& e1, const xmasked_value<T2, B2>& e2)                                    \
        -> common_masked_value_t<T1, T2>                                                                                 \
    {                                                                                                                    \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;                                       \
        return e1.has_value() && e2.has_value() ? masked_value(std::OP(e1.value(), e2.value())) : missing<value_type>(); \
    }

#define DEFINE_BINARY_OPERATOR(OP)  \
    DEFINE_BINARY_OPERATOR_MM(OP)   \
    DEFINE_BINARY_OPERATOR_MT(OP)   \
    DEFINE_BINARY_OPERATOR_TM(OP)   \
    DEFINE_BINARY_OPERATOR_MO(OP)   \
    DEFINE_BINARY_OPERATOR_OM(OP)

#define DEFINE_TERNARY_OPERATOR_MMM(OP)                                                                               \
    template <class T1, class B1, class T2, class B2, class T3, class B3>                                             \
    inline auto OP(const xmasked_value<T1, B1>& e1, const xmasked_value<T2, B2>& e2, const xmasked_value<T3, B3>& e3) \
        -> common_masked_value_t<T1, T2>                                                                              \
    {                                                                                                                 \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>;                  \
        return (e1.has_value() && e2.has_value() && e3.has_value()) ?                                                 \
                masked_value(std::OP(e1.value(), e2.value(), e3.value())) : missing<value_type>();                    \
    }

#define DEFINE_TERNARY_OPERATOR_MMT(OP)                                                              \
    template <class T1, class B1, class T2, class B2, class T3>                                      \
    inline auto OP(const xmasked_value<T1, B1>& e1, const xmasked_value<T2, B2>& e2, const T3& e3)   \
        -> disable_xoptional_like<T3, common_masked_value_t<T1, T2, T3>>                             \
    {                                                                                                \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>; \
        return (e1.has_value() && e2.has_value()) ?                                                  \
                masked_value(std::OP(e1.value(), e2.value(), e3)) : missing<value_type>();           \
    }

#define DEFINE_TERNARY_OPERATOR_MTM(OP)                                                              \
    template <class T1, class B1, class T2, class T3, class B3>                                      \
    inline auto OP(const xmasked_value<T1, B1>& e1, const T2& e2, const xmasked_value<T3, B3>& e3)   \
        -> disable_xoptional_like<T2, common_masked_value_t<T1, T2, T3>>                             \
    {                                                                                                \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>; \
        return (e1.has_value() && e3.has_value()) ?                                                  \
                masked_value(std::OP(e1.value(), e2, e3.value())) : missing<value_type>();           \
    }

#define DEFINE_TERNARY_OPERATOR_TMM(OP)                                                              \
    template <class T1, class T2, class B2, class T3, class B3>                                      \
    inline auto OP(const T1& e1, const xmasked_value<T2, B2>& e2, const xmasked_value<T3, B3>& e3)   \
        -> disable_xoptional_like<T1, common_masked_value_t<T1, T2, T3>>                             \
    {                                                                                                \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>; \
        return (e2.has_value() && e3.has_value()) ?                                                  \
                masked_value(std::OP(e1, e2.value(), e3.value())) : missing<value_type>();           \
    }

#define DEFINE_TERNARY_OPERATOR_TTM(OP)                                                              \
    template <class T1, class T2, class T3, class B3>                                                \
    inline auto OP(const T1& e1, const T2& e2, const xmasked_value<T3, B3>& e3)                      \
        -> disable_both_xoptional_like<T1, T2, common_masked_value_t<T1, T2, T3>>                    \
    {                                                                                                \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>; \
        return e3.has_value() ? masked_value(std::OP(e1, e2, e3.value())) : missing<value_type>();   \
    }

#define DEFINE_TERNARY_OPERATOR_TMT(OP)                                                              \
    template <class T1, class T2, class B2, class T3>                                                \
    inline auto OP(const T1& e1, const xmasked_value<T2, B2>& e2, const T3& e3)                      \
        -> disable_both_xoptional_like<T1, T3, common_masked_value_t<T1, T2, T3>>                    \
    {                                                                                                \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>; \
        return e2.has_value() ? masked_value(std::OP(e1, e2.value(), e3)) : missing<value_type>();   \
    }

#define DEFINE_TERNARY_OPERATOR_MTT(OP)                                                              \
    template <class T1, class B1, class T2, class T3>                                                \
    inline auto OP(const xmasked_value<T1, B1>& e1, const T2& e2, const T3& e3)                      \
        -> disable_both_xoptional_like<T2, T3, common_masked_value_t<T1, T2, T3>>                    \
    {                                                                                                \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>; \
        return e1.has_value() ? masked_value(std::OP(e1.value(), e2, e3)) : missing<value_type>();   \
    }

#define DEFINE_TERNARY_OPERATOR_MMO(OP)                                                                               \
    template <class T1, class B1, class T2, class B2, class T3, class B3>                                             \
    inline auto OP(const xmasked_value<T1, B1>& e1, const xmasked_value<T2, B2>& e2, const xtl::xoptional<T3, B3>& e3)\
        -> common_masked_value_t<T1, T2, T3>                                                                          \
    {                                                                                                                 \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>;                  \
        return (e1.has_value() && e2.has_value() && e3.has_value()) ?                                                 \
                masked_value(std::OP(e1.value(), e2.value(), e3.value())) : missing<value_type>();                    \
    }

#define DEFINE_TERNARY_OPERATOR_MOM(OP)                                                                               \
    template <class T1, class B1, class T2, class B2, class T3, class B3>                                             \
    inline auto OP(const xmasked_value<T1, B1>& e1, const xtl::xoptional<T2, B2>& e2, const xmasked_value<T3, B3>& e3)\
        -> common_masked_value_t<T1, T2, T3>                                                                          \
    {                                                                                                                 \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>;                  \
        return (e1.has_value() && e2.has_value() && e3.has_value()) ?                                                 \
                masked_value(std::OP(e1.value(), e2.value(), e3.value())) : missing<value_type>();                    \
    }

#define DEFINE_TERNARY_OPERATOR_OMM(OP)                                                                               \
    template <class T1, class B1, class T2, class B2, class T3, class B3>                                             \
    inline auto OP(const xtl::xoptional<T1, B1>& e1, const xmasked_value<T2, B2>& e2, const xmasked_value<T3, B3>& e3)\
        -> common_masked_value_t<T1, T2, T3>                                                                          \
    {                                                                                                                 \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>;                  \
        return (e1.has_value() && e2.has_value() && e3.has_value()) ?                                                 \
                masked_value(std::OP(e1.value(), e2.value(), e3.value())) : missing<value_type>();                    \
    }

#define DEFINE_TERNARY_OPERATOR_OOM(OP)                                                                                \
    template <class T1, class B1, class T2, class B2, class T3, class B3>                                              \
    inline auto OP(const xtl::xoptional<T1, B1>& e1, const xtl::xoptional<T2, B2>& e2, const xmasked_value<T3, B3>& e3)\
        -> common_masked_value_t<T1, T2, T3>                                                                           \
    {                                                                                                                  \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>;                   \
        return (e1.has_value() && e2.has_value() && e3.has_value()) ?                                                  \
                masked_value(std::OP(e1.value(), e2.value(), e3.value())) : missing<value_type>();                     \
    }

#define DEFINE_TERNARY_OPERATOR_OMO(OP)                                                                                 \
    template <class T1, class B1, class T2, class B2, class T3, class B3>                                               \
    inline auto OP(const xtl::xoptional<T1, B1>& e1, const xmasked_value<T2, B2>& e2, const xtl::xoptional<T3, B3>& e3) \
        -> common_masked_value_t<T1, T2, T3>                                                                            \
    {                                                                                                                   \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>;                    \
        return (e1.has_value() && e2.has_value() && e3.has_value()) ?                                                   \
                masked_value(std::OP(e1.value(), e2.value(), e3.value())) : missing<value_type>();                      \
    }

#define DEFINE_TERNARY_OPERATOR_MOO(OP)                                                                                 \
    template <class T1, class B1, class T2, class B2, class T3, class B3>                                               \
    inline auto OP(const xmasked_value<T1, B1>& e1, const xtl::xoptional<T2, B2>& e2, const xtl::xoptional<T3, B3>& e3) \
        -> common_masked_value_t<T1, T2, T3>                                                                            \
    {                                                                                                                   \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>;                    \
        return (e1.has_value() && e2.has_value() && e3.has_value()) ?                                                   \
                masked_value(std::OP(e1.value(), e2.value(), e3.value())) : missing<value_type>();                      \
    }

#define DEFINE_TERNARY_OPERATOR_TMO(OP)                                                               \
    template <class T1, class T2, class B2, class T3, class B3>                                       \
    inline auto OP(const T1& e1, const xmasked_value<T2, B2>& e2, const xtl::xoptional<T3, B3>& e3)   \
        -> disable_xoptional_like<T1, common_masked_value_t<T1, T2, T3>>                              \
    {                                                                                                 \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>;  \
        return (e2.has_value() && e3.has_value()) ?                                                   \
                masked_value(std::OP(e1, e2.value(), e3.value())) : missing<value_type>();            \
    }

#define DEFINE_TERNARY_OPERATOR_MTO(OP)                                                               \
    template <class T1, class B1, class T2, class T3, class B3>                                       \
    inline auto OP(const xmasked_value<T1, B1>& e1, const T2& e2, const xtl::xoptional<T3, B3>& e3)   \
        -> disable_xoptional_like<T2, common_masked_value_t<T1, T2, T3>>                              \
    {                                                                                                 \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>;  \
        return (e1.has_value() && e3.has_value()) ?                                                   \
                masked_value(std::OP(e1.value(), e2, e3.value())) : missing<value_type>();            \
    }

#define DEFINE_TERNARY_OPERATOR_MOT(OP)                                                               \
    template <class T1, class B1, class T2, class B2, class T3>                                       \
    inline auto OP(const xmasked_value<T1, B1>& e1, const xtl::xoptional<T2, B2>& e2, const T3& e3)   \
        -> disable_xoptional_like<T3, common_masked_value_t<T1, T2, T3>>                              \
    {                                                                                                 \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>;  \
        return (e1.has_value() && e2.has_value()) ?                                                   \
                masked_value(std::OP(e1.value(), e2.value(), e3)) : missing<value_type>();            \
    }

#define DEFINE_TERNARY_OPERATOR_OMT(OP)                                                               \
    template <class T1, class B1, class T2, class B2, class T3>                                       \
    inline auto OP(const xtl::xoptional<T1, B1>& e1, const xmasked_value<T2, B2>& e2, const T3& e3)   \
        -> disable_xoptional_like<T3, common_masked_value_t<T1, T2, T3>>                              \
    {                                                                                                 \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>;  \
        return (e1.has_value() && e2.has_value()) ?                                                   \
                masked_value(std::OP(e1.value(), e2.value(), e3)) : missing<value_type>();            \
    }

#define DEFINE_TERNARY_OPERATOR_OTM(OP)                                                               \
    template <class T1, class B1, class T2, class T3, class B3>                                       \
    inline auto OP(const xtl::xoptional<T1, B1>& e1, const T2& e2, const xmasked_value<T3, B3>& e3)   \
        -> disable_xoptional_like<T2, common_masked_value_t<T1, T2, T3>>                              \
    {                                                                                                 \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>;  \
        return (e1.has_value() && e3.has_value()) ?                                                   \
                masked_value(std::OP(e1.value(), e2, e3.value())) : missing<value_type>();            \
    }

#define DEFINE_TERNARY_OPERATOR_TOM(OP)                                                               \
    template <class T1, class T2, class B2, class T3, class B3>                                       \
    inline auto OP(const T1& e1, const xtl::xoptional<T2, B2>& e2, const xmasked_value<T3, B3>& e3)   \
        -> disable_xoptional_like<T1, common_masked_value_t<T1, T2, T3>>                              \
    {                                                                                                 \
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>, std::decay_t<T3>>;  \
        return (e2.has_value() && e3.has_value()) ?                                                   \
                masked_value(std::OP(e1, e2.value(), e3.value())) : missing<value_type>();            \
    }

#define DEFINE_TERNARY_OPERATOR(OP) \
    DEFINE_TERNARY_OPERATOR_MMM(OP) \
                                    \
    DEFINE_TERNARY_OPERATOR_MMT(OP) \
    DEFINE_TERNARY_OPERATOR_MTM(OP) \
    DEFINE_TERNARY_OPERATOR_TMM(OP) \
    DEFINE_TERNARY_OPERATOR_TTM(OP) \
    DEFINE_TERNARY_OPERATOR_TMT(OP) \
    DEFINE_TERNARY_OPERATOR_MTT(OP) \
                                    \
    DEFINE_TERNARY_OPERATOR_MMO(OP) \
    DEFINE_TERNARY_OPERATOR_MOM(OP) \
    DEFINE_TERNARY_OPERATOR_OMM(OP) \
    DEFINE_TERNARY_OPERATOR_OOM(OP) \
    DEFINE_TERNARY_OPERATOR_OMO(OP) \
    DEFINE_TERNARY_OPERATOR_MOO(OP) \
                                    \
    DEFINE_TERNARY_OPERATOR_TMO(OP) \
    DEFINE_TERNARY_OPERATOR_MTO(OP) \
    DEFINE_TERNARY_OPERATOR_MOT(OP) \
    DEFINE_TERNARY_OPERATOR_OMT(OP) \
    DEFINE_TERNARY_OPERATOR_OTM(OP) \
    DEFINE_TERNARY_OPERATOR_TOM(OP)

    DEFINE_UNARY_OPERATOR(abs)
    DEFINE_UNARY_OPERATOR(fabs)
    DEFINE_UNARY_OPERATOR(exp)
    DEFINE_UNARY_OPERATOR(exp2)
    DEFINE_UNARY_OPERATOR(expm1)
    DEFINE_UNARY_OPERATOR(log)
    DEFINE_UNARY_OPERATOR(log10)
    DEFINE_UNARY_OPERATOR(log2)
    DEFINE_UNARY_OPERATOR(log1p)
    DEFINE_UNARY_OPERATOR(sqrt)
    DEFINE_UNARY_OPERATOR(cbrt)
    DEFINE_UNARY_OPERATOR(sin)
    DEFINE_UNARY_OPERATOR(cos)
    DEFINE_UNARY_OPERATOR(tan)
    DEFINE_UNARY_OPERATOR(acos)
    DEFINE_UNARY_OPERATOR(asin)
    DEFINE_UNARY_OPERATOR(atan)
    DEFINE_UNARY_OPERATOR(sinh)
    DEFINE_UNARY_OPERATOR(cosh)
    DEFINE_UNARY_OPERATOR(tanh)
    DEFINE_UNARY_OPERATOR(acosh)
    DEFINE_UNARY_OPERATOR(asinh)
    DEFINE_UNARY_OPERATOR(atanh)
    DEFINE_UNARY_OPERATOR(erf)
    DEFINE_UNARY_OPERATOR(erfc)
    DEFINE_UNARY_OPERATOR(tgamma)
    DEFINE_UNARY_OPERATOR(lgamma)
    DEFINE_UNARY_OPERATOR(ceil)
    DEFINE_UNARY_OPERATOR(floor)
    DEFINE_UNARY_OPERATOR(trunc)
    DEFINE_UNARY_OPERATOR(round)
    DEFINE_UNARY_OPERATOR(nearbyint)
    DEFINE_UNARY_OPERATOR(rint)
    DEFINE_UNARY_BOOL_OPERATOR(isfinite)
    DEFINE_UNARY_BOOL_OPERATOR(isinf)
    DEFINE_UNARY_BOOL_OPERATOR(isnan)
    DEFINE_BINARY_OPERATOR(fmod)
    DEFINE_BINARY_OPERATOR(remainder)
    DEFINE_BINARY_OPERATOR(fmax)
    DEFINE_BINARY_OPERATOR(fmin)
    DEFINE_BINARY_OPERATOR(fdim)
    DEFINE_BINARY_OPERATOR(pow)
    DEFINE_BINARY_OPERATOR(hypot)
    DEFINE_BINARY_OPERATOR(atan2)
    DEFINE_TERNARY_OPERATOR(fma)

#undef DEFINE_TERNARY_OPERATOR
#undef DEFINE_TERNARY_OPERATOR_MMM
#undef DEFINE_TERNARY_OPERATOR_MMT
#undef DEFINE_TERNARY_OPERATOR_MTM
#undef DEFINE_TERNARY_OPERATOR_TMM
#undef DEFINE_TERNARY_OPERATOR_TTM
#undef DEFINE_TERNARY_OPERATOR_TMT
#undef DEFINE_TERNARY_OPERATOR_MTT
#undef DEFINE_TERNARY_OPERATOR_MMO
#undef DEFINE_TERNARY_OPERATOR_MOM
#undef DEFINE_TERNARY_OPERATOR_OMM
#undef DEFINE_TERNARY_OPERATOR_OOM
#undef DEFINE_TERNARY_OPERATOR_OMO
#undef DEFINE_TERNARY_OPERATOR_MOO
#undef DEFINE_TERNARY_OPERATOR_TMO
#undef DEFINE_TERNARY_OPERATOR_MTO
#undef DEFINE_TERNARY_OPERATOR_MOT
#undef DEFINE_TERNARY_OPERATOR_OMT
#undef DEFINE_TERNARY_OPERATOR_OTM
#undef DEFINE_TERNARY_OPERATOR_TOM
#undef DEFINE_BINARY_OPERATOR
#undef DEFINE_BINARY_OPERATOR_MM
#undef DEFINE_BINARY_OPERATOR_MT
#undef DEFINE_BINARY_OPERATOR_TM
#undef DEFINE_BINARY_OPERATOR_MO
#undef DEFINE_BINARY_OPERATOR_OM
#undef DEFINE_UNARY_OPERATOR
#undef DEFINE_UNARY_BOOL_OPERATOR
#undef DEFINE_OPERATOR
#undef DEFINE_BOOL_OPERATOR
}

#endif
