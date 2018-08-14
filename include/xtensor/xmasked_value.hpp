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
}

// Monkey patch on the xtl::common_optional_t implementation
namespace xtl
{
    namespace detail
    {
        template <class T1, class T2, class B2>
        struct common_optional_impl<T1, xt::xmasked_value<T2, B2>>
            : common_optional_impl<T1, T2>
        {
        };

        template <class T1, class B1, class T2>
        struct common_optional_impl<xt::xmasked_value<T1, B1>, T2>
            : common_optional_impl<T1, T2>
        {
        };
    }
}

namespace xt
{
    template <class T>
    inline xmasked_value<T, bool> masked() noexcept
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
    }

    template <class E>
    using is_xmasked_value = detail::is_xmasked_value_impl<E>;

    template <class E, class R>
    using disable_xmasked_value = std::enable_if_t<!is_xmasked_value<E>::value, R>;

    template <class E, class R>
    using disable_xoptional_like = std::enable_if_t<!xtl::is_xoptional<E>::value && !is_xmasked_value<E>::value, R>;

    template <class E1, class E2, class R = void>
    using disable_both_xoptional_like = std::enable_if_t<
        !xtl::is_xoptional<E1>::value && !is_xmasked_value<E1>::value &&
            !xtl::is_xoptional<E2>::value && !is_xmasked_value<E2>::value,
        R
    >;

    /****************************
    * xmasked_value declaration *
    *****************************/

    template <class T, class B>
    class xmasked_value
    {
    public:

        using self_type = xmasked_value<T, B>;

        using value_type = T;
        using flag_type = B;

        template <class T1, class B1>
        constexpr xmasked_value(T1&& value, B1&& flag);

        template <class T1>
        explicit constexpr xmasked_value(T1&& value);

        explicit constexpr xmasked_value();

        inline operator value_type() {
            return convert<value_type>();
        }

        std::add_lvalue_reference_t<T> value() & noexcept;
        std::add_lvalue_reference_t<std::add_const_t<T>> value() const & noexcept;
        std::conditional_t<std::is_reference<T>::value, apply_cv_t<T, std::decay_t<T>>&, std::decay_t<T>> value() && noexcept;
        std::conditional_t<std::is_reference<T>::value, const std::decay_t<T>&, std::decay_t<T>> value() const && noexcept;

        std::add_lvalue_reference_t<B> visible() & noexcept;
        std::add_lvalue_reference_t<std::add_const_t<B>> visible() const & noexcept;
        std::conditional_t<std::is_reference<B>::value, apply_cv_t<B, std::decay_t<B>>&, std::decay_t<B>> visible() && noexcept;
        std::conditional_t<std::is_reference<B>::value, const std::decay_t<B>&, std::decay_t<B>> visible() const && noexcept;

        template <class T1, class B1>
        bool equal(const xmasked_value<T1, B1>& rhs) const noexcept;

        template <class T1>
        auto equal(const T1& rhs) const noexcept -> disable_xmasked_value<T1, bool>;

        template <class T1, class B1>
        void swap(xmasked_value<T1, B1>& other);

#define DEFINE_ASSIGN_OPERATOR(OP)                                                            \
        template <class T1>                                                                   \
        inline xmasked_value& operator OP(const T1& rhs)                                      \
        {                                                                                     \
            if (m_visible)                                                                    \
            {                                                                                 \
                m_value OP rhs;                                                               \
            }                                                                                 \
            return *this;                                                                     \
        }                                                                                     \
                                                                                              \
        template <class T1, class B1>                                                         \
        inline xmasked_value& operator OP(const xmasked_value<T1, B1>& rhs)                   \
        {                                                                                     \
            m_visible = m_visible && rhs.visible();                                           \
            if (m_visible)                                                                    \
            {                                                                                 \
                m_value OP rhs.value();                                                       \
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

    private:

        template <class V>
        std::enable_if_t<xtl::is_xoptional<V>::value, V> convert() const;

        template <class V>
        std::enable_if_t<!xtl::is_xoptional<V>::value, V> convert() const;

        value_type m_value;
        flag_type m_visible;
    };

    /********************************
     * xmasked_value implementation *
     ********************************/

    template <class T, class B>
    template <class T1, class B1>
    inline constexpr xmasked_value<T, B>::xmasked_value(T1&& value, B1&& flag)
        : m_value(std::forward<T1>(value)), m_visible(std::forward<B1>(flag))
    {
    }

    template <class T, class B>
    template <class T1>
    inline constexpr xmasked_value<T, B>::xmasked_value(T1&& value)
        : m_value(std::forward<T1>(value)), m_visible(true)
    {
    }

    template <class T, class B>
    inline constexpr xmasked_value<T, B>::xmasked_value()
        : m_value(0), m_visible(true)
    {
    }

    template <class T>
    inline auto masked_value(T&& val)
    {
        return xmasked_value<T>(std::forward<T>(val));
    }

    template <class T, class B>
    inline auto masked_value(T&& val, B&& mask)
    {
        return xmasked_value<T, B>(std::forward<T>(val), std::forward<B>(mask));
    }

    template <class T, class B>
    inline auto xmasked_value<T, B>::value() & noexcept -> std::add_lvalue_reference_t<T>
    {
        return m_value;
    }

    template <class T, class B>
    inline auto xmasked_value<T, B>::value() const & noexcept -> std::add_lvalue_reference_t<std::add_const_t<T>>
    {
        return m_value;
    }

    template <class T, class B>
    inline auto xmasked_value<T, B>::value() && noexcept -> std::conditional_t<std::is_reference<T>::value, apply_cv_t<T, std::decay_t<T>>&, std::decay_t<T>>
    {
        return m_value;
    }

    template <class T, class B>
    inline auto xmasked_value<T, B>::value() const && noexcept -> std::conditional_t<std::is_reference<T>::value, const std::decay_t<T>&, std::decay_t<T>>
    {
        return m_value;
    }

    template <class T, class B>
    inline auto xmasked_value<T, B>::visible() & noexcept -> std::add_lvalue_reference_t<B>
    {
        return m_visible;
    }

    template <class T, class B>
    inline auto xmasked_value<T, B>::visible() const & noexcept -> std::add_lvalue_reference_t<std::add_const_t<B>>
    {
        return m_visible;
    }

    template <class T, class B>
    inline auto xmasked_value<T, B>::visible() && noexcept -> std::conditional_t<std::is_reference<B>::value, apply_cv_t<B, std::decay_t<B>>&, std::decay_t<B>>
    {
        return m_visible;
    }

    template <class T, class B>
    inline auto xmasked_value<T, B>::visible() const && noexcept -> std::conditional_t<std::is_reference<B>::value, const std::decay_t<B>&, std::decay_t<B>>
    {
        return m_visible;
    }

    template <class T, class B>
    template <class T1, class B1>
    inline bool xmasked_value<T, B>::equal(const xmasked_value<T1, B1>& rhs) const noexcept
    {
        return (!m_visible && !rhs.visible()) || (m_value == rhs.value() && (m_visible && rhs.visible()));
    }

    template <class T, class B>
    template <class T1>
    inline auto xmasked_value<T, B>::equal(const T1& rhs) const noexcept -> disable_xmasked_value<T1, bool>
    {
        return m_visible && m_value == rhs;
    }

    template <class T, class B>
    template <class T1, class B1>
    inline void xmasked_value<T, B>::swap(xmasked_value<T1, B1>& other)
    {
        using std::swap;
        swap(m_value, other.m_value);
        swap(m_visible, other.m_visible);
    }

    template <class T, class B>
    template <class V>
    inline auto xmasked_value<T, B>::convert() const -> std::enable_if_t<xtl::is_xoptional<V>::value, V>
    {
        return V(m_value.value(), m_value.has_value() && visible());
    }

    template <class T, class B>
    template <class V>
    inline auto xmasked_value<T, B>::convert() const -> std::enable_if_t<!xtl::is_xoptional<V>::value, V>
    {
        return m_value;
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
        return xmasked_value<std::decay_t<T>, std::decay_t<B>>(e.value(), e.visible());
    }

    template <class T, class B>
    inline auto operator-(const xmasked_value<T, B>& e) noexcept
        -> xmasked_value<std::decay_t<T>, std::decay_t<B>>
    {
        return xmasked_value<std::decay_t<T>, std::decay_t<B>>(-e.value(), e.visible());
    }

    template <class T, class B>
    inline auto operator~(const xmasked_value<T, B>& e) noexcept
        -> xmasked_value<std::decay_t<T>>
    {
        using value_type = std::decay_t<T>;
        return e.visible() ? masked_value(~e.value()) : masked<value_type>();
    }

    template <class T, class B>
    inline auto operator!(const xmasked_value<T, B>& e) noexcept
        -> xmasked_value<bool>
    {
        return e.visible() ? masked_value(!e.value()) : masked<bool>();
    }

    template <class T, class B, class OC, class OT>
    inline std::basic_ostream<OC, OT>& operator<<(std::basic_ostream<OC, OT>& out, xmasked_value<T, B> v)
    {
        if (v.visible())
        {
            out << v.value();
        }
        else
        {
            out << "masked";
        }
        return out;
    }

    template <class T1, class B1, class T2, class B2>
    inline void swap(xmasked_value<T1, B1>& lhs, xmasked_value<T2, B2>& rhs)
    {
        lhs.swap(rhs);
    }

#define DEFINE_OPERATOR(OP)                                                                                                                \
    /* Must be implemented in order to fix ambiguity with operators defined in xoptional */                                                \
    template <class T1, class B1, class T2, class B2>                                                                                      \
    inline auto operator OP(const xmasked_value<T1, B1>& e1, const xtl::xoptional<T2, B2>& e2) noexcept                                    \
        -> std::enable_if_t<xtl::is_xoptional<T1>::value, xmasked_value<xtl::common_optional_t<std::decay_t<T1>, xtl::xoptional<T2, B2>>>> \
    {                                                                                                                                      \
        using value_type = xtl::common_optional_t<std::decay_t<T1>, xtl::xoptional<T2, B2>>;                                               \
        return e1.visible() ? masked_value(e1.value() OP e2) : masked<value_type>();                                                       \
    }                                                                                                                                      \
                                                                                                                                           \
    template <class T1, class B1, class T2, class B2>                                                                                      \
    inline auto operator OP(const xtl::xoptional<T1, B1>& e1, const xmasked_value<T2, B2>& e2) noexcept                                    \
        -> std::enable_if_t<xtl::is_xoptional<T2>::value, xmasked_value<xtl::common_optional_t<xtl::xoptional<T1, B1>, std::decay_t<T2>>>> \
    {                                                                                                                                      \
        using value_type = xtl::common_optional_t<xtl::xoptional<T1, B1>, std::decay_t<T2>>;                                               \
        return e2.visible() ? masked_value(e1 OP e2.value()) : masked<value_type>();                                                       \
    }                                                                                                                                      \
                                                                                                                                           \
    template <class T1, class B1, class T2, class B2>                                                                                      \
    inline auto operator OP(const xmasked_value<T1, B1>& e1, const xmasked_value<T2, B2>& e2) noexcept                                     \
        -> xmasked_value<xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>>                                                           \
    {                                                                                                                                      \
        using value_type = xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>;                                                         \
        return e1.visible() && e2.visible() ? masked_value(e1.value() OP e2.value()) : masked<value_type>();                               \
    }                                                                                                                                      \
                                                                                                                                           \
    template <class T1, class B1, class T2>                                                                                                \
    inline auto operator OP(const xmasked_value<T1, B1>& e1, const T2& e2) noexcept                                                        \
        -> disable_xoptional_like<T2, xmasked_value<xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>>>                               \
    {                                                                                                                                      \
        using value_type = xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>;                                                         \
        return e1.visible() ? masked_value(e1.value() OP e2) : masked<value_type>();                                                       \
    }                                                                                                                                      \
                                                                                                                                           \
    template <class T1, class T2, class B2>                                                                                                \
    inline auto operator OP(const T1& e1, const xmasked_value<T2, B2>& e2) noexcept                                                        \
        -> disable_xoptional_like<T1, xmasked_value<xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>>>                               \
    {                                                                                                                                      \
        using value_type = xt::promote_type_t<std::decay_t<T1>, std::decay_t<T2>>;                                                         \
        return e2.visible() ? masked_value(e1 OP e2.value()) : masked<value_type>();                                                       \
    }

#define DEFINE_BOOL_OPERATOR(OP)                                                                           \
    /* Must be implemented in order to fix ambiguity with operators defined in xoptional */                \
    template <class T1, class B1, class T2, class B2>                                                      \
    inline auto operator OP(const xmasked_value<T1, B1>& e1, const xtl::xoptional<T2, B2>& e2) noexcept    \
        -> std::enable_if_t<xtl::is_xoptional<T1>::value, xmasked_value<xtl::xoptional<bool>>>             \
    {                                                                                                      \
        return e1.visible() ? masked_value(e1.value() OP e2) : masked<xtl::xoptional<bool>>();             \
    }                                                                                                      \
                                                                                                           \
    template <class T1, class B1, class T2, class B2>                                                      \
    inline auto operator OP(const xtl::xoptional<T1, B1>& e1, const xmasked_value<T2, B2>& e2) noexcept    \
        -> std::enable_if_t<xtl::is_xoptional<T2>::value, xmasked_value<xtl::xoptional<bool>>>             \
    {                                                                                                      \
        return e2.visible() ? masked_value(e1 OP e2.value()) : masked<xtl::xoptional<bool>>();             \
    }                                                                                                      \
                                                                                                           \
    template <class T1, class B1, class T2, class B2>                                                      \
    inline auto operator OP(const xmasked_value<T1, B1>& e1, const xmasked_value<T2, B2>& e2) noexcept     \
        -> xmasked_value<decltype(e1.value() OP e2.value())>                                               \
    {                                                                                                      \
        return e1.visible() && e2.visible() ?                                                              \
            masked_value(e1.value() OP e2.value()) :                                                       \
            masked<decltype(e1.value() OP e2.value())>();                                                  \
    }                                                                                                      \
                                                                                                           \
    template <class T1, class B1, class T2>                                                                \
    inline auto operator OP(const xmasked_value<T1, B1>& e1, const T2& e2) noexcept                        \
        -> disable_xoptional_like<T2, xmasked_value<decltype(e1.value() OP e2)>>                           \
    {                                                                                                      \
        return e1.visible() ? masked_value(e1.value() OP e2) : masked<decltype(e1.value() OP e2)>();       \
    }                                                                                                      \
                                                                                                           \
    template <class T1, class T2, class B2>                                                                \
    inline auto operator OP(const T1& e1, const xmasked_value<T2, B2>& e2) noexcept                        \
        -> disable_xoptional_like<T1, xmasked_value<decltype(e1 OP e2.value())>>                           \
    {                                                                                                      \
        return e2.visible() ? masked_value(e1 OP e2.value()) : masked<decltype(e1 OP e2.value())>();       \
    }

#define DEFINE_UNARY_OPERATOR(OP)                                                     \
    template <class T, class B>                                                       \
    inline xmasked_value<std::decay_t<T>> OP(const xmasked_value<T, B>& e)            \
    {                                                                                 \
        using std::OP;                                                                \
        return e.visible() ? masked_value(OP(e.value())) : masked<std::decay_t<T>>(); \
    }

#define DEFINE_UNARY_BOOL_OPERATOR(OP)                                                        \
    template <class T, class B>                                                               \
    inline auto OP(const xmasked_value<T, B>& e)                                              \
    {                                                                                         \
        using std::OP;                                                                        \
        return e.visible() ? masked_value(OP(e.value())) : masked<decltype(OP(e.value()))>(); \
    }

#define DEFINE_BINARY_OPERATOR(OP)                                                                       \
    template <class T1, class B1, class T2, class B2>                                                    \
    inline auto OP(const xmasked_value<T1, B1>& e1, const xmasked_value<T2, B2>& e2)                     \
    {                                                                                                    \
        using std::OP;                                                                                   \
        return e1.visible() && e2.visible() ?                                                            \
            masked_value(OP(e1.value(), e2.value())) :                                                   \
            masked<decltype(OP(e1.value(), e2.value()))>();                                              \
    }                                                                                                    \
                                                                                                         \
    template <class T1, class B1, class T2>                                                              \
    inline auto OP(const xmasked_value<T1, B1>& e1, const T2& e2)                                        \
    {                                                                                                    \
        using std::OP;                                                                                   \
        return e1.visible() ? masked_value(OP(e1.value(), e2)) : masked<decltype(OP(e1.value(), e2))>(); \
    }                                                                                                    \
                                                                                                         \
    template <class T1, class T2, class B2>                                                              \
    inline auto OP(const T1& e1, const xmasked_value<T2, B2>& e2)                                        \
    {                                                                                                    \
        using std::OP;                                                                                   \
        return e2.visible() ? masked_value(OP(e1, e2.value())) : masked<decltype(OP(e1, e2.value()))>(); \
    }

#define DEFINE_TERNARY_OPERATOR_MMM(OP)                                                                               \
    template <class T1, class B1, class T2, class B2, class T3, class B3>                                             \
    inline auto OP(const xmasked_value<T1, B1>& e1, const xmasked_value<T2, B2>& e2, const xmasked_value<T3, B3>& e3) \
    {                                                                                                                 \
        using std::OP;                                                                                                \
        return (e1.visible() && e2.visible() && e3.visible()) ?                                                       \
                masked_value(OP(e1.value(), e2.value(), e3.value())) :                                                \
                masked<decltype(OP(e1.value(), e2.value(), e3.value()))>();                                           \
    }

#define DEFINE_TERNARY_OPERATOR_MMT(OP)                                                              \
    template <class T1, class B1, class T2, class B2, class T3>                                      \
    inline auto OP(const xmasked_value<T1, B1>& e1, const xmasked_value<T2, B2>& e2, const T3& e3)   \
    {                                                                                                \
        using std::OP;                                                                               \
        return (e1.visible() && e2.visible()) ?                                                      \
                masked_value(OP(e1.value(), e2.value(), e3)) :                                       \
                masked<decltype(OP(e1.value(), e2.value(), e3))>();                                  \
    }

#define DEFINE_TERNARY_OPERATOR_MTM(OP)                                                            \
    template <class T1, class B1, class T2, class T3, class B3>                                    \
    inline auto OP(const xmasked_value<T1, B1>& e1, const T2& e2, const xmasked_value<T3, B3>& e3) \
    {                                                                                              \
        using std::OP;                                                                             \
        return (e1.visible() && e3.visible()) ?                                                    \
                masked_value(OP(e1.value(), e2, e3.value())) :                                     \
                masked<decltype(OP(e1.value(), e2, e3.value()))>();                                \
    }

#define DEFINE_TERNARY_OPERATOR_TMM(OP)                                                            \
    template <class T1, class T2, class B2, class T3, class B3>                                    \
    inline auto OP(const T1& e1, const xmasked_value<T2, B2>& e2, const xmasked_value<T3, B3>& e3) \
    {                                                                                              \
        using std::OP;                                                                             \
        return (e2.visible() && e3.visible()) ?                                                    \
                masked_value(OP(e1, e2.value(), e3.value())) :                                     \
                masked<decltype(OP(e1, e2.value(), e3.value()))>();                                \
    }

#define DEFINE_TERNARY_OPERATOR_TTM(OP)                                         \
    template <class T1, class T2, class T3, class B3>                           \
    inline auto OP(const T1& e1, const T2& e2, const xmasked_value<T3, B3>& e3) \
    {                                                                           \
        using std::OP;                                                          \
        return e3.visible() ?                                                   \
            masked_value(OP(e1, e2, e3.value())) :                              \
            masked<decltype(OP(e1, e2, e3.value()))>();                         \
    }

#define DEFINE_TERNARY_OPERATOR_TMT(OP)                                         \
    template <class T1, class T2, class B2, class T3>                           \
    inline auto OP(const T1& e1, const xmasked_value<T2, B2>& e2, const T3& e3) \
    {                                                                           \
        using std::OP;                                                          \
        return e2.visible() ?                                                   \
            masked_value(OP(e1, e2.value(), e3)) :                              \
            masked<decltype(OP(e1, e2.value(), e3))>();                         \
    }

#define DEFINE_TERNARY_OPERATOR_MTT(OP)                                         \
    template <class T1, class B1, class T2, class T3>                           \
    inline auto OP(const xmasked_value<T1, B1>& e1, const T2& e2, const T3& e3) \
    {                                                                           \
        using std::OP;                                                          \
        return e1.visible() ?                                                   \
            masked_value(OP(e1.value(), e2, e3)) :                              \
            masked<decltype(OP(e1.value(), e2, e3))>();                         \
    }

#define DEFINE_TERNARY_OPERATOR(OP) \
    DEFINE_TERNARY_OPERATOR_MMM(OP) \
                                    \
    DEFINE_TERNARY_OPERATOR_MMT(OP) \
    DEFINE_TERNARY_OPERATOR_MTM(OP) \
    DEFINE_TERNARY_OPERATOR_TMM(OP) \
    DEFINE_TERNARY_OPERATOR_TTM(OP) \
    DEFINE_TERNARY_OPERATOR_TMT(OP) \
    DEFINE_TERNARY_OPERATOR_MTT(OP)

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
#undef DEFINE_BINARY_OPERATOR
#undef DEFINE_UNARY_OPERATOR
#undef DEFINE_UNARY_BOOL_OPERATOR
#undef DEFINE_OPERATOR
#undef DEFINE_BOOL_OPERATOR
}

#endif
