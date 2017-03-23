/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XOPTIONAL_HPP
#define XOPTIONAL_HPP

#include <type_traits>
#include <utility>

#include "xtensor/xutils.hpp"

namespace xt
{
    /********************
     * optional helpers *
     ********************/

    template <class T, class B>
    auto optional(T&& t, B&& b) noexcept;

    template <class T>
    auto missing() noexcept;

    /*************************
     * xoptional declaration *
     *************************/

    template <class CT, class CB>
    class xoptional;

    namespace detail
    {
        template <class E>
        struct is_xoptional_impl
        {
            static constexpr bool value = false;
        };

        template <class CT, class CB>
        struct is_xoptional_impl<xoptional<CT, CB>> : std::true_type
        {
        };
    }

    template <class E>
    using is_xoptional = detail::is_xoptional_impl<E>;

    template <class E, class R = void>
    using disable_xoptional = typename std::enable_if<!is_xoptional<E>::value, R>::type;

    /**
     * @class xoptional
     * @brief Closure-type based optional handler.
     *
     * The xoptional is an optional proxy. It holds a closure on a value and a closure on a boolean-convertible type.
     *
     * xoptional is different from std::optional
     *
     *  - no `operator->()` that returns a pointer, since pointer to an rvalue may be an issue.
     *  - no `operator*()` that returns a value.
     *
     * The only way to access the underlying value is with the `value` and `value_or` methods.
     *
     *  - no explicit convertion to bool. This may lead to confusion when the underlying value type is boolean too.
     *
     * @tparam CT Closure type for the value.
     * @tparam CB Closure type for the missing flag. A falsy flag means that the value is missing. 
     */
    template <class CT, class CB=bool>
    class xoptional
    {
    public:
        using value_type = std::decay_t<CT>;
        using flag_type = std::decay_t<CB>;

        // Constructors
        xoptional();
        xoptional(const xoptional&) = default;
        xoptional(xoptional&&) = default;

        template <class CTO, class CBO>
        xoptional(const xoptional<CTO, CBO>&);

        template <class CTO, class CBO>
        xoptional(xoptional<CTO, CBO>&&);

        xoptional(const value_type&);
        xoptional(value_type&&);

        xoptional(value_type&&, flag_type&&);
        xoptional(std::add_lvalue_reference_t<CT>, std::add_lvalue_reference_t<CB>);
        xoptional(value_type&&, std::add_lvalue_reference_t<CB>);
        xoptional(std::add_lvalue_reference_t<CT>, flag_type&&);

        // Assignment
        xoptional& operator=(const xoptional&) = default;

        template <class CTO, class CBO>
        xoptional& operator=(const xoptional<CTO, CBO>&);

        template <class CTO, class CBO>
        xoptional& operator=(xoptional<CTO, CBO>&&);

        xoptional& operator=(const value_type&);
        xoptional& operator=(value_type&&);

        // Operators
        template <class CTO, class CBO>
        xoptional& operator+=(const xoptional<CTO, CBO>&);
        template <class CTO, class CBO>
        xoptional& operator-=(const xoptional<CTO, CBO>&);
        template <class CTO, class CBO>
        xoptional& operator*=(const xoptional<CTO, CBO>&);
        template <class CTO, class CBO>
        xoptional& operator/=(const xoptional<CTO, CBO>&);

        template <class T>
        disable_xoptional<T, xoptional&> operator+=(const T&);
        template <class T>
        disable_xoptional<T, xoptional&> operator-=(const T&);
        template <class T>
        disable_xoptional<T, xoptional&> operator*=(const T&);
        template <class T>
        disable_xoptional<T, xoptional&> operator/=(const T&);

        // Access
        std::add_lvalue_reference_t<CT> value() & noexcept;
        std::add_lvalue_reference_t<std::add_const_t<CT>> value() const & noexcept;
        std::conditional_t<std::is_reference<CT>::value, value_type&, value_type> value() && noexcept;
        std::conditional_t<std::is_reference<CT>::value, const value_type&, const value_type> value() const && noexcept;

        template <class U> 
        value_type value_or(U&&) const & noexcept;
        template <class U> 
        value_type value_or(U&&) const && noexcept;

        // Access
        std::add_lvalue_reference_t<CB> has_value() & noexcept;
        std::add_lvalue_reference_t<std::add_const_t<CB>> has_value() const & noexcept;
        std::conditional_t<std::is_reference<CB>::value, flag_type&, value_type> has_value() && noexcept;
        std::conditional_t<std::is_reference<CB>::value, const flag_type&, const flag_type> has_value() const && noexcept;

        // Swap
        void swap(xoptional& other);

    private:
        template <class CTO, class CBO>
        friend class xoptional;

        CT m_value;
        CB m_flag;
    };


    /***************************************
     * optional and missing implementation *
     ***************************************/
    /**
     * @brief Returns an \ref xoptional holding closure types on the specified parameters
     *
     * @tparam t the optional value
     * @tparam b the boolean flag
     */
    template <class T, class B>
    inline auto optional(T&& t, B&& b) noexcept
    {
        using optional_type = xoptional<closure_t<T>, closure_t<B>>;
        return optional_type(std::forward<T>(t), std::forward<B>(b));
    }

    /**
     * @brief Returns an \ref xoptional for a missig value
     */
    template <class T>
    auto missing() noexcept
    {
        return xoptional<T, bool>(T(), false);
    }

    /****************************
     * xoptional implementation *
     ****************************/

    // Constructors
    template <class CT, class CB>
    xoptional<CT, CB>::xoptional() : m_value(), m_flag(false)
    {
    }

    template <class CT, class CB>
    template <class CTO, class CBO>
    xoptional<CT, CB>::xoptional(const xoptional<CTO, CBO>& opt) : m_value(opt.m_value), m_flag(opt.m_flag)
    {
    }

    template <class CT, class CB>
    template <class CTO, class CBO>
    xoptional<CT, CB>::xoptional(xoptional<CTO, CBO>&& opt) : m_value(std::move(opt.m_value)), m_flag(std::move(opt.m_flag))
    {
    }

    template <class CT, class CB>
    xoptional<CT, CB>::xoptional(const value_type& value) : m_value(value), m_flag(true)
    {
    }

    template <class CT, class CB>
    xoptional<CT, CB>::xoptional(value_type&& value) : m_value(value), m_flag(true)
    {
    }

    template <class CT, class CB>
    xoptional<CT, CB>::xoptional(value_type&& value, flag_type&& flag): m_value(std::move(value)), m_flag(std::move(flag))
    {
    }

    template <class CT, class CB>
    xoptional<CT, CB>::xoptional(std::add_lvalue_reference_t<CT> value, std::add_lvalue_reference_t<CB> flag) : m_value(value), m_flag(flag)
    {
    }

    template <class CT, class CB>
    xoptional<CT, CB>::xoptional(value_type&& value, std::add_lvalue_reference_t<CB> flag) : m_value(std::move(value)), m_flag(flag)
    {
    }

    template <class CT, class CB>
    xoptional<CT, CB>::xoptional(std::add_lvalue_reference_t<CT> value, flag_type&& flag) : m_value(value), m_flag(std::move(flag))
    {
    }

    // Assignment
    template <class CT, class CB>
    template <class CTO, class CBO>
    auto xoptional<CT, CB>::operator=(const xoptional<CTO, CBO>& rhs) -> xoptional&
    {
        m_flag = rhs.m_flag;
        m_value = rhs.m_value;
        return *this;
    }

    template <class CT, class CB>
    template <class CTO, class CBO>
    auto xoptional<CT, CB>::operator=(xoptional<CTO, CBO>&& rhs) -> xoptional&
    {
        m_flag = std::move(rhs.m_flag);
        m_value = std::move(rhs.m_value);
        return *this;
    }

    template <class CT, class CB>
    auto xoptional<CT, CB>::operator=(const value_type& value) -> xoptional&
    {
        m_flag = true;
        m_value = value;
        return *this;
    }

    template <class CT, class CB>
    auto xoptional<CT, CB>::operator=(value_type&& value) -> xoptional&
    {
        m_flag = true;
        m_value = std::move(value);
        return *this;
    }

    // Operators
    template <class CT, class CB>
    template <class CTO, class CBO>
    auto xoptional<CT, CB>::operator+=(const xoptional<CTO, CBO>& rhs) -> xoptional&
    {
        m_flag = m_flag && rhs.m_flag;
        if(m_flag)
            m_value += rhs.m_value;
        return *this;
    }

    template <class CT, class CB>
    template <class CTO, class CBO>
    auto xoptional<CT, CB>::operator-=(const xoptional<CTO, CBO>& rhs) -> xoptional&
    {
        m_flag = m_flag && rhs.m_flag;
        if(m_flag)
            m_value -= rhs.m_value;
        return *this;
    }

    template <class CT, class CB>
    template <class CTO, class CBO>
    auto xoptional<CT, CB>::operator*=(const xoptional<CTO, CBO>& rhs) -> xoptional&
    {
        m_flag = m_flag && rhs.m_flag;
        if(m_flag)
            m_value *= rhs.m_value;
        return *this;
    }

    template <class CT, class CB>
    template <class CTO, class CBO>
    auto xoptional<CT, CB>::operator/=(const xoptional<CTO, CBO>& rhs) -> xoptional&
    {
        m_flag = m_flag && rhs.m_flag;
        if(m_flag)
            m_value /= rhs.m_value;
        return *this;
    }

    template <class CT, class CB>
    template <class T>
    auto xoptional<CT, CB>::operator+=(const T& rhs) -> disable_xoptional<T, xoptional&>
    {
        if(m_flag)
            m_value += rhs;
        return *this;
    }

    template <class CT, class CB>
    template <class T>
    auto xoptional<CT, CB>::operator-=(const T& rhs) -> disable_xoptional<T, xoptional&>
    {
        if(m_flag)
            m_value -= rhs;
        return *this;
    }

    template <class CT, class CB>
    template <class T>
    auto xoptional<CT, CB>::operator*=(const T& rhs) -> disable_xoptional<T, xoptional&> 
    {
        if(m_flag)
            m_value *= rhs;
        return *this;
    }

    template <class CT, class CB>
    template <class T>
    auto xoptional<CT, CB>::operator/=(const T& rhs) -> disable_xoptional<T, xoptional&> 
    {
        if(m_flag)
            m_value /= rhs;
        return *this;
    }

    // Access
    template <class CT, class CB>
    auto xoptional<CT, CB>::value() & noexcept -> std::add_lvalue_reference_t<CT>
    {
        return m_value;
    }

    template <class CT, class CB>
    auto xoptional<CT, CB>::value() const & noexcept -> std::add_lvalue_reference_t<std::add_const_t<CT>> 
    {
        return m_value;
    }

    template <class CT, class CB>
    auto xoptional<CT, CB>::value() && noexcept -> std::conditional_t<std::is_reference<CT>::value, value_type&, value_type> 
    {
        return m_value;
    }

    template <class CT, class CB>
    auto xoptional<CT, CB>::value() const && noexcept -> std::conditional_t<std::is_reference<CT>::value, const value_type&, const value_type> 
    {
        return m_value;
    }

    template <class CT, class CB>
    template <class U> 
    auto xoptional<CT, CB>::value_or(U&& default_value) const & noexcept -> value_type 
    {
        return m_flag ? m_value : std::forward<U>(default_value);
    }

    template <class CT, class CB>
    template <class U> 
    auto xoptional<CT, CB>::value_or(U&& default_value) const && noexcept -> value_type
    {
        return m_flag ? m_value : std::forward<U>(default_value);
    }

    // Access
    template <class CT, class CB>
    auto xoptional<CT, CB>::has_value() & noexcept -> std::add_lvalue_reference_t<CB>
    {
        return m_flag;
    }

    template <class CT, class CB>
    auto xoptional<CT, CB>::has_value() const & noexcept -> std::add_lvalue_reference_t<std::add_const_t<CB>>
    {
        return m_flag;
    }

    template <class CT, class CB>
    auto xoptional<CT, CB>::has_value() && noexcept -> std::conditional_t<std::is_reference<CB>::value, flag_type&, value_type> 
    {
        return m_flag;
    }

    template <class CT, class CB>
    auto xoptional<CT, CB>::has_value() const && noexcept-> std::conditional_t<std::is_reference<CB>::value, const flag_type&, const flag_type>
    {
        return m_flag;
    }

    // Swap
    template <class CT, class CB>
    void xoptional<CT, CB>::swap(xoptional& other)
    {
        std::swap(m_value, other.m_flag);
        std::swap(m_flag, other.m_flag);
    }

    template <class T, class B>
    inline auto operator+(const xoptional<T, B>& e) noexcept
        -> xoptional<std::decay_t<T>>
    {
        return e;
    }

    // Operations
    template <class T, class B>
    inline auto operator-(const xoptional<T, B>& e) noexcept
        -> xoptional<std::decay_t<T>>
    {
        using value_type = std::decay_t<T>;
        if (e.has_value())
        {
            return -e.value();
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class B1, class T2, class B2>
    inline auto operator+(const xoptional<T1, B1>& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e1.has_value() && e2.has_value())
        {
            return e1.value() + e2.value();
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class T2, class B2>
    inline auto operator+(const T1& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e2.has_value())
        {
            return e1 + e2.value();
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class B1, class T2>
    inline auto operator+(const xoptional<T1, B1>& e1, const T2& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e1.has_value())
        {
            return e1.value() + e2;
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class B1, class T2, class B2>
    inline auto operator-(const xoptional<T1, B1>& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e1.has_value() && e2.has_value())
        {
            return e1.value() - e2.value();
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class T2, class B2>
    inline auto operator-(const T1& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e2.has_value())
        {
            return e1 - e2.value();
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class B1, class T2>
    inline auto operator-(const xoptional<T1, B1>& e1, const T2& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e1.has_value())
        {
            return e1.value() - e2;
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class B1, class T2, class B2>
    inline auto operator*(const xoptional<T1, B1>& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e1.has_value() && e2.has_value())
        {
            return e1.value() * e2.value();
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class T2, class B2>
    inline auto operator*(const T1& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e2.has_value())
        {
            return e1 * e2.value();
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class B1, class T2>
    inline auto operator*(const xoptional<T1, B1>& e1, const T2& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e1.has_value())
        {
            return e1.value() * e2;
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class B1, class T2, class B2>
    inline auto operator/(const xoptional<T1, B1>& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e1.has_value() && e2.has_value())
        {
            return e1.value() / e2.value();
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class T2, class B2>
    inline auto operator/(const T1& e1, const xoptional<T2, B2>& e2) noexcept
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e2.has_value())
        {
            return e1 / e2.value();
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class B1, class T2>
    inline auto operator/(const xoptional<T1, B1>& e1, const T2& e2) noexcept
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e1.has_value())
        {
            return e1.value() / e2;
        }
        else
        {
            return missing<value_type>(); 
        }
    }

    template <class T1, class B1, class T2, class B2>
    inline auto operator||(const xoptional<T1, B1>& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e1.has_value() && e2.has_value())
        {
            return e1.value() || e2.value();
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class T2, class B2>
    inline auto operator||(const T1& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e2.has_value())
        {
            return e1 || e2.value();
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class B1, class T2>
    inline auto operator||(const xoptional<T1, B1>& e1, const T2& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e1.has_value())
        {
            return e1.value() || e2;
        }
        else
        {
            return missing<value_type>();
        }
    }


    template <class T1, class B1, class T2, class B2>
    inline auto operator&&(const xoptional<T1, B1>& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e1.has_value() && e2.has_value())
        {
            return e1.value() && e2.value();
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class T2, class B2>
    inline auto operator&&(const T1& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e2.has_value())
        {
            return e1 && e2.value();
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T1, class B1, class T2>
    inline auto operator&&(const xoptional<T1, B1>& e1, const T2& e2) noexcept
       -> xoptional<std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e1.has_value())
        {
            return e1.value() && e2;
        }
        else
        {
            return missing<value_type>();
        }
    }

    template <class T, class B>
    inline auto operator!(const xoptional<T, B>& e) noexcept
       -> xoptional<bool>
    {
        if (e.has_value())
        {
            return !e.value();
        }
        else
        {
            return missing<bool>();
        }
    }

    template <class T1, class B1, class T2, class B2>
    inline auto operator<(const xoptional<T1, B1>& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<bool>
    {
        if (e1.has_value() && e2.has_value())
        {
            return e1.value() < e2.value();
        }
        else
        {
            return missing<bool>(); 
        }
    }

    template <class T1, class T2, class B2>
    inline auto operator<(const T1& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<bool>
    {
        if (e2.has_value())
        {
            return e1 < e2.value();
        }
        else
        {
            return missing<bool>(); 
        }
    }

    template <class T1, class B1, class T2>
    inline auto operator<(const xoptional<T1, B1>& e1, const T2& e2) noexcept
       -> xoptional<bool>
    {
        if (e1.has_value())
        {
            return e1.value() < e2;
        }
        else
        {
            return missing<bool>(); 
        }
    }

    template <class T1, class B1, class T2, class B2>
    inline auto operator<=(const xoptional<T1, B1>& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<bool>
    {
        using value_type = std::common_type_t<std::decay_t<T1>, std::decay_t<T2>>;
        if (e1.has_value() && e2.has_value())
        {
            return e1.value() <= e2.value();
        }
        else
        {
            return missing<bool>();
        }
    }

    template <class T1, class T2, class B2>
    inline auto operator<=(const T1& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<bool>
    {
        if (e2.has_value())
        {
            return e1 <= e2.value();
        }
        else
        {
            return missing<bool>();
        }
    }

    template <class T1, class B1, class T2>
    inline auto operator<=(const xoptional<T1, B1>& e1, const T2& e2) noexcept
       -> xoptional<bool>
    {
        if (e1.has_value())
        {
            return e1.value() <= e2;
        }
        else
        {
            return missing<bool>();
        }
    }

    template <class T1, class B1, class T2, class B2>
    inline auto operator>(const xoptional<T1, B1>& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<bool>
    {
        if (e1.has_value() && e2.has_value())
        {
            return e1.value() > e2.value();
        }
        else
        {
            return missing<bool>();
        }
    }

    template <class T1, class T2, class B2>
    inline auto operator>(const T1& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<bool>
    {
        if (e2.has_value())
        {
            return e1 > e2.value();
        }
        else
        {
            return missing<bool>();
        }
    }

    template <class T1, class B1, class T2>
    inline auto operator>(const xoptional<T1, B1>& e1, const T2& e2) noexcept
       -> xoptional<bool>
    {
        if (e1.has_value())
        {
            return e1.value() > e2;
        }
        else
        {
            return missing<bool>();
        }
    }

    template <class T1, class B1, class T2, class B2>
    inline auto operator>=(const xoptional<T1, B1>& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<bool>
    {
        if (e1.has_value() && e2.has_value())
        {
            return e1.value() >= e2.value();
        }
        else
        {
            return missing<bool>();
        }
    }

    template <class T1, class T2, class B2>
    inline auto operator>=(const T1& e1, const xoptional<T2, B2>& e2) noexcept
       -> xoptional<bool>
    {
        if (e2.has_value())
        {
            return e1 >= e2.value();
        }
        else
        {
            return missing<bool>(); 
        }
    }

    template <class T1, class B1, class T2>
    inline auto operator>=(const xoptional<T1, B1>& e1, const T2& e2) noexcept
       -> xoptional<bool>
    {
        if (e1.has_value())
        {
            return e1.value() >= e2;
        }
        else
        {
            return missing<bool>(); 
        }
    }

}

#endif
