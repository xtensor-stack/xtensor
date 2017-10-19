/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ANY_HPP
#define XTENSOR_ANY_HPP

#include <stdexcept>
#include <utility>

#include "xtl/xany.hpp"
#include "xtl/xclosure.hpp"

#include "xtensor/xexpression.hpp"

namespace xt
{
    class xany;

    inline void swap(xany& lhs, xany& rhs);

    namespace detail
    {
        class xany_impl;
    }

    /********************
     * xany declaration *
     ********************/

    class xany
    {
    public:

        using implementation_type = detail::xany_impl;

        xany();
        ~xany();
        xany(const xany& rhs);
        xany(xany&& rhs);
        template <class D>
        xany(const xexpression<D>& rhs);
        template <class D>
        xany(xexpression<D>&& rhs);
        xany(implementation_type* holder);
        xany& operator=(const xany& rhs);
        xany& operator=(xany&& rhs);
        template <class D>
        xany& operator=(const xexpression<D>& rhs);
        template <class D>
        xany& operator=(xexpression<D>&& rhs);
        void swap(xany& rhs);

        xtl::any value() &;
        const xtl::any value() const &;
        xtl::any value() &&;

        template <class D>
        D& get() &;

        template <class D>
        const D& get() const &;

        template <class D>
        D get() &&;

    private:

        void check_holder() const;

        implementation_type* p_holder;
    };

    /***********************
     * xany implementation *
     ***********************/

    namespace detail
    {
        class xany_impl
        {
        public:
      
            xany_impl() = default;
            xany_impl(xany_impl&&) = delete;
            xany_impl& operator=(const xany_impl&) = delete;
            xany_impl& operator=(xany_impl&&) = delete;
            virtual xany_impl* clone() const = 0;
            virtual ~xany_impl() = default;

            virtual xtl::any value() & = 0;
            virtual const xtl::any value() const & = 0;
            virtual xtl::any value() && = 0;
        
        protected:

            xany_impl(const xany_impl&) = default;
        };

        template <class D>
        class xany_owning : public xany_impl
        {
        public:
        
            using base_type = xany_impl;

            xany_owning(const xexpression<D>& value)
                : base_type(),
                  m_value(value.derived_cast())
            {
            }

            xany_owning(xexpression<D>&& value)
                : base_type(),
                  m_value(std::move(value.derived_cast()))
            {
            }

            virtual ~xany_owning()
            {
            }

            virtual base_type* clone() const override
            {
                return new xany_owning(*this);
            }

            virtual xtl::any value() & override
            {
                return xtl::closure(m_value);
            }

            virtual const xtl::any value() const & override
            {
                return xtl::closure(m_value);
            }

            virtual xtl::any value() && override
            {
                return xtl::closure(std::move(m_value));
            }

        private:

            xany_owning(const xany_owning&) = default;
            xany_owning(xany_owning&&) = default;
            xany_owning& operator=(const xany_owning&) = default;
            xany_owning& operator=(xany_owning&&) = default;

            D m_value;
        };
    }

    /***********************
     * xany implementation *
     ***********************/

    inline xany::xany()
        : p_holder(nullptr)
    {
    }

    inline xany::xany(detail::xany_impl* holder)
        : p_holder(holder)
    {
    }

    inline xany::~xany()
    {
        delete p_holder;
    }

    inline xany::xany(const xany& rhs)
        : p_holder(rhs.p_holder ? rhs.p_holder->clone() : nullptr)
    {
    }

    template <class D>
    xany::xany(const xexpression<D>& rhs)
        : xany(new detail::xany_owning<D>(rhs))
    {
    }

    template <class D>
    xany::xany(xexpression<D>&& rhs)
        : xany(new detail::xany_owning<D>(std::move(rhs)))
    {
    }

    inline xany::xany(xany&& rhs)
        : p_holder(rhs.p_holder)
    {
        rhs.p_holder = nullptr;
    }

    inline xany& xany::operator=(const xany& rhs)
    {
        using std::swap;
        xany tmp(rhs);
        swap(*this, tmp);
        return *this;
    }

    inline xany& xany::operator=(xany&& rhs)
    {
        using std::swap;
        xany tmp(std::move(rhs));
        swap(*this, tmp);
        return *this;
    }

    template <class D>
    xany& xany::operator=(const xexpression<D>& rhs)
    {
        using std::swap;
        xany tmp(rhs);
        swap(tmp, *this);
        return *this;
    }

    template <class D>
    xany& xany::operator=(xexpression<D>&& rhs)
    {
        using std::swap;
        xany tmp(rhs);
        swap(tmp, *this);
        return *this;
    }

    inline void xany::swap(xany& rhs)
    {
        std::swap(p_holder, rhs.p_holder);
    }

    inline xtl::any xany::value() &
    {
        check_holder();
        return p_holder->value();
    }

    inline const xtl::any xany::value() const &
    {
        check_holder();
        return p_holder->value();
    }

    inline xtl::any xany::value() &&
    {
        check_holder();
        return p_holder->value();
    }

    template <class D>
    D& xany::get() &
    {
        return xtl::any_cast<xtl::closure_wrapper<D&>>(this->value()).get();
    }

    template <class D>
    const D& xany::get() const &
    {
        return xtl::any_cast<xtl::closure_wrapper<const D&>>(this->value()).get();
    }

    inline void xany::check_holder() const
    {
        if (p_holder == nullptr)
        {
            throw std::runtime_error("The holder does not contain an expression");
        }
    }

    inline void swap(xany& lhs, xany& rhs)
    {
        lhs.swap(rhs);
    }
}
#endif
