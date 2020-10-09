/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZARRAY_HPP
#define XTENSOR_ZARRAY_HPP

#include <memory>

#include <xtl/xmultimethods.hpp>

#include "xarray.hpp"
#include "zarray_impl.hpp"
#include "zassign.hpp"

namespace xt
{

    /**********
     * zarray *
     **********/

    class zarray;

    template <>
    struct xcontainer_inner_types<zarray>
    {
        using temporary_type = zarray;
    };

    class zarray : public xcontainer_semantic<zarray>
    {
    public:

        using expression_tag = zarray_expression_tag;
        using semantic_base = xcontainer_semantic<zarray>;
        using implementation_ptr = std::unique_ptr<zarray_impl>;

        zarray() = default;
        ~zarray() = default;

        zarray(implementation_ptr&& impl);
        zarray& operator=(implementation_ptr&& impl);

        zarray(const zarray& rhs);
        zarray& operator=(const zarray& rhs);

        zarray(zarray&& rhs);
        zarray& operator=(zarray&& rhs);

        template <class E>
        zarray(const xexpression<E>& e);

        template <class E>
        zarray(xexpression<E>& e);

        template <class E>
        zarray(xexpression<E>&& e);
        
        template <class E>
        zarray& operator=(const xexpression<E>&);

        void swap(zarray& rhs);

        zarray_impl& get_implementation();
        const zarray_impl& get_implementation() const;

        template <class T>
        xarray<T>& get_array();

        template <class T>
        const xarray<T>& get_array() const;

        const zchunked_array& as_chunked_array() const;

    private:

        template <class E>
        void init_implementation(E&& e, xtensor_expression_tag);

        template <class E>
        void init_implementation(const xexpression<E>& e, zarray_expression_tag);

        implementation_ptr p_impl;
    };

    /*************************
     * zarray implementation *
     *************************/

    template <class E>
    inline void zarray::init_implementation(E&& e, xtensor_expression_tag)
    {
        p_impl = implementation_ptr(detail::build_zarray(std::forward<E>(e)));
    }

    template <class E>
    inline void zarray::init_implementation(const xexpression<E>& e, zarray_expression_tag)
    {
        p_impl = nullptr;
        semantic_base::assign(e);
    }

    inline zarray::zarray(implementation_ptr&& impl)
        : p_impl(std::move(impl))
    {
    }

    inline zarray& zarray::operator=(implementation_ptr&& impl)
    {
        p_impl = std::move(impl);
        return *this;
    }

    inline zarray::zarray(const zarray& rhs)
        : p_impl(rhs.p_impl->clone())
    {
    }

    inline zarray& zarray::operator=(const zarray& rhs)
    {
        zarray tmp(rhs);
        swap(tmp);
        return *this;
    }

    inline zarray::zarray(zarray&& rhs)
        : p_impl(std::move(rhs.p_impl))
    {
    }

    template <class E>
    inline zarray::zarray(const xexpression<E>& e)
    {
        init_implementation(e.derived_cast(), extension::get_expression_tag_t<std::decay_t<E>>());
    }

    template <class E>
    inline zarray::zarray(xexpression<E>& e)
    {
        init_implementation(e.derived_cast(), extension::get_expression_tag_t<std::decay_t<E>>());
    }

    template <class E>
    inline zarray::zarray(xexpression<E>&& e)
    {
        init_implementation(std::move(e).derived_cast(), extension::get_expression_tag_t<std::decay_t<E>>());
    }

    inline zarray& zarray::operator=(zarray&& rhs)
    {
        swap(rhs);
        return *this;
    }

    template <class E>
    inline zarray& zarray::operator=(const xexpression<E>& e)
    {
        return semantic_base::operator=(e);
    }
    
    inline void zarray::swap(zarray& rhs)
    {
        std::swap(p_impl, rhs.p_impl);
    }

    inline zarray_impl& zarray::get_implementation()
    {
        return *p_impl;
    }

    inline const zarray_impl& zarray::get_implementation() const
    {
        return *p_impl;
    }

    template <class T>
    inline xarray<T>& zarray::get_array()
    {
        return dynamic_cast<ztyped_array<T>*>(p_impl.get())->get_array();
    }

    template <class T>
    inline const xarray<T>& zarray::get_array() const
    {
        return dynamic_cast<const ztyped_array<T>*>(p_impl.get())->get_array();
    }

    inline const zchunked_array& zarray::as_chunked_array() const
    {
        return dynamic_cast<const zchunked_array&>(*(p_impl.get()));
    }
}

#endif

