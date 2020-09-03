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

namespace xt
{

    /**********
     * zarray *
     **********/

    class zarray : public xexpression<zarray>
    {
    public:

        using expression_tag = zarray_expression_tag;
        using implementation_ptr = std::unique_ptr<zarray_impl>;

        zarray() = default;
        ~zarray() = default;

        template <class E>
        zarray(E&& e);

        zarray(implementation_ptr&& impl);

        zarray(const zarray& rhs);
        zarray& operator=(const zarray& rhs);

        zarray(zarray&& rhs);
        zarray& operator=(zarray&& rhs);

        void swap(zarray& rhs);

        zarray_impl& get_implementation();
        const zarray_impl& get_implementation() const;

        template <class T>
        xarray<T>& get_array();

        template <class T>
        const xarray<T>& get_array() const;

    private:

        implementation_ptr p_impl;
    };

    /*************************
     * zarray implementation *
     *************************/

    template <class E>
    inline zarray::zarray(E&& e)
        : p_impl(detail::build_zarray(std::forward<E>(e)))
    {
    }

    inline zarray::zarray(implementation_ptr&& impl)
        : p_impl(std::move(impl))
    {
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

    inline zarray& zarray::operator=(zarray&& rhs)
    {
        swap(rhs);
        return *this;
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
}

#endif

