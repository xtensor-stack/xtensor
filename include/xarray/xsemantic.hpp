/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSEMANTIC_HPP
#define XSEMANTIC_HPP

#include "xexpression.hpp"
#include "xassign.hpp"

namespace xt
{

    template <class D>
    class xsemantic_base : public xexpression<D>
    {

    public:

        using base_type = xexpression<D>;
        using derived_type = typename base_type::derived_type;

        using temporary_type = typename array_inner_types<D>::temporary_type;

        template <class E>
        disable_xexpression<E, derived_type&> operator+=(const E&);

        template <class E>
        disable_xexpression<E, derived_type&> operator-=(const E&);

        template <class E>
        disable_xexpression<E, derived_type&> operator*=(const E&);

        template <class E>
        disable_xexpression<E, derived_type&> operator/=(const E&);

        template <class E>
        derived_type& operator+=(const xexpression<E>&);

        template <class E>
        derived_type& operator-=(const xexpression<E>&);

        template <class E>
        derived_type& operator*=(const xexpression<E>&);

        template <class E>
        derived_type& operator/=(const xexpression<E>&);

        template <class E>
        derived_type& assign(const xexpression<E>&);

        template <class E>
        derived_type& plus_assign(const xexpression<E>&);

        template <class E>
        derived_type& minus_assign(const xexpression<E>&);

        template <class E>
        derived_type& multiplies_assign(const xexpression<E>&);

        template <class E>
        derived_type& divides_assign(const xexpression<E>&);

    protected:

        xsemantic_base() = default;
        ~xsemantic_base() = default;

        xsemantic_base(const xsemantic_base&) = default;
        xsemantic_base& operator=(const xsemantic_base&) = default;

        xsemantic_base(xsemantic_base&&) = default;
        xsemantic_base& operator=(xsemantic_base&&) = default;

        template <class E>
        derived_type& operator=(const xexpression<E>&);

    private:

        template <class E, class F>
        derived_type& scalar_computed_assign(const E& e, F&& f);
    };


    template <class D>
    class xarray_semantic : public xsemantic_base<D>
    {

    public:

        using base_type = xsemantic_base<D>;
        using derived_type = D;
        using temporary_type = typename base_type::temporary_type;

        derived_type& assign_temporary(temporary_type&);

    protected:

        xarray_semantic() = default;
        ~xarray_semantic() = default;

        xarray_semantic(const xarray_semantic&) = default;
        xarray_semantic& operator=(const xarray_semantic&) = default;

        xarray_semantic(xarray_semantic&&) = default;
        xarray_semantic& operator=(xarray_semantic&&) = default;

        template <class E>
        derived_type& operator=(const xexpression<E>&);
    };


    template <class D>
    class xadaptor_semantic : public xsemantic_base<D>
    {

    public:

        using base_type = xsemantic_base<D>;
        using derived_type = D;
        using temporary_type = typename base_type::temporary_type;

        derived_type& assign_temporary(temporary_type&);

    protected:

        xadaptor_semantic() = default;
        ~xadaptor_semantic() = default;

        xadaptor_semantic(const xadaptor_semantic&) = default;
        xadaptor_semantic& operator=(const xadaptor_semantic&) = default;

        xadaptor_semantic(xadaptor_semantic&&) = default;
        xadaptor_semantic& operator=(xadaptor_semantic&&) = default;

        template <class E>
        derived_type& operator=(const xexpression<E>&);
    };

    /*********************************
     * xsemantic_base implementation *
     *********************************/

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator+=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return scalar_computed_assign(e, std::plus<>());
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator-=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return scalar_computed_assign(e, std::minus<>());
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator*=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return scalar_computed_assign(e, std::multiplies<>());
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator/=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return scalar_computed_assign(e, std::divides<>());
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator+=(const xexpression<E>& e) -> derived_type&
    {
        return operator=(base_type::derived_cast() + e.derived_cast());
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator-=(const xexpression<E>& e) -> derived_type&
    {
        return operator=(base_type::derived_cast() - e.derived_cast());
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator*=(const xexpression<E>& e) -> derived_type&
    {
        return operator=(base_type::derived_cast() * e.derived_cast());
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator/=(const xexpression<E>& e) -> derived_type&
    {
        return operator=(base_type::derived_cast() / e.derived_cast());
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::assign(const xexpression<E>& e) -> derived_type&
    {
        assign_xexpression(*this, e);
        return base_type::derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::plus_assign(const xexpression<E>& e) -> derived_type&
    {
        auto expr = (base_type::derived_cast() + e.derived_cast());
        computed_assign_xexpression(*this, expr);
        return base_type::derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::minus_assign(const xexpression<E>& e) -> derived_type&
    {
        auto expr = (base_type::derived_cast() - e.derived_cast());
        computed_assign_xexpression(*this, expr);
        return base_type::derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::multiplies_assign(const xexpression<E>& e) -> derived_type&
    {
        auto expr = (base_type::derived_cast() * e.derived_cast());
        computed_assign_xexpression(*this, expr);
        return base_type::derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::divides_assign(const xexpression<E>& e) -> derived_type&
    {
        auto expr = (base_type::derived_cast() / e.derived_cast());
        computed_assign_xexpression(*this, expr);
        return base_type::derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator=(const xexpression<E>& e) -> derived_type&
    {
        temporary_type tmp(e);
        return base_type::derived_cast().assign_temporary(tmp);
    }

    template <class D>
    template <class E, class F>
    inline auto xsemantic_base<D>::scalar_computed_assign(const E& e, F&& f) -> derived_type&
    {
        derived_type& d = base_type::derived_cast();
        std::transform(d.storage_begin(), d.storage_end(), d.storage_begin(),
                [e, &f](const auto& v) { return f(v, e); });
        return d;
    }

    /**********************************
     * xarray_semantic implementation *
     **********************************/

    template <class D>
    inline auto xarray_semantic<D>::assign_temporary(temporary_type& tmp) -> derived_type&
    {
        using std::swap;
        swap(this->derived_cast(), tmp);
        return this->derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xarray_semantic<D>::operator=(const xexpression<E>& e) -> derived_type&
    {
        return base_type::operator=(e);
    }

    /**************************************
     * xadaptor_semantic implementation
     **************************************/

    template <class D>
    inline auto xadaptor_semantic<D>::assign_temporary(temporary_type& tmp) -> derived_type&
    {
        this->derived_cast().assign_temporary_impl(tmp);
        return this->derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xadaptor_semantic<D>::operator=(const xexpression<E>& e) -> derived_type&
    {
        return base_type::operator=(e);
    }
}

#endif

