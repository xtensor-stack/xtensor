#ifndef XSEMANTIC_HPP
#define XSEMANTIC_HPP

#include "xexpression.hpp"

namespace qs
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
        derived_type& conformant_assign(const xexpression<E>&);

    protected:

        xsemantic_base() = default;
        ~xsemantic_base() = default;

        xsemantic_base(const xsemantic_base&) = default;
        xsemantic_base& operator=(const xsemantic_base&) = default;

        xsemantic_base(xsemantic_base&&) = default;
        xsemantic_base& operator=(xsemantic_base&&) = default;

        template <class E>
        derived_type& operator=(const xexpression<E>&);
    };


    template <class D>
    class xarray_semantic : public xsemantic_base<D>
    {

    public:

        using derived_type = D;

    protected:

        xarray_semantic() = default;
        ~xarray_semantic() = default;

        xarray_semantic(const xarray_semantic&) = default;
        xarray_semantic& operator=(const xarray_semantic&) = default;

        xarray_semantic(xarray_semantic&&) = default;
        xarray_semantic& operator=(xarray_semantic&&) = default;
    };


    /************************************
     * xsemantic_base implementation
     ************************************/

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator+=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return conformant_assign(*this + e);
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator-=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return conformant_assign(*this - e);
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator*=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return conformant_assign(*this * e);
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator/=(const E& e) -> disable_xexpression<E, derived_type&>
    {
        return conformant_assign(*this / e);
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator+=(const xexpression<E>& e) -> derived_type&
    {
        temporary_type tmp(*this + e);
        return base_type::derived_cast().assign_temporary(tmp);
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator-=(const xexpression<E>& e) -> derived_type&
    {
        temporary_type tmp(*this - e);
        return base_type::derived_cast().assign_temporary(tmp);
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator*=(const xexpression<E>& e) -> derived_type&
    {
        temporary_type tmp(*this * e);
        return base_type::derived_cast().assign_temporary(tmp);
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::operator/=(const xexpression<E>& e) -> derived_type&
    {
        temporary_type tmp(*this / e);
        return base_type::derived_cast().assign_temporary(tmp);
    }

    template <class D>
    template <class E>
    inline auto xsemantic_base<D>::conformant_assign(const xexpression<E>& e) -> derived_type&
    {
        // TODO
    }
}

#endif

