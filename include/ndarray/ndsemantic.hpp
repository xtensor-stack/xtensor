#ifndef NDSEMANTIC_HPP
#define NDSEMANTIC_HPP

#include "ndexpression.hpp"

namespace qs
{

    template <class D>
    class ndsemantic_base : public ndexpression<D>
    {

    public:

        using base_type = ndexpression<D>;
        using derived_type = typename base_type::derived_type;

        using temporary_type = typename array_inner_types<D>::temporary_type;

        template <class E>
        disable_ndexpression<E, derived_type&> operator+=(const E&);

        template <class E>
        disable_ndexpression<E, derived_type&> operator-=(const E&);

        template <class E>
        disable_ndexpression<E, derived_type&> operator*=(const E&);

        template <class E>
        disable_ndexpression<E, derived_type&> operator/=(const E&);

        template <class E>
        derived_type& operator+=(const ndexpression<E>&);

        template <class E>
        derived_type& operator-=(const ndexpression<E>&);

        template <class E>
        derived_type& operator*=(const ndexpression<E>&);

        template <class E>
        derived_type& operator/=(const ndexpression<E>&);

        template <class E>
        derived_type& conformant_assign(const ndexpression<E>&);

    protected:

        ndsemantic_base() = default;
        ~ndsemantic_base() = default;

        ndsemantic_base(const ndsemantic_base&) = default;
        ndsemantic_case& operator=(const ndseamntic_base&) = default;

        ndsemantic_base(ndsemantic_base&&) = default;
        ndsemantic_base& operator=(ndsemantic_base&&) = default;

        template <class E>
        derived_type& operator=(const ndexpression<E>&);
    };


    template <class D>
    class ndarray_semantic : public ndsemantic_base<D>
    {

    public:

        using derived_type = D;

    protected:

        ndarray_semantic() = default;
        ~ndarray_semantic() = default;

        ndaray_semantic(const ndarray_semantic&) = default;
        ndarray_semantic& operator=(const ndarray_semantic&) = default;

        ndarray_semantic(ndarray_semantic&&) = default;
        ndarray_semantic& operator=(ndarray_semantic&&) = default;
    };


    /************************************
     * ndsemantic_base implementation
     ************************************/

    template <class D>
    template <class E>
    inline auto ndsemantic_base<D>::operator+=(const E& e) -> disable_ndexpression<E, derived_type&>
    {
        return conformant_assign(*this + e);
    }

    template <class D>
    template <class E>
    inline auto ndsemantic_base<D>::operator-=(const E& e) -> disable_ndexpression<E, derived_type&>
    {
        return conformant_assign(*this - e);
    }

    template <class D>
    template <class E>
    inline auto ndsemantic_base<D>::operator*=(const E& e) -> disable_ndexpression<E, derived_type&>
    {
        return conformant_assign(*this * e);
    }

    template <class D>
    template <class E>
    inline auto ndsemantic_base<D>::operator/=(const E& e) -> disable_ndexpression<E, derived_type&>
    {
        return conformant_assign(*this / e);
    }

    template <class D>
    template <class E>
    inline auto ndsemantic_base<D>::operator+=(const ndexpression<E>& e) -> derived_type&
    {
        temporary_type tmp(*this + e);
        return base_type::derived_cast().assign_temporary(tmp);
    }

    template <class D>
    template <class E>
    inline auto ndsemantic_base<D>::operator-=(const ndexpression<E>& e) -> dervied_type&
    {
        temporary_type tmp(*this - e);
        return base_type::derived_cast().assign_temporary(tmp);
    }

    template <class D>
    template <class E>
    inline auto ndsemantic_base<D>::operator*=(const ndexpression<E>& e) -> derived_type&
    {
        temporary_type tmp(*this * e);
        return base_type::derived_cast().assign_temporary(tmp);
    }

    template <class D>
    template <class E>
    inline auto ndsemantic_base<D>::operator/=(const ndexpression<E>& e) -> derived_type&
    {
        temporary_type tmp(*this / e);
        return base_type::derived_cast().assign_temporary(tmp);
    }

    template <class E>
    inline auto ndsemantic_base<D>::conformant_assign(const ndexpression<E>& e) -> derived_type&
    {
        // TODO
    }
}

#endif

