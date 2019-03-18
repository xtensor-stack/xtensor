/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ACCESSIBLE_HPP
#define XTENSOR_ACCESSIBLE_HPP

#include "xexception.hpp"
#include "xstrides.hpp"
#include "xtensor_forward.hpp"

namespace xt
{
    /**
     * @class xaccessible
     * @brief Base class for implementation of common expression access methods.
     *
     * The xaccessible class implements access methods common to all expressions.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xaccessible
     *           provides the interface.
     */
    template <class D>
    class xaccessible
    {
    public:

        using derived_type = D;
        using inner_types = xcontainer_inner_types<D>;
        using reference = typename inner_types::reference;
        using const_reference = typename inner_types::const_reference;
        using size_type = typename inner_types::size_type;

        size_type dimension() const noexcept;
        size_type shape(size_type index) const;

        template <class... Args>
        reference at(Args... args);

        template <class... Args>
        const_reference at(Args... args) const;

        template <class... Args>
        reference periodic(Args... args);

        template <class... Args>
        const_reference periodic(Args... args) const;

        template <class... Args>
        bool in_bounds(Args... args) const;

    protected:

        xaccessible() = default;
        ~xaccessible() = default;

        xaccessible(const xaccessible&) = default;
        xaccessible& operator=(const xaccessible&) = default;

        xaccessible(xaccessible&&) = default;
        xaccessible& operator=(xaccessible&&) = default;

    private:

        derived_type& derived_cast() noexcept;
        const derived_type& derived_cast() const noexcept;
    };

    /******************************
     * xaccessible implementation *
     ******************************/

    /**
     * Returns the number of dimensions of the expression.
     */
    template <class D>
    inline auto xaccessible<D>::dimension() const noexcept -> size_type
    {
        return derived_cast().shape().size();
    }

    /**
     * Returns the i-th dimension of the expression.
     */
    template <class D>
    inline auto xaccessible<D>::shape(size_type index) const -> size_type
    {
        return derived_cast().shape()[index];
    }

    /**
     * Returns a reference to the element at the specified position in the expression,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the expression. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the expression.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class D>
    template <class... Args>
    inline auto xaccessible<D>::at(Args... args) -> reference
    {   
        check_access(derived_cast().shape(), static_cast<size_type>(args)...);
        return derived_cast().operator()(args...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the expression. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the expression.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class D>
    template <class... Args>
    inline auto xaccessible<D>::at(Args... args) const -> const_reference
    {
        check_access(derived_cast().shape(), static_cast<size_type>(args)...);
        return derived_cast().operator()(args...);
    }

    /**
     * Returns a reference to the element at the specified position in the expression,
     * after applying periodicity to the indices (negative and 'overflowing' indices are changed).
     * @param args a list of indices specifying the position in the expression. Indices
     * must be integers, the number of indices should be equal to the number of dimensions
     * of the expression.
     * @exception std::out_of_range if the number of argument is greater than the
     * number of dimensions
     */
    template <class D>
    template <class... Args>
    inline auto xaccessible<D>::periodic(Args... args) -> reference
    {
        normalize_periodic(derived_cast().shape(), args...);
        return derived_cast()(static_cast<size_type>(args)...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression,
     * after applying periodicity to the indices (negative and 'overflowing' indices are changed).
     * @param args a list of indices specifying the position in the expression. Indices
     * must be integers, the number of indices should be equal to the number of dimensions
     * of the expression.
     * @exception std::out_of_range if the number of argument is greater than the
     * number of dimensions
     */
    template <class D>
    template <class... Args>
    inline auto xaccessible<D>::periodic(Args... args) const -> const_reference
    {
        normalize_periodic(derived_cast().shape(), args...);
        return derived_cast()(static_cast<size_type>(args)...);
    }
    
    /**
     * Returns ``true`` only if the the specified position is a valid entry in the expression.
     * @param args a list of indices specifying the position in the expression.
     * @return bool
     */
    template <class D>
    template <class... Args>
    inline bool xaccessible<D>::in_bounds(Args... args) const
    {
        return check_in_bounds(derived_cast().shape(), args...);
    }

    template <class D>
    inline auto xaccessible<D>::derived_cast() noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xaccessible<D>::derived_cast() const noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }
}

#endif

