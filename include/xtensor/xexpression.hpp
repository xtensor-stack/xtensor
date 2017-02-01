/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XEXPRESSION_HPP
#define XEXPRESSION_HPP

#include <type_traits>
#include <cstddef>
#include <vector>

#include "xutils.hpp"

namespace xt
{

    using xindex = std::vector<std::size_t>;

    /***************************
     * xexpression declaration *
     ***************************/

    /**
     * @class xexpression
     * @brief Base class for xexpressions
     *
     * The xexpression class is the base class for all classes representing an expression
     * that can be evaluated to a multidimensional container with tensor semantic.
     * Functions that can apply to any xexpression regardless of its specific type should take a
     * xexpression argument.
     *
     * \tparam E The derived type.
     *
     */
    template <class D>
    class xexpression
    {

    public:

        using derived_type = D;

        derived_type& derived_cast() noexcept;
        const derived_type& derived_cast() const noexcept;

    protected:

        xexpression() = default;
        ~xexpression() = default;

        xexpression(const xexpression&) = default;
        xexpression& operator=(const xexpression&) = default;

        xexpression(xexpression&&) = default;
        xexpression& operator=(xexpression&&) = default;
    };

    /******************************
     * xexpression implementation *
     ******************************/

    /**
     * @name Downcast functions
     */
    //@{
    /**
     * Returns a reference to the actual derived type of the xexpression.
     */
    template <class D>
    inline auto xexpression<D>::derived_cast() noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    /**
     * Returns a constant reference to the actual derived type of the xexpression.
     */
    template <class D>
    inline auto xexpression<D>::derived_cast() const noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }
    //@}

    namespace detail
    {
        template <class E>
        struct is_xexpression_impl : std::is_base_of<xexpression<E>, E>
        {
        };

        template <class E>
        struct is_xexpression_impl<xexpression<E>> : std::true_type
        {
        };
    }

    template <class E>
    using is_xexpression = detail::is_xexpression_impl<E>;

    template <class E, class R = void>
    using disable_xexpression = typename std::enable_if<!is_xexpression<E>::value, R>::type;

    template <class... E>
    using has_xexpression = or_<is_xexpression<E>...>;

    /************
     * xclosure *
     ************/

    template <class T>
    class xscalar;

    template <class E, class EN = void>
    struct xclosure
    {
        using xexpression_type = typename std::decay<E>::type;
        using xclosure_type = typename std::conditional<std::is_lvalue_reference<E>::value,
                                                        const xexpression_type&,
                                                        xexpression_type>::type;
    };

    template <class E>
    struct xclosure<E, disable_xexpression<typename std::decay<E>::type>>
    {
        using xexpression_type = xscalar<typename std::decay<E>::type>;
        using xclosure_type = xexpression_type;
    };
    
    /******************
     * get_value_type *
     ******************/

    namespace detail
    {
        template <class E, class enable = void>
        struct get_value_type_impl
        {
            using type = E;
        };

        template <class E>
        struct get_value_type_impl<E, std::enable_if_t<is_xexpression<E>::value>>
        {
            using type = typename E::value_type;
        };
    }

    template <class E>
    using get_value_type = typename detail::get_value_type_impl<E>::type;
    
    /***************
     * get_element *
     ***************/

    namespace detail
    {
        template <class E>
        inline typename E::reference get_element(E& e)
        {
            return e();
        }

        template <class E, class S, class... Args>
        inline typename E::reference get_element(E& e, S i, Args... args)
        {
            if(sizeof...(Args) >= e.dimension())
                return get_element(e, args...);
            return e(i, args...);
        }

        template <class E>
        inline typename E::const_reference get_element(const E& e)
        {
            return e();
        }

        template <class E, class S, class... Args>
        inline typename E::const_reference get_element(const E& e, S i, Args... args)
        {
            if(sizeof...(Args) >= e.dimension())
                return get_element(e, args...);
            return e(i, args...);
        }
    }

}

#endif

