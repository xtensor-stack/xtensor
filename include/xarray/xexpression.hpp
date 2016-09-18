#ifndef XEXPRESSION_HPP
#define XEXPRESSION_HPP

#include <type_traits>
#include "utils.hpp"

namespace qs
{

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


    /*********************************
     * xexpression implementation
     *********************************/

    template <class D>
    inline auto xexpression<D>::derived_cast() noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xexpression<D>::derived_cast() const noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class E>
    using is_xexpression = std::is_base_of<xexpression<E>, E>;
   
    template <class E, class R>
    using disable_xexpression = typename std::enable_if<!is_xexpression<E>::value, R>::type;

    template <class... E>
    using has_xexpression = or_<is_xexpression<E>...>;


    /********************
     * get_expression
     ********************/

    template <class E>
    inline const E& get_xexpression(const xexpression<E>& e) noexcept
    {
        return e.derived_cast();
    }

    template <class E>
    inline disable_xexpression<E, const E&> get_xexpression(const E& e) noexcept
    {
        return e;
    }


    /*********************************
     *  get_closure_type
     *********************************/
    
    template <class T>
    class xscalar;

    namespace detail
    {
        template <class E, class enable=void>
        struct get_closure_type_impl
        {
            using type = xscalar<E>;
        };
        
        template <class E>
        struct get_closure_type_impl<E, std::enable_if_t<is_xexpression<E>::value, void>>
        {
            using type = typename E::closure_type;
        };
    }

    template <class E>
    using get_closure_type = typename detail::get_closure_type_impl<E>::type;
 

    /********************
     * get_value_type
     ********************/

    namespace detail
    {
        template <class E, class enable = void>
        struct get_value_type_impl
        {
            using type = E;
        };

        template <class E>
        struct get_value_type_impl<E, std::enable_if_t<is_xexpression<E>::value, void>>
        {
            using type = typename E::value_type;
        };
    }

    template <class E>
    using get_value_type = typename detail::get_value_type_impl<E>::type;


    /*******************
     * get_size_type
     *******************/

    namespace detail
    {
        template <class E, class enable = void>
        struct get_size_type_impl
        {
            using type = size_t;
        };

        template <class E>
        struct get_size_type_impl<E, std::enable_if_t<is_xexpression<E>::value, void>>
        {
            using type = typename E::size_type;
        };
    }

    template <class E>
    using get_size_type = typename detail::get_size_type_impl<E>::type;


    /*************************
     * get_difference_type
     *************************/

    namespace detail
    {
        template <class E, class enable = void>
        struct get_difference_type_impl
        {
            using type = ptrdiff_t;
        };

        template <class E>
        struct get_difference_type_impl<E, std::enable_if_t<is_xexpression<E>::value, void>>
        {
            using type = typename E::difference_type;
        };
    }

    template <class E>
    using get_difference_type = typename detail::get_difference_type_impl<E>::type;

}

#endif

