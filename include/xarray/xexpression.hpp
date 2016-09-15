#ifndef XEXPRESSION_HPP
#define XEXPRESSION_HPP

namespace qs
{

    template <class D>
    class xexpression
    {

    public:

        using derived_type = D;

        derived_type& derived_cast();
        const derived_type& derived_cast() const;

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
    inline auto xexpression<D>::derived_cast() -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xexpression<D>::derived_cast() const -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class E>
    using is_xexpression = std::is_base_of<xexpression<E>, E>;
   
    template <class E, class R>
    using disable_xexpression = typename std::enable_if<!is_xexpression<E>::value, R>::type;

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
 
}

#endif

