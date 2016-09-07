#ifndef NDEXPRESSION_HPP
#define NDEXPRESSION_HPP

namespace qs
{
    
    template <class D>
    class ndexpression
    {

    public:

        using derived_type = D;

        derived_type& derived_cast();
        const derived_type& derived_cast() const;

    protected:

        ndexpression() = default;
        ~ndexpression() = default;

        ndexpression(const ndexpression&) = default;
        ndexpression& operator=(const ndexpression&) = default;

        ndexpression(ndexpression&&) = default;
        ndexpression& operator=(ndexpression&&) = default;
    };


    /*********************************
     * ndexpression implementation
     *********************************/

    template <class D>
    inline auto ndexpression<D>::derived_cast() -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto ndexpression<D>::derived_cast() const -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

}

#endif

