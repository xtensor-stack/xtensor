#ifndef XSLICE_HPP
#define XSLICE_HPP

#include <utility>
#include <type_traits>
#include <vector>

namespace qs
{

    template <class D>
    class xslice
    {

    public:

        using derived_type = D;

        derived_type& derived_cast() noexcept;
        const derived_type& derived_cast() const noexcept;

    protected:

        xslice() = default;
        ~xslice() = default;

        xslice(const xslice&) = default;
        xslice& operator=(const xslice&) = default;

        xslice(xslice&&) = default;
        xslice& operator=(xslice&&) = default;
    };

    /*********************************
     * xslice implementation
     *********************************/

    template <class D>
    inline auto xslice<D>::derived_cast() noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto xslice<D>::derived_cast() const noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class S>
    using is_xslice = std::is_base_of<xslice<S>, S>;

    template <class E, class R>
    using disable_xslice = typename std::enable_if<!is_xslice<E>::value, R>::type;

    template <class... E>
    using has_xslice = or_<is_xslice<E>...>;

    /****************************
     * xrange slice
     ****************************/

    template <class size_type>
    class xrange : public xslice<xrange<size_type>>
    {

    public:

        xrange() = default;
        ~xrange() = default;
        xrange(const xrange&) = default;
        xrange& operator=(const xrange&) = default;
        xrange(xrange&&) = default;
        xrange& operator=(xrange&&) = default;

        explicit xrange(size_type min, size_type max) noexcept : m_min(min), m_size(max - min) {}

        inline size_type operator()(size_type i) const noexcept
        {
            return m_min + i;
        }

        inline size_type size() const noexcept
        {
            return m_size;
        }

    private:
        size_type m_min;  
        size_type m_size;  
    };

    template <class size_type>
    auto range(size_type min, size_type max)
    {
        return xrange<size_type>(min, max);
    }

    /********************************
     * xall
     ********************************/

    template <class size_type>
    class xall : public xslice<xall<size_type>>
    {

    public:

        xall() = default;
        ~xall() = default;
        xall(const xall&) = default;
        xall& operator=(const xall&) = default;
        xall(xall&&) = default;
        xall& operator=(xall&&) = default;

        explicit xall(size_type size) noexcept : m_size(size) {}

        inline size_type operator()(size_type i) const noexcept
        {
            return i;
        }

        inline size_type size() const noexcept
        {
            return m_size;
        }

    private:
        size_type m_size;
    };

    /*******************************
     * xsqueeze
     *******************************/

    template <class size_type>
    class xsqueeze
    {

    public:

        xsqueeze() = default;
        ~xsqueeze() = default;
        xsqueeze(const xsqueeze&) = default;
        xsqueeze& operator=(const xsqueeze&) = default;
        xsqueeze(xsqueeze&&) = default;
        xsqueeze& operator=(xsqueeze&&) = default;

        explicit xsqueeze(size_type index) noexcept : m_index(index) {}

        inline size_type operator()() const noexcept
        {
            return m_index;
        }

    private:

        const size_type m_index;
    };

    /********************************
     * get_xslice_type
     ********************************/

    template <class S>
    using get_xslice_type = std::conditional_t<is_xslice<S>::value, S, xsqueeze<S>>;

}

#endif
