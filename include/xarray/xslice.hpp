#ifndef XSLICE_HPP
#define XSLICE_HPP

#include <utility>
#include <type_traits>
#include <vector>

namespace qs
{

    /****************************
     * xslice declaration
     ****************************/

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

    template <class S>
    using is_xslice = std::is_base_of<xslice<S>, S>;

    template <class E, class R>
    using disable_xslice = typename std::enable_if<!is_xslice<E>::value, R>::type;

    template <class... E>
    using has_xslice = or_<is_xslice<E>...>;

    /****************************
     * xrange declaration
     ****************************/

    template <class size_type>
    class xrange : public xslice<xrange<size_type>>
    {

    public:

        xrange() = default;
        explicit xrange(size_type min, size_type max) noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;

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
     * xall declaration
     ********************************/

    template <class size_type>
    class xall : public xslice<xall<size_type>>
    {

    public:

        xall() = default;
        explicit xall(size_type size) noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;

    private:

        size_type m_size;
    };

    /******************************************************
     * homogeneous get_size for integral types and slices
     ******************************************************/

    template <class S>
    inline disable_xslice<S, std::size_t> get_size(const S&)
    {
        return 0;
    };

    template <class S>
    inline std::size_t get_size(const xslice<S>& slice)
    {
        return slice.derived_cast().size();
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

    /****************************
     * xrange implementation
     ****************************/

    template <class size_type>
    inline xrange<size_type>::xrange(size_type min, size_type max) noexcept : m_min(min), m_size(max - min)
    {
    }

    template <class size_type>
    inline size_type xrange<size_type>::operator()(size_type i) const noexcept
    {
        return m_min + i;
    }

    template <class size_type>
    inline size_type xrange<size_type>::size() const noexcept
    {
        return m_size;
    }

    /********************************
     * xall implementation
     ********************************/

    template <class size_type>
    inline xall<size_type>::xall(size_type size) noexcept : m_size(size)
    {
    }

    template <class size_type>
    inline size_type xall<size_type>::operator()(size_type i) const noexcept
    {
        return i;
    }

    template <class size_type>
    inline size_type xall<size_type>::size() const noexcept
    {
        return m_size;
    }

}

#endif
