/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSLICE_HPP
#define XSLICE_HPP

#include <cstddef>
#include <utility>
#include <type_traits>
#include "xutils.hpp"

namespace xt
{

    /**********************
     * xslice declaration *
     **********************/

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

    template <class E, class R = void>
    using disable_xslice = typename std::enable_if<!is_xslice<E>::value, R>::type;

    template <class... E>
    using has_xslice = or_<is_xslice<E>...>;

    struct xnone_tag
    {
    };

    inline auto xnone()
    {
        return xnone_tag();
    }

    /**********************
     * xrange declaration *
     **********************/

    template <class T>
    class xrange : public xslice<xrange<T>>
    {

    public:

        using size_type = T;

        xrange() = default;
        xrange(size_type min, size_type max) noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;

    private:

        size_type m_min;
        size_type m_size;
    };

    /**
     * Returns a slice representing an interval, to
     * be used as an argument of view function.
     * @param min the first index of the interval
     * @param max the last index of the interval
     * @sa view
     */
    template <class T, class E = std::enable_if_t<!std::is_same<T, xnone_tag>::value>>
    inline auto range(T min, T max) noexcept
    {
        return xrange<T>(min, max);
    }

    /******************************
     * xstepped_range declaration *
     ******************************/

    template <class T>
    class xstepped_range : public xslice<xstepped_range<T>>
    {

    public:

        using size_type = T;

        xstepped_range() = default;
        xstepped_range(size_type min, size_type max, size_type step) noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;

    private:

        size_type m_min;
        size_type m_size;
        size_type m_step;
    };

    /**
     * Returns a slice representing an interval, to
     * be used as an argument of view function.
     * @param min the first index of the interval
     * @param max the last index of the interval
     * @param step the space between two indices
     * @sa view
     */
    template <class T, class E = std::enable_if_t<!std::is_same<T, xnone_tag>::value>>
    inline auto range(T min, T max, T step) noexcept
    {
        return xstepped_range<T>(min, max, step);
    }

    /********************
     * xall declaration *
     ********************/

    template <class T>
    class xall : public xslice<xall<T>>
    {

    public:

        using size_type = T;

        xall() = default;
        explicit xall(size_type size) noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;

    private:

        size_type m_size;
    };

    struct xall_tag
    {
    };

    /**
     * Returns a slice representing a full dimension,
     * to be used as an argument of view function.
     * @sa view
     */
    inline auto all() noexcept
    {
        return xall_tag();
    }

    /************************
    * xnewaxis declaration *
    ************************/

    template <class T>
    class xnewaxis : public xslice<xnewaxis<T>>
    {

    public:

        using size_type = T;

        xnewaxis() = default;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;
    };

    struct xnewaxis_tag
    {
    };

    /**
    * Returns a slice representing a new axis of length one,
    * to be used as an argument of view function.
    * @sa view
    */
    inline auto newaxis() noexcept
    {
        return xnewaxis_tag();
    }

    template <class A, class B, class C>
    struct xrange_adaptor
    {
        xrange_adaptor(A min, B max, C step)
            : m_min(min), m_max(max), m_step(step)
        {
        }

        template <class MIN = A, class MAX = B, class STEP = C>
        std::enable_if_t<!std::is_integral<MIN>::value && std::is_integral<MAX>::value && std::is_integral<STEP>::value, xstepped_range<int>>
        get(std::size_t size)
        {
            if (m_step > 0)
            {
                return xstepped_range<int>(0, m_max, m_step);
            }
            else
            {
                return xstepped_range<int>(int(size) - 1, m_max, m_step);
            }
        }

        template <class MIN = A, class MAX = B, class STEP = C>
        std::enable_if_t<!std::is_integral<MAX>::value && std::is_integral<MIN>::value && std::is_integral<STEP>::value, xstepped_range<int>>
        get(std::size_t size)
        {
            if (m_step > 0)
            {
                return xstepped_range<int>(m_min, int(size), m_step);
            }
            else
            {
                return xstepped_range<int>(m_min, -1, m_step);
            }
        }

        template <class MIN = A, class MAX = B, class STEP = C>
        std::enable_if_t<!std::is_integral<MIN>::value && !std::is_integral<MAX>::value && std::is_integral<STEP>::value, xstepped_range<int>>
        get(std::size_t size)
        {
            if (m_step > 0)
            {
                return xstepped_range<int>(0, int(size), m_step);
            }
            else
            {
                return xstepped_range<int>(int(size) - 1, -1, m_step);
            }
        }

        template <class MIN = A, class MAX = B, class STEP = C>
        std::enable_if_t<std::is_integral<MIN>::value && !std::is_integral<MAX>::value && !std::is_integral<STEP>::value, xrange<std::size_t>>
        get(std::size_t size)
        {
            return xrange<std::size_t>((std::size_t) m_min, size);
        }

        template <class MIN = A, class MAX = B, class STEP = C>
        std::enable_if_t<!std::is_integral<MIN>::value && std::is_integral<MAX>::value && !std::is_integral<STEP>::value, xrange<std::size_t>>
        get(std::size_t /*size*/)
        {
            return xrange<std::size_t>(0, (std::size_t)m_max);
        }

        template <class MIN = A, class MAX = B, class STEP = C>
        std::enable_if_t<!std::is_integral<MIN>::value && !std::is_integral<MAX>::value && !std::is_integral<STEP>::value, xall<std::size_t>>
        get(std::size_t size)
        {
            return xall<std::size_t>(size);
        }

    private:
        A m_min;
        B m_max;
        C m_step;
    };

    template <class A, class B>
    inline auto range(A min, B max)
    {
        return xrange_adaptor<A, B, xnone_tag>(min, max, xnone_tag());
    }

    template <class A, class B, class C>
    inline auto range(A min, B max, C step)
    {
        return xrange_adaptor<A, B, C>(min, max, step);
    }


    /******************************************************
     * homogeneous get_size for integral types and slices *
     ******************************************************/

    template <class S>
    inline disable_xslice<S, std::size_t> get_size(const S&) noexcept
    {
        return 1;
    };

    template <class S>
    inline auto get_size(const xslice<S>& slice) noexcept
    {
        return slice.derived_cast().size();
    };

    /*******************************************************
     * homogeneous step_size for integral types and slices *
     *******************************************************/

    template <class S>
    inline disable_xslice<S, std::size_t> step_size(const S&) noexcept
    {
        return 0;
    }

    template <class S>
    inline auto step_size(const xslice<S>& slice) noexcept
    {
        return slice.derived_cast().step_size();
    }

    /*********************************************
     * homogeneous value for integral and slices *
     *********************************************/

    template <class S, class I>
    inline disable_xslice<S, std::size_t> value(const S& s, I) noexcept
    {
        return s;
    }

    template <class S, class I>
    inline auto value(const xslice<S>& slice, I i) noexcept
    {
        return slice.derived_cast()(i);
    }

    /****************************************
     * homogeneous get_slice_implementation *
     ****************************************/

    template <class E, class SL>
    inline auto get_slice_implementation(E& /*e*/, SL&& slice, std::size_t /*index*/)
    {
        return std::forward<SL>(slice);
    }

    template <class E>
    inline auto get_slice_implementation(E& e, xall_tag, std::size_t index)
    {
        return xall<typename E::size_type>(e.shape()[index]);
    }
    
    template <class E>
    inline auto get_slice_implementation(E& /*e*/, xnewaxis_tag, std::size_t /*index*/)
    {
        return xnewaxis<typename E::size_type>();
    }

    template <class E, class A, class B, class C>
    inline auto get_slice_implementation(E& e, xrange_adaptor<A, B, C> adaptor, std::size_t index)
    {
        return adaptor.get(e.shape()[index]);
    }


    /******************************
     * homogeneous get_slice_type *
     ******************************/

    namespace detail
    {
        template <class E, class SL>
        struct get_slice_type_impl
        {
            using type = SL;
        };

        template <class E>
        struct get_slice_type_impl<E, xall_tag>
        {
            using type = xall<typename E::size_type>;
        };

        template <class E>
        struct get_slice_type_impl<E, xnewaxis_tag>
        {
            using type = xnewaxis<typename E::size_type>;
        };

        template <class E, class A, class B, class C>
        struct get_slice_type_impl<E, xrange_adaptor<A, B, C>>
        {
            using type = decltype(xrange_adaptor<A, B, C>(A(), B(), C()).get(0));
        };
    }

    template <class E, class SL>
    using get_slice_type = typename detail::get_slice_type_impl<E, std::remove_reference_t<SL>>::type;

    /*************************
     * xslice implementation *
     *************************/

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

    /*************************
     * xrange implementation *
     *************************/

    template <class T>
    inline xrange<T>::xrange(size_type min, size_type max) noexcept
        : m_min(min), m_size(max - min)
    {
    }

    template <class T>
    inline auto xrange<T>::operator()(size_type i) const noexcept -> size_type
    {
        return m_min + i;
    }

    template <class T>
    inline auto xrange<T>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class T>
    inline auto xrange<T>::step_size() const noexcept -> size_type
    {
        return 1;
    }

    /********************************
     * xtepped_range implementation *
     ********************************/

    template <class T>
    inline xstepped_range<T>::xstepped_range(size_type min, size_type max, size_type step) noexcept
        : m_min(min), m_size((size_type) std::ceil(double(max - min)/double(step))), m_step(step)
    {
    }

    template <class T>
    inline auto xstepped_range<T>::operator()(size_type i) const noexcept -> size_type
    {
        return m_min + i * m_step;
    }

    template <class T>
    inline auto xstepped_range<T>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class T>
    inline auto xstepped_range<T>::step_size() const noexcept -> size_type
    {
        return m_step;
    }

    /***********************
     * xall implementation *
     ***********************/

    template <class T>
    inline xall<T>::xall(size_type size) noexcept
        : m_size(size)
    {
    }

    template <class T>
    inline auto xall<T>::operator()(size_type i) const noexcept -> size_type
    {
        return i;
    }

    template <class T>
    inline auto xall<T>::size() const noexcept -> size_type
    {
        return m_size;
    }

    template <class T>
    inline auto xall<T>::step_size() const noexcept -> size_type
    {
        return 1;
    }

    /***************************
    * xnewaxis implementation *
    ***************************/

    template <class T>
    inline auto xnewaxis<T>::operator()(size_type) const noexcept -> size_type
    {
        return 0;
    }

    template <class T>
    inline auto xnewaxis<T>::size() const noexcept -> size_type
    {
        return 1;
    }

    template <class T>
    inline auto xnewaxis<T>::step_size() const noexcept -> size_type
    {
        return 0;
    }
}

#endif
