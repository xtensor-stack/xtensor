/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_SLICE_HPP
#define XTENSOR_SLICE_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include <xtl/xtype_traits.hpp>

#include "xutils.hpp"

namespace xt
{

    namespace placeholders
    {
        // xtensor universal placeholder
        struct xtuph
        {
        };

        constexpr xtuph _{};
    }

    inline auto xnone()
    {
        return placeholders::xtuph();
    }

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
    using has_xslice = xtl::disjunction<is_xslice<E>...>;

    /**********************
     * xrange declaration *
     **********************/

    template <class T>
    class xrange : public xslice<xrange<T>>
    {
    public:

        using size_type = T;
        using self_type = xrange<T>;

        xrange() = default;
        xrange(size_type min_val, size_type max_val) noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;
        size_type step_size(size_type i, size_type n = 1) const noexcept;
        size_type revert_index(size_type i) const noexcept;

        bool contains(size_type i) const noexcept;

        bool operator==(const self_type& rhs) const noexcept;
        bool operator!=(const self_type& rhs) const noexcept;

    private:

        size_type m_min;
        size_type m_size;
    };

    /******************************
     * xstepped_range declaration *
     ******************************/

    template <class T>
    class xstepped_range : public xslice<xstepped_range<T>>
    {
    public:

        using size_type = T;
        using self_type = xstepped_range<T>;

        xstepped_range() = default;
        xstepped_range(size_type min_val, size_type max_val, size_type step) noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;
        size_type step_size(size_type i, size_type n = 1) const noexcept;
        size_type revert_index(size_type i) const noexcept;

        bool contains(size_type i) const noexcept;

        bool operator==(const self_type& rhs) const noexcept;
        bool operator!=(const self_type& rhs) const noexcept;

    private:

        size_type m_min;
        size_type m_size;
        size_type m_step;
    };

    /********************
     * xall declaration *
     ********************/

    template <class T>
    class xall : public xslice<xall<T>>
    {
    public:

        using size_type = T;
        using self_type = xall<T>;

        xall() = default;
        explicit xall(size_type size) noexcept;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;
        size_type step_size(size_type i, size_type n = 1) const noexcept;
        size_type revert_index(size_type i) const noexcept;

        bool contains(size_type i) const noexcept;

        bool operator==(const self_type& rhs) const noexcept;
        bool operator!=(const self_type& rhs) const noexcept;

    private:

        size_type m_size;
    };

    struct xall_tag
    {
    };

    /**
     * Returns a slice representing a full dimension,
     * to be used as an argument of view function.
     * @sa view, dynamic_view
     */
    inline auto all() noexcept
    {
        return xall_tag();
    }

    template <class T>
    class xellipsis : public xslice<xellipsis<T>>
    {
    public:

        using size_type = T;

        xellipsis() = default;

        size_type operator()(size_type i) const noexcept;

        size_type size() const noexcept;
        size_type step_size() const noexcept;
    };

    struct xellipsis_tag
    {
    };

    /**
     * Returns a slice representing all remaining dimensions,
     * and selecting all in these dimensions. Ellipsis will expand
     * to a series of `all()` slices, until the number of slices is
     * equal to the number of dimensions of the source array.
     *
     * Note: ellipsis can only be used in dynamic_view!
     *
     * \code{.cpp}
     * xarray<double> a = xarray<double>::from_shape({5, 5, 1, 1, 5});
     * auto v = xt::dynamic_view(a, {2, xt::ellipsis(), 2});
     * // equivalent to using {2, xt::all(), xt::all(), xt::all(), 2};
     * \endcode
     *
     * @sa dynamic_view
     */
    inline auto ellipsis() noexcept
    {
        return xellipsis_tag();
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
        size_type step_size(size_type i, size_type n = 1) const noexcept;
        size_type revert_index(size_type i) const noexcept;

        bool contains(size_type i) const noexcept;
    };

    struct xnewaxis_tag
    {
    };

    /**
     * Returns a slice representing a new axis of length one,
     * to be used as an argument of view function.
     * @sa view, dynamic_view
     */
    inline auto newaxis() noexcept
    {
        return xnewaxis_tag();
    }

    /******************
     * xrange_adaptor *
     ******************/

    template <class A, class B, class C>
    struct xrange_adaptor
    {
        xrange_adaptor(A min_val, B max_val, C step)
            : m_min(min_val), m_max(max_val), m_step(step)
        {
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<std::is_integral<MI>::value &&
                                    std::is_integral<MA>::value &&
                                    std::is_integral<STEP>::value,
                                xstepped_range<std::ptrdiff_t>>
            get(std::size_t /*size*/) const
        {
            return xstepped_range<std::ptrdiff_t>(m_min, m_max, m_step);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<!std::is_integral<MI>::value &&
                                    std::is_integral<MA>::value &&
                                    std::is_integral<STEP>::value,
                                xstepped_range<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return xstepped_range<std::ptrdiff_t>(m_step > 0 ? 0 : int(size) - 1, m_max, m_step);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<std::is_integral<MI>::value &&
                                    !std::is_integral<MA>::value &&
                                    std::is_integral<STEP>::value,
                                xstepped_range<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return xstepped_range<std::ptrdiff_t>(m_min, m_step > 0 ? int(size) : -1, m_step);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<std::is_integral<MI>::value &&
                                    std::is_integral<MA>::value &&
                                    !std::is_integral<STEP>::value,
                                xrange<std::ptrdiff_t>>
            get(std::size_t /*size*/) const
        {
            return xrange<std::ptrdiff_t>(static_cast<std::ptrdiff_t>(m_min), static_cast<std::ptrdiff_t>(m_max));
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<!std::is_integral<MI>::value &&
                                    !std::is_integral<MA>::value &&
                                    std::is_integral<STEP>::value,
                                xstepped_range<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            int min_val_arg = m_step > 0 ? 0 : int(size) - 1;
            int max_val_arg = m_step > 0 ? int(size) : -1;
            return xstepped_range<std::ptrdiff_t>(min_val_arg, max_val_arg, m_step);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<std::is_integral<MI>::value &&
                                    !std::is_integral<MA>::value &&
                                    !std::is_integral<STEP>::value,
                                xrange<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return xrange<std::ptrdiff_t>(std::size_t(m_min), size);
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<!std::is_integral<MI>::value &&
                                    std::is_integral<MA>::value &&
                                    !std::is_integral<STEP>::value,
                                xrange<std::ptrdiff_t>>
            get(std::size_t /*size*/) const
        {
            return xrange<std::ptrdiff_t>(0, std::size_t(m_max));
        }

        template <class MI = A, class MA = B, class STEP = C>
        inline std::enable_if_t<!std::is_integral<MI>::value &&
                                    !std::is_integral<MA>::value &&
                                    !std::is_integral<STEP>::value,
                                xall<std::ptrdiff_t>>
        get(std::size_t size) const
        {
            return xall<std::ptrdiff_t>(size);
        }

    private:

        A m_min;
        B m_max;
        C m_step;
    };

    namespace detail
    {
        template <class T, class E = void>
        struct cast_if_integer
        {
            using type = T;

            type operator()(T t)
            {
                return t;
            }
        };

        template <class T>
        struct cast_if_integer<T, std::enable_if_t<std::is_integral<T>::value>>
        {
            using type = std::ptrdiff_t;

            type operator()(T t)
            {
                return static_cast<type>(t);
            }
        };

        template <class T>
        using cast_if_integer_t = typename cast_if_integer<T>::type;
    }

    /**
     * Select a range from min_val to max_val.
     * You can use the shorthand `_` syntax to select from the start or until the end.
     *
     * \code{.cpp}
     * using namespace xt::placeholders;  // to enable _ syntax
     *
     * range(3, _)  // select from index 3 to the end
     * range(_, 5)  // select from index o to 5
     * range(_, _)  // equivalent to `all()`
     * \endcode
     *
     * @sa view, dynamic_view
     */
    template <class A, class B>
    inline auto range(A min_val, B max_val)
    {
        return xrange_adaptor<detail::cast_if_integer_t<A>, detail::cast_if_integer_t<B>, placeholders::xtuph>(
            detail::cast_if_integer<A>{}(min_val), detail::cast_if_integer<B>{}(max_val), placeholders::xtuph());
    }

    /**
     * Select a range from min_val to max_val with step
     * You can use the shorthand `_` syntax to select from the start or until the end.
     *
     * \code{.cpp}
     * using namespace xt::placeholders;  // to enable _ syntax
     * range(3, _, 5)  // select from index 3 to the end with stepsize 5
     * \endcode
     *
     * @sa view, dynamic_view
     */
    template <class A, class B, class C>
    inline auto range(A min_val, B max_val, C step)
    {
        return xrange_adaptor<detail::cast_if_integer_t<A>, detail::cast_if_integer_t<B>, detail::cast_if_integer_t<C>>(
            detail::cast_if_integer<A>{}(min_val), detail::cast_if_integer<B>{}(max_val), detail::cast_if_integer<C>{}(step));
    }


    /******************************************************
     * homogeneous get_size for integral types and slices *
     ******************************************************/

    template <class S>
    inline disable_xslice<S, std::size_t> get_size(const S&) noexcept
    {
        return 1;
    }

    template <class S>
    inline auto get_size(const xslice<S>& slice) noexcept
    {
        return slice.derived_cast().size();
    }

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
        return static_cast<std::size_t>(s);
    }

    template <class S, class I>
    inline auto value(const xslice<S>& slice, I i) noexcept
    {
        using ST = typename S::size_type;
        return slice.derived_cast()(static_cast<ST>(i));
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
    inline auto get_slice_implementation(E& /*e*/, xellipsis_tag, std::size_t /*index*/)
    {
        return xellipsis<typename E::size_type>();
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
        struct get_slice_type_impl<E, xellipsis_tag>
        {
            using type = xellipsis<typename E::size_type>;
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
    inline xrange<T>::xrange(size_type min_val, size_type max_val) noexcept
        : m_min(min_val), m_size(max_val - min_val)
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

    template <class T>
    inline auto xrange<T>::step_size(size_type /*i*/, size_type n) const noexcept -> size_type
    {
        return n;
    }

    template <class T>
    inline auto xrange<T>::revert_index(size_type i) const noexcept -> size_type
    {
        return i - m_min;
    }

    template <class T>
    inline bool xrange<T>::contains(size_type i) const noexcept
    {
        return i >= m_min && i < m_min + m_size;
    }

    template <class T>
    inline bool xrange<T>::operator==(const self_type& rhs) const noexcept
    {
        return (m_min == rhs.m_min) && (m_size == rhs.m_size);
    }

    template <class T>
    inline bool xrange<T>::operator!=(const self_type& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    /********************************
     * xtepped_range implementation *
     ********************************/

    template <class T>
    inline xstepped_range<T>::xstepped_range(size_type min_val, size_type max_val, size_type step) noexcept
        : m_min(min_val), m_size(size_type(std::ceil(double(max_val - min_val) / double(step)))), m_step(step)
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

    template <class T>
    inline auto xstepped_range<T>::step_size(size_type /*i*/, size_type n) const noexcept -> size_type
    {
        return m_step * n;
    }

    template <class T>
    inline auto xstepped_range<T>::revert_index(size_type i) const noexcept -> size_type
    {
        return (i - m_min) / m_step;
    }

    template <class T>
    inline bool xstepped_range<T>::contains(size_type i) const noexcept
    {
        return i >= m_min && i < m_min + m_size * m_step && ((i - m_min) % m_step == 0);
    }

    template <class T>
    inline bool xstepped_range<T>::operator==(const self_type& rhs) const noexcept
    {
        return (m_min == rhs.m_min) && (m_size == rhs.m_size) && (m_step == rhs.m_step);
    }

    template <class T>
    inline bool xstepped_range<T>::operator!=(const self_type& rhs) const noexcept
    {
        return !(*this == rhs);
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

    template <class T>
    inline auto xall<T>::step_size(size_type /*i*/, size_type n) const noexcept -> size_type
    {
        return n;
    }

    template <class T>
    inline auto xall<T>::revert_index(size_type i) const noexcept -> size_type
    {
        return i;
    }

    template <class T>
    inline bool xall<T>::contains(size_type i) const noexcept
    {
        return i < m_size;
    }

    template <class T>
    inline bool xall<T>::operator==(const self_type& rhs) const noexcept
    {
        return m_size == rhs.m_size;
    }

    template <class T>
    inline bool xall<T>::operator!=(const self_type& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    /****************************
     * xellipsis implementation *
     ****************************/

    template <class T>
    inline auto xellipsis<T>::operator()(size_type /*i*/) const noexcept -> size_type
    {
        return 0;
    }

    template <class T>
    inline auto xellipsis<T>::size() const noexcept -> size_type
    {
        return 0;
    }

    template <class T>
    inline auto xellipsis<T>::step_size() const noexcept -> size_type
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

    template <class T>
    inline auto xnewaxis<T>::step_size(size_type /*i*/, size_type /*n*/) const noexcept -> size_type
    {
        return 0;
    }

    template <class T>
    inline auto xnewaxis<T>::revert_index(size_type i) const noexcept -> size_type
    {
        return i;
    }

    template <class T>
    inline bool xnewaxis<T>::contains(size_type i) const noexcept
    {
        return i == 0;
    }
}

#endif
