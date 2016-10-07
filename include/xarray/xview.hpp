#ifndef XVIEW_HPP
#define XVIEW_HPP

#include <utility>
#include <type_traits>
#include <tuple>
#include <algorithm>

#include "xexpression.hpp"
#include "xutils.hpp"
#include "xslice.hpp"
#include "xindex.hpp"

namespace qs
{

    /*********************************
     * xview declaration
     *********************************/

    template <class E, class... S>
    class xview : public xexpression<xview<E, S...>>
    {

    public:

        using self_type = xview<E, S...>;

        using value_type = typename E::value_type;
        using reference = typename E::reference;
        using const_reference = typename E::const_reference;
        using pointer = typename E::pointer;
        using const_pointer = typename E::const_pointer;
        using size_type = typename E::size_type;
        using difference_type = typename E::difference_type;

        using shape_type = xshape<size_type>;
        using closure_type = const self_type&;

        xview(E&& e, S&&... slices) noexcept;

        size_type dimension() const noexcept;
        
        auto shape() const noexcept;

        bool broadcast_shape(shape_type& shape) const;

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

    private:

        E& m_e;
        std::tuple<S...> m_slices;
        shape_type m_shape;

        template <size_type... I, class... Args>
        reference access_impl(std::index_sequence<I...>, Args... args);

        template <size_type... I, class... Args>
        const_reference access_impl(std::index_sequence<I...>, Args... args) const;

        template <size_type I, class... Args>
        std::enable_if_t<(I<sizeof...(S)), size_type> index(Args... args) const;

        template <size_type I, class... Args>
        std::enable_if_t<(I>=sizeof...(S)), size_type> index(Args... args) const;

        template<size_type I, class T, class... Args>
        size_type sliced_access(const xslice<T>& slice, Args... args) const;

        template<size_type I, class T, class... Args>
        disable_xslice<T, size_type> sliced_access(const T& squeeze, Args...) const;

    };

    template <class E, class... S>
    inline xview<E, S...> make_xview(E& e, S&&... slices);

    // number of integral types in the specified sequence of types
    template <class... S>
    inline constexpr size_t squeeze_count();

    // number of integral types in the specified sequence of types before specified index.
    template <class... S>
    inline constexpr size_t squeeze_count_before(size_t i);

    // index in the specified sequence of types of the ith non-integral type.
    template <class... S>
    inline constexpr size_t non_squeeze(size_t i);

    /*********************************
     * xview implementation
     *********************************/

    template <class E, class... S>
    inline xview<E, S...>::xview(E&& e, S&&... slices) noexcept : m_e(e), m_slices(slices...)
    {
        auto func = [](auto s) { return get_size(s); };
        m_shape.resize(dimension());
        for (size_type i = 0; i != dimension(); ++i)
        {
            size_type index = non_squeeze<S...>(i);
            if (index < sizeof...(S))
            {
                m_shape[i] = apply<size_t>(index, func, std::forward<S>(slices)...);
            }
            else
            {
                m_shape[i] = m_e.shape()[index];
            }
        }
    }

    template <class E, class... S>
    inline auto xview<E, S...>::dimension() const noexcept -> size_type
    {
        return m_e.dimension() - squeeze_count<S...>();
    }
        
    template <class E, class... S>
    inline auto xview<E, S...>::shape() const noexcept
    {
        return m_shape;
    }

    template <class E, class... S>
    inline auto xview<E, S...>::broadcast_shape(shape_type& shape) const -> bool
    {
        return false;
    }

    template <class E, class... S>
    template <class... Args>
    inline auto xview<E, S...>::operator()(Args... args) -> reference
    {
        return access_impl(std::make_index_sequence<sizeof...(Args) + squeeze_count<S...>()>(), args...);
    }

    template <class E, class... S>
    template <class... Args>
    inline auto xview<E, S...>::operator()(Args... args) const -> const_reference
    {
        return access_impl(std::make_index_sequence<sizeof...(Args) + squeeze_count<S...>()>(), args...);
    }

    template <class E, class... S>
    template <typename E::size_type... I, class... Args>
    inline auto xview<E, S...>::access_impl(std::index_sequence<I...>, Args... args) -> reference
    {
        return m_e(index<I>(args...)...);
    }

    template <class E, class... S>
    template <typename E::size_type... I, class... Args>
    inline auto xview<E, S...>::access_impl(std::index_sequence<I...>, Args... args) const -> const_reference
    {
        return m_e(index<I>(args...)...);
    }

    template <class E, class... S>
    template <typename E::size_type I, class... Args>
    inline auto xview<E, S...>::index(Args... args) const -> std::enable_if_t<(I<sizeof...(S)), size_type>
    {
        return sliced_access<I - squeeze_count_before<S...>(I)>(std::get<I>(m_slices), args...);
    }

    template <class E, class... S>
    template <typename E::size_type I, class... Args>
    inline auto xview<E, S...>::index(Args... args) const -> std::enable_if_t<(I>=sizeof...(S)), size_type>
    {
        return argument<I - squeeze_count_before<S...>(I)>(args...);
    }

    template <class E, class... S>
    template<typename E::size_type I, class T, class... Args>
    inline auto xview<E, S...>::sliced_access(const xslice<T>& slice, Args... args) const -> size_type
    {
        return slice.derived_cast()(argument<I>(args...));
    }

    template <class E, class... S>
    template<typename E::size_type I, class T, class... Args>
    inline auto xview<E, S...>::sliced_access(const T& squeeze, Args...) const -> disable_xslice<T, size_type>
    {
        return squeeze;
    }

    template <class E, class... S>
    inline xview<E, S...> make_xview(E& e, S&&... slices)
    {
        return xview<E, S...>(std::forward<E>(e), std::forward<S>(slices)...);
    }

    /*********************************
     * number of integral types
     *********************************/

    namespace detail
    {

        template <class T, class... S>
        struct squeeze_count_before_impl
        {
            static inline constexpr size_t count(size_t i) noexcept
            {
                return i ? (squeeze_count_before_impl<S...>::count(i - 1) + (std::is_integral<std::remove_reference_t<T>>::value ? 1 : 0)) : 0;
            } 
        };

        template <>
        struct squeeze_count_before_impl<void>
        {
            static inline constexpr size_t count(size_t i) noexcept
            {
                return i;
            } 
        };
    }

    template <class... S>
    inline constexpr size_t squeeze_count()
    {
        return detail::squeeze_count_before_impl<S..., void>::count(sizeof...(S));
    }

    template <class... S>
    inline constexpr size_t squeeze_count_before(size_t i)
    {
        return detail::squeeze_count_before_impl<S..., void>::count(i);
    }

    /**************************************
     * index of ith non-integral type
     **************************************/

    namespace detail
    {

        template <class T, class... S>
        struct non_squeeze_impl
        {
            static inline constexpr size_t count(size_t i) noexcept
            {
                if (i == 0)
                {
                    return std::is_integral<std::remove_reference_t<T>>::value ? 1 + non_squeeze_impl<S...>::count(i) : 0;
                }
                else
                {
                    return 1 + (
                        std::is_integral<std::remove_reference_t<T>>::value ? 
                            non_squeeze_impl<S...>::count(i) :
                            non_squeeze_impl<S...>::count(i - 1)
                    );
                }
            }
        };

        template <>
        struct non_squeeze_impl<void>
        {
            static inline constexpr size_t count(size_t i) noexcept
            {
                return i;
            }
        };
    }

    template <class... S>
    inline constexpr size_t non_squeeze(size_t i)
    {
        return detail::non_squeeze_impl<S..., void>::count(i);
    }

}

#endif
