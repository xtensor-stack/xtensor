#ifndef XVIEW_HPP
#define XVIEW_HPP

#include <utility>
#include <type_traits>
#include <tuple>
#include <algorithm>

#include "xexpression.hpp"
#include "utils.hpp"
#include "xslice.hpp"
#include "xindex.hpp"

namespace qs
{

    /*********************************
     * squeeze count
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
    constexpr size_t squeeze_count()
    {
        return detail::squeeze_count_before_impl<S..., void>::count(sizeof...(S));
    }

    template <class... S>
    constexpr size_t squeeze_count_before(size_t i)
    {
        return detail::squeeze_count_before_impl<S..., void>::count(i);
    }

    /**************************************
     * index of n-th non-squeeze
     **************************************/
    namespace detail
    {

        template <class T, class... S>
        struct non_squeeze_impl
        {
            static inline constexpr size_t count(size_t i) noexcept
            {
                return 1 + (std::is_integral<std::remove_reference_t<T>>::value ? non_squeeze_impl<S...>::count(i) : non_squeeze_impl<S...>::count(i - 1));
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
    constexpr size_t non_squeeze(size_t i)
    {
        return detail::non_squeeze_impl<S..., void>::count(i);
    }

    /******************************************
     * Views on xexpressions
     ******************************************/

    // Initialize an xview with an xexpression and a list of slices

    // The xview is valid as long as the underlying expression has not
    // been reshaped.

    // If more slices are provided than the dimension of the underlying
    // expression, the behavior is undefined.

    template <class S>
    disable_xslice<S, size_t> get_size(const S&)
    {
        return 0;
    };

    template <class S>
    size_t get_size(const xslice<S>& slice)
    {
        return slice.derived_cast().size();
    };

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

        xview(E&& e, S&&... slices) noexcept
            : m_e(e), m_slices(get_xslice_type<S>(slices)...)
        {
            auto func = [](auto s) { return get_size(s); };
            m_shape.reserve(dimension());
            for (size_type i=0; i!=sizeof...(S); ++i)
            {
                m_shape.push_back(apply<size_t>(non_squeeze<S...>(i), func, std::forward<S>(slices)...));
            }
            for (size_type i = sizeof...(S); i!=dimension(); ++i)
            {
                m_shape.push_back(m_e.shape()[non_squeeze<S...>(i)]);
            }
        }

        inline size_type dimension() const noexcept
        {
            return m_e.dimension() - squeeze_count<S...>();
        }
        
        auto shape() const noexcept
        {
            return m_shape;
        }

        inline bool broadcast_shape(shape_type& shape) const
        {
            return false;
        }

        template <class... Args>
        reference operator()(Args... args)
        {
            return access_impl(std::make_index_sequence<sizeof...(Args) + squeeze_count<S...>()>(), args...);
        }

        template <class... Args>
        const_reference operator()(Args... args) const
        {
            return access_impl(std::make_index_sequence<sizeof...(Args) + squeeze_count<S...>()>(), args...);
        }

    private:

        E& m_e;
        std::tuple<get_xslice_type<S>...> m_slices;
        shape_type m_shape;

        template <size_type... I, class... Args>
        reference access_impl(std::index_sequence<I...>, Args... args)
        {
            return m_e(index<I>(args...)...);
        }

        template <size_type... I, class... Args>
        const_reference access_impl(std::index_sequence<I...>, Args... args) const
        {
            return m_e(index<I>(args...)...);
        }

        template <size_type I, class... Args>
        std::enable_if_t<(I<sizeof...(S)), size_type> index(Args... args) const
        {
            return sliced_access<I - squeeze_count_before<S...>(I)>(std::get<I>(m_slices), args...);
        }

        template <size_type I, class... Args>
        std::enable_if_t<(I>=sizeof...(S)), size_type> index(Args... args) const
        {
            return argument<I - squeeze_count_before<S...>(I)>(args...);
        }

        template<size_type I, class T, class... Args>
        size_type sliced_access(const xslice<T>& slice, Args... args) const
        {
            return slice.derived_cast()(argument<I>(args...));
        }

        template<size_type I, class Squeeze, class... Args>
        disable_xslice<Squeeze, size_type> sliced_access(const Squeeze& squeeze, Args...) const
        {
            return squeeze();
        }

    };

    template <class E, class... S>
    xview<E, S...> make_xview(E& e, S&&... slices)
    {
        return xview<E, S...>(std::forward<E>(e), std::forward<S>(slices)...);
    }

}

#endif
