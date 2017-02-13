#include "xtensor.hpp"
#include "xarray.hpp"

namespace xt
{
    namespace detail
    {
        template <class T>
        using is_container = std::is_base_of<xcontainer<std::remove_const_t<T>>, T>;

        template <class T, class S>
        inline auto eval_impl(T&& t, const S& /*s*/) -> std::enable_if_t<is_container<std::decay_t<T>>::value, xclosure_t<T>>
        {
            return xclosure_t<T>(std::forward<T>(t));
        }

        template <class T, class N, std::size_t O>
        inline auto eval_impl(const T& t, const std::array<N, O>& s) -> std::enable_if_t<!is_container<T>::value, xclosure_t<xtensor<typename T::value_type, O>>>
        {
            return xtensor<typename T::value_type, O>(t);
        }

        template <class T, class S>
        inline auto eval_impl(const T& t, const S& s) -> std::enable_if_t<!is_container<T>::value, xclosure_t<xt::xarray<typename T::value_type>>>
        {
            return xarray<typename T::value_type>(t);
        }
    }

    template <class T>
    inline auto eval(T&& t)
    {
        return detail::eval_impl(std::forward<T>(t), t.shape());
    }
}