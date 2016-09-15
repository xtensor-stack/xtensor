#ifndef XFUNCTION_HPP
#define XFUNCTION_HPP

#include <type_traits>
#include <utility>
#include <tuple>
#include <algorithm>

#include "utils.hpp"
#include "xexpression.hpp"
#include "broadcast.hpp"

namespace qs
{

    /******************************************
     * Operation on multiple expressions
     ******************************************/
    
    namespace detail
    {
        template <class... Args>
        using common_size_type = std::common_type_t<typename Args::size_type...>;

        template <class... Args>
        using common_difference_type = std::common_type_t<typename Args::difference_type...>;

        template <class... Args>
        using common_value_type = std::common_type_t<typename Args::value_type...>;
    }

    template <class F, class R, class... E>
    class xfunction : public xexpression<xfunction<F, R, E...>>
    {

    public:

        using self_type = xfunction<F, E...>;
        using functor_type = F;

        using value_type = R;
        using const_reference = value_type;
        using const_pointer = const value_type*;
        using size_type = detail::common_size_type<E...>;
        using difference_type = detail::common_difference_type<E...>;

        using shape_type = array_shape<size_type>;
        using closure_type = const self_type;

        //using const_iterator = xfunction_iterator<F, E...>;

        template <class Func>
        inline xfunction(Func&& f, const E&...e) noexcept
            : m_f(std::forward<Func>(f)), m_e(e...)
        {
        }

        inline size_type dimension() const
        {
            auto func = [](size_type d, auto&& e) { return std::max(d, e.dimension()); };
            return accumulate(func, 0, m_e);
        }

        inline bool broadcast_shape(shape_type& shape) const
        {
            auto func = [&shape](bool b, auto&& e) { return b && e.broadcast_shape(shape); };
            return accumulate(func, true, m_e);
        }

        template <class... Args>
        const_reference operator()(Args... args) const
        {
            return access_impl(std::make_index_sequence<sizeof...(E)>(), args...);
        }

    private:

        std::tuple<get_closure_type<E>...> m_e;
        F m_f;

        template <size_t... I, class... Args>
        const_reference access_impl(std::index_sequence<I...>, Args... args) const
        {
            return m_f(std::get<I>(m_e)(args...)...);
        }
     };

}

#endif

