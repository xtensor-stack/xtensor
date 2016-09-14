#ifndef XFUNCTION_HPP
#define XFUNCTION_HPP

#include <type_traits>
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

    template <class F, class... E>
    class xfunction : public xexpression<xfunction<F, E...>>
    {

    public:

        using self_type = xfunction<F, E...>;
        using functor_type = F;

        using value_type = typename functor_type::result_type;
        using const_reference = value_type;
        using const_pointer = const value_type*;
        using size_type = detail::common_size_type<E...>;
        using difference_type = detail::common_difference_type<E...>;

        using shape_type = array_shape<size_type>;
        using closure_type = const self_type;

        //using const_iterator = xfunction_iterator<F, E...>;

        inline xfunction(const E&...e)
            : m_e(e...)  // m_e(wrap_scalar(e)...)
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

        // xfunction does not have a non const version of operator()
        template <class... Args>
        const_reference operator()(Args... args) const
        {
            return access_impl(std::make_index_sequence<sizeof...(E)>(), args...);
        }

    private:

        std::tuple<const E&...> m_e;

        template <class... Args, size_t... I>
        const_reference access_impl(std::index_sequence<I...>, Args... args) const
        {
            return F::apply(std::get<I>(m_e)(args...)...);
        }
     };

    template <template <class...> class F, class... E>
    using xfunction_op = xfunction<F<typename E::value_type...>, E...>;

}

#endif

