#ifndef NDOPERATION_EXPRESSION_HPP
#define NDOPERATION_EXPRESSION_HPP

#include <type_traits>
#include "ndutils.hpp"
#include "ndarray_expression.hpp"
#include "ndbroadcast.hpp"

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
    class ndfunction : public ndexpression<ndfunction<F, E...>>
    {

    public:

        using self_type = ndfunction<F, E...>;
        using functor_type = F;

        using value_type = typename functor_type::result_type;
        using const_reference = value_type;
        using const_pointer = const value_type*;
        using size_type = detail::common_size_type<E...>;
        using difference_type = detail::common_difference_type<E...>;

        using shape_type = array_shape<size_type>;
        using closure_type = const self_type;

        using const_iterator = ndfunction_iterator<F, E...>;

        inline ndfunction(const E&...e)
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
            auto func = [](bool b, auto&& e) { return b && e.broadcast_shape(shape); };
            return accumulate(func, True, e);
        }

    private:

        std::tuple<const E&...> m_e;
    };

    template <template <class, class> class F, class... E>
    using ndfunction_op = ndfunction<F<typename E::value_type...>, E...>;

}

#endif

