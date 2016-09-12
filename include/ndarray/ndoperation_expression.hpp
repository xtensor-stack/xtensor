#ifndef NDOPERATION_EXPRESSION_HPP
#define NDOPERATION_EXPRESSION_HPP

#include <type_traits>
#include "ndarray_expression.hpp"
#include "ndbroadcast.hpp"

namespace qs
{
    
    namespace detail
    {
        template <class... Args>
        using common_size_type = std::common_type_t<typename Args::size_type...>;

        template <class... Args>
        using common_difference_type = std::common_type_t<typename Args::difference_type...>;

        template <class... Args>
        using common_value_type = std::common_type_t<typename Args::value_type...>;
    }


    /***************************************
     * Unary operation on a ndexpression
     ***************************************/

    template <class F, class E>
    class ndunary : public ndexpression<ndunary<F, E>>
    {

    public:

        using self_type = ndunary<F, E>;
        using functor_type = F;
        
        using value_type = typename functor_type::result_type;
        using const_reference = value_type;
        using const_pointer = const value_type*;
        using size_type = typename E::size_type;
        using difference_type = typename E::difference_type;

        using shape_type = array_shape<size_type>;

        using expression_closure_type = typename E::closure_type;
        using closure_type = const self_type;

        using subiterator_type = typename E::const_iterator;
        using const_iterator = ndunary_iterator<F, E>;

        inline ndunary(const E& e) : m_e(e)
        {
        }

        inline size_type nb_dim() const
        {
            return m_e.nb_dim();
        }

        inline bool broadcast_shape(shape_type& shape) const
        {
            return m_e.broadcast_shape(shape);
        }

        inline const_iterator begin() const { return const_iterator(m_e.begin()); }
        inline const_iterator end() const { return const_iterator(m_e.end()); }

        inline const_iterator cbegin() const { return const_iterator(m_e.cbegin()); }
        inline const_iterator cend() const { return conost_iterator(m_e.cend()); }

    private:

        expression_closure_type m_e;
    };

    template <template <class> class F, class E>
    using ndunary_op = ndunary<F<typename E::value_type>, E>;


    /******************************************
     * Binary operation on two ndexpression
     ******************************************/

    template <class F, class E1, class E2>
    class ndbinary : public ndexpression<ndbinary<F, E1, E2>>
    {

    public:

        using self_type = ndbinary<F, E1, E2>;
        using functor_type = F;

        using value_type = typename functor_type::result_type;
        using const_reference = value_type;
        using const_pointer = const value_type*;
        using size_type = detail::common_size_type<E1, E2>;
        using difference_type = detail::common_difference_type<E1, E2>;

        using shape_type = array_shape<size_type>;

        using expression1_closure_type = typename E1::closure_type;
        using expression2_closure_type = typename E2::closure_type;
        using closure_type = const self_type;

        using subiterator1_type = typename E1::const_iterator;
        using subiterator2_type = typename E2::const_iterator;
        using const_iterator = ndbinary_iterator<F, E1, E2>;

        inline ndbinary(const E1& e1, const E2& e2)
            : m_e1(e1), m_e2(e2)
        {
        }

        inline size_type nb_dim() const
        {
            return std::max(m_e1.nb_dim(), m_e2.nb_dim());
        }

        inline bool broadcast_shape(shape_type& shape) const
        {
            bool res1 = m_e1.broadcast_shape(shape);
            bool res2 = m_e2.broadcast_shape(shape);
            return res1 && res2;
        }

    private:

        expression1_closure_type m_e1;
        expression2_closure_type m_e2;
    };

    template <template <class, class> class F, class E1, class E2>
    using ndbinary_op = ndbinary<F<typename E1::value_type, typename E2::value_type>, E1, E2>;

}

#endif

