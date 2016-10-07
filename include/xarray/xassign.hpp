#ifndef XASSIGN_HPP
#define XASSIGN_HPP

#include "xbroadcast.hpp"
#include "xindex.hpp"

namespace qs
{
    template <class E>
    class xexpression;


    /**********************
     * Assign functions
     **********************/

    template <class E1, class E2>
    inline void assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial);

    template <class E1, class E2>
    inline bool reshape(xexpression<E1>& e1, const xexpression<E2>& e2);

    template <class E1, class E2>
    inline void assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2);

    template <class E1, class E2>
    inline void computed_assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2);


    /*******************
     * data_assigner
     *******************/

    template <class E1, class E2>
    class data_assigner
    {

    public:

        using lhs_iterator = typename E1::stepper;
        using rhs_iterator = typename E2::const_stepper;
        using shape_type = typename E1::shape_type;
        using size_type = typename lhs_iterator::size_type;

        data_assigner(lhs_iterator lhs_begin, rhs_iterator rhs_begin,
                      rhs_iterator rhs_end, const shape_type& shape);

        void run();

        void step(size_type i);
        void reset(size_type i);

        void to_end();
        
    private:

        lhs_iterator m_lhs;
        rhs_iterator m_rhs;
        rhs_iterator m_rhs_end;

        shape_type m_shape;
        shape_type m_index;
    };


    /*************************************
     * Assign functions implementation
     *************************************/

    template <class E1, class E2>
    inline void assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial)
    {
        E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();
        bool trivial_broadcast = trivial && de2.is_trivial_broadcast(de1.strides());
        if(trivial_broadcast)
        {
            std::copy(de2.storage_begin(), de2.storage_end(), de1.storage_begin());
        }
        else
        {
            const auto& shape = de1.shape();
            data_assigner<E1, E2> assigner(de1.stepper_begin(shape), de2.stepper_begin(shape), de2.stepper_end(shape), shape);
            assigner.run();
        }
    }

    template <class E1, class E2>
    inline bool reshape(xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        using shape_type = typename E1::shape_type;
        using size_type = typename E1::size_type;
        const E2& de2 = e2.derived_cast();
        size_type size = de2.dimension();
        shape_type shape(size, size_type(1));
        bool trivial_broadcast = de2.broadcast_shape(shape);
        e1.derived_cast().reshape(shape);
        return trivial_broadcast;
    }

    template <class E1, class E2>
    inline void assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        bool trivial_broadcast = reshape(e1, e2);
        assign_data(e1, e2, trivial_broadcast);
    }

    template <class E1, class E2>
    inline void computed_assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2)
    {
        using shape_type = typename E1::shape_type;
        using size_type = typename E1::size_type;

        E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();

        size_type dim = de2.dimension();
        shape_type shape(dim, size_type(1));
        bool trivial_broadcast = de2.broadcast_shape(shape);

        if(dim > de1.dimension() || shape > de1.shape())
        {
            typename E1::temporary_type tmp(shape);
            assign_data(tmp, e2, trivial_broadcast);
            de1.assign_temporary(tmp);
        }
        else
        {
            assign_data(e1, e2, trivial_broadcast);
        }
    }


    /**********************************
     * data_assigner implementation
     **********************************/

    template <class E1, class E2>
    inline data_assigner<E1, E2>::data_assigner(lhs_iterator lhs_begin, rhs_iterator rhs_begin,
                                                rhs_iterator rhs_end, const shape_type& shape)
        : m_lhs(lhs_begin), m_rhs(rhs_begin), m_rhs_end(rhs_end), m_shape(shape),
          m_index(shape.size(), size_type(0))
    {
    }

    template <class E1, class E2>
    inline void data_assigner<E1, E2>::run()
    {
        while(m_rhs != m_rhs_end)
        {
            *m_lhs = *m_rhs;
            increment_stepper(*this, m_index, m_shape);
        }
    }

    template <class E1, class E2>
    inline void data_assigner<E1, E2>::step(size_type i)
    {
        m_lhs.step(i);
        m_rhs.step(i);
    }

    template <class E1, class E2>
    inline void data_assigner<E1, E2>::reset(size_type i)
    {
        m_lhs.reset(i);
        m_rhs.reset(i);
    }

    template <class E1, class E2>
    inline void data_assigner<E1, E2>::to_end()
    {
        m_lhs.to_end();
        m_rhs.to_end();
    }
}

#endif

