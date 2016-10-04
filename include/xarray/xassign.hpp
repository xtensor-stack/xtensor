#ifndef XASSIGN_HPP
#define XASSIGN_HPP

#include "broadcast.hpp"
#include "xindex.hpp"

namespace qs
{
    template <class E>
    class xexpression;

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
            // TODO : improve performance, this implementation allocates
            // two xiterators, and thus two xshape
            std::copy(de2.xbegin(de1.shape()), de2.xend(de1.shape()), de1.begin());
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
}

#endif

