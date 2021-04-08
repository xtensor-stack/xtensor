/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_CHUNKED_ASSIGN_HPP
#define XTENSOR_CHUNKED_ASSIGN_HPP

#include "xnoalias.hpp"
#include "xstrided_view.hpp"

namespace xt
{
    template <class T, class chunk_storage>
    class xchunked_assigner
    {
    public:

        using temporary_type = T;

        template <class E, class DST>
        void build_and_assign_temporary(const xexpression<E>& e, DST& dst);
    };

    /*********************************
     * xchunked_semantic declaration *
     *********************************/
    
    template <class D>
    class xchunked_semantic : public xsemantic_base<D>
    {
    public:

        using base_type = xsemantic_base<D>;
        using derived_type = D;
        using temporary_type = typename base_type::temporary_type;

        template <class E>
        derived_type& assign_xexpression(const xexpression<E>& e);

        template <class E>
        derived_type& computed_assign(const xexpression<E>& e);

        template <class E, class F>
        derived_type& scalar_computed_assign(const E& e, F&& f);

    protected:

        xchunked_semantic() = default;
        ~xchunked_semantic() = default;

        xchunked_semantic(const xchunked_semantic&) = default;
        xchunked_semantic& operator=(const xchunked_semantic&) = default;

        xchunked_semantic(xchunked_semantic&&) = default;
        xchunked_semantic& operator=(xchunked_semantic&&) = default;

        template <class E>
        derived_type& operator=(const xexpression<E>& e);

    private:

        template <class CS>
        xchunked_assigner<temporary_type, CS> get_assigner(const CS&) const;
    };

    /************************************
     * xchunked_semantic implementation *
     ************************************/

    template <class T, class CS>
    template <class E, class DST>
    inline void xchunked_assigner<T, CS>::build_and_assign_temporary(const xexpression<E>& e, DST& dst)
    {
        temporary_type tmp(e, CS(), dst.chunk_shape());
        dst = std::move(tmp);
    }

    template <class D>
    template <class E>
    inline auto xchunked_semantic<D>::assign_xexpression(const xexpression<E>& e) -> derived_type&
    {
        using shape_type = std::decay_t<decltype(this->derived_cast().shape())>;
        using size_type = typename shape_type::size_type;
        const auto& chunk_shape = this->derived_cast().chunk_shape();
        auto& chunks = this->derived_cast().chunks();
        size_t dimension = this->derived_cast().dimension();
        xstrided_slice_vector sv(chunk_shape.size());  // element slice corresponding to chunk
        std::transform(chunk_shape.begin(), chunk_shape.end(), sv.begin(),
                       [](auto size) { return range(0, size); });
        shape_type ic(dimension);  // index of chunk, initialized to 0...
        size_type ci = 0;
        for (auto& chunk: chunks)
        {
            auto rhs = strided_view(e.derived_cast(), sv);
            auto rhs_shape = rhs.shape();
            if (rhs_shape != chunk_shape)
            {
                xstrided_slice_vector esv(chunk_shape.size());  // element slice in edge chunk
                std::transform(rhs_shape.begin(), rhs_shape.end(), esv.begin(),
                               [](auto size) { return range(0, size); });
                noalias(strided_view(chunk, esv)) = rhs;
            }
            else
            {
                noalias(chunk) = rhs;
            }
            bool last_chunk = ci == chunks.size() - 1;
            if (!last_chunk)
            {
                size_type di = dimension - 1;
                while (true)
                {
                    if (ic[di] + 1 == chunks.shape()[di])
                    {
                        ic[di] = 0;
                        sv[di] = range(0, chunk_shape[di]);
                        if (di == 0)
                        {
                            break;
                        }
                        else
                        {
                            di--;
                        }
                    }
                    else
                    {
                        ic[di] += 1;
                        sv[di] = range(ic[di] * chunk_shape[di], (ic[di] + 1) * chunk_shape[di]);
                        break;
                    }
                }
            }
            ++ci;
        }
        return this->derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xchunked_semantic<D>::computed_assign(const xexpression<E>& e) -> derived_type&
    {
        D& d = this->derived_cast();
        if (e.derived_cast().dimension() > d.dimension()
            || e.derived_cast().shape() > d.shape())
        {
            return operator=(e);
        }
        else
        {
            return assign_xexpression(e);
        }
    }

    template <class D>
    template <class E, class F>
    inline auto xchunked_semantic<D>::scalar_computed_assign(const E& e, F&& f) -> derived_type&
    {
        for (auto& c: this->derived_cast().chunks())
        {
            c.scalar_computed_assign(e, f);
        }
        return this->derived_cast();
    }

    template <class D>
    template <class E>
    inline auto xchunked_semantic<D>::operator=(const xexpression<E>& e) -> derived_type&
    {
        D& d = this->derived_cast();
        get_assigner(d.chunks()).build_and_assign_temporary(e, d);
        return d;
    }

    template <class D>
    template <class CS>
    inline auto xchunked_semantic<D>::get_assigner(const CS&) const -> xchunked_assigner<temporary_type, CS>
    {
        return xchunked_assigner<temporary_type, CS>();
    }
}

#endif

