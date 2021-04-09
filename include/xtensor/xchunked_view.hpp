/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_CHUNKED_VIEW_HPP
#define XTENSOR_CHUNKED_VIEW_HPP

#include <xtl/xsequence.hpp>

#include "xnoalias.hpp"
#include "xstorage.hpp"
#include "xstrided_view.hpp"

namespace xt
{

    /*****************
     * xchunked_view *
     *****************/

    template <class E>
    class xchunk_iterator;

    template <class E>
    class xchunked_view
    {
    public:
        
        using self_type = xchunked_view<E>;
        using expression_type = std::decay_t<E>;
        using size_type = size_t;
        using shape_type = svector<size_type>;
        using chunk_iterator_type = xchunk_iterator<self_type>;

        template <class OE, class S>
        xchunked_view(OE&& e, S&& chunk_shape);

        template <class OE>
        xchunked_view<E>& operator=(const OE& e);

        size_type dimension() const noexcept;
        const shape_type& shape() const noexcept;
        const shape_type& chunk_shape() const noexcept;
        size_type grid_size() const noexcept;
        const shape_type& grid_shape() const noexcept;

        expression_type& expression() noexcept;
        const expression_type& expression() const noexcept;

        chunk_iterator_type chunk_begin();
        chunk_iterator_type chunk_end();

    private:

        E m_expression;
        shape_type m_shape;
        shape_type m_chunk_shape;
        shape_type m_grid_shape;
        size_type m_chunk_nb;
    };

    template <class E, class S>
    xchunked_view<E> as_chunked(E&& e, S&& chunk_shape);

    /********************************
     * xchunked_view implementation *
     ********************************/

    template <class E>
    template <class OE, class S>
    inline xchunked_view<E>::xchunked_view(OE&& e, S&& chunk_shape)
        : m_expression(std::forward<OE>(e))
        , m_chunk_shape(xtl::forward_sequence<shape_type, S>(chunk_shape))
    {
        m_shape.resize(e.dimension());
        const auto& s = e.shape();
        std::copy(s.cbegin(), s.cend(), m_shape.begin());
        // compute chunk number in each dimension
        m_grid_shape.resize(m_shape.size());
        std::transform
        (
            m_shape.cbegin(), m_shape.cend(),
            m_chunk_shape.cbegin(),
            m_grid_shape.begin(),
            [](auto s, auto cs)
            {
                std::size_t cn = s / cs;
                if (s % cs > 0)
                {
                    cn++; // edge_chunk
                }
                return cn;
            }
        );
        m_chunk_nb = std::accumulate(std::begin(m_grid_shape), std::end(m_grid_shape), std::size_t(1), std::multiplies<>());
    }

    template <class E>
    template <class OE>
    xchunked_view<E>& xchunked_view<E>::operator=(const OE& e)
    {
        for (auto it = chunk_begin(); it != chunk_end(); it++)
        {
            auto el = *it;
            noalias(el) = strided_view(e, it.get_slice_vector());
        }
        return *this;
    }

    template <class E>
    inline auto xchunked_view<E>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    template <class E>
    inline auto xchunked_view<E>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    template <class E>
    inline auto xchunked_view<E>::chunk_shape() const noexcept -> const shape_type&
    {
        return m_chunk_shape;
    }

    template <class E>
    inline auto xchunked_view<E>::grid_size() const noexcept -> size_type
    {
        return m_chunk_nb;
    }

    template <class E>
    inline auto xchunked_view<E>::grid_shape() const noexcept -> const shape_type&
    {
        return m_grid_shape;
    }

    template <class E>
    inline auto xchunked_view<E>::expression() noexcept -> expression_type&
    {
        return m_expression;
    }

    template <class E>
    inline auto xchunked_view<E>::expression() const noexcept -> const expression_type&
    {
        return m_expression;
    }

    template <class E>
    inline auto xchunked_view<E>::chunk_begin() -> chunk_iterator_type
    {
        shape_type chunk_index(m_shape.size(), size_type(0));
        return chunk_iterator_type(*this, std::move(chunk_index), 0u);
    }

    template <class E>
    inline auto xchunked_view<E>::chunk_end() -> chunk_iterator_type
    {
        return chunk_iterator_type(*this, shape_type(grid_shape()), grid_size());
    }

    template <class E, class S>
    inline xchunked_view<E> as_chunked(E&& e, S&& chunk_shape)
    {
        return xchunked_view<E>(std::forward<E>(e), std::forward<S>(chunk_shape));
    }
}

#endif
