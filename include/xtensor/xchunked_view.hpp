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

#include "xstrided_view.hpp"
#include "xnoalias.hpp"

namespace xt
{

    template <class T>
    inline void xchunked_view_init(T* that)
    {
        that->m_shape.resize(that->m_expression.dimension());
        const auto& s = that->m_expression.shape();
        std::copy(s.cbegin(), s.cend(), that->m_shape.begin());
        // compute chunk number in each dimension
        that->m_shape_of_chunks.resize(that->m_shape.size());
        std::transform
        (
            that->m_shape.cbegin(), that->m_shape.cend(),
            that->m_chunk_shape.cbegin(),
            that->m_shape_of_chunks.begin(),
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
        that->m_ic.resize(that->m_shape.size());
        that->m_chunk_nb = std::accumulate(std::begin(that->m_shape_of_chunks), std::end(that->m_shape_of_chunks), std::size_t(1), std::multiplies<>());
        that->m_sv.resize(that->m_shape.size());
    }

    template <class E, class V>
    class xchunk_iterator
    {
    public:
        xchunk_iterator(V& chunked_view, std::size_t chunk_idx);
        xchunk_iterator() = default;

        xchunk_iterator<E, V>& operator++();
        xchunk_iterator<E, V> operator++(int);
        bool operator==(const xchunk_iterator& other) const;
        bool operator!=(const xchunk_iterator& other) const;
        auto operator*();

    private:
        V* m_pcv;
        std::size_t m_ci;
    };

    template <class E>
    class xchunked_view
    {
    public:
        template <class OE>
        xchunked_view(OE&& e, std::vector<std::size_t>& chunk_shape);

        xchunk_iterator<E, xchunked_view<E>> begin();
        xchunk_iterator<E, xchunked_view<E>> end();

        template <class OE>
        xchunked_view<E>& operator=(const OE& e);

        E m_expression;
        std::vector<std::size_t> m_shape;
        std::vector<std::size_t> m_chunk_shape;
        std::vector<std::size_t> m_shape_of_chunks;
        std::vector<std::size_t> m_ic;
        std::size_t m_chunk_nb;
        xstrided_slice_vector m_sv;

    private:

        friend class xchunk_iterator<E, xchunked_view<E>>;
    };

    template <class E>
    class xchunked_view_on_chunked
    {
    public:
        template <class OE>
        xchunked_view_on_chunked(OE&& e);

        xchunk_iterator<E, xchunked_view_on_chunked<E>> begin();
        xchunk_iterator<E, xchunked_view_on_chunked<E>> end();

        template <class OE>
        xchunked_view_on_chunked<E>& operator=(const OE& e);

        E m_expression;
        std::vector<std::size_t> m_shape;
        std::vector<std::size_t> m_chunk_shape;
        std::vector<std::size_t> m_shape_of_chunks;
        std::vector<std::size_t> m_ic;
        std::size_t m_chunk_nb;
        xstrided_slice_vector m_sv;

    private:

        friend class xchunk_iterator<E, xchunked_view_on_chunked<E>>;
    };

    /////////////////////
    // xchunk_iterator //
    /////////////////////

    template <class E, class V>
    inline xchunk_iterator<E, V>::xchunk_iterator(V& chunked_view, std::size_t chunk_idx)
        : m_pcv(&chunked_view)
        , m_ci(chunk_idx)
    {
    }

    template <class E, class V>
    inline xchunk_iterator<E, V>& xchunk_iterator<E, V>::operator++()
    {
        if (m_ci != m_pcv->m_chunk_nb - 1)
        {
            std::size_t di = m_pcv->m_shape.size() - 1;
            while (true)
            {
                if (m_pcv->m_ic[di] + 1 == m_pcv->m_shape_of_chunks[di])
                {
                    m_pcv->m_ic[di] = 0;
                    m_pcv->m_sv[di] = range(0, m_pcv->m_chunk_shape[di]);
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
                    m_pcv->m_ic[di] += 1;
                    m_pcv->m_sv[di] = range(m_pcv->m_ic[di] * m_pcv->m_chunk_shape[di], (m_pcv->m_ic[di] + 1) * m_pcv->m_chunk_shape[di]);
                    break;
                }
            }
        }
        m_ci++;
        return *this;
    }

    template <class E, class V>
    inline xchunk_iterator<E, V> xchunk_iterator<E, V>::operator++(int)
    {
        xchunk_iterator<E, V> it = *this;
        ++(*this);
        return it;
    }

    template <class E, class V>
    inline bool xchunk_iterator<E, V>::operator==(const xchunk_iterator& other) const
    {
        return m_ci == other.m_ci;
    }

    template <class E, class V>
    inline bool xchunk_iterator<E, V>::operator!=(const xchunk_iterator& other) const
    {
        return !(*this == other);
    }

    template <class E, class V>
    inline auto xchunk_iterator<E, V>::operator*()
    {
        return m_pcv->m_sv;
    }

    ///////////////////
    // xchunked_view //
    ///////////////////

    template <class E>
    template <class OE>
    inline xchunked_view<E>::xchunked_view(OE&& e, std::vector<std::size_t>& chunk_shape)
        : m_expression(std::forward<OE>(e))
        , m_chunk_shape(chunk_shape)
    {
        xchunked_view_init(this);
    }

    template <class E>
    inline xchunk_iterator<E, xchunked_view<E>> xchunked_view<E>::begin()
    {
        auto it = xchunk_iterator<E, xchunked_view<E>>(*this, 0);
        std::transform(m_chunk_shape.begin(), m_chunk_shape.end(), m_sv.begin(),
                       [](auto size) { return range(0, size); });
        std::fill(m_ic.begin(), m_ic.end(), std::size_t(0));
        return it;
    }

    template <class E>
    inline xchunk_iterator<E, xchunked_view<E>> xchunked_view<E>::end()
    {
        auto it = xchunk_iterator<E, xchunked_view<E>>(*this, m_chunk_nb);
        return it;
    }

    template <class E>
    template <class OE>
    xchunked_view<E>& xchunked_view<E>::operator=(const OE& e)
    {
        for (auto it = begin(); it != end(); it++)
        {
            auto sv = *it;
            noalias(strided_view(m_expression, sv)) = strided_view(e, sv);
        }
        return *this;
    }

    //////////////////////////////
    // xchunked_view_on_chunked //
    //////////////////////////////

    template <class E>
    template <class OE>
    inline xchunked_view_on_chunked<E>::xchunked_view_on_chunked(OE&& e)
        : m_expression(std::forward<OE>(e))
    {
        m_chunk_shape.resize(e.dimension());
        const auto& s = e.chunk_shape();
        std::copy(s.cbegin(), s.cend(), m_chunk_shape.begin());
        xchunked_view_init(this);
    }

    template <class E>
    inline xchunk_iterator<E, xchunked_view_on_chunked<E>> xchunked_view_on_chunked<E>::begin()
    {
        auto it = xchunk_iterator<E, xchunked_view_on_chunked<E>>(*this, 0);
        std::transform(m_chunk_shape.begin(), m_chunk_shape.end(), m_sv.begin(),
                       [](auto size) { return range(0, size); });
        std::fill(m_ic.begin(), m_ic.end(), std::size_t(0));
        return it;
    }

    template <class E>
    inline xchunk_iterator<E, xchunked_view_on_chunked<E>> xchunked_view_on_chunked<E>::end()
    {
        auto it = xchunk_iterator<E, xchunked_view_on_chunked<E>>(*this, m_chunk_nb);
        return it;
    }
    template <class E>
    template <class OE>
    xchunked_view_on_chunked<E>& xchunked_view_on_chunked<E>::operator=(const OE& e)
    {
        auto& chunks = m_expression.chunks();
        auto chunk_it = chunks.begin();
        for (auto it = begin(); it != end(); it++)
        {
            auto sv = *it;

            auto rhs = strided_view(e, sv);
            auto rhs_shape = rhs.shape();
            if (rhs_shape != m_expression.chunk_shape())
            {
                xstrided_slice_vector esv(m_chunk_shape.size());  // element slice in edge chunk
                std::transform(rhs_shape.begin(), rhs_shape.end(), esv.begin(),
                               [](auto size) { return range(0, size); });
                noalias(strided_view(*chunk_it, esv)) = rhs;
            }
            else
            {
                noalias(*chunk_it) = rhs;
            }
            chunk_it++;
        }
        return *this;
    }

    ////////////////
    // as_chunked //
    ////////////////

    template <class E>
    auto as_chunked(E&& e, std::vector<std::size_t>& chunk_shape)
    {
        return xchunked_view<E>(std::forward<E>(e), chunk_shape);
    }

    template <class E>
    auto as_chunked(E&& e)
    {
        return xchunked_view_on_chunked<E>(std::forward<E>(e));
    }

}

#endif
