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

    template <class E>
    class xchunked_view;

    template <class E>
    class xchunk_iterator
    {
    public:
        xchunk_iterator(xchunked_view<E>& chunked_view, std::size_t chunk_idx);
        xchunk_iterator() = default;

        xchunk_iterator<E>& operator++();
        xchunk_iterator<E> operator++(int);
        bool operator==(const xchunk_iterator& other) const;
        bool operator!=(const xchunk_iterator& other) const;
        auto operator*();

    private:
        xchunked_view<E>* m_pcv;
        std::size_t m_ci;
    };

    template <class E>
    class xchunked_view
    {
    public:
        template <class OE>
        xchunked_view(OE&& e, std::vector<std::size_t>& chunk_shape);

        xchunk_iterator<E> begin();
        xchunk_iterator<E> end();

        template <class OE>
        xchunked_view<E>& operator=(const OE& e);

    private:
        E m_expression;
        std::vector<std::size_t> m_shape;
        std::vector<std::size_t> m_chunk_shape;
        std::vector<std::size_t> m_shape_of_chunks;
        std::vector<std::size_t> m_ic;
        std::size_t m_chunk_nb;
        xstrided_slice_vector m_sv;

        friend class xchunk_iterator<E>;
    };

    template <class E>
    inline xchunk_iterator<E>::xchunk_iterator(xchunked_view<E>& chunked_view, std::size_t chunk_idx)
        : m_pcv(&chunked_view)
        , m_ci(chunk_idx)
    {
    }

    template <class E>
    inline xchunk_iterator<E>& xchunk_iterator<E>::operator++()
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

    template <class E>
    inline xchunk_iterator<E> xchunk_iterator<E>::operator++(int)
    {
        xchunk_iterator<E> it = *this;
        ++(*this);
        return it;
    }

    template <class E>
    inline bool xchunk_iterator<E>::operator==(const xchunk_iterator& other) const
    {
        return m_ci == other.m_ci;
    }

    template <class E>
    inline bool xchunk_iterator<E>::operator!=(const xchunk_iterator& other) const
    {
        return !(*this == other);
    }

    template <class E>
    inline auto xchunk_iterator<E>::operator*()
    {
        auto chunk = strided_view(m_pcv->m_expression, m_pcv->m_sv);
        return std::make_pair(chunk, m_pcv->m_sv);
    }

    template <class E>
    template <class OE>
    inline xchunked_view<E>::xchunked_view(OE&& e, std::vector<std::size_t>& chunk_shape)
        : m_expression(std::forward<OE>(e))
        , m_chunk_shape(chunk_shape)
    {
        m_shape.resize(e.dimension());
        const auto& s = e.shape();
        std::copy(s.cbegin(), s.cend(), m_shape.begin());
        // compute chunk number in each dimension
        m_shape_of_chunks.resize(m_shape.size());
        std::transform
        (
            m_shape.cbegin(), m_shape.cend(),
            m_chunk_shape.cbegin(),
            m_shape_of_chunks.begin(),
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
        m_ic.resize(m_shape.size());
        m_chunk_nb = std::accumulate(std::begin(m_shape_of_chunks), std::end(m_shape_of_chunks), std::size_t(1), std::multiplies<>());
        m_sv.resize(m_shape.size());
    }

    template <class E>
    inline xchunk_iterator<E> xchunked_view<E>::begin()
    {
        auto it = xchunk_iterator<E>(*this, 0);
        std::transform(m_chunk_shape.begin(), m_chunk_shape.end(), m_sv.begin(),
                       [](auto size) { return range(0, size); });
        std::fill(m_ic.begin(), m_ic.end(), std::size_t(0));
        return it;
    }

    template <class E>
    inline xchunk_iterator<E> xchunked_view<E>::end()
    {
        auto it = xchunk_iterator<E>(*this, m_chunk_nb);
        return it;
    }

    template <class E>
    template <class OE>
    xchunked_view<E>& xchunked_view<E>::operator=(const OE& e)
    {
        for (auto it = begin(); it != end(); it++)
        {
            auto el = *it;
            noalias(el.first) = strided_view(e, el.second);
        }
        return *this;
    }

    template <class E>
    auto as_chunked(E&& e, std::vector<std::size_t>& chunk_shape)
    {
        return xchunked_view<E>(std::forward<E>(e), chunk_shape);
    }

}

#endif
