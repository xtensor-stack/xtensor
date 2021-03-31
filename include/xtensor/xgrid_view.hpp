/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_GRID_VIEW_HPP
#define XTENSOR_GRID_VIEW_HPP

#include "xstrided_view.hpp"
#include "xnoalias.hpp"

namespace xt
{

    template <class E>
    class xchunk_generator
    {
    public:
        template <class OE>
        xchunk_generator(OE&& e, std::vector<std::size_t>& shape, std::vector<std::size_t>& chunk_shape);

        xstrided_slice_vector& next();
        bool last_chunk();
        xstrided_slice_vector get_slice();
        auto get_chunk();

        template <class OE>
        xchunk_generator<E>& operator=(const OE& e);

    private:
        std::vector<std::size_t> m_shape;
        std::vector<std::size_t> m_chunk_shape;
        E m_expression;
        bool m_last_chunk;
        std::vector<std::size_t> m_shape_of_chunks;
        std::vector<std::size_t> m_ic;
        xstrided_slice_vector m_sv;
        std::size_t m_ci;
        std::size_t m_chunk_nb;
    };

    template <class E>
    template <class OE>
    inline xchunk_generator<E>::xchunk_generator(OE&& e, std::vector<std::size_t>& shape, std::vector<std::size_t>& chunk_shape)
        : m_shape(shape)
        , m_chunk_shape(chunk_shape)
        , m_expression(std::forward<OE>(e))
    {
        // compute chunk number in each dimension
        m_shape_of_chunks.resize(shape.size());
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

        // initialize index of chunk to 0...
        m_ic.resize(shape.size());
        std::fill(m_ic.begin(), m_ic.end(), std::size_t(0));

        m_last_chunk = false;
        m_ci = 0;
        m_chunk_nb = std::accumulate(std::begin(m_shape_of_chunks), std::end(m_shape_of_chunks), std::size_t(1), std::multiplies<>());
        m_sv.resize(m_shape.size());
        std::transform(m_chunk_shape.begin(), m_chunk_shape.end(), m_sv.begin(),
                       [](auto size) { return range(0, size); });
    }

    template <class E>
    inline auto xchunk_generator<E>::get_chunk()
    {
        return strided_view(m_expression, m_sv);
    }

    template <class E>
    inline xstrided_slice_vector xchunk_generator<E>::get_slice()
    {
        return m_sv;
    }

    template <class E>
    inline bool xchunk_generator<E>::last_chunk()
    {
        return m_last_chunk;
    }

    template <class E>
    inline xstrided_slice_vector& xchunk_generator<E>::next()
    {
        m_last_chunk = m_ci == m_chunk_nb - 1;
        if (!m_last_chunk)
        {
            std::size_t di = m_shape.size() - 1;
            while (true)
            {
                if (m_ic[di] + 1 == m_shape_of_chunks[di])
                {
                    m_ic[di] = 0;
                    m_sv[di] = range(0, m_chunk_shape[di]);
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
                    m_ic[di] += 1;
                    m_sv[di] = range(m_ic[di] * m_chunk_shape[di], (m_ic[di] + 1) * m_chunk_shape[di]);
                    break;
                }
            }
        }
        ++m_ci;
        return m_sv;
    }

    template <class E>
    template <class OE>
    xchunk_generator<E>& xchunk_generator<E>::operator=(const OE& e)
    {
        while (!last_chunk())
        {
            auto sv = get_slice();
            auto chunk = get_chunk();
            noalias(chunk) = strided_view(e, sv);
            next();
        }
        return *this;
    }

    template <class E>
    auto grid_view(E&& e, std::vector<std::size_t>& shape, std::vector<std::size_t>& chunk_shape)
    {
        return xchunk_generator<E>(std::forward<E>(e), shape, chunk_shape);
    }

}

#endif
