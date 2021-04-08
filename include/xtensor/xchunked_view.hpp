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
        
        using size_type = size_t;
        using shape_type = svector<size_type>;

        template <class OE, class S>
        xchunked_view(OE&& e, S&& chunk_shape);

        xchunk_iterator<E> chunk_begin();
        xchunk_iterator<E> chunk_end();

        template <class OE>
        xchunked_view<E>& operator=(const OE& e);

    private:

        E m_expression;
        shape_type m_shape;
        shape_type m_chunk_shape;
        shape_type m_grid_shape;
        size_type m_chunk_nb;

        friend class xchunk_iterator<E>;
    };

    template <class E, class S>
    xchunked_view<E> as_chunked(E&& e, S&& chunk_shape);

    /*********************
     * xchunked_iterator *
     *********************/

    template <class E>
    class xchunk_iterator
    {
    public:

        using view_type = xchunked_view<E>;
        using size_type = typename view_type::size_type;
        using shape_type = typename view_type::shape_type;
        using slice_vector = xstrided_slice_vector;

        xchunk_iterator() = default;
        xchunk_iterator(view_type& view,
                        shape_type&& chunk_index,
                        size_type chunk_linear_index);

        xchunk_iterator<E>& operator++();
        xchunk_iterator<E> operator++(int);
        auto operator*();

        bool operator==(const xchunk_iterator& other) const;
        bool operator!=(const xchunk_iterator& other) const;

        const slice_vector& get_slice_vector() const;

    private:

        view_type* p_chunked_view;
        shape_type m_chunk_index;
        size_type m_chunk_linear_index;
        xstrided_slice_vector m_slice_vector;
    };

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
    inline xchunk_iterator<E> xchunked_view<E>::chunk_begin()
    {
        shape_type chunk_index(m_shape.size(), size_type(0));
        return xchunk_iterator<E>(*this, std::move(chunk_index), 0u);
    }

    template <class E>
    inline xchunk_iterator<E> xchunked_view<E>::chunk_end()
    {
        auto it = xchunk_iterator<E>(*this, shape_type(m_grid_shape), m_chunk_nb);
        return it;
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

    template <class E, class S>
    inline xchunked_view<E> as_chunked(E&& e, S&& chunk_shape)
    {
        return xchunked_view<E>(std::forward<E>(e), std::forward<S>(chunk_shape));
    }

    /**********************************
     * xchunk_iterator implementation *
     **********************************/

    template <class E>
    inline xchunk_iterator<E>::xchunk_iterator(view_type& view, shape_type&& chunk_index, size_type chunk_linear_index)
        : p_chunked_view(&view)
        , m_chunk_index(std::move(chunk_index))
        , m_chunk_linear_index(chunk_linear_index)
        , m_slice_vector(m_chunk_index.size())
    {
        for (size_type i = 0; i < m_chunk_index.size(); ++i)
        {
            if (m_chunk_index[i] == 0)
            {
                m_slice_vector[i] = range(0, p_chunked_view->m_chunk_shape[i]);
            }
            else
            {
                m_slice_vector[i] = range(m_chunk_index[i] * p_chunked_view->m_chunk_shape[i],
                                          (m_chunk_index[i] + 1) * p_chunked_view->m_chunk_shape[i]);
            }
        }
    }

    template <class E>
    inline xchunk_iterator<E>& xchunk_iterator<E>::operator++()
    {
        if (m_chunk_linear_index != p_chunked_view->m_chunk_nb - 1)
        {
            size_type di = p_chunked_view->m_shape.size() - 1;
            while (true)
            {
                if (m_chunk_index[di] + 1 == p_chunked_view->m_grid_shape[di])
                {
                    m_chunk_index[di] = 0;
                    m_slice_vector[di] = range(0, p_chunked_view->m_chunk_shape[di]);
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
                    m_chunk_index[di] += 1;
                    m_slice_vector[di] = range(m_chunk_index[di] * p_chunked_view->m_chunk_shape[di],
                                                     (m_chunk_index[di] + 1) * p_chunked_view->m_chunk_shape[di]);
                    break;
                }
            }
        }
        m_chunk_linear_index++;
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
    inline auto xchunk_iterator<E>::operator*()
    {
        return strided_view(p_chunked_view->m_expression, m_slice_vector);
    }

    template <class E>
    inline bool xchunk_iterator<E>::operator==(const xchunk_iterator& other) const
    {
        return m_chunk_linear_index == other.m_chunk_linear_index;
    }

    template <class E>
    inline bool xchunk_iterator<E>::operator!=(const xchunk_iterator& other) const
    {
        return !(*this == other);
    }

    template <class E>
    inline auto xchunk_iterator<E>::get_slice_vector() const -> const slice_vector& 
    {
        return m_slice_vector;
    }
}

#endif
