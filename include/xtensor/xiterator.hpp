/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XITERATOR_HPP
#define XITERATOR_HPP

#include <utility>
#include <tuple>
#include <type_traits>
#include <iterator>
#include <array>
#include <algorithm>

#include "xindex.hpp"
#include "xutils.hpp"
#include "xexception.hpp"

namespace xt
{

    /***********************
     * broadcast functions *
     ***********************/

    template <class S>
    bool broadcast_shape(const xshape<S>& input, xshape<S>& output);

    /************
     * xstepper *
     ************/

    namespace detail
    {
        template <class C>
        struct get_storage_iterator_impl
        {
            using type = typename C::storage_iterator;
        };

        template <class C>
        struct get_storage_iterator_impl<const C>
        {
            using type = typename C::const_storage_iterator;
        };
    }

    template <class C>
    using get_storage_iterator = typename detail::get_storage_iterator_impl<C>::type;

    template <class C>
    class xstepper
    {

    public:

        using container_type = C;
        using subiterator_type = get_storage_iterator<C>;
        using value_type = typename subiterator_type::value_type;
        using reference = typename subiterator_type::reference;
        using pointer = typename subiterator_type::pointer;
        using difference_type = typename subiterator_type::difference_type;
        using size_type = typename container_type::size_type;

        xstepper(container_type* c, subiterator_type it, size_type offset);

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);

        void to_end();

        bool equal(const xstepper& rhs) const;

    private:

        container_type* p_c;
        subiterator_type m_it;
        size_type m_offset;
    };

    template <class C>
    bool operator==(const xstepper<C>& lhs,
                    const xstepper<C>& rhs);

    template <class C>
    bool operator!=(const xstepper<C>& lhs,
                    const xstepper<C>& rhs);

    template <class S>
    void increment_stepper(S& stepper,
                           xshape<typename S::size_type>& index,
                           const xshape<typename S::size_type>& shape);

    /*************
     * xiterator *
     *************/

    template <class It>
    class xiterator
    {

    public:

        using self_type = xiterator<It>;

        using subiterator_type = It;
        using value_type = typename subiterator_type::value_type;
        using reference = typename subiterator_type::reference;
        using pointer = typename subiterator_type::pointer;
        using difference_type = typename subiterator_type::difference_type;
        using size_type = typename subiterator_type::size_type;
        using iterator_category = std::input_iterator_tag;

        using shape_type = xshape<size_type>;

        xiterator(It it, const shape_type& shape);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;

        bool equal(const xiterator& rhs) const;

    private:

        subiterator_type m_it;
        shape_type m_shape;
        shape_type m_index;
    };

    template <class It>
    bool operator==(const xiterator<It>& lhs,
                    const xiterator<It>& rhs);

    template <class It>
    bool operator!=(const xiterator<It>& lhs,
                    const xiterator<It>& rhs);

    /**************************************
     * broadcast functions implementation *
     **************************************/

    template <class S>
    inline bool broadcast_shape(const xshape<S>& input, xshape<S>& output)
    {
        auto size = output.size();
        bool trivial_broadcast = (input.size() == output.size());
        auto output_iter = output.rbegin();
        auto input_rend = input.rend();
        for(auto input_iter = input.rbegin(); input_iter != input_rend;
            ++input_iter, ++output_iter)
        {
            if(*output_iter == 1)
            {
                *output_iter = *input_iter;
            }
            else if((*input_iter != 1) && (*output_iter != *input_iter))
            {
                throw broadcast_error<S>(output, input);
            }
            trivial_broadcast = trivial_broadcast && (*output_iter == *input_iter);
        }
        return trivial_broadcast;
    }

    /***************************
     * xstepper implementation *
     ***************************/

    template <class C>
    inline xstepper<C>::xstepper(container_type* c, subiterator_type it, size_type offset)
        : p_c(c), m_it(it), m_offset(offset)
    {
    }

    template <class C>
    inline auto xstepper<C>::operator*() const -> reference
    {
        return *m_it;
    }

    template <class C>
    inline void xstepper<C>::step(size_type dim, size_type n)
    {
        if(dim >= m_offset)
            m_it += n * p_c->strides()[dim - m_offset];
    }

    template <class C>
    inline void xstepper<C>::step_back(size_type dim, size_type n)
    {
        if(dim >= m_offset)
            m_it -= n * p_c->strides()[dim - m_offset];
    }

    template <class C>
    inline void xstepper<C>::reset(size_type dim)
    {
        if(dim >= m_offset)
            m_it -= p_c->backstrides()[dim - m_offset];
    }

    template <class C>
    inline void xstepper<C>::to_end()
    {
        m_it = p_c->storage_end();
    }

    template <class C>
    inline bool xstepper<C>::equal(const xstepper& rhs) const
    {
        return p_c == rhs.p_c && m_it == rhs.m_it && m_offset == rhs.m_offset;
    }

    template <class C>
    inline bool operator==(const xstepper<C>& lhs,
                           const xstepper<C>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class C>
    inline bool operator!=(const xstepper<C>& lhs,
                           const xstepper<C>& rhs)
    {
        return !(lhs.equal(rhs));
    }

    template <class S>
    void increment_stepper(S& stepper,
                           xshape<typename S::size_type>& index,
                           const xshape<typename S::size_type>& shape)
    {
        using size_type = typename S::size_type;
        for(size_type j = index.size(); j != 0; --j)
        {
            size_type i = j-1;
            if(++index[i] != shape[i])
            {
                stepper.step(i);
                break;
            }
            else if (i == 0)
            {
                stepper.to_end();
            }
            else
            {
                index[i] = 0;
                stepper.reset(i);
            }
        }
    }

    /****************************
     * xiterator implementation *
     ****************************/

    template <class It>
    inline xiterator<It>::xiterator(It it, const shape_type& shape)
        : m_it(it), m_shape(shape), m_index(shape.size(), size_type(0))
    {
    }

    template <class It>
    inline auto xiterator<It>::operator++() -> self_type&
    {
        increment_stepper(m_it, m_index, m_shape);
        return *this;
    }

    template <class It>
    inline auto xiterator<It>::operator++(int) -> self_type
    {
        self_type tmp(*this);
        ++(*this);
        return tmp;
    }

    template <class It>
    inline auto xiterator<It>::operator*() const -> reference
    {
        return *m_it;
    }

    template <class It>
    inline bool xiterator<It>::equal(const xiterator& rhs) const
    {
        return m_it == rhs.m_it && m_shape == rhs.m_shape;
    }

    template <class It>
    inline bool operator==(const xiterator<It>& lhs,
                           const xiterator<It>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class It>
    inline bool operator!=(const xiterator<It>& lhs,
                           const xiterator<It>& rhs)
    {
        return !(lhs.equal(rhs));
    }
}

#endif

