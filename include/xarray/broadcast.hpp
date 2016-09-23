#ifndef BROADCAST_HPP
#define BROADCAST_HPP

#include <utility>
#include <tuple>
#include <type_traits>
#include <stdexcept>
#include <iterator>
#include <array>
#include <algorithm>

#include "xindex.hpp"
#include "utils.hpp"

namespace qs
{

    /*************************
     * Broadcast functions
     *************************/

    template <class S, size_t N>
    S broadcast_dim(std::array<S, N>& dim_list);

    template <class S>
    bool broadcast_shape(const array_shape<S>& input, array_shape<S>& output);

    template <class S>
    bool check_trivial_broadcast(const array_strides<S>& strides1,
                                 const array_strides<S>& strides2);


    /***************************
     * broadcasting_iterator
     ***************************/

    template <class C>
    class broadcasting_iterator
    {

    public:

        using container_type = C;
        using subiterator_type = typename container_type::const_iterator;
        using value_type = typename subiterator_type::value_type;
        using reference = typename subiterator_type::reference;
        using pointer = typename subiterator_type::pointer;
        using difference_type = typename subiterator_type::difference_type;
        using size_type = typename container_type::size_type;
        using iterator_category = std::input_iterator_tag;

        broadcasting_iterator(const container_type* c, subiterator_type it);
        reference operator*() const;

        void increment(size_type i);
        void reset(size_type i);

        bool equal(const broadcasting_iterator& rhs) const;

    private:

        const container_type* p_c;
        subiterator_type m_it;
    };

    template <class C>
    bool operator==(const broadcasting_iterator<C>& lhs,
                    const broadcasting_iterator<C>& rhs);

    template <class C>
    bool operator!=(const broadcasting_iterator<C>& lhs,
                    const broadcasting_iterator<C>& rhs);


    /**********************
     * indexed_iterator
     **********************/

    template <class E>
    class indexed_iterator
    {

    public:

        using self_type = indexed_iterator<E>;

        using subiterator_type = typename E::broadcasting_iterator;
        using value_type = typename subiterator_type::value_type;
        using reference = typename subiterator_type::reference;
        using pointer = typename subiterator_type::pointer;
        using difference_type = typename subiterator_type::difference_type;
        using size_type = typename E::size_type;
        using iterator_category = std::input_iterator_tag;
        
        using shape_type = array_shape<size_type>;
        
        indexed_iterator(const E& e, const shape_type& shape);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;

        bool equal(const indexed_iterator& rhs) const;

    private:

        subiterator_type m_it;
        shape_type m_shape;
        shape_type m_index;
    };

    template <class E>
    bool operator==(const indexed_iterator<E>& lhs,
                    const indexed_iterator<E>& rhs);

    template <class E>
    bool operator!=(const indexed_iterator<E>& lhs,
                    const indexed_iterator<E>& rhs);


    /****************************************
     * Broadcast functions implementation
     ****************************************/

    template <class S, size_t N>
    inline S broadcast_dim(const std::array<S, N>& dim_list)
    {
        S ndim = std::accumulate(dim_list.begin(), dim_list.end(), S(0),
                [](S res, const S& dim) { return std::max(dim, res); });
        return ndim;
    }

    template <class S>
    inline bool broadcast_shape(const array_shape<S>& input, array_shape<S>& output)
    {
        size_t size = output.size();
        bool trivial_broadcast = (input.size() == output.size());
        auto output_iter = output.rbegin();
        for(auto input_iter = input.rbegin(); input_iter != input.rend();
            ++input_iter, ++output_iter)
        {
            if(*output_iter == 1)
            {
                *output_iter = *input_iter;
            }
            else if((*input_iter != 1) && (*output_iter != *input_iter))
            {
                throw std::runtime_error("broadcast error : incompatible dimension of inputs");
            }
            trivial_broadcast = trivial_broadcast && (*output_iter == *input_iter);
        }
        return trivial_broadcast;
    }

    template <class S>
    inline bool check_trivial_broadcast(const array_strides<S>& strides1,
                                        const array_strides<S>& strides2)
    {
        return strides1 == strides2;
    }


    /******************************************
     * broadcasting_iterator implementation
     ******************************************/

    template <class C>
    inline broadcasting_iterator<C>::broadcasting_iterator(const container_type* c, subiterator_type it)
        : p_c(c), m_it(it)
    {
    }

    template <class C>
    inline auto broadcasting_iterator<C>::operator*() const -> reference
    {
        return *m_it;
    }

    template <class C>
    inline void broadcasting_iterator<C>::increment(size_type dim)
    {
        if(dim < p_c->dimension())
            m_it += p_c->strides()[dim];
    }

    template <class C>
    inline void broadcasting_iterator<C>::reset(size_type dim)
    {
        if(dim < p_c->dimension())
            m_it -= p_c->backstrides()[dim];
    }

    template <class C>
    inline bool broadcasting_iterator<C>::equal(const broadcasting_iterator& rhs) const
    {
        return p_c == rhs.p_c && m_it == rhs.m_it;
    }

    template <class C>
    inline bool operator==(const broadcasting_iterator<C>& lhs,
                           const broadcasting_iterator<C>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class C>
    inline bool operator!=(const broadcasting_iterator<C>& lhs,
                           const broadcasting_iterator<C>& rhs)
    {
        return !(lhs.equal(rhs));
    }


    /*************************************
     * indexed_iterator implementation
     *************************************/

    template <class E>
    inline indexed_iterator<E>::indexed_iterator(const E& e, const shape_type& shape)
        : m_it(e.begin(shape)), m_shape(shape), m_index(shape.size(), size_type(0))
    {
    }

    template <class E>
    inline auto indexed_iterator<E>::operator++() -> self_type&
    {
        for(size_type j = m_index.size(); j != 0; --j)
        {
            size_type i = j-1;
            if(++m_index[i] != m_shape[i])
            {
                m_it.increment(i);
                break;
            }
            else
            {
                m_index[i] = 0;
                m_it.reset(i);
            }
        }
    }

    template <class E>
    inline auto indexed_iterator<E>::operator++(int) -> self_type
    {
        self_type tmp(*this);
        ++(*this);
        return tmp;
    }

    template <class E>
    inline auto indexed_iterator<E>::operator*() const -> reference
    {
        return *m_it;
    }

    template <class E>
    inline bool indexed_iterator<E>::equal(const indexed_iterator& rhs) const
    {
        return m_it == rhs.m_it && m_shape == rhs.m_shape && m_index == rhs.m_index;
    }

    template <class E>
    inline bool operator==(const indexed_iterator<E>& lhs,
                           const indexed_iterator<E>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class E>
    inline bool operator!=(const indexed_iterator<E>& lhs,
                           const indexed_iterator<E>& rhs)
    {
        return !(lhs.equal(rhs));
    }

}

#endif

