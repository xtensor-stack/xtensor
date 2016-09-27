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
    bool broadcast_shape(const xshape<S>& input, xshape<S>& output);

    template <class S>
    bool check_trivial_broadcast(const xstrides<S>& strides1,
                                 const xstrides<S>& strides2);


    /***************************
     * xstepper
     ***************************/

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

        void step(size_type i);
        void reset(size_type i);

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


    /**********************
     * xiterator
     **********************/

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
    inline bool broadcast_shape(const xshape<S>& input, xshape<S>& output)
    {
        size_t size = output.size();
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
                throw std::runtime_error("broadcast error : incompatible dimension of arrays");
            }
            trivial_broadcast = trivial_broadcast && (*output_iter == *input_iter);
        }
        return trivial_broadcast;
    }

    template <class S>
    inline bool check_trivial_broadcast(const xstrides<S>& strides1,
                                        const xstrides<S>& strides2)
    {
        return strides1 == strides2;
    }


    /*****************************
     * xstepper implementation
     *****************************/

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
    inline void xstepper<C>::step(size_type dim)
    {
        if(dim >= m_offset)
            m_it += p_c->strides()[dim - m_offset];
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
        return p_c == rhs.p_c && m_it == rhs.m_it;
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


    /*************************************
     * xiterator implementation
     *************************************/

    template <class It>
    inline xiterator<It>::xiterator(It it, const shape_type& shape)
        : m_it(it), m_shape(shape), m_index(shape.size(), size_type(0))
    {
    }

    template <class It>
    inline auto xiterator<It>::operator++() -> self_type&
    {
        for(size_type j = m_index.size(); j != 0; --j)
        {
            size_type i = j-1;
            if(++m_index[i] != m_shape[i])
            {
                m_it.step(i);
                break;
            }
            else if (i == 0)
            {
                m_it.to_end();
            }
            else
            {
                m_index[i] = 0;
                m_it.reset(i);
            }
        }
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

