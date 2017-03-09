/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XREDUCER_HPP
#define XREDUCER_HPP

#include <algorithm>
#include "xexpression.hpp"
#include "xiterator.hpp"
#include "xutils.hpp"

namespace xt
{

    template <class ST, std::size_t N>
    struct xreducer_shape_type;

    namespace detail
    {
        template <class F, class CT, std::size_t N>
        class reducing_iterator;
    }

    /************************
     * xreducer declaration *
     ************************/

    template <class F, class CT, std::size_t N>
    class xreducer : public xexpression<xreducer<F, CT, N>>
    {

    public:

        using self_type = xreducer<F, CT, N>;
        using functor_type = typename std::remove_reference<F>::type;
        using xexpression_type = std::decay_t<CT>;
        using axis_storage = std::array<std::size_t, N>;

        using value_type = typename xexpression_type::value_type;
        using reference = value_type;
        using const_reference = value_type;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using shape_type = typename xreducer_shape_type<typename xexpression_type::shape_type, N>::type;

        template <class Func>
        xreducer(Func&& func, CT e, const axis_storage& axis);

        size_type dimension() const noexcept;
        const shape_type& shape() const;

        template <class... Args>
        const_reference operator()(Args... args) const;

        const_reference operator[](const xindex& index) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool broadcast_shape(S& shape) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const noexcept;

    private:

        CT m_e;
        functor_type m_f;
        axis_storage m_axis;
        shape_type m_shape;

        using index_type = xindex_type_t<shape_type>;
        mutable index_type m_index;

        friend class detail::reducing_iterator<F, CT, N>;
    };

    // meta-function returning the shape type for an xreducer
    template <class ST, std::size_t N>
    struct xreducer_shape_type
    {
        using type = ST;
    };

    template <class I, std::size_t N1, std::size_t N2>
    struct xreducer_shape_type<std::array<I, N1>, N2>
    {
        using type = std::array<I, N1 - N2>;
    };

    /***************************
     * xreducer implementation *
     ***************************/

    namespace detail
    {
        template <class InputIt, class ExcludeIt, class OutputIt>
        inline void excluding_copy(InputIt first, InputIt last,
                                   ExcludeIt e_first, ExcludeIt e_last,
                                   OutputIt d_first)
        {
            InputIt iter = first;
            while (iter != last && e_first != e_last)
            {
                if (std::distance(first, iter) != *e_first)
                {
                    *d_first++ = *iter++;
                }
                else
                {
                    ++iter;
                    ++e_first;
                }
            }
            std::copy(iter, last, d_first);
        }

        template <class InputIt, class ExcludeIt, class OutputIt, class T>
        inline void inject(InputIt first, InputIt last,
                           ExcludeIt e_first, ExcludeIt e_last,
                           OutputIt d_first, T default_value)
        {
            OutputIt d_first_bu = d_first;
            while (first != last && e_first != e_last)
            {
                if (std::distance(d_first_bu, d_first) != *e_first)
                {
                    *d_first++ = *first++;
                }
                else
                {
                    *d_first++ = default_value;
                    ++e_first;
                }
            }
            std::copy(first, last, d_first);
        }

        // This is not a true iterator since two instances
        // of reducing_iterator on the same xreducer share
        // the same state. However this allows optimization
        // and is not problematic since not in the public
        // interface.
        template <class F, class CT, std::size_t N>
        class reducing_iterator
        {

        public:

            using self_type = reducing_iterator<F, CT, N>;
            using reducer_type = xreducer<F, CT, N>;
            using value_type = typename reducer_type::value_type;
            using reference = typename reducer_type::reference;
            using pointer = typename reducer_type::pointer;
            using difference_type = typename reducer_type::difference_type;
            using size_type = typename reducer_type::size_type;
            using iterator_category = std::forward_iterator_tag;

            reducing_iterator() = default;
            reducing_iterator(const reducer_type& reducer, bool end = false);

            self_type& operator++();
            self_type operator++(int);

            reference operator*() const;

            bool equal(const self_type& rhs) const;

        private:

            void increment();

            size_type axis(size_type index) const;
            size_type shape(size_type index) const;

            const reducer_type& m_reducer;
            bool m_end;
        };

        template <class F, class CT, std::size_t N>
        inline bool operator==(const reducing_iterator<F, CT, N>& lhs,
                               const reducing_iterator<F, CT, N>& rhs)
        {
            return lhs.equal(rhs);
        }

        template <class F, class CT, std::size_t N>
        inline bool operator!=(const reducing_iterator<F, CT, N>& lhs,
                               const reducing_iterator<F, CT, N>& rhs)
        {
            return !(lhs.equal(rhs));
        }

        /*************************************
         * xreducing_iterator implementation *
         *************************************/

        template <class F, class CT, std::size_t N>
        inline reducing_iterator<F, CT, N>::reducing_iterator(const reducer_type& reducer, bool end)
            : m_reducer(reducer), m_end(end)
        {
        }

        template <class F, class CT, std::size_t N>
        inline auto reducing_iterator<F, CT, N>::operator++() -> self_type&
        {
            increment();
            return *this;
        }

        template <class F, class CT, std::size_t N>
        inline auto reducing_iterator<F, CT, N>::operator++(int) -> self_type
        {
            self_type tmp(*this);
            ++(*this);
            return tmp;
        }

        template <class F, class CT, std::size_t N>
        inline auto reducing_iterator<F, CT, N>::operator*() const -> reference
        {
            return m_reducer.m_e.element(m_reducer.m_index.cbegin(), m_reducer.m_index.cend());
        }

        template <class F, class CT, std::size_t N>
        inline bool reducing_iterator<F, CT, N>::equal(const self_type& rhs) const
        {
            return &m_reducer == &(rhs.m_reducer) && m_end == rhs.m_end;
        }

        template <class F, class CT, std::size_t N>
        inline void reducing_iterator<F, CT, N>::increment()
        {
            size_type i = m_reducer.m_axis.size();
            while (i != 0)
            {
                --i;
                if (++(m_reducer.m_index[axis(i)]) != shape(axis(i)))
                {
                    return;
                }
                else
                {
                    m_reducer.m_index[axis(i)] = 0;
                }
            }
            if (i == 0)
            {
                m_end = true;
            }
        }

        template <class F, class CT, std::size_t N>
        inline auto reducing_iterator<F, CT, N>::axis(size_type index) const -> size_type
        {
            return m_reducer.m_axis[index];
        }

        template <class F, class CT, std::size_t N>
        inline auto reducing_iterator<F, CT, N>::shape(size_type index) const -> size_type
        {
            return m_reducer.m_e.shape()[index];
        }
    }

    /***************************
     * xreducer implementation *
     ***************************/

    template <class F, class CT, std::size_t N>
    template <class Func>
    inline xreducer<F, CT, N>::xreducer(Func&& func, CT e, const axis_storage& axis)
        : m_e(e), m_f(std::forward<Func>(func)), m_axis(axis),
          m_shape(make_sequence<shape_type>(m_e.dimension() - axis.size(), 0)),
          m_index(make_sequence<index_type>(m_e.dimension(), 0))
    {
        std::sort(m_axis.begin(), m_axis.end());
        detail::excluding_copy(m_e.shape().begin(), m_e.shape().end(),
                               m_axis.begin(), m_axis.end(),
                               m_shape.begin());
    }

    template <class F, class CT, std::size_t N>
    inline auto xreducer<F, CT, N>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    template <class F, class CT, std::size_t N>
    inline auto xreducer<F, CT, N>::shape() const -> const shape_type&
    {
        return m_shape;
    }

    template <class F, class CT, std::size_t N>
    template <class... Args>
    inline auto xreducer<F, CT, N>::operator()(Args... args) const -> const_reference
    {
        std::array<std::size_t, sizeof...(Args)> arg_array = { static_cast<std::size_t>(args)... };
        return element(arg_array.cbegin(), arg_array.cend());
    }

    template <class F, class CT, std::size_t N>
    inline auto xreducer<F, CT, N>::operator[](const xindex& index) const -> const_reference
    {
        return element(index.cbegin(), index.cend());
    }

    template <class F, class CT, std::size_t N>
    template <class It>
    inline auto xreducer<F, CT, N>::element(It first, It last) const -> const_reference
    {
        detail::inject(first, last, m_axis.cbegin(), m_axis.cend(),
                       m_index.begin(), size_type(0));
        using iter_type = detail::reducing_iterator<F, CT, N>;
        iter_type iter = iter_type(*this);
        iter_type iter_end = iter_type(*this, true);
        value_type init_value = *iter;
        value_type res = std::accumulate(++iter, iter_end, init_value, m_f);
        return res;
    }

    template <class F, class CT, std::size_t N>
    template <class S>
    inline bool xreducer<F, CT, N>::broadcast_shape(S& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    template <class F, class CT, std::size_t N>
    template <class S>
    inline bool xreducer<F, CT, N>::is_trivial_broadcast(const S& /*strides*/) const noexcept
    {
        return false;
    }
}

#endif