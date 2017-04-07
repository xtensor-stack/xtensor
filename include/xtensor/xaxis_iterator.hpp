/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XAXIS_ITERATOR_HPP
#define XAXIS_ITERATOR_HPP

#include "xview.hpp"

namespace xt
{

    /******************
     * xaxis_iterator *
     ******************/

    template <class CT>
    class xaxis_iterator
    {

    public:

        using self_type = xaxis_iterator<CT>;

        using xexpression_type = std::decay_t<CT>;
        using value_type = typename xexpression_type::value_type;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;
        using reference = xview<CT, size_type>;
        using const_reference = const xview<CT, size_type>;
        using pointer = reference*;
        using const_pointer = const reference*;


        using iterator_category = std::forward_iterator_tag;

        template <class CTA>
        xaxis_iterator(CTA&& e, size_type index);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;
        pointer operator->() const;

        bool equal(const self_type& rhs) const;

    private:

        reference m_view;
    };

    template <class CT>
    bool operator==(const xaxis_iterator<CT>& lhs, const xaxis_iterator<CT>& rhs);

    template <class CT>
    bool operator!=(const xaxis_iterator<CT>& lhs, const xaxis_iterator<CT>& rhs);

    template <class E>
    auto axis_begin(E&& e);

    template <class E>
    auto axis_end(E&& e);

    /*********************************
     * xaxis_iterator implementation *
     *********************************/

    template <class CT>
    template <class CTA>
    inline xaxis_iterator<CT>::xaxis_iterator(CTA&& e, size_type index)
        : m_view(view(std::forward<CTA>(e), index))
    {
    }

    template <class CT>
    inline auto xaxis_iterator<CT>::operator++() -> self_type&
    {
        ++std::get<0>(m_view.m_slices);
        return *this;
    }

    template <class CT>
    inline auto xaxis_iterator<CT>::operator++(int) -> self_type
    {
        self_type tmp(*this);
        ++(*this);
        return tmp;
    }

    template <class CT>
    inline auto xaxis_iterator<CT>::operator*() const -> reference
    {
        return m_view;
    }

    template <class CT>
    inline auto xaxis_iterator<CT>::operator->() const -> pointer
    {
        return const_cast<pointer>(&m_view);
    }

    template<class CT>
    inline bool xaxis_iterator<CT>::equal(const self_type& rhs) const
    {
        return &m_view.m_e == &(rhs.m_view.m_e)
            && (std::get<0>(m_view.m_slices) == std::get<0>(rhs.m_view.m_slices));
    }

    template <class CT>
    inline bool operator==(const xaxis_iterator<CT>& lhs, const xaxis_iterator<CT>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class CT>
    inline bool operator!=(const xaxis_iterator<CT>& lhs, const xaxis_iterator<CT>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class E>
    inline auto axis_begin(E&& e)
    {
        using return_type = xaxis_iterator<closure_t<E>>;
        using size_type = typename std::decay_t<E>::size_type;
        return return_type(std::forward<E>(e), size_type(0));
    }

    template <class E>
    inline auto axis_end(E&& e)
    {
        using return_type = xaxis_iterator<closure_t<E>>;
        using size_type = typename std::decay_t<E>::size_type;
        return return_type(std::forward<E>(e), size_type(e.shape()[0]));
    }
}

#endif
