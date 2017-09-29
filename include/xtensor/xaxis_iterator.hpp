/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XAXIS_ITERATOR_HPP
#define XAXIS_ITERATOR_HPP

#include "xtl/xclosure.hpp"

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
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;
        using value_type = xview<CT, size_type>;
        using reference = std::remove_reference_t<apply_cv_t<CT, value_type>>;
        using pointer = std::nullptr_t;

        using iterator_category = std::forward_iterator_tag;

        xaxis_iterator();
        template <class CTA>
        xaxis_iterator(CTA&& e, size_type index);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;
        pointer operator->() const;

        bool equal(const self_type& rhs) const;

    private:

        mutable xtl::closure_wrapper<CT> p_expression;
        size_type m_index;
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
    inline xaxis_iterator<CT>::xaxis_iterator()
        : p_expression(nullptr), m_index(0)
    {
    }

    template <class CT>
    template <class CTA>
    inline xaxis_iterator<CT>::xaxis_iterator(CTA&& e, size_type index)
        : p_expression(std::forward<CTA>(e)), m_index(index)
    {
    }

    template <class CT>
    inline auto xaxis_iterator<CT>::operator++() -> self_type&
    {
        ++m_index;
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
        return view(p_expression.get(), size_type(m_index));
    }

    template <class CT>
    inline auto xaxis_iterator<CT>::operator->() const -> pointer
    {
        return nullptr;
    }

    template <class CT>
    inline bool xaxis_iterator<CT>::equal(const self_type& rhs) const
    {
        return p_expression == rhs.p_expression && m_index == rhs.m_index;
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
        using return_type = xaxis_iterator<xtl::closure_type_t<E>>;
        using size_type = typename std::decay_t<E>::size_type;
        return return_type(std::forward<E>(e), size_type(0));
    }

    template <class E>
    inline auto axis_end(E&& e)
    {
        using return_type = xaxis_iterator<xtl::closure_type_t<E>>;
        using size_type = typename std::decay_t<E>::size_type;
        return return_type(std::forward<E>(e), size_type(e.shape()[0]));
    }
}

#endif
