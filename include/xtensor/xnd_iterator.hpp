/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_AXIS_ITERATOR_HPP
#define XTENSOR_AXIS_ITERATOR_HPP

#include <xtl/xclosure.hpp>

#include "xview.hpp"

namespace xt
{

    /******************
     * xnd_iterator *
     ******************/

    template <class CT>
    class xnd_iterator
    {
    public:

        using self_type = xnd_iterator<CT>;

        using xexpression_type = std::decay_t<CT>;
        using size_type = typename xexpression_type::size_type;
        using shape_type = typename xexpression_type::shape_type;
        using strided_view_type = xstrided_view<CT, shape_type>;
        using reference = std::remove_reference_t<apply_cv_t<CT, strided_view_type>>;
        using pointer = xtl::xclosure_pointer<std::remove_reference_t<apply_cv_t<CT, strided_view_type>>>;

        template <class CTA>
        xnd_iterator(CTA&& e, size_type index);
        template <class CTA>
        xnd_iterator(CTA&& e, size_type axis, size_type offset);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;
        pointer operator->() const;

        bool equal(const self_type& rhs) const;

    private:

        using storing_type = xtl::ptr_closure_type_t<CT>;
        mutable storing_type p_expression;
        size_type m_add_offset;
        strided_view_type m_sv;

        template <class T, class CTA>
        std::enable_if_t<std::is_pointer<T>::value, T>
            get_storage_init(CTA&& e) const;

        template <class T, class CTA>
        std::enable_if_t<!std::is_pointer<T>::value, T>
            get_storage_init(CTA&& e) const;
    };

    template <class CT>
    bool operator==(const xnd_iterator<CT>& lhs, const xnd_iterator<CT>& rhs);

    template <class CT>
    bool operator!=(const xnd_iterator<CT>& lhs, const xnd_iterator<CT>& rhs);

    template <class E>
    auto xnd_axis_begin(E&& e);

    template <class E>
    auto xnd_axis_begin(E&& e, typename std::decay_t<E>::size_type axis);

    template <class E>
    auto xnd_axis_end(E&& e);

    template <class E>
    auto xnd_axis_end(E&& e, typename std::decay_t<E>::size_type axis);

    /*********************************
     * xnd_iterator implementation *
     *********************************/
    template <class CT>
    static auto derive_xstrided_view(CT&& e, typename std::decay_t<CT>::size_type axis, typename std::decay_t<CT>::size_type offset)
    {
        auto e_shape = e.shape();
        using xexpression_type = std::decay_t<CT>;
        using size_type = typename xexpression_type::size_type;
        using shape_type = typename xexpression_type::shape_type;
        using strides_type = typename xexpression_type::strides_type;

        shape_type shape;
        for (size_type i = 0; i < e_shape.size(); ++i)
        {
            if (i != axis)
                shape.push_back(e_shape[i]);
        }

        auto e_strides = e.strides();
        strides_type strides;

        for (size_type i = 0; i < e_strides.size(); ++i)
        {
            if (i != axis)
                strides.push_back(e_strides[i]);
        }

        return strided_view(std::forward<CT>(e), std::forward<shape_type>(shape),
            std::forward<strides_type>(strides), offset, e.layout());
    }

    template <class CT>
    template <class T, class CTA>
    inline std::enable_if_t<std::is_pointer<T>::value, T>
        xnd_iterator<CT>::get_storage_init(CTA&& e) const
    {
        return &e;
    }

    template <class CT>
    template <class T, class CTA>
    inline std::enable_if_t<!std::is_pointer<T>::value, T>
        xnd_iterator<CT>::get_storage_init(CTA&& e) const
    {
        return e;
    }

    template <class CT>
    template <class CTA>
    inline xnd_iterator<CT>::xnd_iterator(CTA&& e, size_type axis)
        : p_expression(get_storage_init<storing_type>(std::forward<CTA>(e))), m_add_offset(e.strides()[axis]), m_sv(derive_xstrided_view<CTA>(std::forward<CTA>(e), axis, e.data_offset()))
    {
    }

    template <class CT>
    template <class CTA>
    inline xnd_iterator<CT>::xnd_iterator(CTA&& e, size_type axis, size_type offset)
        : p_expression(get_storage_init<storing_type>(std::forward<CTA>(e))), m_add_offset(e.strides()[axis]), m_sv(derive_xstrided_view<CTA>(std::forward<CTA>(e), axis, offset))
    {
    }

    template <class CT>
    inline auto xnd_iterator<CT>::operator++() -> self_type&
    {
        m_sv.set_offset(m_sv.data_offset() + m_add_offset);
        return *this;
    }

    template <class CT>
    inline auto xnd_iterator<CT>::operator++(int) -> self_type
    {
        self_type tmp(*this);
        ++(*this);
        return tmp;
    }

    template <class CT>
    inline auto xnd_iterator<CT>::operator*() const -> reference
    {
        return m_sv;
    }

    template <class CT>
    inline auto xnd_iterator<CT>::operator->() const -> pointer
    {
        return xtl::closure_pointer(operator*());
    }

    template <class CT>
    inline bool xnd_iterator<CT>::equal(const self_type& rhs) const
    {
        return p_expression == rhs.p_expression && m_sv.data_offset() == rhs.m_sv.data_offset();
    }

    template <class CT>
    inline bool operator==(const xnd_iterator<CT>& lhs, const xnd_iterator<CT>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class CT>
    inline bool operator!=(const xnd_iterator<CT>& lhs, const xnd_iterator<CT>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class E>
    inline auto xnd_axis_begin(E&& e)
    {
        using return_type = xnd_iterator<xtl::closure_type_t<E>>;
        return return_type(std::forward<E>(e), 0);
    }

    template <class E>
    inline auto xnd_axis_begin(E&& e, typename std::decay_t<E>::size_type axis)
    {
        using return_type = xnd_iterator<xtl::closure_type_t<E>>;
        return return_type(std::forward<E>(e), axis);
    }

    template <class E>
    inline auto xnd_axis_end(E&& e)
    {
        using return_type = xnd_iterator<xtl::closure_type_t<E>>;
        return return_type(std::forward<E>(e), 0, e.shape()[0]);
    }

    template <class E>
    inline auto xnd_axis_end(E&& e, typename std::decay_t<E>::size_type axis)
    {
        using return_type = xnd_iterator<xtl::closure_type_t<E>>;
        return return_type(std::forward<E>(e), axis, e.shape()[axis]);
    }
}

#endif
