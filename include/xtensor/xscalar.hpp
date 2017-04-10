/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSCALAR_HPP
#define XSCALAR_HPP

#include <array>
#include <cstddef>
#include <utility>

#include "xexpression.hpp"
#include "xlayout.hpp"

namespace xt
{

    /***********
     * xscalar *
     ***********/

    // xscalar is a cheap wrapper for a scalar value as an xexpression.

    template <bool is_const, class CT>
    class xscalar_stepper;

    template <bool is_const, class CT>
    class xscalar_iterator;

    template <class CT>
    class xscalar : public xexpression<xscalar<CT>>
    {

    public:

        using value_type = std::decay_t<CT>;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using self_type = xscalar<CT>;
        using shape_type = std::array<size_type, 0>;

        using stepper = xscalar_stepper<false, CT>;
        using const_stepper = xscalar_stepper<true, CT>;

        using broadcast_iterator = xscalar_iterator<false, CT>;
        using const_broadcast_iterator = xscalar_iterator<true, CT>;

        using iterator = broadcast_iterator;
        using const_iterator = const_broadcast_iterator;

        static constexpr xt::layout layout_type = xt::layout::any;
        static constexpr bool contiguous_layout = true;

        xscalar(CT value) noexcept;

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const shape_type& shape() const noexcept;
        xt::layout layout() const noexcept;

        template <class... Args>
        reference operator()(Args...) noexcept;
        reference operator[](const xindex&) noexcept;
        reference operator[](size_type) noexcept;

        template <class... Args>
        const_reference operator()(Args...) const noexcept;
        const_reference operator[](const xindex&) const noexcept;
        const_reference operator[](size_type) const noexcept;

        template <class It>
        reference element(It, It) noexcept;

        template <class It>
        const_reference element(It, It) const noexcept;

        template <class S>
        bool broadcast_shape(S& shape) const noexcept;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const noexcept;

        iterator begin() noexcept;
        iterator end() noexcept;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        broadcast_iterator xbegin() noexcept;
        broadcast_iterator xend() noexcept;

        const_broadcast_iterator xbegin() const noexcept;
        const_broadcast_iterator xend() const noexcept;
        const_broadcast_iterator cxbegin() const noexcept;
        const_broadcast_iterator cxend() const noexcept;

        template <class S>
        broadcast_iterator xbegin(const S& shape) noexcept;
        template <class S>
        broadcast_iterator xend(const S& shape) noexcept;

        template <class S>
        const_broadcast_iterator xbegin(const S& shape) const noexcept;
        template <class S>
        const_broadcast_iterator xend(const S& shape) const noexcept;
        template <class S>
        const_broadcast_iterator cxbegin(const S& shape) const noexcept;
        template <class S>
        const_broadcast_iterator cxend(const S& shape) const noexcept;

        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape) noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape) const noexcept;

    private:

        CT m_value;
    };

    template <class T>
    xscalar<T&> xref(T& t);

    template <class T>
    xscalar<const T&> xcref(T& t);

    /*******************
     * xscalar_stepper *
     *******************/

    template <bool is_const, class CT>
    class xscalar_stepper
    {

    public:

        using self_type = xscalar_stepper<is_const, CT>;
        using container_type = std::conditional_t<is_const,
                                                  const xscalar<CT>,
                                                  xscalar<CT>>;

        using value_type = typename container_type::value_type;
        using reference = std::conditional_t<is_const,
                                             typename container_type::const_reference,
                                             typename container_type::reference>;
        using pointer = std::conditional_t<is_const,
                                           typename container_type::const_pointer,
                                           typename container_type::pointer>;
        using size_type = typename container_type::size_type;
        using difference_type = typename container_type::difference_type;

        xscalar_stepper(container_type* c) noexcept;

        reference operator*() const noexcept;

        void step(size_type dim, size_type n = 1) noexcept;
        void step_back(size_type dim, size_type n = 1) noexcept;
        void reset(size_type dim) noexcept;

        void to_end() noexcept;

        bool equal(const self_type& rhs) const noexcept;

    private:

        container_type* p_c;
    };

    template <bool is_const, class CT>
    bool operator==(const xscalar_stepper<is_const, CT>& lhs,
                    const xscalar_stepper<is_const, CT>& rhs) noexcept;

    template <bool is_const, class CT>
    bool operator!=(const xscalar_stepper<is_const, CT>& lhs,
                    const xscalar_stepper<is_const, CT>& rhs) noexcept;

    /********************
     * xscalar_iterator *
     ********************/

    template <bool is_const, class CT>
    class xscalar_iterator
    {

    public:

        using self_type = xscalar_iterator<is_const, CT>;
        using container_type = std::conditional_t<is_const,
                                                  const xscalar<CT>,
                                                  xscalar<CT>>;

        using value_type = typename container_type::value_type;
        using reference = std::conditional_t<is_const,
                                             typename container_type::const_reference,
                                             typename container_type::reference>;
        using pointer = std::conditional_t<is_const,
                                           typename container_type::const_pointer,
                                           typename container_type::pointer>;
        using difference_type = typename container_type::difference_type;
        using iterator_category = std::forward_iterator_tag;

        explicit xscalar_iterator(container_type* c) noexcept;

        self_type& operator++() noexcept;
        self_type operator++(int) noexcept;

        reference operator*() const noexcept;

        bool equal(const self_type& rhs) const noexcept;

    private:

        container_type* p_c;
    };

    template <bool is_const, class CT>
    bool operator==(const xscalar_iterator<is_const, CT>& lhs,
                    const xscalar_iterator<is_const, CT>& rhs) noexcept;

    template <bool is_const, class CT>
    bool operator!=(const xscalar_iterator<is_const, CT>& lhs,
                    const xscalar_iterator<is_const, CT>& rhs) noexcept;

    /**************************
     * xscalar implementation *
     **************************/

    template <class CT>
    inline xscalar<CT>::xscalar(CT value) noexcept
        : m_value(value)
    {
    }

    template <class CT>
    inline auto xscalar<CT>::size() const noexcept -> size_type
    {
        return 1;
    }

    template <class CT>
    inline auto xscalar<CT>::dimension() const noexcept -> size_type
    {
        return 0;
    }

    template <class CT>
    inline auto xscalar<CT>::shape() const noexcept -> const shape_type&
    {
        static std::array<size_type, 0> zero_shape;
        return zero_shape;
    }

    template <class CT>
    inline xt::layout xscalar<CT>::layout() const noexcept
    {
        return layout_type;
    }

    template <class CT>
    template <class... Args>
    inline auto xscalar<CT>::operator()(Args...) noexcept -> reference
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::operator[](const xindex&) noexcept -> reference
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::operator[](size_type) noexcept -> reference
    {
        return m_value;
    }

    template <class CT>
    template <class... Args>
    inline auto xscalar<CT>::operator()(Args...) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::operator[](const xindex&) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class CT>
    inline auto xscalar<CT>::operator[](size_type) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class CT>
    template <class It>
    inline auto xscalar<CT>::element(It, It) noexcept -> reference
    {
        return m_value;
    }

    template <class CT>
    template <class It>
    inline auto xscalar<CT>::element(It, It) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class CT>
    template <class S>
    inline bool xscalar<CT>::broadcast_shape(S&) const noexcept
    {
        return true;
    }

    template <class CT>
    template <class S>
    inline bool xscalar<CT>::is_trivial_broadcast(const S&) const noexcept
    {
        return true;
    }

    template <class CT>
    inline auto xscalar<CT>::begin() noexcept -> iterator
    {
        return iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::end() noexcept -> iterator
    {
        return iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::begin() const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::end() const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::cbegin() const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::cend() const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::xbegin() noexcept -> broadcast_iterator
    {
        return broadcast_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::xend() noexcept -> broadcast_iterator
    {
        return broadcast_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::xbegin() const noexcept -> const_broadcast_iterator
    {
        return const_broadcast_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::xend() const noexcept -> const_broadcast_iterator
    {
        return const_broadcast_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::cxbegin() const noexcept -> const_broadcast_iterator
    {
        return xbegin();
    }

    template <class CT>
    inline auto xscalar<CT>::cxend() const noexcept -> const_broadcast_iterator
    {
        return xend();
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::xbegin(const S&) noexcept -> broadcast_iterator
    {
        return broadcast_iterator(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::xend(const S&) noexcept -> broadcast_iterator
    {
        return broadcast_iterator(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::xbegin(const S&) const noexcept -> const_broadcast_iterator
    {
        return const_broadcast_iterator(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::xend(const S&) const noexcept -> const_broadcast_iterator
    {
        return const_broadcast_iterator(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::cxbegin(const S&) const noexcept -> const_broadcast_iterator
    {
        return const_broadcast_iterator(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::cxend(const S&) const noexcept -> const_broadcast_iterator
    {
        return const_broadcast_iterator(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::stepper_begin(const S&) noexcept -> stepper
    {
        return stepper(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::stepper_end(const S&) noexcept -> stepper
    {
        return stepper(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::stepper_begin(const S&) const noexcept -> const_stepper
    {
        return const_stepper(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::stepper_end(const S&) const noexcept -> const_stepper
    {
        return const_stepper(this + 1);
    }

    template <class T>
    inline xscalar<T&> xref(T& t)
    {
        return xscalar<T&>(t);
    }

    template <class T>
    inline xscalar<const T&> xcref(T& t)
    {
        return xscalar<const T&>(t);
    }

    /**********************************
     * xscalar_stepper implementation *
     **********************************/

    template <bool is_const, class CT>
    inline xscalar_stepper<is_const, CT>::xscalar_stepper(container_type* c) noexcept
        : p_c(c)
    {
    }

    template <bool is_const, class CT>
    inline auto xscalar_stepper<is_const, CT>::operator*() const noexcept -> reference
    {
        return p_c->operator()();
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::step(size_type /*dim*/, size_type /*n*/) noexcept
    {
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::step_back(size_type /*dim*/, size_type /*n*/) noexcept
    {
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::reset(size_type /*dim*/) noexcept
    {
    }

    template <bool is_const, class CT>
    inline void xscalar_stepper<is_const, CT>::to_end() noexcept
    {
        p_c = p_c->stepper_end(p_c->shape()).p_c;
    }

    template <bool is_const, class CT>
    inline bool xscalar_stepper<is_const, CT>::equal(const self_type& rhs) const noexcept
    {
        return (p_c == rhs.p_c);
    }

    template <bool is_const, class CT>
    inline bool operator==(const xscalar_stepper<is_const, CT>& lhs,
                           const xscalar_stepper<is_const, CT>& rhs) noexcept
    {
        return lhs.equal(rhs);
    }

    template <bool is_const, class CT>
    inline bool operator!=(const xscalar_stepper<is_const, CT>& lhs,
                           const xscalar_stepper<is_const, CT>& rhs) noexcept
    {
        return !(lhs.equal(rhs));
    }

    /***********************************
     * xscalar_iterator implementation *
     ***********************************/

    template <bool is_const, class CT>
    inline xscalar_iterator<is_const, CT>::xscalar_iterator(container_type* c) noexcept
        : p_c(c)
    {
    }

    template <bool is_const, class CT>
    inline auto xscalar_iterator<is_const, CT>::operator++() noexcept -> self_type&
    {
        return *this;
    }

    template <bool is_const, class CT>
    inline auto xscalar_iterator<is_const, CT>::operator++(int) noexcept -> self_type
    {
        self_type tmp(*this);
        ++(*this);
        return tmp;
    }

    template <bool is_const, class CT>
    inline auto xscalar_iterator<is_const, CT>::operator*() const noexcept -> reference
    {
        return p_c->operator()();
    }

    template <bool is_const, class CT>
    inline bool xscalar_iterator<is_const, CT>::equal(const self_type& rhs) const noexcept
    {
        return p_c == rhs.p_c;
    }

    template <bool is_const, class CT>
    inline bool operator==(const xscalar_iterator<is_const, CT>& lhs,
                           const xscalar_iterator<is_const, CT>& rhs) noexcept
    {
        return lhs.equal(rhs);
    }

    template <bool is_const, class CT>
    inline bool operator!=(const xscalar_iterator<is_const, CT>& lhs,
                           const xscalar_iterator<is_const, CT>& rhs) noexcept
    {
        return !(lhs.equal(rhs));
    }
}

#endif
