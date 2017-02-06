/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XSCALAR_HPP
#define XSCALAR_HPP

#include <utility>
#include <cstddef>
#include <array>

#include "xexpression.hpp"

namespace xt
{

    /***********
     * xscalar *
     ***********/

    // xscalar is a cheap wrapper for a scalar value as an xexpression.

    template <class CT>
    class xscalar_stepper;

    template <class CT>
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

        using const_stepper = xscalar_stepper<CT>;
        using const_iterator = xscalar_iterator<CT>;
        using const_storage_iterator = xscalar_iterator<CT>;

        xscalar(CT value) noexcept;

        size_type size() const noexcept;
        size_type dimension() const noexcept;

        const shape_type& shape() const noexcept;

        template <class... Args>
        const_reference operator()(Args... args) const noexcept;

        template <class It>
        const_reference element(It, It) const noexcept;

        template <class S>
        bool broadcast_shape(S& shape) const noexcept;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const noexcept;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        template <class S>
        const_iterator xbegin(const S& shape) const noexcept;
        template <class S>
        const_iterator xend(const S& shape) const noexcept;
        template <class S>
        const_iterator cxbegin(const S& shape) const noexcept;
        template <class S>
        const_iterator cxend(const S& shape) const noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape) const noexcept;

        const_storage_iterator storage_begin() const noexcept;
        const_storage_iterator storage_end() const noexcept;

        const_storage_iterator storage_cbegin() const noexcept;
        const_storage_iterator storage_cend() const noexcept;

    private:

        CT m_value;
    };

    /*******************
     * xscalar_stepper *
     *******************/

    template <class CT>
    class xscalar_stepper
    {

    public:

        using self_type = xscalar_stepper<CT>;
        using container_type = xscalar<CT>;

        using value_type = typename container_type::value_type;
        using reference = typename container_type::const_reference;
        using pointer = typename container_type::const_pointer;
        using size_type = typename container_type::size_type;
        using difference_type = typename container_type::difference_type;

        xscalar_stepper(const container_type* c) noexcept;

        reference operator*() const noexcept;

        void step(size_type dim, size_type n = 1) noexcept;
        void step_back(size_type dim, size_type n = 1) noexcept;
        void reset(size_type dim) noexcept;

        void to_end() noexcept;

        bool equal(const self_type& rhs) const noexcept;

    private:

        const container_type* p_c;
    };

    template <class CT>
    bool operator==(const xscalar_stepper<CT>& lhs,
                    const xscalar_stepper<CT>& rhs) noexcept;

    template <class CT>
    bool operator!=(const xscalar_stepper<CT>& lhs,
                    const xscalar_stepper<CT>& rhs) noexcept;

    /********************
     * xscalar_iterator *
     ********************/

    template <class CT>
    class xscalar_iterator
    {

    public:

        using self_type = xscalar_iterator<CT>;
        using container_type = xscalar<CT>;

        using value_type = typename container_type::value_type;
        using reference = typename container_type::const_reference;
        using pointer = typename container_type::const_pointer;
        using difference_type = typename container_type::difference_type;
        using iterator_category = std::input_iterator_tag;

        explicit xscalar_iterator(const container_type* c) noexcept;

        self_type& operator++() noexcept;
        self_type operator++(int) noexcept;

        reference operator*() const noexcept;

        bool equal(const self_type& rhs) const noexcept;

    private:

        const container_type* p_c;
    };

    template <class CT>
    bool operator==(const xscalar_iterator<CT>& lhs,
                    const xscalar_iterator<CT>& rhs) noexcept;

    template <class CT>
    bool operator!=(const xscalar_iterator<CT>& lhs,
                    const xscalar_iterator<CT>& rhs) noexcept;

    /**************************
     * xscalar implementation *
     **************************/

    template <class CT>
    inline xscalar<CT>::xscalar(CT value) noexcept
        : m_value(std::forward<const value_type>(value))
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
    template <class... Args>
    inline auto xscalar<CT>::operator()(Args...) const noexcept -> const_reference
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
    template <class S>
    inline auto xscalar<CT>::xbegin(const S&) const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::xend(const S&) const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::cxbegin(const S&) const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class CT>
    template <class S>
    inline auto xscalar<CT>::cxend(const S&) const noexcept -> const_iterator
    {
        return const_iterator(this);
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

    template <class CT>
    inline auto xscalar<CT>::storage_begin() const noexcept -> const_storage_iterator
    {
        return const_storage_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::storage_end() const noexcept -> const_storage_iterator
    {
        return const_storage_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::storage_cbegin() const noexcept -> const_storage_iterator
    {
        return const_storage_iterator(this);
    }

    template <class CT>
    inline auto xscalar<CT>::storage_cend() const noexcept -> const_storage_iterator
    {
        return const_storage_iterator(this);
    }

    /**********************************
     * xscalar_stepper implementation *
     **********************************/

    template <class CT>
    inline xscalar_stepper<CT>::xscalar_stepper(const container_type* c) noexcept
        : p_c(c)
    {
    }

    template <class CT>
    inline auto xscalar_stepper<CT>::operator*() const noexcept -> reference
    {
        return p_c->operator()();
    }

    template <class CT>
    inline void xscalar_stepper<CT>::step(size_type /*dim*/, size_type /*n*/) noexcept
    {
    }

    template <class CT>
    inline void xscalar_stepper<CT>::step_back(size_type /*dim*/, size_type /*n*/) noexcept
    {
    }

    template <class CT>
    inline void xscalar_stepper<CT>::reset(size_type /*dim*/) noexcept
    {
    }

    template <class CT>
    inline void xscalar_stepper<CT>::to_end() noexcept
    {
        p_c = p_c->stepper_end(p_c->shape()).p_c;
    }

    template <class CT>
    inline bool xscalar_stepper<CT>::equal(const self_type& rhs) const noexcept
    {
        return (p_c == rhs.p_c);
    }

    template <class CT>
    inline bool operator==(const xscalar_stepper<CT>& lhs,
                           const xscalar_stepper<CT>& rhs) noexcept
    {
        return lhs.equal(rhs);
    }

    template <class CT>
    inline bool operator!=(const xscalar_stepper<CT>& lhs,
                           const xscalar_stepper<CT>& rhs) noexcept
    {
        return !(lhs.equal(rhs));
    }

    /***********************************
     * xscalar_iterator implementation *
     ***********************************/

    template <class CT>
    inline xscalar_iterator<CT>::xscalar_iterator(const container_type* c) noexcept
        : p_c(c)
    {
    }

    template <class CT>
    inline auto xscalar_iterator<CT>::operator++() noexcept -> self_type&
    {
        return *this;
    }

    template <class CT>
    inline auto xscalar_iterator<CT>::operator++(int) noexcept -> self_type
    {
        self_type tmp(*this);
        ++(*this);
        return tmp;
    }

    template <class CT>
    inline auto xscalar_iterator<CT>::operator*() const noexcept -> reference
    {
        return p_c->operator()();
    }

    template <class CT>
    inline bool xscalar_iterator<CT>::equal(const self_type& rhs) const noexcept
    {
        return p_c == rhs.p_c;
    }

    template <class CT>
    inline bool operator==(const xscalar_iterator<CT>& lhs,
                           const xscalar_iterator<CT>& rhs) noexcept
    {
        return lhs.equal(rhs);
    }

    template <class CT>
    inline bool operator!=(const xscalar_iterator<CT>& lhs,
                           const xscalar_iterator<CT>& rhs) noexcept
    {
        return !(lhs.equal(rhs));
    }
}

#endif

