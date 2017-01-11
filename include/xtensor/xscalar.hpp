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

    template <class T>
    class xscalar_stepper;

    template <class T>
    class xscalar_iterator;

    template <class T>
    class xscalar : public xexpression<xscalar<T>>
    {

    public:

        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using self_type = xscalar<T>;
        using shape_type = std::array<size_type, 0>;

        using closure_type = const self_type;
        using const_stepper = xscalar_stepper<T>;
        using const_iterator = xscalar_iterator<T>;
        using const_storage_iterator = xscalar_iterator<T>;

        xscalar(T value) noexcept;

        size_type size() const noexcept;
        size_type dimension() const noexcept;

        const shape_type& shape() const noexcept;

        template <class... Args>
        const_reference operator()(Args... args) const noexcept;
        const_reference operator[](const xindex& /*idx*/) const noexcept;

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

        value_type m_value;
    };

    /*******************
     * xscalar_stepper *
     *******************/

    template <class T>
    class xscalar_stepper
    {

    public:

        using self_type = xscalar_stepper<T>;
        using container_type = xscalar<T>;

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

    template <class T>
    bool operator==(const xscalar_stepper<T>& lhs,
                    const xscalar_stepper<T>& rhs) noexcept;

    template <class T>
    bool operator!=(const xscalar_stepper<T>& lhs,
                    const xscalar_stepper<T>& rhs) noexcept;

    /********************
     * xscalar_iterator *
     ********************/

    template <class T>
    class xscalar_iterator
    {

    public:

        using self_type = xscalar_iterator<T>;
        using container_type = xscalar<T>;

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

    template <class T>
    bool operator==(const xscalar_iterator<T>& lhs,
                    const xscalar_iterator<T>& rhs) noexcept;

    template <class T>
    bool operator!=(const xscalar_iterator<T>& lhs,
                    const xscalar_iterator<T>& rhs) noexcept;

    /**************************
     * xscalar implementation *
     **************************/

    template <class T>
    inline xscalar<T>::xscalar(T value) noexcept
        : m_value(std::move(value))
    {
    }

    template <class T>
    inline auto xscalar<T>::size() const noexcept -> size_type
    {
        return 1;
    }

    template <class T>
    inline auto xscalar<T>::dimension() const noexcept -> size_type
    {
        return 0;
    }

    template <class T>
    inline auto xscalar<T>::shape() const noexcept -> const shape_type&
    {
        static std::array<size_type, 0> zero_shape;
        return zero_shape;
    }

    template <class T>
    template <class... Args>
    inline auto xscalar<T>::operator()(Args...) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class T>
    template <class It>
    inline auto xscalar<T>::element(It, It) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class T>
    inline auto xscalar<T>::operator[](const xindex& /*idx*/) const noexcept -> const_reference
    {
        return m_value;
    }

    template <class T>
    template <class S>
    inline bool xscalar<T>::broadcast_shape(S&) const noexcept
    {
        return true;
    }

    template <class T>
    template <class S>
    inline bool xscalar<T>::is_trivial_broadcast(const S&) const noexcept
    {
        return true;
    }

    template <class T>
    inline auto xscalar<T>::begin() const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class T>
    inline auto xscalar<T>::end() const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class T>
    inline auto xscalar<T>::cbegin() const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class T>
    inline auto xscalar<T>::cend() const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class T>
    template <class S>
    inline auto xscalar<T>::xbegin(const S& shape) const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class T>
    template <class S>
    inline auto xscalar<T>::xend(const S& shape) const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class T>
    template <class S>
    inline auto xscalar<T>::cxbegin(const S& shape) const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class T>
    template <class S>
    inline auto xscalar<T>::cxend(const S& shape) const noexcept -> const_iterator
    {
        return const_iterator(this);
    }

    template <class T>
    template <class S>
    inline auto xscalar<T>::stepper_begin(const S&) const noexcept -> const_stepper
    {
        return const_stepper(this);
    }

    template <class T>
    template <class S>
    inline auto xscalar<T>::stepper_end(const S&) const noexcept -> const_stepper
    {
        return const_stepper(this + 1);
    }

    template <class T>
    inline auto xscalar<T>::storage_begin() const noexcept -> const_storage_iterator
    {
        return const_storage_iterator(this);
    }

    template <class T>
    inline auto xscalar<T>::storage_end() const noexcept -> const_storage_iterator
    {
        return const_storage_iterator(this);
    }

    template <class T>
    inline auto xscalar<T>::storage_cbegin() const noexcept -> const_storage_iterator
    {
        return const_storage_iterator(this);
    }

    template <class T>
    inline auto xscalar<T>::storage_cend() const noexcept -> const_storage_iterator
    {
        return const_storage_iterator(this);
    }

    /**********************************
     * xscalar_stepper implementation *
     **********************************/

    template <class T>
    inline xscalar_stepper<T>::xscalar_stepper(const container_type* c) noexcept
        : p_c(c)
    {
    }

    template <class T>
    inline auto xscalar_stepper<T>::operator*() const noexcept -> reference
    {
        return p_c->operator()();
    }

    template <class T>
    inline void xscalar_stepper<T>::step(size_type /*dim*/, size_type /*n*/) noexcept
    {
    }

    template <class T>
    inline void xscalar_stepper<T>::step_back(size_type /*dim*/, size_type /*n*/) noexcept
    {
    }

    template <class T>
    inline void xscalar_stepper<T>::reset(size_type /*dim*/) noexcept
    {
    }

    template <class T>
    inline void xscalar_stepper<T>::to_end() noexcept
    {
        p_c = p_c->stepper_end(p_c->shape()).p_c;
    }

    template <class T>
    inline bool xscalar_stepper<T>::equal(const self_type& rhs) const noexcept
    {
        return (p_c == rhs.p_c);
    }

    template <class T>
    inline bool operator==(const xscalar_stepper<T>& lhs,
                           const xscalar_stepper<T>& rhs) noexcept
    {
        return lhs.equal(rhs);
    }

    template <class T>
    inline bool operator!=(const xscalar_stepper<T>& lhs,
                           const xscalar_stepper<T>& rhs) noexcept
    {
        return !(lhs.equal(rhs));
    }

    /***********************************
     * xscalar_iterator implementation *
     ***********************************/

    template <class T>
    inline xscalar_iterator<T>::xscalar_iterator(const container_type* c) noexcept
        : p_c(c)
    {
    }

    template <class T>
    inline auto xscalar_iterator<T>::operator++() noexcept -> self_type&
    {
        return *this;
    }

    template <class T>
    inline auto xscalar_iterator<T>::operator++(int) noexcept -> self_type
    {
        self_type tmp(*this);
        ++(*this);
        return tmp;
    }

    template <class T>
    inline auto xscalar_iterator<T>::operator*() const noexcept -> reference
    {
        return p_c->operator()();
    }

    template <class T>
    inline bool xscalar_iterator<T>::equal(const self_type& rhs) const noexcept
    {
        return p_c == rhs.p_c;
    }

    template <class T>
    inline bool operator==(const xscalar_iterator<T>& lhs,
                           const xscalar_iterator<T>& rhs) noexcept
    {
        return lhs.equal(rhs);
    }

    template <class T>
    inline bool operator!=(const xscalar_iterator<T>& lhs,
                           const xscalar_iterator<T>& rhs) noexcept
    {
        return !(lhs.equal(rhs));
    }
}

#endif

