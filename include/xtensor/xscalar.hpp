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
        using strides_type = std::array<size_type, 0>;

        using closure_type = const self_type;
        using const_stepper = xscalar_stepper<T>;
        using const_storage_iterator = xscalar_iterator<T>;

        xscalar(const T& value);

        size_type size() const;
        size_type dimension() const;

        shape_type shape() const;
        strides_type strides() const;
        strides_type backstrides() const;

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class S>
        bool broadcast_shape(S& shape) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const;

        template <class S>
        const_stepper stepper_begin(const S& shape) const;
        template <class S>
        const_stepper stepper_end(const S& shape) const;

        const_storage_iterator storage_begin() const;
        const_storage_iterator storage_end() const;

    private:

        const T& m_value;
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

        explicit xscalar_stepper(const container_type* c);

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);

        void to_end();

        bool equal(const self_type& rhs) const;

    private:

        const container_type* p_c;
    };

    template <class T>
    bool operator==(const xscalar_stepper<T>& lhs,
                    const xscalar_stepper<T>& rhs);

    template <class T>
    bool operator!=(const xscalar_stepper<T>& lhs,
                    const xscalar_stepper<T>& rhs);

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

        explicit xscalar_iterator(const container_type* c);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;

        bool equal(const self_type& rhs) const;

    private:

        const container_type* p_c;
    };

    template <class T>
    bool operator==(const xscalar_iterator<T>& lhs,
                    const xscalar_iterator<T>& rhs);

    template <class T>
    bool operator!=(const xscalar_iterator<T>& lhs,
                    const xscalar_iterator<T>& rhs);

    /**************************
     * xscalar implementation *
     **************************/

    template <class T>
    inline xscalar<T>::xscalar(const T& value)
        : m_value(value)
    {
    }

    template <class T>
    inline auto xscalar<T>::size() const -> size_type
    {
        return 1;
    }

    template <class T>
    inline auto xscalar<T>::dimension() const -> size_type
    {
        return 0;
    }

    template <class T>
    inline auto xscalar<T>::shape() const -> shape_type
    {
        return {};
    }

    template <class T>
    inline auto xscalar<T>::strides() const -> strides_type
    {
        return {};
    }

    template <class T>
    inline auto xscalar<T>::backstrides() const -> strides_type
    {
        return {};
    }

    template <class T>
    template <class... Args>
    inline auto xscalar<T>::operator()(Args...) const -> const_reference
    {
        return m_value;
    }

    template <class T>
    template <class S>
    inline bool xscalar<T>::broadcast_shape(S&) const
    {
        return true;
    }

    template <class T>
    template <class S>
    inline bool xscalar<T>::is_trivial_broadcast(const S&) const
    {
        return true;
    }

    template <class T>
    template <class S>
    inline auto xscalar<T>::stepper_begin(const S&) const -> const_stepper
    {
        return const_stepper(this);
    }

    template <class T>
    template <class S>
    inline auto xscalar<T>::stepper_end(const S&) const -> const_stepper
    {
        return const_stepper(this);
    }

    template <class T>
    inline auto xscalar<T>::storage_begin() const -> const_storage_iterator
    {
        return const_storage_iterator(this);
    }

    template <class T>
    inline auto xscalar<T>::storage_end() const -> const_storage_iterator
    {
        return const_storage_iterator(this);
    }

    /**********************************
     * xscalar_stepper implementation *
     **********************************/

    template <class T>
    inline xscalar_stepper<T>::xscalar_stepper(const container_type* c)
        : p_c(c)
    {
    }

    template <class T>
    inline auto xscalar_stepper<T>::operator*() const -> reference
    {
        return p_c->operator()();
    }

    template <class T>
    inline void xscalar_stepper<T>::step(size_type /*dim*/, size_type /*n*/)
    {
    }

    template <class T>
    inline void xscalar_stepper<T>::step_back(size_type /*dim*/, size_type /*n*/)
    {
    }

    template <class T>
    inline void xscalar_stepper<T>::reset(size_type /*dim*/)
    {
    }

    template <class T>
    inline void xscalar_stepper<T>::to_end()
    {
    }

    template <class T>
    inline bool xscalar_stepper<T>::equal(const self_type& rhs) const
    {
        return p_c == rhs.p_c;
    }

    template <class T>
    inline bool operator==(const xscalar_stepper<T>& lhs,
                           const xscalar_stepper<T>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class T>
    inline bool operator!=(const xscalar_stepper<T>& lhs,
                           const xscalar_stepper<T>& rhs)
    {
        return !(lhs.equal(rhs));
    }

    /***********************************
     * xscalar_iterator implementation *
     ***********************************/

    template <class T>
    inline xscalar_iterator<T>::xscalar_iterator(const container_type* c)
        : p_c(c)
    {
    }

    template <class T>
    inline auto xscalar_iterator<T>::operator++() -> self_type&
    {
        return *this;
    }

    template <class T>
    inline auto xscalar_iterator<T>::operator++(int) -> self_type
    {
        self_type tmp(*this);
        ++(*this);
        return tmp;
    }

    template <class T>
    inline auto xscalar_iterator<T>::operator*() const -> reference
    {
        return p_c->operator()();
    }

    template <class T>
    inline bool xscalar_iterator<T>::equal(const self_type& rhs) const
    {
        return p_c == rhs.p_c;
    }

    template <class T>
    inline bool operator==(const xscalar_iterator<T>& lhs,
                           const xscalar_iterator<T>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class T>
    inline bool operator!=(const xscalar_iterator<T>& lhs,
                           const xscalar_iterator<T>& rhs)
    {
        return !(lhs.equal(rhs));
    }
}

#endif

