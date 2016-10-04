#ifndef XSCALAR_HPP
#define XSCALAR_HPP

#include <utility>

#include "xexpression.hpp"
#include "xindex.hpp"

namespace qs
{

    /*************************
     * xscalar
     *************************/

    // xscalar is a cheap wrapper for a scalar value as an xexpression.

    template <class T>
    class xscalar_stepper;

    template <class T>
    class xscalar : public xexpression<xscalar<T>>
    {

    public:

        using value_type = T;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;
        using size_type = size_t;
        using difference_type = ptrdiff_t;

        using self_type = xscalar<T>;
        using shape_type = xshape<size_type>;
        using strides_type = xstrides<size_type>;

        using closure_type = const self_type;
        using const_stepper = xscalar_stepper<T>;

        xscalar(const T& value);

        size_type size() const;
        size_type dimension() const;

        shape_type shape() const;
        strides_type strides() const;
        strides_type backstrides() const;

        template <class... Args>
        const_reference operator()(Args... args) const;

        bool broadcast_shape(shape_type& shape) const;
        bool is_trivial_broadcast(const strides_type& strides) const;

        const_stepper stepper_begin(const shape_type& shape) const;
        const_stepper stepper_end(const shape_type& shape) const;

    private:

        const T& m_value;
    };


    /*********************
     * xscalar_stepper
     *********************/

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

        xscalar_stepper(const container_type* c, bool end = false);

        reference operator*() const;

        void step(size_type i);
        void reset(size_type i);

        void to_end();

        bool equal(const self_type& rhs) const;

    private:

        const container_type* p_c;
        bool m_end;
    };

    template <class T>
    bool operator==(const xscalar_stepper<T>& lhs,
                    const xscalar_stepper<T>& rhs);

    template <class T>
    bool operator!=(const xscalar_stepper<T>& lhs,
                    const xscalar_stepper<T>& rhs);


    /****************************
     * xscalar implementation
     ****************************/

    template <class T>
    inline xscalar<T>::xscalar(const T& value)
        : m_value(value)
    {
    }

    template <class T>
    inline typename xscalar<T>::size_type xscalar<T>::size() const
    {
        return 1;
    }

    template <class T>
    inline typename xscalar<T>::size_type xscalar<T>::dimension() const
    {
        return 0;
    }

    template <class T>
    inline typename xscalar<T>::shape_type xscalar<T>::shape() const
    {
        return {};
    }

    template <class T>
    inline typename xscalar<T>::strides_type xscalar<T>::strides() const
    {
        return {};
    }

    template <class T>
    inline typename xscalar<T>::strides_type xscalar<T>::backstrides() const
    {
        return {};
    }

    template <class T>
    template <class... Args>
    inline typename xscalar<T>::const_reference xscalar<T>::operator()(Args... args) const
    {
        return m_value;
    }

    template <class T>
    inline bool xscalar<T>::broadcast_shape(shape_type&) const
    {
        return true;
    }

    template <class T>
    inline bool xscalar<T>::is_trivial_broadcast(const strides_type&) const
    {
        return true;
    }


    /************************************
     * xscalar_stepper implementation
     ************************************/

    template <class T>
    inline xscalar_stepper<T>::xscalar_stepper(const container_type* c, bool end)
        : p_c(c), m_end(end)
    {
    }

    template <class T>
    inline auto xscalar_stepper<T>::operator*() const -> reference
    {
        return p_c->operator()();
    }

    template <class T>
    inline void xscalar_stepper<T>::step(size_type i)
    {
    }

    template <class T>
    inline void xscalar_stepper<T>::reset(size_type i)
    {
    }

    template <class T>
    inline void xscalar_stepper<T>::to_end()
    {
        m_end = true;
    }

    template <class T>
    inline bool xscalar_stepper<T>::equal(const self_type& rhs) const
    {
        return p_c = rhs.p_c && m_end = rhs.m_end;
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

}

#endif

