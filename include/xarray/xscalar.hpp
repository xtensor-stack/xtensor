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
        using shape_type = array_shape<size_type>;
        using strides_type = array_strides<size_type>;

        xscalar(const T& value);

        size_type size() const;
        size_type dimension() const;

        shape_type shape() const;
        strides_type strides() const;
        strides_type backstrides() const;

        template <class... Args>
        const_reference operator()(Args... args) const;

        bool broadcast_shape(shape_type& shape) const;

    private:

        const T& m_value;
    };


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
    inline bool xscalar<T>::broadcast_shape(xscalar<T>::shape_type& shape) const
    {
        return true;
    }

}

#endif

