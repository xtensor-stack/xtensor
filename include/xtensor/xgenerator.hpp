/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef xgenerator_HPP
#define xgenerator_HPP

#include <cstddef>
#include <type_traits>
#include <utility>
#include <tuple>
#include <algorithm>

#include "xexpression.hpp"
#include "xiterator.hpp"
#include "xutils.hpp"

namespace xt
{

    template <class F, class R>
    class xgenerator_stepper;

    /**************
     * xgenerator *
     **************/

    /**
     * @class xgenerator
     * @brief Multidimensional function operating on indices.
     *
     * Th xgenerator class implements a multidimensional function
     * operating on the supplied indices.
     *
     * @tparam F the function type
     * @tparam R the return type of the function
     */
    template <class F, class R>
    class xgenerator : public xexpression<xgenerator<F, R>>
    {

    public:

        using self_type = xgenerator<F, R>;
        using functor_type = typename std::remove_reference<F>::type;

        using value_type = R;
        using reference = value_type;
        using const_reference = value_type;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using shape_type = std::vector<size_type>;
        using strides_type = std::vector<size_type>;
        using closure_type = const self_type;

        using const_stepper = xgenerator_stepper<F, R>;
        using const_iterator = xiterator<const_stepper, shape_type>;
        using const_storage_iterator = const_iterator;

        template <class Func, class S>
        xgenerator(Func&& f, const S& shape) noexcept;

        size_type dimension() const noexcept;
        shape_type shape() const;

        template <class... Args>
        const_reference operator()(Args... args) const;
        const_reference operator[](const xindex& index) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool broadcast_shape(S& shape) const;

        template <class S>
        bool is_trivial_broadcast(const S& /*strides*/) const noexcept;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        template <class S>
        xiterator<const_stepper, S> xbegin(const S& shape) const noexcept;
        template <class S>
        xiterator<const_stepper, S> xend(const S& shape) const noexcept;
        template <class S>
        xiterator<const_stepper, S> cxbegin(const S& shape) const noexcept;
        template <class S>
        xiterator<const_stepper, S> cxend(const S& shape) const noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;

        template <class S>
        const_stepper stepper_end(const S& shape) const noexcept;

        const_storage_iterator storage_begin() const noexcept;
        const_storage_iterator storage_end() const noexcept;

        const_storage_iterator storage_cbegin() const noexcept;
        const_storage_iterator storage_cend() const noexcept;

    private:

        template <class Func, std::size_t... I>
        const_stepper build_stepper(Func&& f, std::index_sequence<I...>) const noexcept;

        functor_type m_f;
        shape_type m_shape;
        friend class xgenerator_stepper<F, R>;
    };

    /***************************
     * xgenerator_stepper *
     ***************************/

    template <class F, class R>
    class xgenerator_stepper
    {

    public:

        using self_type = xgenerator_stepper<F, R>;
        using functor_type = typename std::remove_reference<F>::type;
        using xfunction_type = xgenerator<F, R>;

        using value_type = typename xfunction_type::value_type;
        using reference = typename xfunction_type::value_type;
        using pointer = typename xfunction_type::const_pointer;
        using size_type = typename xfunction_type::size_type;
        using difference_type = typename xfunction_type::difference_type;
        using iterator_category = std::input_iterator_tag;

        using shape_type = typename xfunction_type::shape_type;

        xgenerator_stepper(const xfunction_type* func, const shape_type& shape) noexcept;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);

        void to_end();

        reference operator*() const;

        bool equal(const self_type& rhs) const;

    private:
        const xfunction_type* p_f;
        shape_type m_shape;
        shape_type m_index;
    };

    template <class F, class R>
    bool operator==(const xgenerator_stepper<F, R>& it1,
                    const xgenerator_stepper<F, R>& it2);

    template <class F, class R>
    bool operator!=(const xgenerator_stepper<F, R>& it1,
                    const xgenerator_stepper<F, R>& it2);

    /**********************************
     * xgenerator implementation *
     **********************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xgenerator applying the specified function to the given
     * arguments.
     * @param f the function to apply
     * @param e the \ref xexpression arguments
     */
    template <class F, class R>
    template <class Func, class S>
    inline xgenerator<F, R>::xgenerator(Func&& f, const S& shape) noexcept
        : m_f(std::forward<Func>(f)), m_shape(shape)
    {
    }
    //@}

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of dimensions of the function.
     */
    template <class F, class R>
    inline auto xgenerator<F, R>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    /**
     * Returns the shape of the xgenerator.
     */
    template <class F, class R>
    inline auto xgenerator<F, R>::shape() const -> shape_type
    {
        return m_shape;
    }
    //@}

    /**
     * @name Data
     */
    /**
     * Returns the evaluated element at the specified position in the function.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the function.
     */
    template <class F, class R>
    template <class... Args>
    inline auto xgenerator<F, R>::operator()(Args... args) const -> const_reference
    {
        return m_f(args...);
    }

    template <class F, class R>
    inline auto xgenerator<F, R>::operator[](const xindex& index) const -> const_reference
    {
        return m_f[index];
    }

    /**
     * Returns a constant reference to the element at the specified position in the function.
     * @param first iterator starting the sequence of indices
     * @param second iterator starting the sequence of indices
     * The number of indices in the squence should be equal or greater
     * than the number of dimensions of the container.
     */
    template <class F, class R>
    template <class It>
    inline auto xgenerator<F, R>::element(It first, It last) const -> const_reference
    {
        return m_f.element(first, last);
    }
    //@}
    
    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the function to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class R>
    template <class S>
    inline bool xgenerator<F, R>::broadcast_shape(S& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class R>
    template <class S>
    inline bool xgenerator<F, R>::is_trivial_broadcast(const S& /*strides*/) const noexcept
    {
        return false;
    }
    //@}

    /**
     * @name Iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the function.
     */
    template <class F, class R>
    inline auto xgenerator<F, R>::begin() const noexcept -> const_iterator
    {
        return xbegin(m_shape);
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the function.
     */
    template <class F, class R>
    inline auto xgenerator<F, R>::end() const noexcept -> const_iterator
    {
        return xend(m_shape);
    }

    /**
     * Returns a constant iterator to the first element of the function.
     */
    template <class F, class R>
    inline auto xgenerator<F, R>::cbegin() const noexcept -> const_iterator
    {
        return begin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the function.
     */
    template <class F, class R>
    inline auto xgenerator<F, R>::cend() const noexcept -> const_iterator
    {
        return end();
    }

    /**
     * Returns a constant iterator to the first element of the function. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class F, class R>
    template <class S>
    inline auto xgenerator<F, R>::xbegin(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xiterator<const_stepper, S>(stepper_begin(shape), shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * function. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class F, class R>
    template <class S>
    inline auto xgenerator<F, R>::xend(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xiterator<const_stepper, S>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the function. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class F, class R>
    template <class S>
    inline auto xgenerator<F, R>::cxbegin(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xbegin(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * function. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class F, class R>
    template <class S>
    inline auto xgenerator<F, R>::cxend(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xend(shape);
    }
    //@}

    template <class F, class R>
    template <class S>
    inline auto xgenerator<F, R>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        return xgenerator_stepper<F, R>(this, shape);
    }

    template <class F, class R>
    template <class S>
    inline auto xgenerator<F, R>::stepper_end(const S& shape) const noexcept -> const_stepper
    {
        auto stepper = xgenerator_stepper<F, R>(this, shape);
        stepper.to_end();
        return stepper;
    }

    /**
     * @name Storage iterators
     */
    /**
     * Returns an iterator to the first element of the buffer
     * containing the elements of the function.
     */
    template <class F, class R>
    inline auto xgenerator<F, R>::storage_begin() const noexcept -> const_storage_iterator
    {
        return cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the function.
     */
    template <class F, class R>
    inline auto xgenerator<F, R>::storage_end() const noexcept -> const_storage_iterator
    {
        return cend();
    }

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the function.
     */
    template <class F, class R>
    inline auto xgenerator<F, R>::storage_cbegin() const noexcept -> const_storage_iterator
    {
        return cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the function.
     */
    template <class F, class R>
    inline auto xgenerator<F, R>::storage_cend() const noexcept -> const_storage_iterator
    {
        return cend();
    }
    //@}

    /******************************************
     * xgenerator_stepper implementation *
     ******************************************/

    template <class F, class R>
    inline xgenerator_stepper<F, R>::xgenerator_stepper(const xfunction_type* func, const shape_type& shape) noexcept
        : p_f(func), m_shape(shape), m_index(make_sequence<shape_type>(shape.size(), size_type(0)))
    {
    }

    template <class F, class R>
    inline void xgenerator_stepper<F, R>::step(size_type dim, size_type n)
    {
        m_index[dim] += n;
    }

    template <class F, class R>
    inline void xgenerator_stepper<F, R>::step_back(size_type dim, size_type n)
    {
        m_index[dim] -= 1;
    }

    template <class F, class R>
    inline void xgenerator_stepper<F, R>::reset(size_type dim)
    {
        m_index[dim] = 0;
    }

    template <class F, class R>
    inline void xgenerator_stepper<F, R>::to_end()
    {
        m_index = m_shape;
    }

    template <class F, class R>
    inline bool xgenerator_stepper<F, R>::equal(const self_type& rhs) const
    {
        return p_f == rhs.p_f && std::equal(m_index.begin(), m_index.end(), rhs.m_index.begin());
    }

    template <class F, class R>
    inline auto xgenerator_stepper<F, R>::operator*() const -> reference
    {
        return (*p_f)[m_index];
    }

    template <class F, class R>
    inline bool operator==(const xgenerator_stepper<F, R>& it1,
                           const xgenerator_stepper<F, R>& it2)
    {
        return it1.equal(it2);
    }

    template <class F, class R>
    inline bool operator!=(const xgenerator_stepper<F, R>& it1,
                           const xgenerator_stepper<F, R>& it2)
    {
        return !(it1.equal(it2));
    }
}

#endif