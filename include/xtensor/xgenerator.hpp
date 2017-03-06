/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XGENERATOR_HPP
#define XGENERATOR_HPP

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

    template <class F, class R, class S>
    class xgenerator_stepper;

    /**************
     * xgenerator *
     **************/

    /**
     * @class xgenerator
     * @brief Multidimensional function operating on indices.
     *
     * The xgenerator class implements a multidimensional function,
     * generating a value from the supplied indices.
     *
     * @tparam F the function type
     * @tparam R the return type of the function
     * @tparam S the shape type of the generator
     */
    template <class F, class R, class S>
    class xgenerator : public xexpression<xgenerator<F, R, S>>
    {

    public:

        using self_type = xgenerator<F, R, S>;
        using functor_type = typename std::remove_reference<F>::type;

        using value_type = R;
        using reference = value_type;
        using const_reference = value_type;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using shape_type = S;
        using strides_type = S;

        using const_stepper = xgenerator_stepper<F, R, S>;
        using stepper = const_stepper;

        using const_broadcast_iterator = xiterator<const_stepper, shape_type*>;
        using broadcast_iterator = const_broadcast_iterator;

        using const_iterator = const_broadcast_iterator;
        using iterator = const_iterator;

        template <class Func>
        xgenerator(Func&& f, const S& shape) noexcept;

        size_type dimension() const noexcept;
        const shape_type& shape() const;

        template <class... Args>
        const_reference operator()(Args... args) const;
        const_reference operator[](const xindex& index) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class O>
        bool broadcast_shape(O& shape) const;

        template <class O>
        bool is_trivial_broadcast(const O& /*strides*/) const noexcept;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        const_broadcast_iterator xbegin() const noexcept;
        const_broadcast_iterator xend() const noexcept;
        const_broadcast_iterator cxbegin() const noexcept;
        const_broadcast_iterator cxend() const noexcept;

        template <class O>
        xiterator<const_stepper, O> xbegin(const O& shape) const noexcept;
        template <class O>
        xiterator<const_stepper, O> xend(const O& shape) const noexcept;
        template <class O>
        xiterator<const_stepper, O> cxbegin(const O& shape) const noexcept;
        template <class O>
        xiterator<const_stepper, O> cxend(const O& shape) const noexcept;

        template <class O>
        const_stepper stepper_begin(const O& shape) const noexcept;
        template <class O>
        const_stepper stepper_end(const O& shape) const noexcept;

    private:

        functor_type m_f;
        shape_type m_shape;
        friend class xgenerator_stepper<F, R, S>;
    };

    /**********************
     * xgenerator_stepper *
     **********************/

    template <class F, class R, class S>
    class xgenerator_stepper
    {

    public:

        using self_type = xgenerator_stepper<F, R, S>;
        using functor_type = typename std::remove_reference<F>::type;
        using xgenerator_type = xgenerator<F, R, S>;

        using value_type = typename xgenerator_type::value_type;
        using reference = typename xgenerator_type::value_type;
        using pointer = typename xgenerator_type::const_pointer;
        using size_type = typename xgenerator_type::size_type;
        using difference_type = typename xgenerator_type::difference_type;
        using iterator_category = std::input_iterator_tag;

        using shape_type = typename xgenerator_type::shape_type;
        using index_type = xindex_type_t<shape_type>;

        xgenerator_stepper() = default;
        xgenerator_stepper(const xgenerator_type* func, size_type offset, bool end = false) noexcept;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);

        void to_end();

        reference operator*() const;

        bool equal(const self_type& rhs) const;

    private:

        const xgenerator_type* p_f;
        index_type m_index;
        size_type m_offset;
    };

    template <class F, class R, class S>
    bool operator==(const xgenerator_stepper<F, R, S>& it1,
                    const xgenerator_stepper<F, R, S>& it2);

    template <class F, class R, class S>
    bool operator!=(const xgenerator_stepper<F, R, S>& it1,
                    const xgenerator_stepper<F, R, S>& it2);

    /*****************************
     * xgenerator implementation *
     *****************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xgenerator applying the specified function over the 
     * given shape.
     * @param f the function to apply
     * @param shape the shape of the xgenerator
     */
    template <class F, class R, class S>
    template <class Func>
    inline xgenerator<F, R, S>::xgenerator(Func&& f, const S& shape) noexcept
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
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    /**
     * Returns the shape of the xgenerator.
     */
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::shape() const -> const shape_type&
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
    template <class F, class R, class S>
    template <class... Args>
    inline auto xgenerator<F, R, S>::operator()(Args... args) const -> const_reference
    {
        return m_f(args...);
    }

    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::operator[](const xindex& index) const -> const_reference
    {
        return m_f[index];
    }

    /**
     * Returns a constant reference to the element at the specified position in the function.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the squence should be equal to or greater
     * than the number of dimensions of the container.
     */
    template <class F, class R, class S>
    template <class It>
    inline auto xgenerator<F, R, S>::element(It first, It last) const -> const_reference
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
    template <class F, class R, class S>
    template <class O>
    inline bool xgenerator<F, R, S>::broadcast_shape(O& shape) const
    {
        return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class R, class S>
    template <class O>
    inline bool xgenerator<F, R, S>::is_trivial_broadcast(const O& /*strides*/) const noexcept
    {
        return false;
    }
    //@}

    /**
     * @name Iterators
     */
    /**
     * Returns an iterator to the first element of the buffer
     * containing the elements of the function.
     */
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::begin() const noexcept -> const_iterator
    {
        return cxbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the function.
     */
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::end() const noexcept -> const_iterator
    {
        return cxend();
    }

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the function.
     */
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::cbegin() const noexcept -> const_iterator
    {
        return cxbegin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the function.
     */
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::cend() const noexcept -> const_iterator
    {
        return cxend();
    }
    //@}

    /**
     * @name Broadcast iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the function.
     */
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::xbegin() const noexcept -> const_broadcast_iterator
    {
        return const_broadcast_iterator(stepper_begin(m_shape), &m_shape);
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the function.
     */
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::xend() const noexcept -> const_broadcast_iterator
    {
        return const_broadcast_iterator(stepper_end(m_shape), &m_shape);
    }

    /**
     * Returns a constant iterator to the first element of the function.
     */
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::cxbegin() const noexcept -> const_broadcast_iterator
    {
        return xbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the function.
     */
    template <class F, class R, class S>
    inline auto xgenerator<F, R, S>::cxend() const noexcept -> const_broadcast_iterator
    {
        return xend();
    }

    /**
     * Returns a constant iterator to the first element of the function. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class F, class R, class S>
    template <class O>
    inline auto xgenerator<F, R, S>::xbegin(const O& shape) const noexcept -> xiterator<const_stepper, O>
    {
        return xiterator<const_stepper, S>(stepper_begin(shape), shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * function. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class F, class R, class S>
    template <class O>
    inline auto xgenerator<F, R, S>::xend(const O& shape) const noexcept -> xiterator<const_stepper, O>
    {
        return xiterator<const_stepper, S>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the function. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class F, class R, class S>
    template <class O>
    inline auto xgenerator<F, R, S>::cxbegin(const O& shape) const noexcept -> xiterator<const_stepper, O>
    {
        return xbegin(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * function. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class F, class R, class S>
    template <class O>
    inline auto xgenerator<F, R, S>::cxend(const O& shape) const noexcept -> xiterator<const_stepper, O>
    {
        return xend(shape);
    }
    //@}

    template <class F, class R, class S>
    template <class O>
    inline auto xgenerator<F, R, S>::stepper_begin(const O& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return xgenerator_stepper<F, R, S>(this, offset);
    }

    template <class F, class R, class S>
    template <class O>
    inline auto xgenerator<F, R, S>::stepper_end(const O& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return xgenerator_stepper<F, R, S>(this, offset, true);
    }

    /******************************************
     * xgenerator_stepper implementation *
     ******************************************/

    template <class F, class R, class S>
    inline xgenerator_stepper<F, R, S>::xgenerator_stepper(const xgenerator_type* func, size_type offset, bool end) noexcept
        : p_f(func), m_index(make_sequence<index_type>(func->shape().size(), size_type(0))), m_offset(offset)
    {
        if (end)
        {
            m_index = p_f->shape();
        }
    }

    template <class F, class R, class S>
    inline void xgenerator_stepper<F, R, S>::step(size_type dim, size_type n)
    {
        if(dim >= m_offset)
            m_index[dim - m_offset] += n;
    }

    template <class F, class R, class S>
    inline void xgenerator_stepper<F, R, S>::step_back(size_type dim, size_type n)
    {
        if(dim >= m_offset)
            m_index[dim - m_offset] -= n;
    }

    template <class F, class R, class S>
    inline void xgenerator_stepper<F, R, S>::reset(size_type dim)
    {
        if(dim >= m_offset)
            m_index[dim - m_offset] = 0;
    }

    template <class F, class R, class S>
    inline void xgenerator_stepper<F, R, S>::to_end()
    {
        m_index = p_f->shape();
    }

    template <class F, class R, class S>
    inline bool xgenerator_stepper<F, R, S>::equal(const self_type& rhs) const
    {
        return (p_f == rhs.p_f) && (m_index == rhs.m_index);
    }

    template <class F, class R, class S>
    inline auto xgenerator_stepper<F, R, S>::operator*() const -> reference
    {
        return p_f->element(m_index.begin(), m_index.end());
    }

    template <class F, class R, class S>
    inline bool operator==(const xgenerator_stepper<F, R, S>& it1,
                           const xgenerator_stepper<F, R, S>& it2)
    {
        return it1.equal(it2);
    }

    template <class F, class R, class S>
    inline bool operator!=(const xgenerator_stepper<F, R, S>& it1,
                           const xgenerator_stepper<F, R, S>& it2)
    {
        return !(it1.equal(it2));
    }

    namespace detail
    {
#ifdef X_OLD_CLANG
        template <class Functor, class I>
        inline auto make_xgenerator(Functor&& f, std::initializer_list<I> shape) noexcept
        {
            using shape_type = std::vector<std::size_t>;
            using type = xgenerator<Functor, typename Functor::value_type, shape_type>;
            return type(std::forward<Functor>(f), forward_sequence<shape_type>(shape));
        }
#else
        template <class Functor, class I, std::size_t L>
        inline auto make_xgenerator(Functor&& f, const I(&shape)[L]) noexcept
        {
            using shape_type = std::array<std::size_t, L>;
            using type = xgenerator<Functor, typename Functor::value_type, shape_type>;
            return type(std::forward<Functor>(f), forward_sequence<shape_type>(shape));
        }
#endif
        
        template <class Functor, class S>
        inline auto make_xgenerator(Functor&& f, S&& shape) noexcept
        {
            using type = xgenerator<Functor, typename Functor::value_type, std::decay_t<S>>;
            return type(std::forward<Functor>(f), std::forward<S>(shape));
        }
    }
}

#endif
