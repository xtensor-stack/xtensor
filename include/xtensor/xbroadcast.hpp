/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XBROADCAST_HPP
#define XBROADCAST_HPP

#include <utility>
#include <initializer_list>
#include <vector>
#include <algorithm>

#include "xtensor/xutils.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xiterator.hpp"

namespace xt
{

    /*************
     * broadcast *
     *************/

    template <class E, class S>
    auto broadcast(E&& e, const S& s);

    template <class E, class I, std::size_t L>
    auto broadcast(E&& e, const I(&s)[L]);

    /**************
     * xbroadcast *
     **************/

    /**
     * @class xbroadcast
     * @brief Broadcasted xexpression to a specified shape.
     *
     * Th xbroadcast class implements the broadcasting of an xexpression
     * to a specified shape
     *
     * @tparam E the type of the xexpression to broadcast
     * @tparam S the type of the shape.
     */
    template <class E, class X>
    class xbroadcast : public xexpression<xbroadcast<E, X>>
    {

    public:

        using self_type = xbroadcast<E, X>;

        using value_type = typename E::value_type;
        using reference = typename E::reference;
        using const_reference = typename E::const_reference;
        using pointer = typename E::pointer;
        using const_pointer = typename E::const_pointer;
        using size_type = typename E::size_type;
        using difference_type = typename E::difference_type;
        
        using shape_type = promote_shape_t<typename E::shape_type, X>;
        using strides_type = promote_strides_t<typename E::strides_type, X>;
        using closure_type = const self_type;

        using const_stepper = typename E::const_stepper;
        using const_iterator = xiterator<const_stepper, shape_type>;
        using const_storage_iterator = const_iterator;

        template <class S>
        xbroadcast(E e, S s) noexcept;

        size_type dimension() const noexcept;
        const shape_type & shape() const noexcept;

        template <class... Args>
        const_reference operator()(Args... args) const;
        const_reference operator[](const xindex& index) const;

        template <class S>
        bool broadcast_shape(S& shape) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const noexcept;

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

        E m_e;
        shape_type m_shape;
    };

    /****************************
     * broadcast implementation *
     ****************************/

    namespace detail
    {
        template <class R, class A>
        struct shape_forwarder
        {
            static inline R run(const A& r)
            {
                return R(r.cbegin(), r.cend());
            }
        };

        template <class R, class T, std::size_t L>
        struct shape_forwarder<R, T(&)[L]>
        {
            static inline R run(const T(&r)[L])
            {
                return R(r, r + L);
            }
        };        

        template <class R>
        struct shape_forwarder<R, R>
        {
            static inline const R& run(const R& r)
            {
                return r;
            }
        };
    }

    template <class E, class S>
    inline auto broadcast(E&& e, const S& s)
    {
        using broadcast_type = xbroadcast<get_xexpression_type<E>, S>;
        return broadcast_type(std::forward<E>(e), detail::shape_forwarder<typename broadcast_type::shape_type, S>::run(s));
    }

    template <class E, class I, std::size_t L>
    inline auto broadcast(E&& e, const I(&s)[L])
    {
        using broadcast_type = xbroadcast<get_xexpression_type<E>, std::array<I, L>>;
        return broadcast_type(std::forward<E>(e), detail::shape_forwarder<typename broadcast_type::shape_type, I(&)[L]>::run(s));
    }

    /*****************************
     * xbroadcast implementation *
     *****************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xbroadcast broadcasting the specified expression to the given
     * shape
     *
     * @param e the expression to broadcast
     * @param s the shape to apply
     */
    template <class E, class X>
    template <class S>
    inline xbroadcast<E, X>::xbroadcast(E e, S s) noexcept
        : m_e(std::move(e)), m_shape(std::move(s))
    {
        xt::broadcast_shape(e.shape(), m_shape);
    }
    //@}


    template <class E, class X>
    inline auto xbroadcast<E, X>::dimension() const noexcept -> size_type
    {
        return m_shape.size();
    }

    template <class E, class X>
    inline auto xbroadcast<E, X>::shape() const noexcept -> const shape_type &
    {
        return m_shape;
    }

    template <class E, class X>
    template <class... Args>
    inline auto xbroadcast<E, X>::operator()(Args... args) const -> const_reference
    {
        return detail::get_element(m_e, args...);
    }

    template <class E, class X>
    inline auto xbroadcast<E, X>::operator[](const xindex& index) const -> const_reference
    {
        // TODO:: depile
        return m_e[index];
    }

    template <class E, class X>
    template <class S>
    inline bool xbroadcast<E, X>::broadcast_shape(S& shape) const
    { 
        return xt::broadcast_shape(m_shape, shape);
    }

    template <class E, class X>
    template <class S>
    inline bool xbroadcast<E, X>::is_trivial_broadcast(const S& strides) const noexcept
    {
        return dimension() == m_e.dimension() && std::equal(m_shape.cbegin(), m_shape.cend(), m_e.shape().cbegin()) &&
               m_e.is_trivial_broadcast(strides);
    }

    template <class E, class X>
    inline auto xbroadcast<E, X>::begin() const noexcept -> const_iterator
    {
        return cxbegin(shape());
    }

    template <class E, class X>
    inline auto xbroadcast<E, X>::end() const noexcept -> const_iterator
    {
        return cxend(shape());
    }

    template <class E, class X>
    inline auto xbroadcast<E, X>::cbegin() const noexcept -> const_iterator
    {
        return cxbegin(shape());
    }

    template <class E, class X>
    inline auto xbroadcast<E, X>::cend() const noexcept -> const_iterator
    {
        return cxend(shape());
    }

    template <class E, class X>
    template <class S>
    inline auto xbroadcast<E, X>::xbegin(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        // Could check if (broadcastable(shape, m_shape)
        return cxbegin(shape);
    }

    template <class E, class X>
    template <class S>
    inline auto xbroadcast<E, X>::xend(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        // Could check if (broadcastable(shape, m_shape)
        return cxend(shape);
    }

    template <class E, class X>
    template <class S>
    inline auto xbroadcast<E, X>::cxbegin(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        // Could check if (broadcastable(shape, m_shape)
        return cxbegin(shape);
    }

    template <class E, class X>
    template <class S>
    inline auto xbroadcast<E, X>::cxend(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        // Could check if (broadcastable(shape, m_shape)
        return cxend(shape);
    }

    template <class E, class X>
    template <class S>
    inline auto xbroadcast<E, X>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        // Could check if (broadcastable(shape, m_shape)
        return m_e.stepper_begin(shape);
    }

    template <class E, class X>
    template <class S>
    inline auto xbroadcast<E, X>::stepper_end(const S& shape) const noexcept -> const_stepper
    {
        // Could check if (broadcastable(shape, m_shape)
        return m_e.stepper_begin(shape);
    }

    template <class E, class X>
    inline auto xbroadcast<E, X>::storage_begin() const noexcept -> const_storage_iterator
    {
        return cbegin();
    }

    template <class E, class X>
    inline auto xbroadcast<E, X>::storage_end() const noexcept -> const_storage_iterator
    {
        return cend();
    }

    template <class E, class X>
    inline auto xbroadcast<E, X>::storage_cbegin() const noexcept -> const_storage_iterator
    {
        return cbegin();
    }

    template <class E, class X>
    inline auto xbroadcast<E, X>::storage_cend() const noexcept -> const_storage_iterator
    {
        return cend();
    }

}

#endif

