/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XOFFSETVIEW_HPP
#define XOFFSETVIEW_HPP

#include <utility>
#include <algorithm>
#include <type_traits>
#include <iterator>
#include <cstddef>
#include <array>

#include "xtensor/xutils.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xiterator.hpp"
#include "xtensor/xsemantic.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"

namespace xt
{

    /***************************
     * xoffsetview declaration *
     ***************************/

    template <class It, class M, std::size_t I>
    class xoffset_iterator;

    template <class St, class M, std::size_t I>
    class xoffset_stepper;

    template <class CT, class M, std::size_t I>
    class xoffsetview;


    /*****************************
     * offsetview_temporary_type *
     *****************************/

    namespace detail
    {
        template <class S, class M, std::size_t I>
        struct offsetview_temporary_type_impl
        {
            using type = xarray<M>;
        };

        template <class T, std::size_t L, class M, std::size_t I>
        struct offsetview_temporary_type_impl<std::array<T, L>, M, I>
        {
            using type = xtensor<M, L>;
        };
    }

    template <class E, class M, std::size_t I>
    struct offsetview_temporary_type
    {
        using type = typename detail::offsetview_temporary_type_impl<typename E::shape_type, M, I>::type;
    };

    template <class CT, class M, std::size_t I>
    struct xcontainer_inner_types<xoffsetview<CT, M, I>>
    {
        using xexpression_type = std::decay_t<CT>;
        using temporary_type = typename offsetview_temporary_type<xexpression_type, M, I>::type;
    };

    template <class CT, class M, std::size_t I>
    class xoffsetview : public xview_semantic<xoffsetview<CT, M, I>>
    {

    public:

        using self_type = xoffsetview<CT, M, I>;
        using xexpression_type = std::decay_t<CT>;
        using semantic_base = xview_semantic<self_type>;

        using value_type = M;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;
        
        using shape_type = typename xexpression_type::shape_type;

        using stepper = xoffset_stepper<typename xexpression_type::stepper, M, I>;
        using const_stepper = xoffset_stepper<typename xexpression_type::const_stepper, M, I>;

        using broadcast_iterator = xoffset_iterator<typename xexpression_type::broadcast_iterator, M, I>;
        using const_broadcast_iterator = xoffset_iterator<typename xexpression_type::const_broadcast_iterator, M, I>;
        
        using iterator = xoffset_iterator<typename xexpression_type::iterator, M, I>;
        using const_iterator = xoffset_iterator<typename xexpression_type::const_iterator, M, I>;

        xoffsetview(CT e) noexcept;

        template <class E>
        self_type& operator=(const xexpression<E>& e);

        template <class E>
        disable_xexpression<E, self_type>& operator=(const E& e);

        size_type dimension() const noexcept;
        const shape_type & shape() const noexcept;

        template <class... Args>
        const_reference operator()(Args... args) const;
        const_reference operator[](const xindex& index) const;

        template <class It>
        const_reference element(It, It last) const;

        template <class S>
        bool broadcast_shape(S& shape) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const;

        iterator begin();
        iterator end();

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

        broadcast_iterator xbegin();
        broadcast_iterator xend();

        const_broadcast_iterator xbegin() const;
        const_broadcast_iterator xend() const;
        const_broadcast_iterator cxbegin() const;
        const_broadcast_iterator cxend() const;

        template <class S>
        xoffset_iterator<xiterator<typename xexpression_type::stepper, S>, M, I>
            xbegin(const S& shape);
        template <class S>
        xoffset_iterator<xiterator<typename xexpression_type::stepper, S>, M, I>
            xend(const S& shape);

        template <class S>
        xoffset_iterator<xiterator<typename xexpression_type::const_stepper, S>, M, I> 
            xbegin(const S& shape) const;
        template <class S>
        xoffset_iterator<xiterator<typename xexpression_type::const_stepper, S>, M, I>
            xend(const S& shape) const;
        template <class S>
        xoffset_iterator<xiterator<typename xexpression_type::const_stepper, S>, M, I>
            cxbegin(const S& shape) const;
        template <class S>
        xoffset_iterator<xiterator<typename xexpression_type::const_stepper, S>, M, I>
            cxend(const S& shape) const;

        template <class S>
        stepper stepper_begin(const S& shape);
        template <class S>
        stepper stepper_end(const S& shape);
        template <class S>
        const_stepper stepper_begin(const S& shape) const;
        template <class S>
        const_stepper stepper_end(const S& shape) const;

    private:

        CT m_e;

        using temporary_type = typename xcontainer_inner_types<self_type>::temporary_type;
        void assign_temporary_impl(temporary_type& tmp);
        friend class xview_semantic<xoffsetview<CT, M, I>>;
    };

    /********************************
     * xoffset_iterator declaration *
     ********************************/

    template <class It, class M, std::size_t I>
    class xoffset_iterator
    {
    public:
        using member_type = M;
        using value_type = member_type;
        using reference = apply_cv_t<typename It::reference, value_type>;
        using pointer = std::remove_reference_t<reference>*;
        using difference_type = typename It::difference_type;
        using iterator_category = typename It::iterator_category;

        using self_type = xoffset_iterator<It, M, I>;

        xoffset_iterator() = default;
        xoffset_iterator(const It& it);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;

        bool equal(const xoffset_iterator& rhs) const;

    private:
        It m_it;

        template <class It_, class M_, std::size_t I_>
        friend xoffset_iterator<It_, M_, I_> operator+(xoffset_iterator<It_, M_, I_>, xoffset_iterator<It_, M_, I_>);

        template <class It_, class M_, std::size_t I_>
        friend typename xoffset_iterator<It_, M_, I_>::difference_type operator-(xoffset_iterator<It_, M_, I_>, xoffset_iterator<It_, M_, I_>);
    };

    template <class It, class M, std::size_t I>
    bool operator==(const xoffset_iterator<It, M, I>& lhs,
                    const xoffset_iterator<It, M, I>& rhs);

    template <class It, class M, std::size_t I>
    bool operator!=(const xoffset_iterator<It, M, I>& lhs,
                    const xoffset_iterator<It, M, I>& rhs);

    template <class It, class M, std::size_t I>
    xoffset_iterator<It, M, I> operator+(xoffset_iterator<It, M, I> it1, xoffset_iterator<It, M, I> it2)
    {
        return xoffset_iterator<It, M, I>(it1.m_it + it2.m_it);
    }

    template <class It, class M, std::size_t I>
    typename xoffset_iterator<It, M, I>::difference_type operator-(xoffset_iterator<It, M, I> it1, xoffset_iterator<It, M, I> it2)
    {
        return it1.m_it - it2.m_it;
    }

    /*******************************
     * xoffset_stepper declaration *
     *******************************/

    template <class St, class M, std::size_t I>
    class xoffset_stepper
    {
    public:

        using value_type = M;
        using reference = apply_cv_t<typename St::reference, value_type>;
        using pointer = std::remove_reference_t<reference>*;
        using size_type = typename St::size_type;
        using difference_type = typename St::difference_type;

        xoffset_stepper() = default;
        xoffset_stepper(const St& stepper);

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);
        void to_end();

        bool equal(const xoffset_stepper& rhs) const;

    private:
        St m_stepper;
    };

    template <class St, class M, std::size_t I>
    bool operator==(const xoffset_stepper<St, M, I>& lhs,
                    const xoffset_stepper<St, M, I>& rhs);

    template <class St, class M, std::size_t I>
    bool operator!=(const xoffset_stepper<St, M, I>& lhs,
                    const xoffset_stepper<St, M, I>& rhs);

    /******************************
     * xoffsetview implementation *
     ******************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xoffsetview expression wrappering the specified \ref xexpression.
     *
     * @param e the expression to broadcast
     * @param s the shape to apply
     */
    template <class CT, class M, std::size_t I>
    inline xoffsetview<CT, M, I>::xoffsetview(CT e) noexcept
        : m_e(e)
    {
    }
    //@}

    /**
     * @name Extended copy semantic
     */
    //@{
    /**
     * The extended assignment operator.
     */
    template <class CT, class M, std::size_t I>
    template <class E>
    inline auto xoffsetview<CT, M, I>::operator=(const xexpression<E>& e) -> self_type&
    {
        bool cond = (e.derived_cast().shape().size() == dimension())
                    && std::equal(shape().begin(), shape().end(), e.derived_cast().shape().begin());
        if(!cond)
        {
            semantic_base::operator=(broadcast(e.derived_cast(), shape()));
        }
        else
        {
            semantic_base::operator=(e);
        }
        return *this;
    }
    //@}

    template <class CT, class M, std::size_t I>
    template <class E>
    inline auto xoffsetview<CT, M, I>::operator=(const E& e) -> disable_xexpression<E, self_type>&
    {
        std::fill(begin(), end(), e);
        return *this;
    }

    template <class CT, class M, std::size_t I>
    inline void xoffsetview<CT, M, I>::assign_temporary_impl(temporary_type& tmp)
    {
        std::copy(tmp.cbegin(), tmp.cend(), xbegin());
    }

    /**
     * @name Size and shape
     */
    /**
     * Returns the number of dimensions of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::dimension() const noexcept -> size_type
    {
        return m_e.dimension();
    }

    /**
     * Returns the shape of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::shape() const noexcept -> const shape_type &
    {
        return m_e.shape();
    }
    //@}

    /**
     * @name Data
     */
    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the expression.
     */
    template <class CT, class M, std::size_t I>
    template <class... Args>
    inline auto xoffsetview<CT, M, I>::operator()(Args... args) const -> const_reference
    {
        return forward_offset<value_type, I>(m_e(args...));
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param index a sequence of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices in the sequence should be equal or greater
     * than the number of dimensions of the container.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::operator[](const xindex& index) const -> const_reference
    {
        return forward_offset<value_type, I>(m_e[index]);
    }
    
    /**
     * Returns a constant reference to the element at the specified position in the expression.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the squence should be equal to or greater
     * than the number of dimensions of the function.
     */
    template <class CT, class M, std::size_t I>
    template <class It>
    inline auto xoffsetview<CT, M, I>::element(It first, It last) const -> const_reference
    {
        return forward_offset<value_type, I>(m_e.element(first, last));
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
    template <class CT, class M, std::size_t I>
    template <class S>
    inline bool xoffsetview<CT, M, I>::broadcast_shape(S& shape) const
    { 
        return m_e.broadcast_shape(shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class CT, class M, std::size_t I>
    template <class S>
    inline bool xoffsetview<CT, M, I>::is_trivial_broadcast(const S& strides) const
    {
        return m_e.is_trivial_broadcast(strides);
    }
    //@}

    /**
     * @name Iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::begin() -> iterator
    {
        return m_e.begin();
    }

    /**
     * Returns an iterator to the element following the last element
     * of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::end() -> iterator
    {
        return m_e.end();
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::begin() const -> const_iterator
    {
        return m_e.cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::end() const -> const_iterator
    {
        return m_e.cend();
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::cbegin() const -> const_iterator
    {
        return m_e.cbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::cend() const -> const_iterator
    {
        return m_e.cend();
    }

    /**
     * Returns an iterator to the first element of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::xbegin() -> broadcast_iterator
    {
        return m_e.xbegin();
    }

    /**
     * Returns an iterator to the element following the last element
     * of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::xend() -> broadcast_iterator
    {
        return m_e.xend();
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::xbegin() const -> const_broadcast_iterator
    {
        return m_e.cxend();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::xend() const -> const_broadcast_iterator
    {
        return m_e.cxend();
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::cxbegin() const -> const_broadcast_iterator
    {
        return m_e.cxbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class CT, class M, std::size_t I>
    inline auto xoffsetview<CT, M, I>::cxend() const -> const_broadcast_iterator
    {
        return m_e.cxend();
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class CT, class M, std::size_t I>
    template <class S>
    inline auto xoffsetview<CT, M, I>::xbegin(const S& shape)
        -> xoffset_iterator<xiterator<typename xexpression_type::stepper, S>, M, I>
    {
        return m_e.xbegin(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class CT, class M, std::size_t I>
    template <class S>
    inline auto xoffsetview<CT, M, I>::xend(const S& shape)
        -> xoffset_iterator<xiterator<typename xexpression_type::stepper, S>, M, I>
    {
        return m_e.xend(shape);
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class CT, class M, std::size_t I>
    template <class S>
    inline auto xoffsetview<CT, M, I>::xbegin(const S& shape) const
        -> xoffset_iterator<xiterator<typename xexpression_type::const_stepper, S>, M, I>
    {
        return m_e.cxbegin(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class CT, class M, std::size_t I>
    template <class S>
    inline auto xoffsetview<CT, M, I>::xend(const S& shape) const
        -> xoffset_iterator<xiterator<typename xexpression_type::const_stepper, S>, M, I>
    {
        return m_e.cxend(shape);
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class CT, class M, std::size_t I>
    template <class S>
    inline auto xoffsetview<CT, M, I>::cxbegin(const S& shape) const
        -> xoffset_iterator<xiterator<typename xexpression_type::const_stepper, S>, M, I>
    {
        return m_e.cxbegin(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class CT, class M, std::size_t I>
    template <class S>
    inline auto xoffsetview<CT, M, I>::cxend(const S& shape) const
        -> xoffset_iterator<xiterator<typename xexpression_type::const_stepper, S>, M, I>
    {
        return m_e.cxend(shape);
    }
    //@}

    /***************
     * stepper api *
     ***************/

    template <class CT, class M, std::size_t I>
    template <class S>
    inline auto xoffsetview<CT, M, I>::stepper_begin(const S& shape) -> stepper
    {
        return m_e.stepper_begin(shape);
    }

    template <class CT, class M, std::size_t I>
    template <class S>
    inline auto xoffsetview<CT, M, I>::stepper_end(const S& shape) -> stepper
    {
        return m_e.stepper_end(shape);
    }

    template <class CT, class M, std::size_t I>
    template <class S>
    inline auto xoffsetview<CT, M, I>::stepper_begin(const S& shape) const -> const_stepper
    {
        const xexpression_type& t_e = m_e;
        return t_e.stepper_begin(shape);
    }

    template <class CT, class M, std::size_t I>
    template <class S>
    inline auto xoffsetview<CT, M, I>::stepper_end(const S& shape) const -> const_stepper
    {
        const xexpression_type& t_e = m_e;
        return t_e.stepper_end(shape);
    }

    /***********************************
     * xoffset_iterator implementation *
     ***********************************/

    template <class It, class M, std::size_t I>
    xoffset_iterator<It, M, I>::xoffset_iterator(const It& it) : m_it(it)
    {
    }

    template <class It, class M, std::size_t I>
    auto xoffset_iterator<It, M, I>::operator++() -> self_type&
    {
        ++m_it;
        return *this;
    }

    template <class It, class M, std::size_t I>
    auto xoffset_iterator<It, M, I>::operator++(int) -> self_type
    {
        self_type tmp(*this);
        ++m_it;
        return tmp;
    }

    template <class It, class M, std::size_t I>
    auto xoffset_iterator<It, M, I>::operator*() const -> reference
    {
        return forward_offset<M, I>(*m_it);
    }

    template <class It, class M, std::size_t I>
    auto xoffset_iterator<It, M, I>::equal(const xoffset_iterator& rhs) const -> bool
    {
        return m_it == rhs.m_it;
    }

    template <class It, class M, std::size_t I>
    bool operator==(const xoffset_iterator<It, M, I>& lhs,
                    const xoffset_iterator<It, M, I>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class It, class M, std::size_t I>
    bool operator!=(const xoffset_iterator<It, M, I>& lhs,
                    const xoffset_iterator<It, M, I>& rhs)
    {
        return !lhs.equal(rhs);
    }

    /**********************************
     * xoffset_stepper implementation *
     **********************************/

    template <class St, class M, std::size_t I>
    xoffset_stepper<St, M, I>::xoffset_stepper(const St& stepper) : m_stepper(stepper)
    {
    }

    template <class St, class M, std::size_t I>
    auto xoffset_stepper<St, M, I>::operator*() const -> reference
    {
        return forward_offset<M, I>(*m_stepper);
    }

    template <class St, class M, std::size_t I>
    void xoffset_stepper<St, M, I>::step(size_type dim, size_type n)
    {
        m_stepper.step(dim, n);
    }

    template <class St, class M, std::size_t I>
    void xoffset_stepper<St, M, I>::step_back(size_type dim, size_type n)
    {
        m_stepper.step_back(dim, n);
    }

    template <class St, class M, std::size_t I>
    void xoffset_stepper<St, M, I>::reset(size_type dim)
    {
        m_stepper.reset(dim);
    }

    template <class St, class M, std::size_t I>
    void xoffset_stepper<St, M, I>::to_end()
    {
        m_stepper.to_end();
    }

    template <class St, class M, std::size_t I>
    auto xoffset_stepper<St, M, I>::equal(const xoffset_stepper& rhs) const -> bool
    {
        return m_stepper == rhs.m_stepper;
    }

    template <class St, class M, std::size_t I>
    bool operator==(const xoffset_stepper<St, M, I>& lhs,
                    const xoffset_stepper<St, M, I>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class St, class M, std::size_t I>
    bool operator!=(const xoffset_stepper<St, M, I>& lhs,
                    const xoffset_stepper<St, M, I>& rhs)
    {
        return !lhs.equal(rhs);
    }

}
#endif

