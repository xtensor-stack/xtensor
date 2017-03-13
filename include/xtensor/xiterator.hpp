/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XITERATOR_HPP
#define XITERATOR_HPP

#include <iterator>

#include "xutils.hpp"
#include "xexception.hpp"

namespace xt
{

    /***********************
     * broadcast functions *
     ***********************/

    template <class S1, class S2>
    bool broadcast_shape(const S1& input, S2& output);

    template <class S1, class S2>
    bool broadcastable(const S1& s1, S2& s2);

    /***********************
     * iterator meta utils *
     ***********************/

    namespace detail
    {
        template <class C>
        struct get_iterator_impl
        {
            using type = typename C::iterator;
        };

        template <class C>
        struct get_iterator_impl<const C>
        {
            using type = typename C::const_iterator;
        };
    }

    template <class C>
    using get_iterator = typename detail::get_iterator_impl<C>::type;
 
    namespace detail
    {
        template <class ST>
        struct index_type_impl
        {
            using type = std::vector<typename ST::value_type>;
        };

        template <class V, std::size_t L>
        struct index_type_impl<std::array<V, L>>
        {
            using type = std::array<V, L>;
        };
    }

    template <class C>
    using xindex_type_t = typename detail::index_type_impl<C>::type;
 
    /************
     * xstepper *
     ************/

    template <class C>
    class xstepper
    {

    public:

        using container_type = C;
        using subiterator_type = get_iterator<C>;
        using subiterator_traits = std::iterator_traits<subiterator_type>;
        using value_type = typename subiterator_traits::value_type;
        using reference = typename subiterator_traits::reference;
        using pointer = typename subiterator_traits::pointer;
        using difference_type = typename subiterator_traits::difference_type;
        using size_type = typename container_type::size_type;
        using shape_type = typename container_type::shape_type;

        xstepper() = default;
        xstepper(container_type* c, subiterator_type it, size_type offset) noexcept;

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);

        void to_end();

        bool equal(const xstepper& rhs) const;

    private:

        container_type* p_c;
        subiterator_type m_it;
        size_type m_offset;
    };

    template <class C>
    bool operator==(const xstepper<C>& lhs,
                    const xstepper<C>& rhs);

    template <class C>
    bool operator!=(const xstepper<C>& lhs,
                    const xstepper<C>& rhs);

    template <class S, class IT, class ST>
    void increment_stepper(S& stepper,
                           IT& index,
                           const ST& shape);

    /********************
     * xindexed_stepper *
     ********************/

    template <class E, bool is_const = true>
    class xindexed_stepper
    {

    public:

        using self_type = xindexed_stepper<E, is_const>;
        using xexpression_type = std::conditional_t<is_const, const E, E>;

        using value_type = typename xexpression_type::value_type;
        using reference = std::conditional_t<is_const,
                                             typename xexpression_type::const_reference,
                                             typename xexpression_type::reference>;
        using pointer = std::conditional_t<is_const,
                                           typename xexpression_type::const_pointer,
                                           typename xexpression_type::pointer>;
        using size_type = typename xexpression_type::size_type;
        using difference_type = typename xexpression_type::difference_type;

        using shape_type = typename xexpression_type::shape_type;
        using index_type = xindex_type_t<shape_type>;

        xindexed_stepper() = default;
        xindexed_stepper(xexpression_type* e, size_type offset, bool end = false) noexcept;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);

        void to_end();

        reference operator*() const;

        bool equal(const self_type& rhs) const;

    private:

        xexpression_type* p_e;
        index_type m_index;
        size_type m_offset;
    };

    template <class C, bool is_const>
    bool operator==(const xindexed_stepper<C, is_const>& lhs,
                    const xindexed_stepper<C, is_const>& rhs);

    template <class C, bool is_const>
    bool operator!=(const xindexed_stepper<C, is_const>& lhs,
                    const xindexed_stepper<C, is_const>& rhs);

    /*************
     * xiterator *
     *************/

    namespace detail
    {
        template <class S>
        class shape_storage
        {

        public:

            using shape_type = S;
            using param_type = const S&;

            shape_storage() = default;
            shape_storage(param_type shape);
            const S& shape() const;

        private:

            S m_shape;
        };

        template <class S>
        class shape_storage<S*>
        {

        public:

            using shape_type = S;
            using param_type = const S*;

            shape_storage(param_type shape = 0);
            const S& shape() const;

        private:

            const S* p_shape;
        };
    }

    template <class It, class S>
    class xiterator : detail::shape_storage<S>
    {

    public:

        using self_type = xiterator<It, S>;

        using subiterator_type = It;
        using value_type = typename subiterator_type::value_type;
        using reference = typename subiterator_type::reference;
        using pointer = typename subiterator_type::pointer;
        using difference_type = typename subiterator_type::difference_type;
        using size_type = typename subiterator_type::size_type;
        using iterator_category = std::forward_iterator_tag;

        using private_base = detail::shape_storage<S>;
        using shape_type = typename private_base::shape_type;
        using shape_param_type = typename private_base::param_type;
        using index_type = xindex_type_t<shape_type>;

        xiterator() = default;
        xiterator(It it, shape_param_type shape);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;
        pointer operator->() const;

        bool equal(const xiterator& rhs) const;

    private:

        subiterator_type m_it;
        index_type m_index;
    };

    template <class It, class S>
    bool operator==(const xiterator<It, S>& lhs,
                    const xiterator<It, S>& rhs);

    template <class It, class S>
    bool operator!=(const xiterator<It, S>& lhs,
                    const xiterator<It, S>& rhs);

    /*******************
     * xconst_iterable *
     *******************/

    template <class D>
    struct xiterable_inner_types;

    /**
     * @class xconst_iterable
     * @brief Base class for multidimensional iterable constant expressions
     *
     * The xconst_iterable class defines the interface for multidimensional
     * constant expressions that can be iterated.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xconst_iterable
     *           provides the interface.
     */
    template <class D>
    class xconst_iterable
    {

    public:

        using derived_type = D;
        
        using iterable_types = xiterable_inner_types<D>;
        using shape_type = typename iterable_types::shape_type;
        using stepper = typename iterable_types::stepper;
        using const_stepper = typename iterable_types::const_stepper;
        using iterator = typename iterable_types::iterator;
        using const_iterator = typename iterable_types::const_iterator;
        using broadcast_iterator = typename iterable_types::broadcast_iterator;
        using const_broadcast_iterator = typename iterable_types::const_broadcast_iterator;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        const_broadcast_iterator xbegin() const noexcept;
        const_broadcast_iterator xend() const noexcept;
        const_broadcast_iterator cxbegin() const noexcept;
        const_broadcast_iterator cxend() const noexcept;

        template <class S>
        xiterator<const_stepper, S> xbegin(const S& shape) const noexcept;
        template <class S>
        xiterator<const_stepper, S> xend(const S& shape) const noexcept;
        template <class S>
        xiterator<const_stepper, S> cxbegin(const S& shape) const noexcept;
        template <class S>
        xiterator<const_stepper, S> cxend(const S& shape) const noexcept;

    protected:

        const shape_type& get_shape() const;

    private:

        template <class S>
        const_stepper get_stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper get_stepper_end(const S& shape) const noexcept;

        const derived_type& derived_cast() const;
    };

    /*************
     * xiterable *
     *************/

    /**
     * @class xiterable
     * @brief Base class for multidimensional iterable expressions
     *
     * The xconst_iterable class defines the interface for multidimensional
     * expressions that can be iterated.
     *
     * @tparam D The derived type, i.e. the inheriting class for which xiterable
     *           provides the interface.
     */
    template <class D>
    class xiterable : public xconst_iterable<D>
    {

    public:

        using derived_type = D;

        using base_type = xconst_iterable<D>;
        using shape_type = typename base_type::shape_type;
        using stepper = typename base_type::stepper;
        using const_stepper = typename base_type::const_stepper;
        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;
        using broadcast_iterator = typename base_type::broadcast_iterator;
        using const_broadcast_iterator = typename base_type::const_broadcast_iterator;

        iterator begin() noexcept;
        iterator end() noexcept;

        using base_type::begin;
        using base_type::end;

        broadcast_iterator xbegin() noexcept;
        broadcast_iterator xend() noexcept;

        using base_type::xbegin;
        using base_type::xend;

        template <class S>
        xiterator<stepper, S> xbegin(const S& shape) noexcept;
        template <class S>
        xiterator<stepper, S> xend(const S& shape) noexcept;

    private:

        template <class S>
        stepper get_stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper get_stepper_end(const S& shape) noexcept;

        derived_type& derived_cast();
    };

    /**************************************
     * broadcast functions implementation *
     **************************************/

    template <class S1, class S2>
    inline bool broadcast_shape(const S1& input, S2& output)
    {
        bool trivial_broadcast = (input.size() == output.size());
        auto input_iter = input.crbegin();
        auto output_iter = output.rbegin();
        for(;input_iter != input.crend(); ++input_iter, ++output_iter)
        {
            if(*output_iter == 1)
            {
                *output_iter = *input_iter;
            }
            else if((*input_iter != 1) && (*output_iter != *input_iter))
            {
                throw broadcast_error(output, input);
            }
            trivial_broadcast = trivial_broadcast && (*output_iter == *input_iter);
        }
        return trivial_broadcast;
    }

    template <class S1, class S2>
    inline bool broadcastable(const S1& s1, const S2& s2)
    {
        auto iter1 = s1.crbegin();
        auto iter2 = s2.crbegin();
        for(;iter1 != s1.crend() && iter2 != s2.crend(); ++iter1, ++iter2)
        {
            if((*iter2 != 1) && (*iter1 != 1) && (*iter2 != *iter1))
            {
                return false;
            }
        }
        return true;
    }

    /***************************
     * xstepper implementation *
     ***************************/

    template <class C>
    inline xstepper<C>::xstepper(container_type* c, subiterator_type it, size_type offset) noexcept
        : p_c(c), m_it(it), m_offset(offset)
    {
    }

    template <class C>
    inline auto xstepper<C>::operator*() const -> reference
    {
        return *m_it;
    }

    template <class C>
    inline void xstepper<C>::step(size_type dim, size_type n)
    {
        if(dim >= m_offset)
            m_it += n * p_c->strides()[dim - m_offset];
    }

    template <class C>
    inline void xstepper<C>::step_back(size_type dim, size_type n)
    {
        if(dim >= m_offset)
            m_it -= n * p_c->strides()[dim - m_offset];
    }

    template <class C>
    inline void xstepper<C>::reset(size_type dim)
    {
        if(dim >= m_offset)
            m_it -= p_c->backstrides()[dim - m_offset];
    }

    template <class C>
    inline void xstepper<C>::to_end()
    {
        m_it = p_c->end();
    }

    template <class C>
    inline bool xstepper<C>::equal(const xstepper& rhs) const
    {
        return p_c == rhs.p_c && m_it == rhs.m_it && m_offset == rhs.m_offset;
    }

    template <class C>
    inline bool operator==(const xstepper<C>& lhs,
                           const xstepper<C>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class C>
    inline bool operator!=(const xstepper<C>& lhs,
                           const xstepper<C>& rhs)
    {
        return !(lhs.equal(rhs));
    }

    template <class S, class IT, class ST>
    void increment_stepper(S& stepper,
                           IT& index,
                           const ST& shape)
    {
        using size_type = typename S::size_type;
        size_type i = index.size();
        while(i != 0)
        {
            --i;
            if(++index[i] != shape[i])
            {
                stepper.step(i);
                return;
            }
            else if(i != 0)
            {
                index[i] = 0;
                stepper.reset(i);
            }
        }
        if(i == 0)
        {
            stepper.to_end();
        }
    }

    /***********************************
     * xindexed_stepper implementation *
     ***********************************/
    
    template <class C, bool is_const>
    inline xindexed_stepper<C, is_const>::xindexed_stepper(xexpression_type* e, size_type offset, bool end) noexcept
        : p_e(e), m_index(make_sequence<index_type>(e->shape().size(), size_type(0))), m_offset(offset)
    {
        if (end)
            to_end();
    }

    template <class C, bool is_const>
    inline void xindexed_stepper<C, is_const>::step(size_type dim, size_type n)
    {
        if (dim >= m_offset)
            m_index[dim - m_offset] += n;
    }

    template <class C, bool is_const>
    inline void xindexed_stepper<C, is_const>::step_back(size_type dim, size_type n)
    {
        if (dim >= m_offset)
            m_index[dim - m_offset] -= n;
    }

    template <class C, bool is_const>
    inline void xindexed_stepper<C, is_const>::reset(size_type dim)
    {
        if (dim >= m_offset)
            m_index[dim - m_offset] = 0;
    }

    template <class C, bool is_const>
    inline void xindexed_stepper<C, is_const>::to_end()
    {
        m_index = p_e->shape();
    }

    template <class C, bool is_const>
    inline auto xindexed_stepper<C, is_const>::operator*() const -> reference
    {
        return p_e->element(m_index.cbegin(), m_index.cend());
    }

    template <class C, bool is_const>
    inline bool xindexed_stepper<C, is_const>::equal(const self_type& rhs) const
    {
        return p_e == rhs.p_e && m_index == rhs.m_index && m_offset == rhs.m_offset;
    }

    template <class C, bool is_const>
    inline bool operator==(const xindexed_stepper<C, is_const>& lhs,
                           const xindexed_stepper<C, is_const>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class C, bool is_const>
    inline bool operator!=(const xindexed_stepper<C, is_const>& lhs,
                           const xindexed_stepper<C, is_const>& rhs)
    {
        return !lhs.equal(rhs);
    }

    /****************************
     * xiterator implementation *
     ****************************/

    namespace detail
    {
        template <class S>
        inline shape_storage<S>::shape_storage(param_type shape)
            : m_shape(shape)
        {
        }

        template <class S>
        inline const S& shape_storage<S>::shape() const
        {
            return m_shape;
        }

        template <class S>
        inline shape_storage<S*>::shape_storage(param_type shape)
            : p_shape(shape)
        {
        }

        template <class S>
        inline const S& shape_storage<S*>::shape() const
        {
            return *p_shape;
        }
    }

    template <class It, class S>
    inline xiterator<It, S>::xiterator(It it, shape_param_type shape)
        : private_base(shape), m_it(it),
          m_index(make_sequence<index_type>(this->shape().size(), size_type(0)))
    {
    }

    template <class It, class S>
    inline auto xiterator<It, S>::operator++() -> self_type&
    {
        increment_stepper(m_it, m_index, this->shape());
        return *this;
    }

    template <class It, class S>
    inline auto xiterator<It, S>::operator++(int) -> self_type
    {
        self_type tmp(*this);
        ++(*this);
        return tmp;
    }

    template <class It, class S>
    inline auto xiterator<It, S>::operator*() const -> reference
    {
        return *m_it;
    }

    template <class It, class S>
    inline auto xiterator<It, S>::operator->() const -> pointer
    {
        return &(*m_it);
    }

    template <class It, class S>
    inline bool xiterator<It, S>::equal(const xiterator& rhs) const
    {
        return m_it == rhs.m_it && this->shape() == rhs.shape();
    }

    template <class It, class S>
    inline bool operator==(const xiterator<It, S>& lhs,
                           const xiterator<It, S>& rhs)
    {
        return lhs.equal(rhs);
    }

    template <class It, class S>
    inline bool operator!=(const xiterator<It, S>& lhs,
                           const xiterator<It, S>& rhs)
    {
        return !(lhs.equal(rhs));
    }

    /**********************************
     * xconst_iterable implementation *
     **********************************/

     /**
      * @name Constant Iterators
      */
     /**
      * Returns a constant iterator to the first element of the expression.
      */
    template <class D>
    inline auto xconst_iterable<D>::begin() const noexcept -> const_iterator
    {
        return cxbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class D>
    inline auto xconst_iterable<D>::end() const noexcept -> const_iterator
    {
        return cxend();
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     */
    template <class D>
    inline auto xconst_iterable<D>::cbegin() const noexcept -> const_iterator
    {
        return cxbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class D>
    inline auto xconst_iterable<D>::cend() const noexcept -> const_iterator
    {
        return cxend();
    }
    //@}

    /**
     * @name Constant broadcast iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the expression.
     */
    template <class D>
    inline auto xconst_iterable<D>::xbegin() const noexcept -> const_broadcast_iterator
    {
        return cxbegin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class D>
    inline auto xconst_iterable<D>::xend() const noexcept -> const_broadcast_iterator
    {
        return cxend();
    }

    /**
     * Returns a constant iterator to the first element of the expression.
     */
    template <class D>
    inline auto xconst_iterable<D>::cxbegin() const noexcept ->const_broadcast_iterator
    {
        return const_broadcast_iterator(get_stepper_begin(get_shape()), &get_shape());
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the expression.
     */
    template <class D>
    inline auto xconst_iterable<D>::cxend() const noexcept -> const_broadcast_iterator
    {
        return const_broadcast_iterator(get_stepper_end(get_shape()), &get_shape());
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class D>
    template <class S>
    inline auto xconst_iterable<D>::xbegin(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return cxbegin(shape);
    }

    /**
    * Returns a constant iterator to the element following the last element of the
    * expression. The iteration is broadcasted to the specified shape.
    * @param shape the shape used for broadcasting
    */
    template <class D>
    template <class S>
    inline auto xconst_iterable<D>::xend(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return cxend(shape);
    }

    /**
     * Returns a constant iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class D>
    template <class S>
    inline auto xconst_iterable<D>::cxbegin(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xiterator<const_stepper, S>(get_stepper_begin(shape), shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class D>
    template <class S>
    inline auto xconst_iterable<D>::cxend(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xiterator<const_stepper, S>(get_stepper_end(shape), shape);
    }
    //@}

    template <class D>
    template <class S>
    inline auto xconst_iterable<D>::get_stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        return derived_cast().stepper_begin(shape);
    }
    template <class D>
    template <class S>
    inline auto xconst_iterable<D>::get_stepper_end(const S& shape) const noexcept -> const_stepper
    {
        return derived_cast().stepper_end(shape);
    }
    
    template <class D>
    inline auto xconst_iterable<D>::get_shape() const -> const shape_type&
    {
        return derived_cast().shape();
    }

    template <class D>
    inline auto xconst_iterable<D>::derived_cast() const -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    /****************************
     * xiterable implementation *
     ****************************/

    /**
     * @name Iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the expression.
     */
    template <class D>
    inline auto xiterable<D>::begin() noexcept -> iterator
    {
        return xbegin();
    }

    /**
     * Returns an iterator to the element following the last element of
     * the expression.
     */
    template <class D>
    inline auto xiterable<D>::end() noexcept -> iterator
    {
        return xend();
    }
    //@}

    /**
     * @name Broadcast iterators
     */
    //@{
    /**
     * Returns an iterator to the first element of the expression.
     */
    template <class D>
    inline auto xiterable<D>::xbegin() noexcept -> broadcast_iterator
    {
        return broadcast_iterator(get_stepper_begin(this->get_shape()), &(this->get_shape()));
    }

    /**
     * Returns an iterator to the element following the last element
     * of the expression.
     */
    template <class D>
    inline auto xiterable<D>::xend() noexcept -> broadcast_iterator
    {
        return broadcast_iterator(get_stepper_end(this->get_shape()), &(this->get_shape()));
    }

    /**
     * Returns an iterator to the first element of the expression. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class D>
    template <class S>
    inline auto xiterable<D>::xbegin(const S& shape) noexcept -> xiterator<stepper, S>
    {
        return xiterator<stepper, S>(get_stepper_begin(shape), shape);
    }

    /**
     * Returns an iterator to the element following the last element of the
     * expression. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class D>
    template <class S>
    inline auto xiterable<D>::xend(const S& shape) noexcept -> xiterator<stepper, S>
    {
        return xiterator<stepper, S>(get_stepper_end(shape), shape);
    }
    //@}

    template <class D>
    template <class S>
    inline auto xiterable<D>::get_stepper_begin(const S& shape) noexcept -> stepper
    {
        return derived_cast().stepper_begin(shape);
    }

    template <class D>
    template <class S>
    inline auto xiterable<D>::get_stepper_end(const S& shape) noexcept -> stepper
    {
        return derived_cast().stepper_end(shape);
    }

    template <class D>
    inline auto xiterable<D>::derived_cast() -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

}

#endif
