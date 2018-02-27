/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_GENERATOR_HPP
#define XTENSOR_GENERATOR_HPP

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

#include "xtl/xsequence.hpp"

#include "xexpression.hpp"
#include "xiterable.hpp"
#include "xstrides.hpp"
#include "xutils.hpp"

namespace xt
{
    namespace detail
    {
        /**
         * Generator Functor Base implementation
         * Stores a shape and the required functions
         */
        template <class S>
        class generator_functor_base 
        {
        public:
            using shape_type = S;

            template <class IS>
            generator_functor_base(IS&& s)
                : m_shape(std::forward<IS>(s))
            {
            }

            inline const S& shape() const
            {
                return m_shape;
            }

            inline std::size_t dimension() const
            {
                return m_shape.size();
            }

            template <class OS>
            inline bool broadcast_shape(OS& shape) const
            {
                return xt::broadcast_shape(m_shape, shape);
            }

        private:
            S m_shape;
        };
    }

    /**************
     * xgenerator *
     **************/

    template <class F, class R>
    class xgenerator;

    template <class C, class R>
    struct xiterable_inner_types<xgenerator<C, R>>
    {
        using inner_shape_type = typename std::decay_t<C>::shape_type;
        using const_stepper = xindexed_stepper<xgenerator<C, R>>;
        using stepper = const_stepper;
    };

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
    template <class F, class R>
    class xgenerator : public xexpression<xgenerator<F, R>>,
                       public xconst_iterable<xgenerator<F, R>>
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

        using iterable_base = xconst_iterable<self_type>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;
        using strides_type = shape_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        static constexpr layout_type static_layout = layout_type::any;
        static constexpr bool contiguous_layout = false;

        xgenerator(F&& f) noexcept;

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const inner_shape_type& shape() const noexcept;
        layout_type layout() const noexcept;

        template <class... Args>
        const_reference operator()(Args... args) const;
        template <class... Args>
        const_reference at(Args... args) const;
        template <class OS>
        disable_integral_t<OS, const_reference> operator[](const OS& index) const;
        template <class I>
        const_reference operator[](std::initializer_list<I> index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class O>
        bool broadcast_shape(O& shape) const;

        template <class O>
        bool is_trivial_broadcast(const O& /*strides*/) const noexcept;

        template <class O>
        const_stepper stepper_begin(const O& shape) const noexcept;
        template <class O>
        const_stepper stepper_end(const O& shape, layout_type) const noexcept;

    private:

        functor_type m_f;
    };

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
    template <class F, class R>
    inline xgenerator<F, R>::xgenerator(F&& f) noexcept
        : m_f(std::forward<F>(f))
    {
    }
    //@}

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the size of the expression.
     */
    template <class F, class R>
    inline auto xgenerator<F, R>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    /**
     * Returns the number of dimensions of the function.
     */
    template <class F, class R>
    inline auto xgenerator<F, R>::dimension() const noexcept -> size_type
    {
        return m_f.dimension();
        // return m_shape.size();
    }

    /**
     * Returns the shape of the xgenerator.
     */
    template <class F, class R>
    inline auto xgenerator<F, R>::shape() const noexcept -> const inner_shape_type&
    {
        return m_f.shape();
        // return m_shape;
    }

    template <class F, class R>
    inline layout_type xgenerator<F, R>::layout() const noexcept
    {
        return static_layout;
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
        XTENSOR_TRY(check_index(shape(), args...));
        return m_f(args...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the expression.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class F, class R>
    template <class... Args>
    inline auto xgenerator<F, R>::at(Args... args) const -> const_reference
    {
        check_access(shape(), args...);
        return this->operator()(args...);
    }

    template <class F, class R>
    template <class OS>
    inline auto xgenerator<F, R>::operator[](const OS& index) const
        -> disable_integral_t<OS, const_reference>
    {
        return element(index.cbegin(), index.cend());
    }

    template <class F, class R>
    template <class I>
    inline auto xgenerator<F, R>::operator[](std::initializer_list<I> index) const
        -> const_reference
    {
        return element(index.begin(), index.end());
    }

    template <class F, class R>
    inline auto xgenerator<F, R>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a constant reference to the element at the specified position in the function.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the container.
     */
    template <class F, class R>
    template <class It>
    inline auto xgenerator<F, R>::element(It first, It last) const -> const_reference
    {
        XTENSOR_TRY(check_element_index(shape(), first, last));
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
    template <class O>
    inline bool xgenerator<F, R>::broadcast_shape(O& shape) const
    {
        return m_f.broadcast_shape(shape);
        // return xt::broadcast_shape(m_shape, shape);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class R>
    template <class O>
    inline bool xgenerator<F, R>::is_trivial_broadcast(const O& /*strides*/) const noexcept
    {
        return false;
    }
    //@}

    template <class F, class R>
    template <class O>
    inline auto xgenerator<F, R>::stepper_begin(const O& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset);
    }

    template <class F, class R>
    template <class O>
    inline auto xgenerator<F, R>::stepper_end(const O& shape, layout_type) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - dimension();
        return const_stepper(this, offset, true);
    }

    namespace detail
    {
        template <class Functor>
        inline auto make_xgenerator(Functor&& f) noexcept
        {
            using type = xgenerator<Functor, typename Functor::value_type>;
            return type(std::forward<Functor>(f));
        }
    }
}

#endif
