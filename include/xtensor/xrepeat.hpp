/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_XREPEAT
#define XTENSOR_XREPEAT

#include <vector>
#include <utility>

#include "xaccessible.hpp"
#include "xiterable.hpp"


namespace xt
{
    template <class CT, class R> class xrepeat;
    template <class S, class R> class xrepeat_stepper;

    template <class CT, class R>
    struct xcontainer_inner_types<xrepeat<CT, R>>
    {
        using xexpression_type = std::decay_t<CT>;
        using reference = inner_reference_t<CT>;
        using const_reference = typename xexpression_type::const_reference;
        using size_type = typename xexpression_type::size_type;
        using temporary_type = typename xexpression_type::temporary_type;

        static constexpr bool is_const = std::is_const<std::remove_reference_t<CT>>::value;

        using extract_storage_type = xtl::mpl::eval_if_t<has_data_interface<xexpression_type>,
            detail::expr_storage_type<xexpression_type>,
            make_invalid_type<>>;
        using storage_type = std::conditional_t<is_const, const extract_storage_type, extract_storage_type>;
    };

    template <class CT, class R>
    struct xiterable_inner_types<xrepeat<CT, R>>
    {
        using xexpression_type = std::decay_t<CT>;
        using repeats_type = std::decay_t<R>;
        using inner_shape_type = typename xexpression_type::inner_shape_type;
        using stepper = xrepeat_stepper<typename xexpression_type::stepper, repeats_type>;
        using const_stepper = xrepeat_stepper<typename xexpression_type::stepper, repeats_type>;
    };

    template <class S, class R>
    class xrepeat_stepper
    {
    public:
        using repeats_type = R;
        using storage_type = typename S::storage_type;
        using subiterator_type = typename S::subiterator_type;
        using subiterator_traits = typename S::subiterator_traits;
        using value_type = typename subiterator_traits::value_type;
        using reference = typename subiterator_traits::reference;
        using pointer = typename subiterator_traits::pointer;
        using difference_type = typename subiterator_traits::difference_type;
        using size_type = typename storage_type::size_type;
        using shape_type = typename storage_type::shape_type;
        using simd_value_type = xt_simd::simd_type<value_type>;

        template <class requested_type>
        using simd_return_type = xt_simd::simd_return_type<value_type, requested_type>;

        xrepeat_stepper(S&& s, const shape_type& shape, const repeats_type& repeats, const size_type& axis);

        reference operator*() const;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);
        void reset_back(size_type dim);

        void to_begin();
        void to_end(layout_type l);

        template <class T>
        simd_return_type<T> step_simd();

        void step_leading();

        template <class V>
        void store_simd(const V& vec);

    private:
        S m_substepper;
        const shape_type& m_shape;

        std::ptrdiff_t m_repeating_steps;
        std::vector<size_type> m_positions;
        size_type m_subposition;

        const size_type& m_repeating_axis;
        const repeats_type& m_repeats;

        void make_step(size_type dim, size_type n);
        void make_step_back(size_type dim, size_type n);

        std::vector<size_type> get_next_positions(const size_type dim, const size_type steps_to_go) const;
        std::vector<size_type> get_next_positions_back(const size_type dim, const size_type steps_to_go) const;
    };

    template <class CT, class R>
    class xrepeat :
        public xiterable<xrepeat<CT, R>>,
        public xaccessible<xrepeat<CT, R>>
    {
    public:
        using xexpression_type = std::decay_t<CT>;
        using value_type = typename xexpression_type::value_type;
        using shape_type = typename xexpression_type::shape_type;
        using repeats_type = xtl::const_closure_type_t<R>;

        using container_type = xcontainer_inner_types<xrepeat<CT, R>>;
        using reference = typename container_type::reference;
        using const_reference = typename container_type::const_reference;
        using size_type = typename container_type::size_type;
        using temporary_type = typename container_type::temporary_type;

        static constexpr layout_type static_layout = xexpression_type::static_layout;
        using bool_load_type = typename xexpression_type::bool_load_type;
        using pointer = typename xexpression_type::pointer;
        using const_pointer = typename xexpression_type::const_pointer;
        using difference_type = typename xexpression_type::difference_type;

        using iterable_type = xiterable<xrepeat<CT, R>>;
        using stepper = typename iterable_type::stepper;
        using const_stepper = typename iterable_type::stepper;

        template<class CTA>
        explicit xrepeat(CTA&& e, R&& repeats, size_type axis);

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        const shape_type& shape() const noexcept;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        stepper stepper_begin() const;

        stepper stepper_begin(const shape_type& s) const;

        stepper stepper_end(const layout_type l) const;

        stepper stepper_end(const shape_type& s, const layout_type l) const;

    private:
        CT m_e;
        const size_type m_repeating_axis;
        repeats_type m_repeats;
        shape_type m_shape;

        reference access();

        template <class Arg, class... Args>
        reference access(Arg arg, Args... args);

        const_reference access() const;

        template <class Arg, class... Args>
        const_reference access(Arg arg, Args... args) const;

        template<size_type I, class Arg, class... Args>
        inline const_reference access_impl(stepper&& s, Arg arg, Args... args) const
        {
            s.step(I, static_cast<size_type>(arg));
            return access_impl<I+1>(std::forward<stepper>(s), args...);
        }

        template<size_type I>
        inline const_reference access_impl(stepper&& s) const
        {
            return *s;
        }

        template<size_type I, class Arg, class... Args>
        inline reference access_impl(stepper&& s, Arg arg, Args... args)
        {
            s.step(I, static_cast<size_type>(arg));
            return access_impl<I+1>(std::forward<stepper>(s), args...);
        }

        template<size_type I>
        inline reference access_impl(stepper&& s)
        {
            return *s;
        }
    };

    template <class CT, class R>
    template <class CTA>
    xrepeat<CT, R>::xrepeat(CTA&& e, R&& repeats, size_type axis)
        : m_e(std::forward<CTA>(e))
        , m_repeating_axis(axis)
        , m_repeats(std::forward<R>(repeats))
        , m_shape(e.shape())
    {
        using value_type = typename shape_type::value_type;
        m_shape[axis] = static_cast<value_type>(std::accumulate(m_repeats.begin(), m_repeats.end(), 0));
    }

    template <class CT, class R>
    template <class... Args>
    inline auto xrepeat<CT, R>::operator()(Args... args) -> reference
    {
        return access(args...);
    }

    template <class CT, class R>
    template <class... Args>
    inline auto xrepeat<CT, R>::operator()(Args... args) const -> const_reference
    {
        return access(args...);
    }

    template <class CT, class R>
    inline auto xrepeat<CT, R>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    /**
    * Returns a reference to the element at the specified position in the view.
    * @param first iterator starting the sequence of indices
    * @param last iterator ending the sequence of indices
    * The number of indices in the sequence should be equal to or greater than the the number
    * of dimensions of the view..
    */
    template <class CT, class R>
    template <class It>
    inline auto xrepeat<CT, R>::element(It first, It last) -> reference
    {
        auto stepper = stepper_begin();
        auto dimension = 0;
        auto iter = first;
        while (iter != last)
        {
            stepper.step(dimension, *iter);
            ++dimension;
            ++first;
        }
        return access_impl<0>(stepper);
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater than the the number
     * of dimensions of the view..
     */
    template <class CT, class R>
    template <class It>
    inline auto xrepeat<CT, R>::element(It first, It last) const -> const_reference
    {
        auto s = stepper_begin(m_e.shape());
        auto dimension = 0;
        auto iter = first;
        while (iter != last)
        {
            s.step(dimension, *iter);
            ++dimension;
            ++iter;
        }
        return access_impl<0>(std::forward<stepper>(s));
    }

    template <class CT, class R>
    inline auto xrepeat<CT, R>::access() -> reference
    {
        return access_impl<0>(stepper_begin(m_e.shape()));
    }

    template <class CT, class R>
    template <class Arg, class... Args>
    inline auto xrepeat<CT, R>::access(Arg arg, Args... args) -> reference
    {
        constexpr size_t number_of_arguments = 1 + sizeof...(Args);
        if (number_of_arguments > this->dimension())
        {
            return access(args...);
        }
        return access_impl<0>(stepper_begin(m_e.shape()), arg, args...);
    }

    template <class CT, class R>
    inline auto xrepeat<CT, R>::access() const -> const_reference
    {
        return access_impl<0>(stepper_begin(m_e.shape()));
    }

    template <class CT, class R>
    template <class Arg, class... Args>
    inline auto xrepeat<CT, R>::access(Arg arg, Args... args) const -> const_reference
    {
        constexpr size_t number_of_arguments = 1 + sizeof...(Args);
        if (number_of_arguments > this->dimension())
        {
            return access(args...);
        }
        return access_impl<0>(stepper_begin(m_e.shape()), arg, args...);
    }

    template <class CT, class R>
    inline auto xrepeat<CT, R>::stepper_begin() const -> stepper
    {
        return stepper_begin(m_e.shape());
    }

    template <class CT, class R>
    inline auto xrepeat<CT, R>::stepper_begin(const shape_type& s) const -> stepper
    {
        return stepper(m_e.stepper_begin(s), m_shape, m_repeats, m_repeating_axis);
    }

    template <class CT, class R>
    inline auto xrepeat<CT, R>::stepper_end(const layout_type l) const -> stepper
    {
        return stepper_end(m_e.shape(), l);
    }

    template <class CT, class R>
    inline auto xrepeat<CT, R>::stepper_end(const shape_type& s, const layout_type l) const -> stepper
    {
        auto st = stepper(m_e.stepper_begin(s), m_shape, m_repeats, m_repeating_axis);
        st.to_end(l);
        return st;
    }

    template<class S, class R>
    xrepeat_stepper<S, R>::xrepeat_stepper(S&& s, const shape_type& shape, const repeats_type& repeats, const size_type& axis)
        : m_substepper(std::forward<S>(s))
        , m_shape(shape)
        , m_repeating_steps(0)
        , m_positions(shape.size())
        , m_subposition(0)
        , m_repeating_axis(axis)
        , m_repeats(repeats)
    {}

    template<class S, class R>
    inline auto xrepeat_stepper<S, R>::operator*() const -> reference
    {
        return m_substepper.operator*();
    }

    template<class S, class R>
    inline void xrepeat_stepper<S, R>::step(size_type dim, size_type steps_to_go)
    {
        if (m_positions[dim] + steps_to_go >= m_shape[dim])
        {
            const auto next_positions = get_next_positions(dim, steps_to_go);
            if (next_positions[dim] > m_positions[dim])
            {
                make_step(dim, next_positions[dim] - m_positions[dim]);
            }
            else
            {
                make_step_back(dim, m_positions[dim] - next_positions[dim]);
            }
            for (size_type d = 0; d < dim; ++d)
            {
                make_step(d, next_positions[d] - m_positions[d]);
            }
        }
        else
        {
            make_step(dim, steps_to_go);
        }
    }

    template<class S, class R>
    inline void xrepeat_stepper<S, R>::step_back(size_type dim, size_type steps_to_go)
    {
        if (m_positions[dim] < steps_to_go)
        {
            const auto next_positions = get_next_positions_back(dim, steps_to_go);
            if (next_positions[dim] < m_positions[dim])
            {
                make_step_back(dim, m_positions[dim] - next_positions[dim]);
            }
            else
            {
                make_step(dim, next_positions[dim] - m_positions[dim]);
            }
            for (size_type d = 0; d < dim; ++d)
            {
                make_step_back(d, m_positions[d] - next_positions[d]);
            }
        }
        else
        {
            make_step_back(dim, steps_to_go);
        }
    }

    template<class S, class R>
    inline void xrepeat_stepper<S, R>::reset(size_type dim)
    {
        m_substepper.reset(dim);
        m_positions[dim] = 0;
        if (dim == m_repeating_axis)
        {
            m_subposition = 0;
            m_repeating_steps = 0;
        }
    }

    template<class S, class R>
    inline void xrepeat_stepper<S, R>::reset_back(size_type dim)
    {
        m_substepper.reset_back(dim);
        m_positions[dim] = m_shape[dim] - 1;
        if (dim == m_repeating_axis)
        {
            m_subposition = m_repeats.size() - 1;
            m_repeating_steps = static_cast<std::ptrdiff_t>(m_repeats.back()) - 1;
        }
    }

    template<class S, class R>
    inline void xrepeat_stepper<S, R>::to_begin()
    {
        m_substepper.to_begin();
        std::fill(m_positions.begin(), m_positions.end(), 0);
        m_subposition = 0;
        m_repeating_steps = 0;
    }

    template<class S, class R>
    inline void xrepeat_stepper<S, R>::to_end(layout_type l)
    {
        m_substepper.to_end(l);
        std::transform(m_shape.begin(), m_shape.end(), m_positions.begin(), [](auto value) {
            return value - 1;
        });
        if (layout_type::row_major == l)
        {
            ++m_positions.front();
        }
        else
        {
            ++m_positions.back();
        }
        m_subposition = m_repeats.size();
        m_repeating_steps = 0;
    }

    template<class S, class R>
    inline void xrepeat_stepper<S, R>::step_leading()
    {
        step(m_shape.size() - 1, 1);
    }

    template<class S, class R>
    inline void xrepeat_stepper<S, R>::make_step(size_type dim, size_type steps_to_go)
    {
        if (steps_to_go > 0)
        {
            if (dim == m_repeating_axis)
            {
                size_type subposition = m_subposition;
                m_repeating_steps += static_cast<std::ptrdiff_t>(steps_to_go);
                while (m_repeating_steps >= static_cast<ptrdiff_t>(m_repeats[subposition]))
                {
                    m_repeating_steps -= static_cast<ptrdiff_t>(m_repeats[subposition]);
                    ++subposition;
                }
                m_substepper.step(dim, subposition - m_subposition);
                m_subposition = subposition;
            }
            else
            {
                m_substepper.step(dim, steps_to_go);
            }
            m_positions[dim] += steps_to_go;
        }
    }

    template<class S, class R>
    inline void xrepeat_stepper<S, R>::make_step_back(size_type dim, size_type steps_to_go)
    {
        if (steps_to_go > 0)
        {
            if (dim == m_repeating_axis)
            {
                size_type subposition = m_subposition;
                m_repeating_steps -= static_cast<std::ptrdiff_t>(steps_to_go);
                while (m_repeating_steps < 0)
                {
                    --subposition;
                    m_repeating_steps += static_cast<ptrdiff_t>(m_repeats[subposition]);
                }
                m_substepper.step_back(dim, m_subposition - subposition);
                m_subposition = subposition;
            }
            else
            {
                m_substepper.step_back(dim, steps_to_go);
            }
            m_positions[dim] -= steps_to_go;
        }
    }

    template<class S, class R>
    inline auto xrepeat_stepper<S, R>::get_next_positions(const size_type dim, const size_type steps_to_go) const -> std::vector<size_type>
    {
        size_type next_position_for_dim = m_positions[dim] + steps_to_go;
        if (dim > 0)
        {
            size_type steps_in_previous_dim = 0;
            while (next_position_for_dim >= m_shape[dim])
            {
                next_position_for_dim -= m_shape[dim];
                ++steps_in_previous_dim;
            }
            if (steps_in_previous_dim > 0)
            {
                auto next_positions = get_next_positions(dim - 1, steps_in_previous_dim);
                next_positions[dim] = next_position_for_dim;
                return next_positions;
            }
        }
        std::vector<size_type> next_positions = m_positions;
        next_positions[dim] = next_position_for_dim;
        return next_positions;
    }

    template<class S, class R>
    inline auto xrepeat_stepper<S, R>::get_next_positions_back(const size_type dim, const size_type steps_to_go) const -> std::vector<size_type>
    {
        auto next_position_for_dim = static_cast<std::ptrdiff_t>(m_positions[dim] - steps_to_go);
        if (dim > 0)
        {
            size_type steps_in_previous_dim = 0;
            while (next_position_for_dim < 0)
            {
                next_position_for_dim += static_cast<std::ptrdiff_t>(m_shape[dim]);
                ++steps_in_previous_dim;
            }
            if (steps_in_previous_dim > 0)
            {
                auto next_positions = get_next_positions_back(dim - 1, steps_in_previous_dim);
                next_positions[dim] = static_cast<size_type>(next_position_for_dim);
                return next_positions;
            }
        }
        std::vector<size_type> next_positions = m_positions;
        next_positions[dim] = static_cast<size_type>(next_position_for_dim);
        return next_positions;
    }
}

#endif
