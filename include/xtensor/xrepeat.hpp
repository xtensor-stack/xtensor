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
    template <class CT> class xrepeat;
    template <class S> class xrepeat_stepper;

    template <class CT>
    struct xcontainer_inner_types<xrepeat<CT>>
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

    template <class CT>
    struct xiterable_inner_types<xrepeat<CT>>
    {
        using xexpression_type = std::decay_t<CT>;
        using inner_shape_type = typename xexpression_type::inner_shape_type;
        using stepper = xrepeat_stepper<typename xexpression_type::stepper>;
        using const_stepper = xrepeat_stepper<typename xexpression_type::stepper>;
    };

    template <class S>
    class xrepeat_stepper
    {
    public:
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

        xrepeat_stepper(S&& s, const std::vector<std::ptrdiff_t>& repeats);

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

        template <class R>
        void store_simd(const R& vec);

        struct repeated_index
        {
            repeated_index(const std::ptrdiff_t repeats)
                : m_repeats(repeats)
                , m_index(0)
            {}

            size_type step(const size_type steps)
            {
                m_index += steps;
                size_type inner_steps{ 0 };
                while (m_index >= m_repeats)
                {
                    ++inner_steps;
                    m_index -= m_repeats;
                }
                return inner_steps;
            }

            size_type step_back(const size_type steps)
            {
                m_index = m_index - steps;
                size_type inner_steps{ 0 };
                while (m_index < 0)
                {
                    ++inner_steps;
                    m_index += m_repeats;
                }
                return inner_steps;
            }

            void reset()
            {
                m_index = 0;
            }

            void reset_back()
            {
                m_index = m_repeats - 1;
            }

        private:
            const std::ptrdiff_t m_repeats;
            std::ptrdiff_t m_index;
        };

    private:
        S m_substepper;
        std::vector<repeated_index> m_indicies;
    };

    template <class CT>
    class xrepeat :
        public xiterable<xrepeat<CT>>,
        public xaccessible<xrepeat<CT>>
    {
    public:
        using xexpression_type = std::decay_t<CT>;
        using value_type = typename xexpression_type::value_type;
        using shape_type = typename xexpression_type::shape_type;

        using container_type = xcontainer_inner_types<xrepeat<CT>>;
        using reference = typename container_type::reference;
        using const_reference = typename container_type::const_reference;
        using size_type = typename container_type::size_type;
        using temporary_type = typename container_type::temporary_type;

        static constexpr layout_type static_layout = xexpression_type::static_layout;
        using bool_load_type = typename xexpression_type::bool_load_type;
        using pointer = typename xexpression_type::pointer;
        using const_pointer = typename xexpression_type::const_pointer;
        using difference_type = typename xexpression_type::difference_type;

        using iterable_type = xiterable<xrepeat<CT>>;
        using stepper = typename iterable_type::stepper;
        using const_stepper = typename iterable_type::stepper;

        template<class CTA>
        explicit xrepeat(CTA&& e,
            std::vector<std::ptrdiff_t>&& repeats,
            std::vector<std::ptrdiff_t>&& axes);

        template <class... Args>
        reference operator()(Args... args);

        template <class... Args>
        const_reference operator()(Args... args) const;

        const shape_type& shape() const noexcept;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        stepper stepper_begin(const shape_type& s) const;

        stepper stepper_end(const shape_type& s, const layout_type l) const;

    private:
        CT m_e;
        std::vector<std::ptrdiff_t> m_repeat_lookup;
        shape_type m_shape;

        reference access();

        template <class Arg, class... Args>
        reference access(Arg arg, Args... args);

        const_reference access() const;

        template <class Arg, class... Args>
        const_reference access(Arg arg, Args... args) const;

        template<typename std::decay_t<CT>::size_type I>
        struct access_impl
        {

            template<class Arg, class... Args>
            inline const_reference operator()(stepper& s, Arg arg, Args... args) const
            {
                s.step(I, arg);
                const access_impl<I + 1> access_next_dimension;
                return access_next_dimension(s, args...);
            }

            inline const_reference operator()(stepper& s) const
            {
                return *s;
            }

            template<class Arg, class... Args>
            inline reference operator()(stepper& s, Arg arg, Args... args)
            {
                s.step(I, arg);
                access_impl<I + 1> access_next_dimension;
                return access_next_dimension(s, args...);
            }

            inline reference operator()(stepper& s)
            {
                return *s;
            }
        };

        access_impl<0> m_access_impl;
    };

    template <class CT>
    template <class CTA>
    xrepeat<CT>::xrepeat(CTA&& e, std::vector<std::ptrdiff_t>&& repeats, std::vector<std::ptrdiff_t>&& axes)
        : m_e(std::forward<CTA>(e))
        , m_repeat_lookup(e.dimension())
        , m_shape(e.shape())
    {
        std::fill(m_repeat_lookup.begin(), m_repeat_lookup.end(), 1);
        for (auto index = 0; index < axes.size(); ++index)
        {
            const auto axis = axes.at(index);
            const auto number_of_repeats = repeats.at(index);
            m_repeat_lookup.at(axis) = number_of_repeats;
            m_shape.at(axis) *= number_of_repeats;
        }
    }

    template <class CT>
    template <class... Args>
    inline auto xrepeat<CT>::operator()(Args... args) -> reference
    {
        return access(args...);
    }

    template <class CT>
    template <class... Args>
    inline auto xrepeat<CT>::operator()(Args... args) const -> const_reference
    {
        return access(args...);
    }

    template <class CT>
    inline auto xrepeat<CT>::shape() const noexcept -> const shape_type&
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
    template <class ct>
    template <class It>
    inline auto xrepeat<ct>::element(It first, It last) -> reference
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
        return m_access_impl(stepper);
    }

    /**
     * Returns a constant reference to the element at the specified position in the view.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater than the the number
     * of dimensions of the view..
     */
    template <class CT>
    template <class It>
    inline auto xrepeat<CT>::element(It first, It last) const -> const_reference
    {
        auto stepper = stepper_begin(m_e.shape());
        auto dimension = 0;
        It iter = first;
        while (iter != last)
        {
            stepper.step(dimension, *iter);
            ++dimension;
            ++iter;
        }
        return m_access_impl(stepper);
    }

    template <class CT>
    inline auto xrepeat<CT>::access() -> reference
    {
        return m_access_impl(stepper_begin(m_e.shape()));
    }

    template <class CT>
    template <class Arg, class... Args>
    inline auto xrepeat<CT>::access(Arg arg, Args... args) -> reference
    {
        constexpr size_t number_of_arguments = 1 + sizeof...(Args);
        if (number_of_arguments > this->dimension())
        {
            return access(args...);
        }
        return m_access_impl(stepper_begin(m_e.shape()), arg, args...);
    }

    template <class CT>
    inline auto xrepeat<CT>::access() const -> const_reference
    {
        return m_access_impl(stepper_begin(m_e.shape()));
    }

    template <class CT>
    template <class Arg, class... Args>
    inline auto xrepeat<CT>::access(Arg arg, Args... args) const -> const_reference
    {
        constexpr size_t number_of_arguments = 1 + sizeof...(Args);
        if (number_of_arguments > this->dimension())
        {
            return access(args...);
        }
        return m_access_impl(stepper_begin(m_e.shape()), arg, args...);
    }

    template <class CT>
    inline auto xrepeat<CT>::stepper_begin(const shape_type& s) const -> stepper
    {
        return stepper(m_e.stepper_begin(s), m_repeat_lookup);
    }

    template <class CT>
    inline auto xrepeat<CT>::stepper_end(const shape_type& s, const layout_type l) const -> stepper
    {
        auto st = stepper(m_e.stepper_begin(s), m_repeat_lookup);
        st.to_end(l);
        return st;
    }

    template<class S>
    xrepeat_stepper<S>::xrepeat_stepper(S&& s, const std::vector<std::ptrdiff_t>& repeats)
        : m_substepper(std::forward<S>(s))
    {
        for (auto number_of_repeat : repeats)
        {
            m_indicies.emplace_back(number_of_repeat);
        }
    }

    template<class S>
    inline auto xrepeat_stepper<S>::operator*() const -> reference
    {
        return m_substepper.operator*();
    }

    template<class S>
    inline void xrepeat_stepper<S>::step(size_type dim, size_type n)
    {
        size_type substeps = m_indicies.at(dim).step(n);
        m_substepper.step(dim, substeps);
    }

    template<class S>
    inline void xrepeat_stepper<S>::step_back(size_type dim, size_type n)
    {
        size_type substeps = m_indicies.at(dim).step_back(n);
        m_substepper.step_back(dim, substeps);
    }

    template<class S>
    inline void xrepeat_stepper<S>::reset(size_type dim)
    {
        m_substepper.reset(dim);
        m_indicies.at(dim).reset();
    }

    template<class S>
    inline void xrepeat_stepper<S>::reset_back(size_type dim)
    {
        m_substepper.reset_back(dim);
        m_indicies.at(dim).reset_back();
    }

    template<class S>
    inline void xrepeat_stepper<S>::to_begin()
    {
        m_substepper.to_begin();
        for (auto index : m_indicies)
        {
            index.reset();
        }
    }

    template<class S>
    inline void xrepeat_stepper<S>::to_end(layout_type l)
    {
        m_substepper.to_end(l);
        for (auto index : m_indicies)
        {
            index.reset_back();
        }
    }

    template<class S>
    inline void xrepeat_stepper<S>::step_leading()
    {
        m_substepper.step_leading();
    }
}

#endif