/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_ASSIGN_HPP
#define XTENSOR_ASSIGN_HPP

#include <algorithm>
#include <functional>
#include <type_traits>
#include <utility>

#include <xtl/xcomplex.hpp>
#include <xtl/xsequence.hpp>

#include "../core/xassign_proxy.hpp"
#include "../core/xassigner.hpp"
#include "../core/xexpression.hpp"
#include "../core/xfunction.hpp"
#include "../core/xiterator.hpp"
#include "../core/xstrides.hpp"
#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../utils/xutils.hpp"

namespace xt
{

    /********************
     * Assign functions *
     ********************/

    template <class E1, class E2, class P>
    void computed_assign(xexpression<E1>& e1, const xexpression<E2>& e2, P p);

    template <class E1, class E2, class F, class P>
    void scalar_computed_assign(xexpression<E1>& e1, const E2& e2, F&& f, P p);

    template <class E1, class E2, class P>
    void assign_to_container(xexpression<E1>& e1, const xexpression<E2>& e2, P p);

    template <class E1, class E2, class P>
    void assign_to_view(xexpression<E1>& e1, const xexpression<E2>& e2, P p);

    /************************
     * xexpression_assigner *
     ************************/

    template <class Tag>
    class xexpression_assigner_base;

    template <>
    class xexpression_assigner_base<xtensor_expression_tag>
    {
    public:

        template <class E1, class E2, class P>
        static void assign_data(xexpression<E1>& e1, const xexpression<E2>& e2, bool trivial, P p);
    };

    template <class Tag>
    class xexpression_assigner : public xexpression_assigner_base<Tag>
    {
    public:

        using base_type = xexpression_assigner_base<Tag>;

        template <class E1, class E2, class P>
        static void computed_assign(xexpression<E1>& e1, const xexpression<E2>& e2, P p);

        template <class E1, class E2, class F, class P>
        static void scalar_computed_assign(xexpression<E1>& e1, const E2& e2, F&& f, P p);

        template <class E1, class E2, class P>
        static void assign_to_container(xexpression<E1>& e1, const xexpression<E2>& e2, P p);

        template <class E1, class E2, class P>
        static void assign_to_view(xexpression<E1>& e1, const xexpression<E2>& e2, P p);

    private:

        template <class S, class E>
        static S get_nobroadcast_shape(const E& e);

        template <class S, class F, class... CT>
        static S get_nobroadcast_shape(const xfunction<F, CT...>& f);

        template <class S, size_t I, class... CT>
        static S get_nobroadcast_shape(const std::tuple<CT...>& args);

        template <class E1, class E2, class P>
        static bool resize(E1& e1, const E2& e2, P);

        template <class E1, class F, class... CT, class P>
        static bool resize(E1& e1, const xfunction<F, CT...>& e2, P);

        template <class E1, class E2, class P>
        static void assert_compatible_shape(const xexpression<E1>& e1, const xexpression<E2>& e2, P);
    };

    /***********************************
     * Assign functions implementation *
     ***********************************/

    template <class E1, class E2, class P>
    inline void computed_assign(xexpression<E1>& e1, const xexpression<E2>& e2, P p)
    {
        using tag = xexpression_tag_t<E1, E2>;
        xexpression_assigner<tag>::computed_assign(e1, e2, p);
    }

    template <class E1, class E2, class F, class P>
    inline void scalar_computed_assign(xexpression<E1>& e1, const E2& e2, F&& f, P p)
    {
        using tag = xexpression_tag_t<E1, E2>;
        xexpression_assigner<tag>::scalar_computed_assign(e1, e2, std::forward<F>(f), p);
    }

    template <class E1, class E2, class P>
    inline void assign_to_container(xexpression<E1>& e1, const xexpression<E2>& e2, P p)
    {
        if constexpr (assignable_to_expression<E1, E2>)
        {
            e2.derived_cast().assign_to(e1);
        }
        else
        {
            using tag = xexpression_tag_t<E1, E2>;
            xexpression_assigner<tag>::assign_to_container(e1, e2, p);
        }
    }

    template <class E1, class E2, class P>
    inline void assign_to_view(xexpression<E1>& e1, const xexpression<E2>& e2, P p)
    {
        using tag = xexpression_tag_t<E1, E2>;
        xexpression_assigner<tag>::assign_to_view(e1, e2, p);
    }

    /***************************************
     * xexpression_assigner implementation *
     ***************************************/

    namespace detail
    {
        template <class E>
        inline bool is_dense_storage(const E& e)
        {
            if constexpr (strided_expression<E>)
            {
                const auto& shape = e.shape();
                const auto& strides = e.strides();
                using size_type = typename E::size_type;
                size_type remaining = e.size();
                size_type stride = 1;
                // Custom strides can cover dense storage in any axis order.
                while (remaining > 1)
                {
                    bool found = false;
                    for (std::size_t axis = 0; axis < shape.size(); ++axis)
                    {
                        if (shape[axis] > 1 && std::cmp_equal(strides[axis], stride))
                        {
                            remaining /= shape[axis];
                            stride *= shape[axis];
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        return false;
                    }
                }
                return true;
            }
            else
            {
                return false;
            }
        }

        template <class E1, class E2>
        constexpr bool linear_static_layout()
        {
            // A row_major or column_major container with a dimension <= 1 is computed as
            // layout any, leading to some performance improvements, for example when
            // assigning a col-major vector to a row-major vector etc
            return compute_layout(
                       select_layout<E1::static_layout, typename E1::shape_type>::value,
                       select_layout<E2::static_layout, typename E2::shape_type>::value
                   )
                   != layout_type::dynamic;
        }

        template <class E1, class E2>
        inline auto is_linear_assign(const E1& e1, const E2& e2)
            -> std::enable_if_t<strided_expression<E1>, bool>
        {
            return (E1::contiguous_layout && E2::contiguous_layout && linear_static_layout<E1, E2>())
                   || (e1.is_contiguous() && e2.has_linear_assign(e1.strides()));
        }

        template <class E1, class E2>
        inline auto is_linear_assign(const E1&, const E2&) -> std::enable_if_t<!strided_expression<E1>, bool>
        {
            return false;
        }

        template <class E1, class E2>
        inline bool linear_dynamic_layout(const E1& e1, const E2& e2)
        {
            return e1.is_contiguous() && e2.is_contiguous()
                   && compute_layout(e1.layout(), e2.layout()) != layout_type::dynamic;
        }

        template <class E, class = void>
        struct has_step_leading : std::false_type
        {
        };

        template <class E>
        struct has_step_leading<E, void_t<decltype(std::declval<E>().step_leading())>> : std::true_type
        {
        };

        template <class T>
        struct use_strided_loop
        {
            static constexpr bool stepper_deref()
            {
                return std::is_reference<typename T::stepper::reference>::value;
            }

            static constexpr bool value = strided_expression<T>
                                          && has_step_leading<typename T::stepper>::value && stepper_deref();
        };

        template <class T>
        struct use_strided_loop<xscalar<T>>
        {
            static constexpr bool value = true;
        };

        template <class F, class... CT>
        struct use_strided_loop<xfunction<F, CT...>>
        {
            static constexpr bool value = std::conjunction<use_strided_loop<std::decay_t<CT>>...>::value;
        };
    }

    template <class E1, class E2>
    class xassign_traits
    {
    private:

        using e1_value_type = typename E1::value_type;
        using e2_value_type = typename E2::value_type;

        template <class T>
        using is_bool = std::is_same<T, bool>;

        static constexpr bool is_bool_conversion()
        {
            return is_bool<e2_value_type>::value && !is_bool<e1_value_type>::value;
        }

        static constexpr bool contiguous_layout()
        {
            return E1::contiguous_layout && E2::contiguous_layout;
        }

        static constexpr bool convertible_types()
        {
            return std::is_convertible<e2_value_type, e1_value_type>::value && !is_bool_conversion();
        }

        static constexpr bool use_xsimd()
        {
            return xt_simd::simd_traits<int8_t>::size > 1;
        }

        template <class T>
        static constexpr bool simd_size_impl()
        {
            return xt_simd::simd_traits<T>::size > 1 || (is_bool<T>::value && use_xsimd());
        }

        static constexpr bool simd_size()
        {
            return simd_size_impl<e1_value_type>() && simd_size_impl<e2_value_type>();
        }

        static constexpr bool simd_interface()
        {
            using rvt = requested_value_type_t<E1, E2>;
            return has_simd_interface<E1, rvt>() && has_simd_interface<E2, rvt>();
        }

    public:

        // constexpr methods instead of constexpr data members avoid the need of definitions at namespace
        // scope of these data members (since they are odr-used).

        static constexpr bool simd_assign()
        {
            return convertible_types() && simd_size() && simd_interface();
        }

        static constexpr bool linear_assign(const E1& e1, const E2& e2, bool trivial)
        {
            return trivial && detail::is_linear_assign(e1, e2);
        }

        static constexpr bool strided_assign()
        {
            return detail::use_strided_loop<E1>::value && detail::use_strided_loop<E2>::value;
        }

        static constexpr bool simd_linear_assign()
        {
            return contiguous_layout() && simd_assign();
        }

        static constexpr bool simd_strided_assign()
        {
            return strided_assign() && simd_assign();
        }

        static constexpr bool simd_linear_assign(const E1& e1, const E2& e2)
        {
            return simd_assign() && detail::linear_dynamic_layout(e1, e2);
        }
    };

    template <class E1, class E2, class P>
    inline void xexpression_assigner_base<xtensor_expression_tag>::assign_data(
        xexpression<E1>& e1,
        const xexpression<E2>& e2,
        bool trivial,
        P
    )
    {
        E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();
        using traits = xassign_traits<E1, E2>;

        bool linear_assign = traits::linear_assign(de1, de2, trivial);
        constexpr bool simd_assign = traits::simd_assign();
        constexpr bool simd_linear_assign = traits::simd_linear_assign();
        constexpr bool simd_strided_assign = traits::simd_strided_assign();

        parallel_assigner_t<E1, E2, typename P::parallelism> assigner;

        if (linear_assign)
        {
            if (simd_linear_assign || traits::simd_linear_assign(de1, de2))
            {
                // Do not use linear_assigner<true> here since it will make the compiler
                // instantiate this branch even if the runtime condition is false, resulting
                // in compilation error for expressions that do not provide a SIMD interface.
                // simd_assign is true if simd_linear_assign() or simd_linear_assign(de1, de2)
                // is true.
                assigner.template linear_assign<simd_assign>(de1, de2);
            }
            else
            {
                assigner.template linear_assign<false>(de1, de2);
            }
        }
        else
        {
            if constexpr (simd_strided_assign)
            {
                assigner.strided_loop_assign(de1, de2);
            }
            else
            {
                assigner.template stepper_assign<default_assignable_layout(E1::static_layout)>(de1, de2);
            }
        }
    }

    template <class Tag>
    template <class E1, class E2, class P>
    inline void
    xexpression_assigner<Tag>::assign_to_container(xexpression<E1>& e1, const xexpression<E2>& e2, P p)
    {
        bool trivial_broadcast = resize(e1.derived_cast(), e2.derived_cast(), p);
        base_type::assign_data(e1, e2, trivial_broadcast, p);
    }

    namespace detail
    {
        template <class F, class P>
        bool get_rhs_triviality(const F&, P)
        {
            return true;
        }

        template <class F, class R, class... CT, class P>
        bool get_rhs_triviality(const xfunction<F, R, CT...>& rhs, P)
        {
            if constexpr (has_broadcasting<P>)
            {
                using index_type = xindex_type_t<typename xfunction<F, R, CT...>::shape_type>;
                using size_type = typename index_type::size_type;
                size_type size = rhs.dimension();
                index_type shape = uninitialized_shape<index_type>(size);
                bool trivial_broadcast = rhs.broadcast_shape(shape, true);
                return trivial_broadcast;
            }
            else
            {
                return true;
            }
        }
    }

    template <class Tag>
    template <class E1, class E2, class P>
    inline void xexpression_assigner<Tag>::assign_to_view(xexpression<E1>& e1, const xexpression<E2>& e2, P p)
    {
        assert_compatible_shape(e1, e2, p);
        base_type::assign_data(e1, e2, detail::get_rhs_triviality(e2.derived_cast(), p), p);
    }

    template <class Tag>
    template <class E1, class E2, class P>
    inline void xexpression_assigner<Tag>::computed_assign(xexpression<E1>& e1, const xexpression<E2>& e2, P p)
    {
        using shape_type = typename E1::shape_type;
        using comperator_type = std::greater<typename shape_type::value_type>;

        using size_type = typename E1::size_type;

        E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();

        if constexpr (has_broadcasting<P>)
        {
            size_type dim2 = de2.dimension();
            shape_type shape = uninitialized_shape<shape_type>(dim2);

            bool trivial_broadcast = de2.broadcast_shape(shape, true);

            auto&& de1_shape = de1.shape();
            if (dim2 > de1.dimension()
                || std::lexicographical_compare(
                    shape.begin(),
                    shape.end(),
                    de1_shape.begin(),
                    de1_shape.end(),
                    comperator_type()
                ))
            {
                typename E1::temporary_type tmp(shape);
                base_type::assign_data(tmp, e2, trivial_broadcast, p);
                de1.assign_temporary(std::move(tmp));
            }
            else
            {
                base_type::assign_data(e1, e2, trivial_broadcast, p);
            }
        }
        else
        {
            XTENSOR_ASSERT(same_shape(de1.shape(), de2.shape()));
            base_type::assign_data(e1, e2, true, p);
        }
    }

    template <class Tag>
    template <class E1, class E2, class F, class P>
    inline void xexpression_assigner<Tag>::scalar_computed_assign(xexpression<E1>& e1, const E2& e2, F&& f, P)
    {
        E1& d = e1.derived_cast();
        using size_type = typename E1::size_type;
        auto assign = [&](auto dst)
        {
            const size_type n = d.size();
            for (size_type i = 0; i < n; ++i)
            {
                *dst = f(*dst, e2);
                ++dst;
            }
        };
        if (E1::contiguous_layout || d.is_contiguous() || detail::is_dense_storage(d))
        {
            assign(d.storage().begin());
        }
        else
        {
            assign(d.begin());
        }
    }

    namespace detail
    {
        template <bool B, class... CT>
        struct static_trivial_broadcast;

        template <class... CT>
        struct static_trivial_broadcast<true, CT...>
        {
            static constexpr bool value = detail::promote_index<typename std::decay_t<CT>::shape_type...>::value;
        };

        template <class... CT>
        struct static_trivial_broadcast<false, CT...>
        {
            static constexpr bool value = false;
        };
    }

    template <class Tag>
    template <class E1, class E2, class P>
    inline bool xexpression_assigner<Tag>::resize(E1& e1, const E2& e2, P)
    {
        // If our RHS is not a xfunction, we know that the RHS is at least potentially trivial
        // We check the strides of the RHS in detail::is_trivial_broadcast to see if they match up!
        // So we can skip a shape copy and a call to broadcast_shape(...)
        e1.resize(e2.shape());
        return true;
    }

    template <class Tag>
    template <class S, class E>
    inline S xexpression_assigner<Tag>::get_nobroadcast_shape(const E& e)
    {
        const auto& shape = e.shape();
        return xtl::forward_sequence<S, decltype(shape)>(shape);
    }

    template <class Tag>
    template <class S, class F, class... CT>
    inline S xexpression_assigner<Tag>::get_nobroadcast_shape(const xfunction<F, CT...>& f)
    {
        return get_nobroadcast_shape<S, 0>(f.arguments());
    }

    template <class Tag>
    template <class S, size_t I, class... CT>
    inline S xexpression_assigner<Tag>::get_nobroadcast_shape(const std::tuple<CT...>& args)
    {
        if constexpr (I == sizeof...(CT))
        {
            return S{};
        }
        else if constexpr (xscalar_concept<std::tuple_element_t<I, std::tuple<CT...>>>)
        {
            return get_nobroadcast_shape<S, I + 1>(args);
        }
        else
        {
            return get_nobroadcast_shape<S>(std::get<I>(args));
        }
    }

    template <class Tag>
    template <class E1, class F, class... CT, class P>
    inline bool xexpression_assigner<Tag>::resize(E1& e1, const xfunction<F, CT...>& e2, P)
    {
        if constexpr (detail::is_fixed<typename xfunction<F, CT...>::shape_type>::value)
        {
            /*
             * If the shape of the xfunction is statically known, we can compute the broadcast triviality
             * at compile time plus we can resize right away.
             */
            // resize in case LHS is not a fixed size container. If it is, this is a NOP
            e1.resize(typename xfunction<F, CT...>::shape_type{});
            return detail::static_trivial_broadcast<
                detail::is_fixed<typename xfunction<F, CT...>::shape_type>::value,
                CT...>::value;
        }
        else
        {
            if constexpr (has_broadcasting<P>)
            {
                using index_type = xindex_type_t<typename E1::shape_type>;
                using size_type = typename E1::size_type;
                size_type size = e2.dimension();
                index_type shape = uninitialized_shape<index_type>(size);
                bool trivial_broadcast = e2.broadcast_shape(shape, true);
                e1.resize(std::move(shape));
                return trivial_broadcast;
            }
            else
            {
                using shape_type = typename E1::shape_type;
                shape_type shape = get_nobroadcast_shape<shape_type>(e2);
                e1.resize(std::move(shape));
                return true;
            }
        }
    }

    template <class Tag>
    template <class E1, class E2, class P>
    inline void
    xexpression_assigner<Tag>::assert_compatible_shape(const xexpression<E1>& e1, const xexpression<E2>& e2, P)
    {
        const E1& de1 = e1.derived_cast();
        const E2& de2 = e2.derived_cast();
        if constexpr (has_broadcasting<P>)
        {
            if (!broadcastable(de2.shape(), de1.shape()))
            {
                throw_broadcast_error(de2.shape(), de1.shape());
            }
        }
        else
        {
            if (!same_shape(de1.shape(), de2.shape()))
            {
                throw_broadcast_error(de2.shape(), de1.shape());
            }
        }
    }
}

#endif
