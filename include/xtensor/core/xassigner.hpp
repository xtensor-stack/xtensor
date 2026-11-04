/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
 * Copyright (c) QuantStack                                                 *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XTENSOR_ASSIGNER_HPP
#define XTENSOR_ASSIGNER_HPP

#include "../core/xassign_proxy.hpp"
#include "../core/xfunction.hpp"

#if defined(XTENSOR_USE_TBB)
#include <tbb/tbb.h>
#endif

#if defined(XTENSOR_USE_OPENMP)
#include <omp.h>
#endif

namespace xt
{
    /*************
     * Assigners *
     *************/

    template <class E1, class E2>
    class noparallel_assigner
    {
    public:

        template <bool simd_assign>
        void linear_assign(E1& e1, const E2& e2) const;

        template <layout_type L>
        void stepper_assign(E1& e1, const E2& e2) const;

        void strided_loop_assign(E1& e1, const E2& e2) const;

    private:

        void linear_assign_no_simd(E1& e1, const E2& e2) const;
    };

#if defined(XTENSOR_USE_TBB)
    template <class E1, class E2>
    class tbb_assigner
    {
    public:

        template <bool simd_assign>
        void linear_assign(E1& e1, const E2& e2) const;

        template <layout_type L>
        void stepper_assign(E1& e1, const E2& e2) const;

        void strided_loop_assign(E1& e1, const E2& e2) const;

    private:

        void linear_assign_no_simd(E1& e1, const E2& e2) const;
    };
#endif

#if defined(XTENSOR_USE_OPENMP)
    template <class E1, class E2>
    class openmp_assigner
    {
    public:

        template <bool simd_assign>
        void linear_assign(E1& e1, const E2& e2) const;

        template <layout_type L>
        void stepper_assign(E1& e1, const E2& e2) const;

        void strided_loop_assign(E1& e1, const E2& e2) const;

    private:

        void linear_assign_no_simd(E1& e1, const E2& e2) const;
    };
#endif

    template <class E1, class E2>
    class waterfall_assigner
    {
    public:

        template <bool simd_assign>
        void linear_assign(E1& e1, const E2& e2) const;

        template <layout_type L>
        void stepper_assign(E1& e1, const E2& e2) const;

        void strided_loop_assign(E1& e1, const E2& e2) const;
    };

    /****************************
     * tags - assigners mapping *
     ****************************/

    template <class E1, class E2, class Tag>
    struct parallel_assigner;

    template <class E1, class E2>
    struct parallel_assigner<E1, E2, noparallel_tag>
    {
        using type = noparallel_assigner<E1, E2>;
    };

#if defined(XTENSOR_USE_TBB)
    template <class E1, class E2>
    struct parallel_assigner<E1, E2, tbb_tag>
    {
        using type = tbb_assigner<E1, E2>;
    };
#endif

#if defined(XTENSOR_USE_OPENMP)
    template <class E1, class E2>
    struct parallel_assigner<E1, E2, openmp_tag>
    {
        using type = openmp_assigner<E1, E2>;
    };
#endif

    template <class E1, class E2>
    struct parallel_assigner<E1, E2, default_parallel_tag>
    {
        using type = waterfall_assigner<E1, E2>;
    };

    template <class E1, class E2, class Tag>
    using parallel_assigner_t = typename parallel_assigner<E1, E2, Tag>::type;

    /***********
     * Helpers *
     ***********/

    template <class FROM, class TO>
    struct is_narrowing_conversion
    {
        using argument_type = std::decay_t<FROM>;
        using result_type = std::decay_t<TO>;

        static const bool value = xtl::is_arithmetic<result_type>::value
                                  && (sizeof(result_type) < sizeof(argument_type)
                                      || (xtl::is_integral<result_type>::value
                                          && std::is_floating_point<argument_type>::value));
    };

    template <class FROM, class TO>
    struct has_sign_conversion
    {
        using argument_type = std::decay_t<FROM>;
        using result_type = std::decay_t<TO>;

        static const bool value = xtl::is_signed<argument_type>::value != xtl::is_signed<result_type>::value;
    };

    template <class FROM, class TO>
    struct has_assign_conversion
    {
        using argument_type = std::decay_t<FROM>;
        using result_type = std::decay_t<TO>;

        static const bool value = is_narrowing_conversion<argument_type, result_type>::value
                                  || has_sign_conversion<argument_type, result_type>::value;
    };

    namespace detail
    {
        /**
         * Considering the assignment LHS = RHS, if the requested value type used for
         * loading simd from RHS is not complex while LHS value_type is complex,
         * the assignment fails. The reason is that SIMD batches of complex values cannot
         * be implicitly instantiated from batches of scalar values.
         * Making the constructor implicit does not fix the issue since in the end,
         * the assignment is done with vec.store(buffer) where vec is a batch of scalars
         * and buffer an array of complex. SIMD batches of scalars do not provide overloads
         * of store that accept buffer of complex values and that SHOULD NOT CHANGE.
         * Load and store overloads must accept SCALAR BUFFERS ONLY.
         * Therefore, the solution is to explicitly force the instantiation of complex
         * batches in the assignment mechanism. A common situation that triggers this
         * issue is:
         * xt::xarray<double> rhs = { 1, 2, 3 };
         * xt::xarray<std::complex<double>> lhs = rhs;
         */
        template <class T1, class T2>
        struct conditional_promote_to_complex
        {
            static constexpr bool cond = xtl::is_gen_complex<T1>::value && !xtl::is_gen_complex<T2>::value;
            // Alternative: use std::complex<T2> or xcomplex<T2, T2, bool> depending on T1
            using type = std::conditional_t<cond, T1, T2>;
        };

        template <class T1, class T2>
        using conditional_promote_to_complex_t = typename conditional_promote_to_complex<T1, T2>::type;
    }

    template <class E1, class E2>
    struct requested_value_type
    {
        using e1_value_type = typename E1::value_type;
        using e2_value_type = typename E2::value_type;

        using e2_requested_value_type = std::
            conditional_t<std::same_as<e2_value_type, bool>, typename E2::bool_load_type, e2_value_type>;
        using type = detail::conditional_promote_to_complex_t<e1_value_type, e2_requested_value_type>;
    };

    template <class E1, class E2>
    using requested_value_type_t = typename requested_value_type<E1, E2>::type;

    template <class E1, class E2>
    class linear_assigner
    {
    public:

        using lhs_align_mode = xt_simd::container_alignment_t<E1>;
        static constexpr bool is_aligned = std::is_same<lhs_align_mode, aligned_mode>::value;
        using rhs_align_mode = std::conditional_t<is_aligned, inner_aligned_mode, unaligned_mode>;
        using e1_value_type = typename E1::value_type;
        using e2_value_type = typename E2::value_type;
        using value_type = requested_value_type_t<E1, E2>;
        using simd_type = xt_simd::simd_type<value_type>;
        using size_type = typename E1::size_type;
        static constexpr size_type simd_size = xt_simd::simd_traits<value_type>::size;
        static constexpr bool needs_cast = has_assign_conversion<e2_value_type, e1_value_type>::value;
        static constexpr bool is_convertible = std::is_convertible<e2_value_type, e1_value_type>::value;

        size_type alignment_begin(const E1& e1, size_type size) const;
        size_type alignment_end(size_type align_begin, size_type size) const;

        void assign_elements(E1& e1, const E2& e2, size_type begin, size_type end) const;
        void assign_simd(E1& e1, const E2& e2, size_type begin, size_type end) const;

        // Assigns elements [begin, end) through linear iterators, without SIMD
        void assign_elements_no_simd(E1& e1, const E2& e2, size_type begin, size_type end) const;
    };

    template <class E1, class E2, layout_type L>
    class stepper_assigner
    {
    public:

        using lhs_iterator = typename E1::stepper;
        using rhs_iterator = typename E2::const_stepper;
        using shape_type = typename E1::shape_type;
        using index_type = xindex_type_t<shape_type>;
        using size_type = typename lhs_iterator::size_type;
        using difference_type = typename lhs_iterator::difference_type;

        stepper_assigner(E1& e1, const E2& e2);

        void run();

        void step(size_type i);
        void step(size_type i, size_type n);
        void reset(size_type i);

        void to_end(layout_type);

    private:

        E1& m_e1;

        lhs_iterator m_lhs;
        rhs_iterator m_rhs;

        index_type m_index;
    };

    namespace strided_assign_detail
    {
        struct loop_sizes_t
        {
            bool can_do_strided_assign;
            bool is_row_major;
            std::size_t inner_loop_size;
            std::size_t outer_loop_size;
            std::size_t cut;
            std::size_t dimension;
        };
    }

    template <class E1, class E2>
    class strided_loop_assigner
    {
    public:

        using size_type = std::size_t;
        using index_type = dynamic_shape<std::size_t>;
        using loop_sizes_t = strided_assign_detail::loop_sizes_t;
        using lhs_stepper = typename E1::stepper;
        using rhs_stepper = typename E2::const_stepper;
        using e1_value_type = typename E1::value_type;
        using e2_value_type = typename E2::value_type;
        using value_type = requested_value_type_t<E1, E2>;
        using simd_type = std::conditional_t<
            std::is_same<e1_value_type, bool>::value,
            xt_simd::simd_bool_type<value_type>,
            xt_simd::simd_type<value_type>>;
        static constexpr bool needs_cast = has_assign_conversion<e2_value_type, e1_value_type>::value;

        strided_loop_assigner(E1& e1, const E2& e2);

        bool can_do_strided_assign() const noexcept;
        size_type inner_loop_size() const noexcept;
        size_type outer_loop_size() const noexcept;

        // Assigns the outer iterations [begin, end).
        void run(size_type begin, size_type end) const;

    private:

        void nth_idx(size_type n, index_type& index) const;
        void next_idx(index_type& index) const;

        E1& m_e1;
        const E2& m_e2;
        loop_sizes_t m_loop_sizes;
        index_type m_max_shape;
        size_type m_step_dim;
        size_type m_simd_size;
        size_type m_simd_rest;
    };

    /**************************************
     * noparallel_assigner implementation *
     **************************************/

    template <class E1, class E2>
    template <bool simd_assign>
    void noparallel_assigner<E1, E2>::linear_assign(E1& e1, const E2& e2) const
    {
        if constexpr (!simd_assign)
        {
            linear_assign_no_simd(e1, e2);
        }
        else
        {
            using assigner_t = linear_assigner<E1, E2>;
            using size_type = typename assigner_t::size_type;

            assigner_t helper;
            size_type size = e1.size();
            auto align_begin = helper.alignment_begin(e1, size);
            auto align_end = helper.alignment_end(align_begin, size);

            helper.assign_elements(e1, e2, 0u, align_begin);
            helper.assign_simd(e1, e2, align_begin, align_end);
            helper.assign_elements(e1, e2, align_end, size);
        }
    }

    template <class E1, class E2>
    template <layout_type L>
    void noparallel_assigner<E1, E2>::stepper_assign(E1& e1, const E2& e2) const
    {
        stepper_assigner<E1, E2, L> helper(e1, e2);
        helper.run();
    }

    template <class E1, class E2>
    void noparallel_assigner<E1, E2>::strided_loop_assign(E1& e1, const E2& e2) const
    {
        strided_loop_assigner<E1, E2> helper(e1, e2);
        if (!helper.can_do_strided_assign())
        {
            stepper_assign<default_assignable_layout(E1::static_layout)>(e1, e2);
            return;
        }

        helper.run(0u, helper.outer_loop_size());
    }

    template <class E1, class E2>
    void noparallel_assigner<E1, E2>::linear_assign_no_simd(E1& e1, const E2& e2) const
    {
        linear_assigner<E1, E2>{}.assign_elements_no_simd(e1, e2, 0u, e1.size());
    }

    /*******************************
     * tbb_assigner implementation *
     *******************************/

#if defined(XTENSOR_USE_TBB)
    template <class E1, class E2>
    template <bool simd_assign>
    void tbb_assigner<E1, E2>::linear_assign(E1& e1, const E2& e2) const
    {
        if constexpr (!simd_assign)
        {
            linear_assign_no_simd(e1, e2);
        }
        else
        {
            using assigner_t = linear_assigner<E1, E2>;
            using size_type = typename assigner_t::size_type;
            constexpr size_type simd_size = assigner_t::simd_size;

            assigner_t helper;
            size_type size = e1.size();
            auto align_begin = helper.alignment_begin(e1, size);
            auto align_end = helper.alignment_end(align_begin, size);

            helper.assign_elements(e1, e2, 0u, align_begin);
            if (size >= XTENSOR_TBB_THRESHOLD)
            {
                tbb::static_partitioner ap;
                tbb::parallel_for(
                    align_begin,
                    align_end,
                    simd_size,
                    [&helper, &e1, &e2](size_type i)
                    {
                        helper.assign_simd(e1, e2, i, i + simd_size);
                    },
                    ap
                );
            }
            else
            {
                helper.assign_simd(e1, e2, align_begin, align_end);
            }
            helper.assign_elements(e1, e2, align_end, size);
        }
    }

    template <class E1, class E2>
    template <layout_type L>
    void tbb_assigner<E1, E2>::stepper_assign(E1& e1, const E2& e2) const
    {
        stepper_assigner<E1, E2, L> helper(e1, e2);
        helper.run();
    }

    template <class E1, class E2>
    void tbb_assigner<E1, E2>::strided_loop_assign(E1& e1, const E2& e2) const
    {
        using assigner_t = strided_loop_assigner<E1, E2>;
        using size_type = typename assigner_t::size_type;

        assigner_t helper(e1, e2);
        if (!helper.can_do_strided_assign())
        {
            stepper_assign<default_assignable_layout(E1::static_layout)>(e1, e2);
            return;
        }

        size_type outer_loop_size = helper.outer_loop_size();
        if (outer_loop_size > XTENSOR_TBB_THRESHOLD / helper.inner_loop_size())
        {
            tbb::static_partitioner sp;
            tbb::parallel_for(
                tbb::blocked_range<size_type>(size_type(0), outer_loop_size),
                [&helper](const tbb::blocked_range<size_type>& r)
                {
                    helper.run(r.begin(), r.end());
                },
                sp
            );
        }
        else
        {
            helper.run(0u, outer_loop_size);
        }
    }

    template <class E1, class E2>
    void tbb_assigner<E1, E2>::linear_assign_no_simd(E1& e1, const E2& e2) const
    {
        using assigner_t = linear_assigner<E1, E2>;
        using size_type = typename assigner_t::size_type;

        assigner_t helper;
        tbb::static_partitioner sp;
        tbb::parallel_for(
            tbb::blocked_range<size_type>(size_type(0), e1.size()),
            [&helper, &e1, &e2](const tbb::blocked_range<size_type>& r)
            {
                helper.assign_elements_no_simd(e1, e2, r.begin(), r.end());
            },
            sp
        );
    }
#endif

    /**********************************
     * openmp_assigner implementation *
     **********************************/

#if defined(XTENSOR_USE_OPENMP)
    template <class E1, class E2>
    template <bool simd_assign>
    void openmp_assigner<E1, E2>::linear_assign(E1& e1, const E2& e2) const
    {
        if constexpr (!simd_assign)
        {
            linear_assign_no_simd(e1, e2);
        }
        else
        {
            using assigner_t = linear_assigner<E1, E2>;
            using size_type = typename assigner_t::size_type;
            constexpr size_type simd_size = assigner_t::simd_size;

            assigner_t helper;
            size_type size = e1.size();
            auto align_begin = helper.alignment_begin(e1, size);
            auto align_end = helper.alignment_end(align_begin, size);

            helper.assign_elements(e1, e2, 0u, align_begin);
            if (size >= size_type(XTENSOR_OPENMP_TRESHOLD))
            {
                // Signed loop index for MSVC, which only supports OpenMP 2.0
#pragma omp parallel for default(none) shared(helper, align_begin, align_end, e1, e2)
                for (auto i = static_cast<std::ptrdiff_t>(align_begin);
                     i < static_cast<std::ptrdiff_t>(align_end);
                     i += static_cast<std::ptrdiff_t>(simd_size))
                {
                    size_type ui = static_cast<size_type>(i);
                    helper.assign_simd(e1, e2, ui, ui + simd_size);
                }
            }
            else
            {
                helper.assign_simd(e1, e2, align_begin, align_end);
            }
            helper.assign_elements(e1, e2, align_end, size);
        }
    }

    template <class E1, class E2>
    template <layout_type L>
    void openmp_assigner<E1, E2>::stepper_assign(E1& e1, const E2& e2) const
    {
        stepper_assigner<E1, E2, L> helper(e1, e2);
        helper.run();
    }

    template <class E1, class E2>
    void openmp_assigner<E1, E2>::strided_loop_assign(E1& e1, const E2& e2) const
    {
        using assigner_t = strided_loop_assigner<E1, E2>;
        using size_type = typename assigner_t::size_type;

        assigner_t helper(e1, e2);
        if (!helper.can_do_strided_assign())
        {
            stepper_assign<default_assignable_layout(E1::static_layout)>(e1, e2);
            return;
        }

        size_type outer_loop_size = helper.outer_loop_size();
        if (outer_loop_size >= XTENSOR_OPENMP_TRESHOLD / helper.inner_loop_size())
        {
            // Each thread assigns a contiguous chunk of outer iterations,
            // as schedule(static) would do
#pragma omp parallel default(none) shared(helper, outer_loop_size)
            {
                size_type nb_threads = static_cast<size_type>(omp_get_num_threads());
                size_type thread_id = static_cast<size_type>(omp_get_thread_num());
                size_type chunk = outer_loop_size / nb_threads;
                size_type rest = outer_loop_size % nb_threads;
                size_type begin = thread_id * chunk + std::min(thread_id, rest);
                size_type end = begin + chunk + (thread_id < rest ? 1u : 0u);
                helper.run(begin, end);
            }
        }
        else
        {
            helper.run(0u, outer_loop_size);
        }
    }

    template <class E1, class E2>
    void openmp_assigner<E1, E2>::linear_assign_no_simd(E1& e1, const E2& e2) const
    {
        using assigner_t = linear_assigner<E1, E2>;
        using size_type = typename assigner_t::size_type;

        assigner_t helper;
        size_type n = e1.size();

        if (n >= size_type(XTENSOR_OPENMP_TRESHOLD))
        {
            // Each thread assigns a contiguous chunk, as schedule(static) would do
#pragma omp parallel default(none) shared(helper, e1, e2, n)
            {
                size_type nb_threads = static_cast<size_type>(omp_get_num_threads());
                size_type thread_id = static_cast<size_type>(omp_get_thread_num());
                size_type chunk = n / nb_threads;
                size_type rest = n % nb_threads;
                size_type begin = thread_id * chunk + std::min(thread_id, rest);
                size_type end = begin + chunk + (thread_id < rest ? 1u : 0u);
                helper.assign_elements_no_simd(e1, e2, begin, end);
            }
        }
        else
        {
            helper.assign_elements_no_simd(e1, e2, 0u, n);
        }
    }
#endif

    /*************************************
     * waterfall_assigner implementation *
     *************************************/

    template <class E1, class E2>
    template <bool simd_assign>
    void waterfall_assigner<E1, E2>::linear_assign(E1& e1, const E2& e2) const
    {
#if defined(XTENSOR_USE_TBB)
        tbb_assigner<E1, E2>{}.template linear_assign<simd_assign>(e1, e2);
#elif defined(XTENSOR_USE_OPENMP)
        openmp_assigner<E1, E2>{}.template linear_assign<simd_assign>(e1, e2);
#else
        noparallel_assigner<E1, E2>{}.template linear_assign<simd_assign>(e1, e2);
#endif
    }

    template <class E1, class E2>
    template <layout_type L>
    void waterfall_assigner<E1, E2>::stepper_assign(E1& e1, const E2& e2) const
    {
        stepper_assigner<E1, E2, L> helper(e1, e2);
        helper.run();
    }

    template <class E1, class E2>
    void waterfall_assigner<E1, E2>::strided_loop_assign(E1& e1, const E2& e2) const
    {
#if defined(XTENSOR_USE_TBB)
        tbb_assigner<E1, E2>{}.strided_loop_assign(e1, e2);
#elif defined(XTENSOR_USE_OPENMP)
        openmp_assigner<E1, E2>{}.strided_loop_assign(e1, e2);
#else
        noparallel_assigner<E1, E2>{}.strided_loop_assign(e1, e2);
#endif
    }

    /**********************************
     * linear_assigner implementation *
     **********************************/

    template <class E1, class E2>
    auto linear_assigner<E1, E2>::alignment_begin(const E1& e1, size_type size) const -> size_type
    {
        size_type align_begin = is_aligned ? 0 : xt_simd::get_alignment_offset(e1.data(), size, simd_size);
        return align_begin;
    }

    template <class E1, class E2>
    auto linear_assigner<E1, E2>::alignment_end(size_type align_begin, size_type size) const -> size_type
    {
        size_type align_end = align_begin + ((size - align_begin) & ~(simd_size - 1));
        return align_end;
    }

    template <class E1, class E2>
    void linear_assigner<E1, E2>::assign_elements(E1& e1, const E2& e2, size_type begin, size_type end) const
    {
        for (size_type i = begin; i < end; ++i)
        {
            e1.data_element(i) = conditional_cast<needs_cast, e1_value_type>(e2.data_element(i));
        }
    }

    template <class E1, class E2>
    void linear_assigner<E1, E2>::assign_simd(E1& e1, const E2& e2, size_type begin, size_type end) const
    {
        for (size_type i = begin; i < end; i += simd_size)
        {
            e1.template store_simd<lhs_align_mode>(i, e2.template load_simd<rhs_align_mode, value_type>(i));
        }
    }

    template <class E1, class E2>
    void
    linear_assigner<E1, E2>::assign_elements_no_simd(E1& e1, const E2& e2, size_type begin, size_type end) const
    {
        // If the types are not compatible, this function is still instantiated but never called.
        // The loop must not be instantiated in that case to avoid compilation errors.
        if constexpr (!is_convertible)
        {
            XTENSOR_PRECONDITION(false, "Internal error: linear_assigner called with unrelated types.");
        }
        else
        {
            using difference_type = std::ptrdiff_t;
            auto src = linear_begin(e2) + static_cast<difference_type>(begin);
            auto dst = linear_begin(e1) + static_cast<difference_type>(begin);
            for (size_type i = begin; i < end; ++i)
            {
                *dst = static_cast<e1_value_type>(*src);
                ++src;
                ++dst;
            }
        }
    }

    /***********************************
     * stepper_assigner implementation *
     ***********************************/

    template <class E1, class E2, layout_type L>
    inline stepper_assigner<E1, E2, L>::stepper_assigner(E1& e1, const E2& e2)
        : m_e1(e1)
        , m_lhs(e1.stepper_begin(e1.shape()))
        , m_rhs(e2.stepper_begin(e1.shape()))
        , m_index(xtl::make_sequence<index_type>(e1.shape().size(), size_type(0)))
    {
    }

    template <class E1, class E2, layout_type L>
    inline void stepper_assigner<E1, E2, L>::run()
    {
        using tmp_size_type = typename E1::size_type;
        using argument_type = std::decay_t<decltype(*m_rhs)>;
        using result_type = std::decay_t<decltype(*m_lhs)>;
        constexpr bool needs_cast = has_assign_conversion<argument_type, result_type>::value;

        tmp_size_type s = m_e1.size();
        for (tmp_size_type i = 0; i < s; ++i)
        {
            *m_lhs = conditional_cast<needs_cast, result_type>(*m_rhs);
            stepper_tools<L>::increment_stepper(*this, m_index, m_e1.shape());
        }
    }

    template <class E1, class E2, layout_type L>
    inline void stepper_assigner<E1, E2, L>::step(size_type i)
    {
        m_lhs.step(i);
        m_rhs.step(i);
    }

    template <class E1, class E2, layout_type L>
    inline void stepper_assigner<E1, E2, L>::step(size_type i, size_type n)
    {
        m_lhs.step(i, n);
        m_rhs.step(i, n);
    }

    template <class E1, class E2, layout_type L>
    inline void stepper_assigner<E1, E2, L>::reset(size_type i)
    {
        m_lhs.reset(i);
        m_rhs.reset(i);
    }

    template <class E1, class E2, layout_type L>
    inline void stepper_assigner<E1, E2, L>::to_end(layout_type l)
    {
        m_lhs.to_end(l);
        m_rhs.to_end(l);
    }

    /****************************************
     * strided_loop_assigner implementation *
     ****************************************/

    namespace strided_assign_detail
    {
        template <layout_type layout>
        struct idx_tools;

        template <>
        struct idx_tools<layout_type::row_major>
        {
            template <class T>
            static void next_idx(T& outer_index, const T& outer_shape)
            {
                auto i = outer_index.size();
                for (; i > 0; --i)
                {
                    if (outer_index[i - 1] + 1 >= outer_shape[i - 1])
                    {
                        outer_index[i - 1] = 0;
                    }
                    else
                    {
                        outer_index[i - 1]++;
                        break;
                    }
                }
            }

            template <class T>
            static void nth_idx(size_t n, T& outer_index, const T& outer_shape)
            {
                dynamic_shape<std::size_t> stride_sizes;
                xt::resize_container(stride_sizes, outer_shape.size());
                // compute strides
                using size_type = typename T::size_type;
                for (size_type i = outer_shape.size(); i > 0; i--)
                {
                    stride_sizes[i - 1] = (i == outer_shape.size()) ? 1 : stride_sizes[i] * outer_shape[i];
                }

                // compute index
                for (size_type i = 0; i < outer_shape.size(); i++)
                {
                    auto d_idx = n / stride_sizes[i];
                    outer_index[i] = d_idx;
                    n -= d_idx * stride_sizes[i];
                }
            }
        };

        template <>
        struct idx_tools<layout_type::column_major>
        {
            template <class T>
            static void next_idx(T& outer_index, const T& outer_shape)
            {
                using size_type = typename T::size_type;
                size_type i = 0;
                auto sz = outer_index.size();
                for (; i < sz; ++i)
                {
                    if (outer_index[i] + 1 >= outer_shape[i])
                    {
                        outer_index[i] = 0;
                    }
                    else
                    {
                        outer_index[i]++;
                        break;
                    }
                }
            }

            template <class T>
            static void nth_idx(size_t n, T& outer_index, const T& outer_shape)
            {
                dynamic_shape<std::size_t> stride_sizes;
                xt::resize_container(stride_sizes, outer_shape.size());

                using size_type = typename T::size_type;

                // compute required strides
                for (size_type i = 0; i < outer_shape.size(); i++)
                {
                    stride_sizes[i] = (i == 0) ? 1 : stride_sizes[i - 1] * outer_shape[i - 1];
                }

                // compute index
                for (size_type i = outer_shape.size(); i > 0;)
                {
                    i--;
                    auto d_idx = n / stride_sizes[i];
                    outer_index[i] = d_idx;
                    n -= d_idx * stride_sizes[i];
                }
            }
        };

        template <layout_type L, class S>
        struct check_strides_functor
        {
            using strides_type = S;

            check_strides_functor(const S& strides)
                : m_cut(L == layout_type::row_major ? 0 : strides.size())
                , m_strides(strides)
            {
            }

            template <class T, layout_type LE = L>
            std::enable_if_t<LE == layout_type::row_major, std::size_t> operator()(const T& el)
            {
                // All dimenions less than var have differing strides
                auto var = check_strides_overlap<layout_type::row_major>::get(m_strides, el.strides());
                if (var > m_cut)
                {
                    m_cut = var;
                }
                return m_cut;
            }

            template <class T, layout_type LE = L>
            std::enable_if_t<LE == layout_type::column_major, std::size_t> operator()(const T& el)
            {
                auto var = check_strides_overlap<layout_type::column_major>::get(m_strides, el.strides());
                // All dimensions >= var have differing strides
                if (var < m_cut)
                {
                    m_cut = var;
                }
                return m_cut;
            }

            template <class T>
            std::size_t operator()(const xt::xscalar<T>& /*el*/)
            {
                return m_cut;
            }

            template <class F, class... CT>
            std::size_t operator()(const xt::xfunction<F, CT...>& xf)
            {
                xt::for_each(*this, xf.arguments());
                return m_cut;
            }

        private:

            std::size_t m_cut;
            const strides_type& m_strides;
        };

        template <bool possible = true, class E1, class E2, std::enable_if_t<!strided_expression<E1> || !possible, bool> = true>
        loop_sizes_t get_loop_sizes(const E1& e1, const E2&)
        {
            return {false, true, 1, e1.size(), e1.dimension(), e1.dimension()};
        }

        template <bool possible = true, class E1, class E2, std::enable_if_t<strided_expression<E1> && possible, bool> = true>
        loop_sizes_t get_loop_sizes(const E1& e1, const E2& e2)
        {
            using shape_value_type = typename E1::shape_type::value_type;
            bool is_row_major = true;

            // Try to find a row-major scheme first, where the outer loop is on the first N = `cut`
            // dimensions, and the inner loop is
            is_row_major = true;
            auto is_zero = [](auto i)
            {
                return i == 0;
            };
            auto&& strides = e1.strides();
            auto it_bwd = std::find_if_not(strides.rbegin(), strides.rend(), is_zero);
            bool de1_row_contiguous = it_bwd != strides.rend() && *it_bwd == 1;
            auto it_fwd = std::find_if_not(strides.begin(), strides.end(), is_zero);
            bool de1_col_contiguous = it_fwd != strides.end() && *it_fwd == 1;
            if (de1_row_contiguous)
            {
                is_row_major = true;
            }
            else if (de1_col_contiguous)
            {
                is_row_major = false;
            }
            else
            {
                // No strided loop possible.
                return {false, true, 1, e1.size(), e1.dimension(), e1.dimension()};
            }

            // Cut is the number of dimensions in the outer loop
            std::size_t cut = 0;

            if (is_row_major)
            {
                auto csf = check_strides_functor<layout_type::row_major, decltype(e1.strides())>(e1.strides());
                cut = csf(e2);
                // This makes that only one dimension will be treated in the inner loop.
                if (cut < e1.strides().size() - 1)
                {
                    // Only make the inner loop go over one dimension by default for now
                    cut = e1.strides().size() - 1;
                }
            }
            else if (!is_row_major)
            {
                auto csf = check_strides_functor<layout_type::column_major, decltype(e1.strides())>(e1.strides()
                );
                cut = csf(e2);
                if (cut > 1)
                {
                    // Only make the inner loop go over one dimension by default for now
                    cut = 1;
                }
            }  // can't reach here because this would have already triggered the fallback

            std::size_t outer_loop_size = static_cast<std::size_t>(std::accumulate(
                e1.shape().begin(),
                e1.shape().begin() + static_cast<std::ptrdiff_t>(cut),
                shape_value_type(1),
                std::multiplies<shape_value_type>{}
            ));
            std::size_t inner_loop_size = static_cast<std::size_t>(std::accumulate(
                e1.shape().begin() + static_cast<std::ptrdiff_t>(cut),
                e1.shape().end(),
                shape_value_type(1),
                std::multiplies<shape_value_type>{}
            ));

            if (!is_row_major)
            {
                std::swap(outer_loop_size, inner_loop_size);
            }

            return {inner_loop_size > 1, is_row_major, inner_loop_size, outer_loop_size, cut, e1.dimension()};
        }
    }

    template <class E1, class E2>
    inline strided_loop_assigner<E1, E2>::strided_loop_assigner(E1& e1, const E2& e2)
        : m_e1(e1)
        , m_e2(e2)
        , m_loop_sizes(strided_assign_detail::get_loop_sizes(e1, e2))
        , m_max_shape()
        , m_step_dim(0)
        , m_simd_size(0)
        , m_simd_rest(0)
    {
        if (!m_loop_sizes.can_do_strided_assign)
        {
            return;
        }

        auto cut = static_cast<std::ptrdiff_t>(m_loop_sizes.cut);
        if (m_loop_sizes.is_row_major)
        {
            m_max_shape.assign(e1.shape().begin(), e1.shape().begin() + cut);
        }
        else
        {
            m_max_shape.assign(e1.shape().begin() + cut, e1.shape().end());
            m_step_dim = m_loop_sizes.cut;
        }

        m_simd_size = m_loop_sizes.inner_loop_size / simd_type::size;
        m_simd_rest = m_loop_sizes.inner_loop_size % simd_type::size;
    }

    template <class E1, class E2>
    inline bool strided_loop_assigner<E1, E2>::can_do_strided_assign() const noexcept
    {
        return m_loop_sizes.can_do_strided_assign;
    }

    template <class E1, class E2>
    inline auto strided_loop_assigner<E1, E2>::inner_loop_size() const noexcept -> size_type
    {
        return m_loop_sizes.inner_loop_size;
    }

    template <class E1, class E2>
    inline auto strided_loop_assigner<E1, E2>::outer_loop_size() const noexcept -> size_type
    {
        return m_loop_sizes.outer_loop_size;
    }

    template <class E1, class E2>
    inline void strided_loop_assigner<E1, E2>::run(size_type begin, size_type end) const
    {
        if (begin >= end)
        {
            return;
        }

        const size_type step_dim = m_step_dim;
        const size_type simd_size = m_simd_size;
        const size_type simd_rest = m_simd_rest;

        lhs_stepper lhs = m_e1.stepper_begin(m_e1.shape());
        rhs_stepper rhs = m_e2.stepper_begin(m_e1.shape());
        index_type index = xtl::make_sequence<index_type>(m_max_shape.size(), size_type(0));

        if (begin != size_type(0))
        {
            nth_idx(begin, index);
            for (size_type i = 0; i < index.size(); ++i)
            {
                lhs.step(i + step_dim, index[i]);
                rhs.step(i + step_dim, index[i]);
            }
        }

        for (size_type ox = begin; ox < end; ++ox)
        {
            for (size_type i = 0; i < simd_size; ++i)
            {
                lhs.store_simd(rhs.template step_simd<value_type>());
            }
            for (size_type i = 0; i < simd_rest; ++i)
            {
                *lhs = conditional_cast<needs_cast, e1_value_type>(*rhs);
                lhs.step_leading();
                rhs.step_leading();
            }

            next_idx(index);

            // need to step E1 as well if not contiguous assign (e.g. view)
            rhs.to_begin();
            if (!E1::contiguous_layout)
            {
                lhs.to_begin();
                for (size_type i = 0; i < index.size(); ++i)
                {
                    lhs.step(i + step_dim, index[i]);
                    rhs.step(i + step_dim, index[i]);
                }
            }
            else
            {
                for (size_type i = 0; i < index.size(); ++i)
                {
                    rhs.step(i + step_dim, index[i]);
                }
            }
        }
    }

    template <class E1, class E2>
    inline void strided_loop_assigner<E1, E2>::nth_idx(size_type n, index_type& index) const
    {
        if (m_loop_sizes.is_row_major)
        {
            strided_assign_detail::idx_tools<layout_type::row_major>::nth_idx(n, index, m_max_shape);
        }
        else
        {
            strided_assign_detail::idx_tools<layout_type::column_major>::nth_idx(n, index, m_max_shape);
        }
    }

    template <class E1, class E2>
    inline void strided_loop_assigner<E1, E2>::next_idx(index_type& index) const
    {
        if (m_loop_sizes.is_row_major)
        {
            strided_assign_detail::idx_tools<layout_type::row_major>::next_idx(index, m_max_shape);
        }
        else
        {
            strided_assign_detail::idx_tools<layout_type::column_major>::next_idx(index, m_max_shape);
        }
    }
}

#endif
