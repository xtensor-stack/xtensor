/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZDISPATCHER_HPP
#define XTENSOR_ZDISPATCHER_HPP

#include <xtl/xmultimethods.hpp>

#include "zdispatching_types.hpp"
#include "zmath.hpp"

namespace xt
{
    namespace mpl = xtl::mpl;

    template <class type_list>
    using zrun_dispatcher_impl = xtl::functor_dispatcher
    <
        type_list,
        void,
        xtl::static_caster,
        xtl::basic_fast_dispatcher
    >;

    template <class type_list>
    using ztype_dispatcher_impl = xtl::functor_dispatcher
    <
        type_list,
        size_t,
        xtl::static_caster,
        xtl::basic_fast_dispatcher
    >;

    /**********************
     * zdouble_dispatcher *
     **********************/

    // Double dispatchers are used for unary operations.
    // They dispatch on the single argument and on the
    // result.

    template <class F>
    class zdouble_dispatcher
    {
    public:

        template <class T, class R>
        static void insert();

        template <class T, class R, class... U>
        static void register_dispatching(mpl::vector<mpl::vector<T, R>, U...>);

        static void init();
        static void dispatch(const zarray_impl& z1, zarray_impl& res);
        static size_t get_type_index(const zarray_impl& z1);

    private:

        static zdouble_dispatcher& instance();

        zdouble_dispatcher();
        ~zdouble_dispatcher() = default;

        template <class T, class R>
        void insert_impl();

        template <class T, class R, class...U>
        inline void register_dispatching_impl(mpl::vector<mpl::vector<T, R>, U...>);
        inline void register_dispatching_impl(mpl::vector<>);

        using zfunctor_type = get_zmapped_functor_t<F>;
        using ztype_dispatcher = ztype_dispatcher_impl<mpl::vector<const zarray_impl>>;
        using zrun_dispatcher = zrun_dispatcher_impl<mpl::vector<const zarray_impl, zarray_impl>>;
        
        ztype_dispatcher m_type_dispatcher;
        zrun_dispatcher m_run_dispatcher;
    };


    /**********************
     * ztriple_dispatcher *
     **********************/

    // Triple dispatchers are used for binary operations.
    // They dispatch on both arguments and on the result.

    template <class F>
    class ztriple_dispatcher
    {
    public:

        template <class T1, class T2, class R>
        static void insert();

        template <class T1, class T2, class R, class... U>
        static void register_dispatching(mpl::vector<mpl::vector<T1, T2, R>, U...>);

        static void init();
        static void dispatch(const zarray_impl& z1, const zarray_impl& z2, zarray_impl& res);
        static size_t get_type_index(const zarray_impl& z1, const zarray_impl& z2);

    private:

        static ztriple_dispatcher& instance();

        ztriple_dispatcher();
        ~ztriple_dispatcher() = default;

        template <class T1, class T2, class R>
        void insert_impl();

        template <class T1, class T2, class R, class...U>
        inline void register_dispatching_impl(mpl::vector<mpl::vector<T1, T2, R>, U...>);
        inline void register_dispatching_impl(mpl::vector<>);

        using zfunctor_type = get_zmapped_functor_t<F>;
        using ztype_dispatcher = ztype_dispatcher_impl<mpl::vector<const zarray_impl, const zarray_impl>>;
        using zrun_dispatcher = zrun_dispatcher_impl<mpl::vector<const zarray_impl, const zarray_impl, zarray_impl>>;
        
        ztype_dispatcher m_type_dispatcher;
        zrun_dispatcher m_run_dispatcher;
    };

    /***************
     * zdispatcher *
     ***************/

    template <class F, size_t N>
    struct zdispatcher;

    template <class F>
    struct zdispatcher<F, 1>
    {
        using type = zdouble_dispatcher<F>;
    };

    template <class F>
    struct zdispatcher<F, 2>
    {
        using type = ztriple_dispatcher<F>;
    };

    template <class F, size_t N>
    using zdispatcher_t = typename zdispatcher<F, N>::type;

    /************************
     * zarray_impl_register *
     ************************/

    class zarray_impl_register
    {
    public:

        template <class T>
        static void insert();

        static void init();
        static const zarray_impl& get(size_t index);

    private:

        static zarray_impl_register& instance();

        zarray_impl_register();
        ~zarray_impl_register() = default;

        template <class T>
        void insert_impl();

        size_t m_next_index;
        std::vector<std::unique_ptr<zarray_impl>> m_register;
    };

    /****************
     * init_zsystem *
     ****************/

    // Early initialization of all dispatchers
    // and zarray_impl_register
    // return int so it can be assigned to a 
    // static variable and be automatically
    // called when loading a shared library
    // for instance.

    int init_zsystem();

    /*************************************
     * zdouble_dispatcher implementation *
     *************************************/

    namespace detail
    {
        template <class F>
        struct unary_dispatching_types
        {
            using type = zunary_func_types;
        };

        template <>
        struct unary_dispatching_types<negate>
        {
            using type = zunary_op_types;
        };

        template <>
        struct unary_dispatching_types<identity>
        {
            using type = zunary_op_types;
        };

        template <class F>
        using unary_dispatching_types_t = typename unary_dispatching_types<F>::type;
    }

    template <class F>
    template <class T, class R>
    inline void zdouble_dispatcher<F>::insert()
    {
        instance().template insert_impl<T, R>();
    }
    
    template <class F>
    template <class T, class R, class... U>
    inline void zdouble_dispatcher<F>::register_dispatching(mpl::vector<mpl::vector<T, R>, U...>)
    {
        instance().register_dispatching_impl(mpl::vector<mpl::vector<T, R>, U...>());
    }

    template <class F>
    inline void zdouble_dispatcher<F>::init()
    {
        instance();
    }

    template <class F>
    inline void zdouble_dispatcher<F>::dispatch(const zarray_impl& z1, zarray_impl& res)
    {
        instance().m_run_dispatcher.dispatch(z1, res);
    }
    
    template <class F>
    inline size_t zdouble_dispatcher<F>::get_type_index(const zarray_impl& z1)
    {
        return instance().m_type_dispatcher.dispatch(z1);
    }

    template <class F>
    inline zdouble_dispatcher<F>& zdouble_dispatcher<F>::instance()
    {
        static zdouble_dispatcher<F> inst;
        return inst;
    }

    template <class F>
    inline zdouble_dispatcher<F>::zdouble_dispatcher()
    {
        register_dispatching_impl(detail::unary_dispatching_types_t<F>());
    }

    template <class F>
    template <class T, class R>
    inline void zdouble_dispatcher<F>::insert_impl()
    {
        using arg_type = const ztyped_array<T>;
        using res_type = ztyped_array<R>;
        m_run_dispatcher.template insert<arg_type, res_type>(&zfunctor_type::template run<T, R>);
        m_type_dispatcher.template insert<arg_type>(&zfunctor_type::template index<T>);
    }

    template <class F>
    template <class T, class R, class...U>
    inline void zdouble_dispatcher<F>::register_dispatching_impl(mpl::vector<mpl::vector<T, R>, U...>)
    {
        insert_impl<T, R>();
        register_dispatching_impl(mpl::vector<U...>());
    }

    template <class F>
    inline void zdouble_dispatcher<F>::register_dispatching_impl(mpl::vector<>)
    {
    }
    
    /*************************************
     * ztriple_dispatcher implementation *
     *************************************/

    namespace detail
    {
        using zbinary_func_list = mpl::vector
        <
            math::atan2_fun,
            math::hypot_fun,
            math::pow_fun,
            math::fdim_fun,
            math::fmax_fun,
            math::fmin_fun,
            math::remainder_fun,
            math::fmod_fun
        >;

        template <class F>
        struct binary_dispatching_types
        {
            using type = std::conditional_t<mpl::contains<zbinary_func_list, F>::value,
                                            zbinary_func_types,
                                            zbinary_op_types>;
        };

        template <class F>
        using binary_dispatching_types_t = typename binary_dispatching_types<F>::type;
    }

    template <class F>
    template <class T1, class T2, class R>
    inline void ztriple_dispatcher<F>::insert()
    {
        instance().template insert_impl<T1, T2, R>();
    }
    
    template <class F>
    template <class T1, class T2, class R, class... U>
    inline void ztriple_dispatcher<F>::register_dispatching(mpl::vector<mpl::vector<T1, T2, R>, U...>)
    {
        instance().register_impl(mpl::vector<mpl::vector<T1, T2, R>, U...>());
    }

    template <class F>
    inline void ztriple_dispatcher<F>::init()
    {
        instance();
    }

    template <class F>
    inline void ztriple_dispatcher<F>::dispatch(const zarray_impl& z1, const zarray_impl& z2, zarray_impl& res)
    {
        instance().m_run_dispatcher.dispatch(z1, z2, res);
    }
    
    template <class F>
    inline size_t ztriple_dispatcher<F>::get_type_index(const zarray_impl& z1, const zarray_impl& z2)
    {
        return instance().m_type_dispatcher.dispatch(z1, z2);
    }

    template <class F>
    inline ztriple_dispatcher<F>& ztriple_dispatcher<F>::instance()
    {
        static ztriple_dispatcher<F> inst;
        return inst;
    }

    template <class F>
    inline ztriple_dispatcher<F>::ztriple_dispatcher()
    {
        register_dispatching_impl(detail::binary_dispatching_types_t<F>());
    }

    template <class F>
    template <class T1, class T2, class R>
    inline void ztriple_dispatcher<F>::insert_impl()
    {
        using arg_type1 = const ztyped_array<T1>;
        using arg_type2 = const ztyped_array<T2>;
        using res_type = ztyped_array<R>;
        m_run_dispatcher.template insert<arg_type1, arg_type2, res_type>(&zfunctor_type::template run<T1, T2, R>);
        m_type_dispatcher.template insert<arg_type1, arg_type1>(&zfunctor_type::template index<T1, T2>);
    }


    template <class F>
    template <class T1, class T2, class R, class...U>
    inline void ztriple_dispatcher<F>::register_dispatching_impl(mpl::vector<mpl::vector<T1, T2, R>, U...>)
    {
        insert_impl<T1, T2, R>();
        register_dispatching_impl(mpl::vector<U...>());
    }

    template <class F>
    inline void ztriple_dispatcher<F>::register_dispatching_impl(mpl::vector<>)
    {
    }
    
    /***************************************
     * zarray_impl_register implementation *
     ***************************************/

    template <class T>
    inline void zarray_impl_register::insert()
    {
        instance().template insert_impl<T>();
    }

    inline void zarray_impl_register::init()
    {
        instance();
    }

    inline const zarray_impl& zarray_impl_register::get(size_t index)
    {
        return *(instance().m_register[index]);
    }

    inline zarray_impl_register& zarray_impl_register::instance()
    {
        static zarray_impl_register r;
        return r;
    }
        
    inline zarray_impl_register::zarray_impl_register()
        : m_next_index(0)
    {
        insert_impl<float>();
        insert_impl<double>();
    }
    
    template <class T>
    inline void zarray_impl_register::insert_impl()
    {
        size_t& idx = ztyped_array<T>::get_class_static_index();
        if (idx == SIZE_MAX)
        {
            m_register.resize(++m_next_index);
            idx = m_register.size() - 1u;

        }
        else if (m_register.size() <= idx)
        {
            m_register.resize(idx + 1u);
        }
        m_register[idx] = std::unique_ptr<zarray_impl>(detail::build_zarray(std::move(xarray<T>())));
    }

    /*******************************
     * init_zsystem implementation *
     *******************************/

    namespace detail
    {
        inline void init_zdispatchers()
        {
            zdispatcher_t<identity, 1>::init();
            zdispatcher_t<negate, 1>::init();
            zdispatcher_t<plus, 2>::init();
            zdispatcher_t<minus, 2>::init();
            zdispatcher_t<multiplies, 2>::init();
            zdispatcher_t<divides, 2>::init();
            //zdispatcher_t<modulus, 2>::init();
            zdispatcher_t<logical_or, 2>::init();
            zdispatcher_t<logical_and, 2>::init();
            //zdispatcher_t<logical_not, 2>::init();
            //zdispatcher_t<bitwise_or, 2>::init();
            //zdispatcher_t<bitwise_and, 2>::init();
            //zdispatcher_t<bitwise_xor, 2>::init();
            //zdispatcher_t<bitwise_not, 2>::init();
            //zdispatcher_t<left_shift, 2>::init();
            //zdispatcher_t<right_shift, 2>::init();
            zdispatcher_t<less, 2>::init();
            zdispatcher_t<less_equal, 2>::init();
            zdispatcher_t<greater, 2>::init();
            zdispatcher_t<greater_equal, 2>::init();
            //zdispatcher_t<equal_to, 2>::init();
            //zdispatcher_t<not_equal_to, 2>::init();
        }
    }

    namespace math
    {
        inline void init_zdispatchers()
        {
            zdispatcher_t<fabs_fun, 1>::init();
            zdispatcher_t<fmod_fun, 2>::init();
            zdispatcher_t<remainder_fun, 2>::init();
            zdispatcher_t<fmax_fun, 2>::init();
            zdispatcher_t<fmin_fun, 2>::init();
            zdispatcher_t<fdim_fun, 2>::init();
            zdispatcher_t<exp_fun, 1>::init();
            zdispatcher_t<exp2_fun, 1>::init();
            zdispatcher_t<expm1_fun, 1>::init();
            zdispatcher_t<log_fun, 1>::init();
            zdispatcher_t<log10_fun, 1>::init();
            zdispatcher_t<log2_fun, 1>::init();
            zdispatcher_t<log1p_fun, 1>::init();
            zdispatcher_t<pow_fun, 2>::init();
            zdispatcher_t<sqrt_fun, 1>::init();
            zdispatcher_t<cbrt_fun, 1>::init();
            zdispatcher_t<hypot_fun, 2>::init();
            zdispatcher_t<sin_fun, 1>::init();
            zdispatcher_t<cos_fun, 1>::init();
            zdispatcher_t<tan_fun, 1>::init();
            zdispatcher_t<asin_fun, 1>::init();
            zdispatcher_t<acos_fun, 1>::init();
            zdispatcher_t<atan_fun, 1>::init();
            zdispatcher_t<atan2_fun, 2>::init();
            zdispatcher_t<sinh_fun, 1>::init();
            zdispatcher_t<cosh_fun, 1>::init();
            zdispatcher_t<tanh_fun, 1>::init();
            zdispatcher_t<asinh_fun, 1>::init();
            zdispatcher_t<acosh_fun, 1>::init();
            zdispatcher_t<atanh_fun, 1>::init();
            zdispatcher_t<erf_fun, 1>::init();
            zdispatcher_t<erfc_fun, 1>::init();
            zdispatcher_t<tgamma_fun, 1>::init();
            zdispatcher_t<lgamma_fun, 1>::init();
            /*zdispatcher_t<ceil_fun, 1>::init();
            zdispatcher_t<floor_fun, 1>::init();
            zdispatcher_t<trunc_fun, 1>::init();
            zdispatcher_t<round_fun, 1>::init();
            zdispatcher_t<nearbyint_fun, 1>::init();
            zdispatcher_t<rint_fun, 1>::init();
            zdispatcher_t<isfinite_fun, 1>::init();
            zdispatcher_t<isinf_fun, 1>::init();
            zdispatcher_t<isnan_fun, 1>::init();*/
        }
    }

    inline int init_zsystem()
    {
        detail::init_zdispatchers();
        math::init_zdispatchers();
        return 0;
    }
}

#endif
