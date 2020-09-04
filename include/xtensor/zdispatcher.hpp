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

#include "zmath.hpp"

namespace xt
{
    namespace mpl = xtl::mpl;

    using supported_type = mpl::vector<float, double>;

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

        static void init();
        static void dispatch(const zarray_impl& z1, zarray_impl& res);
        static size_t get_type_index(const zarray_impl& z1);

    private:

        static zdouble_dispatcher& instance();

        zdouble_dispatcher();
        ~zdouble_dispatcher() = default;

        template <class T, class R>
        void insert_impl();

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

        static void init();
        static void dispatch(const zarray_impl& z1, const zarray_impl& z2, zarray_impl& res);
        static size_t get_type_index(const zarray_impl& z1, const zarray_impl& z2);

    private:

        static ztriple_dispatcher& instance();

        ztriple_dispatcher();
        ~ztriple_dispatcher() = default;

        template <class T1, class T2, class R>
        void insert_impl();

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

    /*************************************
     * zdouble_dispatcher implementation *
     *************************************/

    template <class F>
    template <class T, class R>
    inline void zdouble_dispatcher<F>::insert()
    {
        instance().template insert_impl<T, R>();
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
        insert_impl<float, float>();
        insert_impl<double, double>();
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

    /*************************************
     * ztriple_dispatcher implementation *
     *************************************/

    template <class F>
    template <class T1, class T2, class R>
    inline void ztriple_dispatcher<F>::insert()
    {
        instance().template insert_impl<T1, T2, R>();
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
        insert_impl<float, float, float>();
        insert_impl<double, double, double>();
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
}

#endif
