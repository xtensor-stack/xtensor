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
    private:

        using zfunctor_type = get_zmapped_functor_t<F>;
        using ztype_dispatcher = ztype_dispatcher_impl<mpl::vector<const zarray_impl>>;
        using zrun_dispatcher = zrun_dispatcher_impl<mpl::vector<const zarray_impl, zarray_impl>>;
        
        static ztype_dispatcher& type_dispatcher()
        {
            static ztype_dispatcher dispatcher;
            return dispatcher;
        }

        static zrun_dispatcher& run_dispatcher()
        {
            static zrun_dispatcher dispatcher;
            return dispatcher;
        }

    public:

        template <class T, class R>
        static void insert()
        {
            using arg_type = const ztyped_array<T>;
            using res_type = ztyped_array<R>;
            run_dispatcher().template insert<arg_type, res_type>(&zfunctor_type::template run<T, R>);
            type_dispatcher().template insert<arg_type>(&zfunctor_type::template index<T>);
        }

        static void init()
        {
            insert<float, float>();
            insert<double, double>();
        }

        static void dispatch(const zarray_impl& z1, zarray_impl& res)
        {
            run_dispatcher().dispatch(z1, res);
        }

        static size_t get_type_index(const zarray_impl& z1)
        {
            return type_dispatcher().dispatch(z1);
        }
   };

    /**********************
     * ztriple_dispatcher *
     **********************/

    // Triple dispatchers are used for binary operations.
    // They dispatch on both arguments and on the result.

    template <class F>
    class ztriple_dispatcher
    {
    private:

        using zfunctor_type = get_zmapped_functor_t<F>;
        using ztype_dispatcher = ztype_dispatcher_impl<mpl::vector<const zarray_impl, const zarray_impl>>;
        using zrun_dispatcher = zrun_dispatcher_impl<mpl::vector<const zarray_impl, const zarray_impl, zarray_impl>>;
        
        static ztype_dispatcher& type_dispatcher()
        {
            static ztype_dispatcher dispatcher;
            return dispatcher;
        }

        static zrun_dispatcher& run_dispatcher()
        {
            static zrun_dispatcher dispatcher;
            return dispatcher;
        }

    public:

        template <class T1, class T2, class R>
        static void insert()
        {
            using arg_type1 = const ztyped_array<T1>;
            using arg_type2 = const ztyped_array<T2>;
            using res_type = ztyped_array<R>;
            run_dispatcher().template insert<arg_type1, arg_type2, res_type>(&zfunctor_type::template run<T1, T2, R>);
            type_dispatcher().template insert<arg_type1, arg_type1>(&zfunctor_type::template index<T1, T2>);
        }

        static void init()
        {
            insert<float, float, float>();
            insert<double, double, double>();
        }

        static void dispatch(const zarray_impl& z1, const zarray_impl& z2, zarray_impl& res)
        {
            run_dispatcher().dispatch(z1, z2, res);
        }

        static size_t get_type_index(const zarray_impl& z1, const zarray_impl& z2)
        {
            return type_dispatcher().dispatch(z1, z2);
        }
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
        void insert()
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

        const zarray_impl& operator[](size_t index) const
        {
            return *(m_register[index]);
        }

        static zarray_impl_register& instance()
        {
            static zarray_impl_register r;
            return r;
        }

    private:

        zarray_impl_register()
            : m_next_index(0)
        {
            insert<float>();
            insert<double>();
        }

        size_t m_next_index;
        std::vector<std::unique_ptr<zarray_impl>> m_register;
    };


}

#endif
