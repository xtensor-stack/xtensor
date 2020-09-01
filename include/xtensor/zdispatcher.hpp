/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef ZDISPATCHER_HPP
#define ZDISPATCHER_HPP

#include <xtl/xmultimethods.hpp>

#include "zmath.hpp"

namespace xt
{
    namespace mpl = xtl::mpl;

    using supported_type = mpl::vector<float, double>;

    template <class type_list>
    using zdispatcher_impl = xtl::functor_dispatcher
    <
        type_list,
        void,
        xtl::static_caster,
        xtl::basic_fast_dispatcher
    >;

    using zsingle_dispatcher_impl = zdispatcher_impl
    <
        mpl::vector<const zarray_impl, zarray_impl>
    >;

    using zdouble_dispatcher_impl = zdispatcher_impl
    <
        mpl::vector<const zarray_impl, const zarray_impl, zarray_impl>
    >;

    template <class D>
    struct zsingle_dispatcher_base
    {
        static zsingle_dispatcher_impl& get()
        {
            static zsingle_dispatcher_impl dispatcher;
            return dispatcher;
        }

        static void init()
        {
            D::template insert<float, float>();
            D::template insert<double, double>();
        }

        static void dispatch(const zarray_impl& z1, zarray_impl& res)
        {
            get().dispatch(z1, res);
        }
    };

    template <class F>
    class zsingle_dispatcher;

#define DEFINE_SINGLE_DISPATCHER(FUNCTOR, FUNCTION)\
    template <> \
    struct zsingle_dispatcher<FUNCTOR>\
        : private zsingle_dispatcher_base<zsingle_dispatcher<FUNCTOR>> \
    {\
        using base_type = zsingle_dispatcher_base<zsingle_dispatcher<FUNCTOR>>; \
        using base_type::dispatch; \
        using base_type::init; \
        template <class T1, class T2>\
        static void insert() \
        {\
            base_type::get().template insert<const ztyped_array<T1>, ztyped_array<T2>>(&FUNCTION<T1, T2>); \
        }\
    }

DEFINE_SINGLE_DISPATCHER(math::exp_fun, zexp);

}

#endif
