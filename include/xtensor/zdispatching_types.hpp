/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZDISPATCHING_TYPES_HPP
#define XTENSOR_ZDISPATCHING_TYPES_HPP

#include <xtl/xmeta_utils.hpp>

namespace xt
{
    namespace mpl = xtl::mpl;

    // TODO: move to XTL
    namespace detail
    {
        template <class T, class U>
        struct concatenate;

        template <class... T, class... U>
        struct concatenate<mpl::vector<T...>, mpl::vector<U...>>
        {
            using type = mpl::vector<T..., U...>;
        };

        template <class T, class U>
        using concatenate_t = typename concatenate<T, U>::type;
    }

    /*********************
     * Dispatching types *
     *********************/

    using dispatching_int_types = mpl::vector<uint8_t, int8_t,
                                              uint16_t, int16_t,
                                              uint32_t, int32_t,
                                              uint64_t, int64_t>;

    using dispatching_float_types = mpl::vector<float, double>;

    using dispatching_types = detail::concatenate_t<dispatching_int_types,
                                                    dispatching_float_types>;

    /*************************
     * unary operation types *
     *************************/

    template <class T>
    struct build_pair_type
    {
        using type = mpl::vector<T, T>;
    };

    template <class T>
    using build_pair_type_t = typename build_pair_type<T>::type;

    using unary_op_int_types = mpl::transform_t<build_pair_type_t, dispatching_int_types>;
    using unary_op_float_types = mpl::transform_t<build_pair_type_t, dispatching_float_types>;
    using unary_op_same_types = mpl::transform_t<build_pair_type_t, dispatching_types>;

    /**************************
     * binary operation types *
     **************************/

    template <class T>
    struct build_triplet_type
    {
        using type = mpl::vector<T, T, T>;
    };

    template <class T>
    using build_triplet_type_t = typename build_triplet_type<T>::type;

    using binary_op_int_types = mpl::transform_t<build_triplet_type_t, dispatching_int_types>;
    using binary_op_float_types = mpl::transform_t<build_triplet_type_t, dispatching_float_types>;
    using binary_op_same_types = mpl::transform_t<build_triplet_type_t, dispatching_types>;
}

#endif

