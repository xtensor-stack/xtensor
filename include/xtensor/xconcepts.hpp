/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XCONCEPTS_HPP
#define XCONCEPTS_HPP

#include <type_traits>

/**********************************************************/
/*                                                        */
/*   concept checking and type inference functionality    */
/*                                                        */
/**********************************************************/

namespace xt
{

/**********************************************************/
/*                                                        */
/*        XTENSOR_REQUIRE concept checking macro          */
/*                                                        */
/**********************************************************/

struct require_ok {};

template <bool CONCEPTS>
using concept_check = typename std::enable_if<CONCEPTS, require_ok>::type;

#define XTENSOR_REQUIRE typename Require = concept_check

/**********************************************************/
/*                                                        */
/*                  xexpression_concept                   */
/*                                                        */
/**********************************************************/

struct xexpression_tag {};

    // xexpression_concept is fulfilled by data structures that can be used
    // in xexpressions (xarray, xtensor, xfunction). By default, 'T' fulfills
    // the concept if it is derived from xexpression_tag.
    //
    // Alternatively, one can partially specialize xexpression_concept.
template <class T>
struct xexpression_concept
{
    static const bool value = std::is_base_of<xexpression_tag, std::decay_t<T> >::value;
};

/**********************************************************/
/*                                                        */
/*                  tiny_array_concept                    */
/*                                                        */
/**********************************************************/

struct tiny_array_tag {};

    // tiny_array_concept refers to tiny_array_base and tiny_array.
    // By default, 'ARRAY' fulfills the tiny_array_concept if it is derived
    // from tiny_array_tag.
    //
    // Alternatively, one can partially specialize tiny_array_concept.
template <class ARRAY>
struct tiny_array_concept
{
    static const bool value = std::is_base_of<tiny_array_tag, std::decay_t<ARRAY> >::value;
};

/**********************************************************/
/*                                                        */
/*                   iterator_concept                     */
/*                                                        */
/**********************************************************/

    // currently, we apply only the simple rule that class T
    // is either a pointer or array or has an embedded typedef
    // 'iterator_category'. More sophisticated checks should
    // be added when needed.
template <class T>
struct iterator_concept
{
    using V = std::decay_t<T>;

    static char test(...);

    template <class U>
    static int test(U*, typename U::iterator_category * = 0);

    static const bool value =
        std::is_array<T>::value ||
        std::is_pointer<T>::value ||
        std::is_same<decltype(test((V*)0)), int>::value;
};

/**********************************************************/
/*                                                        */
/*                    promote types                       */
/*                                                        */
/**********************************************************/

namespace concepts_detail
{
    using std::sqrt;

    template <class T>
    using real_promote_t = decltype(sqrt(*(std::decay_t<T>*)0));
}

    // result of arithmetic expressions
    // (e.g. unsigned char + unsigned char => int)
template <class T1, class T2 = T1>
using promote_t = decltype(*(std::decay_t<T1>*)0 + *(std::decay_t<T2>*)0);

    // result of algebraic expressions
    // (e.g. sqrt(int) => double)
template <class T>
using real_promote_t = concepts_detail::real_promote_t<T>;

    // replace 'bool' with 'uint8_t', keep everything else
template <class T>
using bool_promote_t = typename std::conditional<std::is_same<T, bool>::value, uint8_t, T>::type;

/**********************************************************/
/*                                                        */
/*       type inference for norm and squared norm         */
/*                                                        */
/**********************************************************/

template<class T>
struct norm_traits;

namespace concepts_detail {

template <class T, bool scalar = std::is_arithmetic<T>::value>
struct norm_of_scalar_impl;

template <class T>
struct norm_of_scalar_impl<T, false>
{
    static const bool value = false;
    using norm_type         = void *;
    using squared_norm_type = void *;
};

template <class T>
struct norm_of_scalar_impl<T, true>
{
    static const bool value = true;
    using norm_type         = decltype(norm(*(T*)0));
    using squared_norm_type = decltype(squared_norm(*(T*)0));
};

template <class T, bool integral = std::is_integral<T>::value,
                   bool floating = std::is_floating_point<T>::value>
struct norm_of_array_elements_impl;

template <>
struct norm_of_array_elements_impl<void *, false, false>
{
    using norm_type         = void *;
    using squared_norm_type = void *;
};

template <class T>
struct norm_of_array_elements_impl<T, false, false>
{
    using norm_type         = typename norm_traits<T>::norm_type;
    using squared_norm_type = typename norm_traits<T>::squared_norm_type;
};

template <class T>
struct norm_of_array_elements_impl<T, true, false>
{
    static_assert(!std::is_same<T, char>::value,
       "'char' is not a numeric type, use 'signed char' or 'unsigned char'.");

    using norm_type         = double;
    using squared_norm_type = uint64_t;
};

template <class T>
struct norm_of_array_elements_impl<T, false, true>
{
    using norm_type         = double;
    using squared_norm_type = double;
};

template <>
struct norm_of_array_elements_impl<long double, false, true>
{
    using norm_type         = long double;
    using squared_norm_type = long double;
};

template <class ARRAY>
struct norm_of_vector_impl
{
    static void * test(...);

    template <class U>
    static typename U::value_type test(U*, typename U::value_type * = 0);

    using T = decltype(test((ARRAY*)0));

    static const bool value = !std::is_same<T, void*>::value;

    using norm_type         = typename norm_of_array_elements_impl<T>::norm_type;
    using squared_norm_type = typename norm_of_array_elements_impl<T>::squared_norm_type;
};

} // namespace concepts_detail

    /* norm_traits<T> implement the following default rules, which are
       designed to minimize the possibility of overflow:
        * T is a built-in type:
               norm_type is the result type of abs(),
               squared_norm_type is the result type of sq()
        * T is a container of 'long double' elements:
               norm_type and squared_norm_type are 'long double'
        * T is a container of another floating-point type:
               norm_type and squared_norm_type are 'double',
        * T is a container of integer elements:
               norm_type is 'double',
               squared_norm_type is 'uint64_t'
        * T is a container of some other type:
               norm_type is the element's norm type,
               squared_norm_type is the element's squared norm type
       Containers are recognized by having an embedded typedef 'value_type'.

       To change the behavior for a particular case or extend it to cases
       not covered here, simply specialize the norm_traits template.
    */
template<class U>
struct norm_traits
{
    using T = std::decay_t<U>;

    static_assert(!std::is_same<T, char>::value,
       "'char' is not a numeric type, use 'signed char' or 'unsigned char'.");

    using norm_of_scalar = concepts_detail::norm_of_scalar_impl<T>;
    using norm_of_vector = concepts_detail::norm_of_vector_impl<T>;

    static const bool value = norm_of_scalar::value || norm_of_vector::value;

    static_assert(value, "norm_traits<T> are undefined for type U.");

    using norm_type =
        typename std::conditional<norm_of_vector::value,
                    typename norm_of_vector::norm_type,
                    typename norm_of_scalar::norm_type>::type;

    using squared_norm_type =
        typename std::conditional<norm_of_vector::value,
                    typename norm_of_vector::squared_norm_type,
                    typename norm_of_scalar::squared_norm_type>::type;
};

template <class T>
using squared_norm_t = typename norm_traits<T>::squared_norm_type;

template <class T>
using norm_t = typename norm_traits<T>::norm_type;

} // namespace xt

#endif // XCONCEPTS_HPP
