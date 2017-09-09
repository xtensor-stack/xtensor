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

        /** @brief Concept checking macro (more redable than sfinae).

            The macro is used as the last argument in a template declaration.
            It must be followed by a static boolean expression in angle brackets.
            The template will only be included in overload resolution when
            this expression evaluates to 'true'.

            Example:
            \code
            template <class T,
                      XTENSOR_REQUIRE<std::is_arithmetic<T>::value>>
            T foo(T t)
            {...}
            \endcode
        */
    #define XTENSOR_REQUIRE typename Require = concept_check

    /**********************************************************/
    /*                                                        */
    /*                   iterator_concept                     */
    /*                                                        */
    /**********************************************************/

        /** @brief Traits class to check if a type is an iterator.

            This is useful in concept checking to make sure that a given template
            is only instantiated when the argument is an iterator.
            Currently, we apply the simple rule that class @tparam T
            is either a pointer or a C-array or has an embedded typedef
            'iterator_category'. More sophisticated checks can easily
            be added when needed.
        */
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

        /** @brief Result type of mixed arithmetic expressions.

            For example, it tells the user that <tt>unsigned char + unsigned char => int</tt>.
        */
    template <class T1, class T2 = T1>
    using promote_t = decltype(*(std::decay_t<T1>*)0 + *(std::decay_t<T2>*)0);

        /** @brief Result type of algebraic expressions.

            For example, it tells the user that <tt>sqrt(int) => double</tt>.
        */
    template <class T>
    using real_promote_t = concepts_detail::real_promote_t<T>;

        /** @brief Traits class to replace 'bool' with 'uint8_t' and keep everything else.

            This is useful for scientific computing, where a boolean mask array is
            usually implemented as an array of bytes.
        */
    template <class T>
    using bool_promote_t = typename std::conditional<std::is_same<T, bool>::value, uint8_t, T>::type;

    /**********************************************************/
    /*                                                        */
    /*       type inference for norm and squared norm         */
    /*                                                        */
    /**********************************************************/

    template<class T>
    struct norm_traits;

    template<class T>
    struct squared_norm_traits;

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
        using norm_type         = T;
        using squared_norm_type = decltype((*(T*)0) * (*(T*)0));
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
        using norm_type         = typename norm_traits<T>::type;
        using squared_norm_type = typename squared_norm_traits<T>::type;
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

    template<class U>
    struct norm_traits_base
    {
        using T = std::decay_t<U>;

        static_assert(!std::is_same<T, char>::value,
           "'char' is not a numeric type, use 'signed char' or 'unsigned char'.");

        using norm_of_scalar = norm_of_scalar_impl<T>;
        using norm_of_vector = norm_of_vector_impl<T>;

        static const bool value = norm_of_scalar::value || norm_of_vector::value;

        static_assert(value, "norm_traits<T> are undefined for type U.");
    };

    } // namespace concepts_detail

        /** @brief Traits class for the result type of the <tt>norm()</tt> function.

            Member 'type' defines the result of <tt>norm(t)</tt>, where <tt>t</tt>
            is of type @tparam T. It implements the following rules designed to
            minimize the potential for overflow:
                - @tparam T is an arithmetic type: 'type' is the result type of <tt>abs(t)</tt>.
                - @tparam T is a container of 'long double' elements: 'type' is <tt>long double</tt>.
                - @tparam T is a container of another arithmetic type: 'type' is <tt>double</tt>.
                - @tparam T is a container of some other type: 'type' is the element's norm type,

           Containers are recognized by having an embedded typedef 'value_type'.
           To change the behavior for a case not covered here, specialize the
           <tt>concepts_detail::norm_traits_base</tt> template.
        */
    template<class T>
    struct norm_traits
    : public concepts_detail::norm_traits_base<T>
    {
        using base_type = concepts_detail::norm_traits_base<T>;

        using type =
            typename std::conditional<base_type::norm_of_vector::value,
                        typename base_type::norm_of_vector::norm_type,
                        typename base_type::norm_of_scalar::norm_type>::type;
    };

        /** Abbreviation of 'typename norm_traits<T>::type'.
        */
    template <class T>
    using norm_t = typename norm_traits<T>::type;

        /** @brief Traits class for the result type of the <tt>squared_norm()</tt> function.

            Member 'type' defines the result of <tt>squared_norm(t)</tt>, where <tt>t</tt>
            is of type @tparam T. It implements the following rules designed to
            minimize the potential for overflow:
                - @tparam T is an arithmetic type: 'type' is the result type of <tt>t*t</tt>.
                - @tparam T is a container of 'long double' elements: 'type' is <tt>long double</tt>.
                - @tparam T is a container of another floating-point type: 'type' is <tt>double</tt>.
                - @tparam T is a container of integer elements: 'type' is <tt>uint64_t</tt>.
                - @tparam T is a container of some other type: 'type' is the element's squared norm type,

           Containers are recognized by having an embedded typedef 'value_type'.
           To change the behavior for a case not covered here, specialize the
           <tt>concepts_detail::norm_traits_base</tt> template.
        */
    template<class T>
    struct squared_norm_traits
    : public concepts_detail::norm_traits_base<T>
    {
        using base_type = concepts_detail::norm_traits_base<T>;

        using type =
            typename std::conditional<base_type::norm_of_vector::value,
                        typename base_type::norm_of_vector::squared_norm_type,
                        typename base_type::norm_of_scalar::squared_norm_type>::type;
    };

        /** Abbreviation of 'typename squared_norm_traits<T>::type'.
        */
    template <class T>
    using squared_norm_t = typename squared_norm_traits<T>::type;

} // namespace xt

#endif // XCONCEPTS_HPP
