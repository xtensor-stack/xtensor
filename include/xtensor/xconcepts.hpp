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

/*****************************************************
 * concept checking and type inference functionality *
 *****************************************************/

namespace xt
{

    /******************************************
     * XTENSOR_REQUIRE concept checking macro *
     ******************************************/

    struct concept_check_successful {};

    template <bool CONCEPTS>
    using concept_check = typename std::enable_if<CONCEPTS, concept_check_successful>::type;

        /** @brief Concept checking macro (more readable than sfinae).

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
    #define XTENSOR_REQUIRE typename = concept_check

    /********************
     * iterator_concept *
     ********************/

        /** @brief Traits class to check if a type is an iterator.

            This is useful in concept checking to make sure that a given template
            is only instantiated when the argument is an iterator.
            Currently, we apply the simple rule that class @tparam T
            is either a pointer or a C-array or has an embedded typedef
            'iterator_category'. More sophisticated checks can easily
            be added when needed.

            If @tparam T is indeed an iterator, the class' <tt>value</tt> member
            is <tt>true</tt>:
            \code
            template <class T,
                      XTENSOR_REQUIRE<iterator_concept<T>::value>>
            T foo(T t)
            {...}
            \endcode
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
} // namespace xt

#endif // XCONCEPTS_HPP
