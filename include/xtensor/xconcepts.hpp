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

namespace xt
{

/**********************************************************/
/*                                                        */
/*        XTENSOR_REQUIRE concept checking macro          */
/*                                                        */
/**********************************************************/

struct require_ok {};

template <bool CONCEPTS>
using ConceptCheck = typename std::enable_if<CONCEPTS, require_ok>::type;

#define XTENSOR_REQUIRE typename Require = ConceptCheck

/**********************************************************/
/*                                                        */
/*                    TinyArrayConcept                    */
/*                                                        */
/**********************************************************/

struct TinyArrayTag {};

    // TinyArrayConcept refers to TinyArrayBase and TinyArray.
    // By default, 'ARRAY' fulfills the TinyArrayConcept if it is derived
    // from TinyArrayTag.
    //
    // Alternatively, one can partially specialize TinyArrayConcept.
template <class ARRAY>
struct TinyArrayConcept
{
    static const bool value = std::is_base_of<TinyArrayTag, ARRAY>::value;
};


/**********************************************************/
/*                                                        */
/*                    IteratorConcept                     */
/*                                                        */
/**********************************************************/

    // currently, we apply only the simple rule that class T
    // must be a pointer or array or has an embedded typedef
    // 'iterator_category'. More sophisticated checks should
    // be added when needed.
template <class T>
struct IteratorConcept
{
    typedef typename std::decay<T>::type V;

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
