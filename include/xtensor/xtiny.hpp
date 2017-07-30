/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_XTINY_HPP
#define XTENSOR_XTINY_HPP

#include "xtags.hpp"
#include "xconcepts.hpp"
#include "xexception.hpp"
#include "xmath.hpp"
#include <iosfwd>
#include <algorithm>
#include <memory>
#include <iterator>
#include <utility>

#ifdef XTENSOR_CHECK_BOUNDS
    #define XTENSOR_ASSERT_INSIDE(array, diff) \
      xtensor_precondition(diff >= 0 && diff < array.size(), "Index out of bounds")
#else
    #define XTENSOR_ASSERT_INSIDE(array, diff)
#endif

namespace xt {

    /** \brief The general type of array indices.

        Note that this is a signed type, so that negative indices
        and index differences work as intuitively expected.
    */
using index_t = std::ptrdiff_t;

template <class VALUETYPE, int M=runtime_size, int ... N>
class tiny_array;

template <class VALUETYPE, int M=runtime_size, int ... N>
class tiny_array_view;

namespace detail  {

template<class T>
struct may_use_uninitialized_memory
{
    static const bool value = std::is_scalar<T>::value || std::is_pod<T>::value;
};

template<class T, int M, int ... N>
struct may_use_uninitialized_memory<tiny_array<T, M, N...>>
{
    static const bool value = may_use_uninitialized_memory<T>::value;
};

template<class T>
struct may_use_uninitialized_memory<tiny_array<T, runtime_size>>
{
    static const bool value = false;
};

template <index_t LEVEL, int ... N>
struct tiny_shape_helper;

template <index_t LEVEL, int N, int ... REST>
struct tiny_shape_helper<LEVEL, N, REST...>
{
    static_assert(N >= 0, "tiny_array_base(): array must have non-negative shape.");
    using next_type = tiny_shape_helper<LEVEL+1, REST...>;

    static const index_t level      = LEVEL;
    static const index_t stride     = next_type::total_size;
    static const index_t total_size = N * stride;
    static const index_t alloc_size = total_size;

    static index_t offset(index_t const * coord)
    {
        return stride*coord[level] + next_type::offset(coord);
    }

    template <class ... V>
    static index_t offset(index_t i, V...rest)
    {
        return stride*i + next_type::offset(rest...);
    }
};

template <index_t LEVEL, int N>
struct tiny_shape_helper<LEVEL, N>
{
    static_assert(N >= 0, "tiny_array_base(): array must have non-negative shape.");
    static const index_t level      = LEVEL;
    static const index_t stride     = 1;
    static const index_t total_size = N;
    static const index_t alloc_size = total_size;

    static index_t offset(index_t const * coord)
    {
        return coord[level];
    }

    static index_t offset(index_t i)
    {
        return i;
    }
};

template <index_t LEVEL>
struct tiny_shape_helper<LEVEL, 0>
{
    static const index_t level      = LEVEL;
    static const index_t stride     = 1;
    static const index_t total_size = 0;
    static const index_t alloc_size = 1;

    static index_t offset(index_t const * coord)
    {
        return coord[level];
    }

    static index_t offset(index_t i)
    {
        return i;
    }
};

template <int ... N>
struct tiny_size_helper
{
    static const index_t value = tiny_shape_helper<0, N...>::total_size;
    static const index_t ndim  = sizeof...(N);
};

template <int N0, int ... N>
struct tiny_array_is_static
{
    static const int ndim = sizeof...(N)+1;
    static const bool value = ndim > 1 || N0 != runtime_size;
};

} // namespace detail

#define XTENSOR_ASSERT_RUNTIME_SIZE(SHAPE, PREDICATE, MESSAGE) \
    if(detail::tiny_array_is_static<SHAPE>::value) {} else \
        xtensor_precondition(PREDICATE, MESSAGE)

/********************************************************/
/*                                                      */
/*                    tiny_array_base                   */
/*                                                      */
/********************************************************/

/** \brief Base class for fixed size vectors and matrices.

    This class contains functionality shared by
    \ref tiny_array and \ref tiny_array_view, and enables these classes
    to be freely mixed within expressions. It is typically not used directly.

    <b>\#include</b> \<vigra/tinyarray.hxx\><br>
    Namespace: vigra
**/
template <class VALUETYPE, class DERIVED, int ... N>
class tiny_array_base
: public tiny_array_tag
{
  protected:
    using shape_helper = detail::tiny_shape_helper<0, N...>;

    static const bool derived_is_view = !std::is_same<DERIVED, tiny_array<VALUETYPE, N...> >::value;
    using data_array_type = typename std::conditional<derived_is_view,
                                                VALUETYPE *,
                                                VALUETYPE[shape_helper::alloc_size]>::type;

    template <int LEVEL, class ... V2>
    void init_impl(VALUETYPE v1, V2... v2)
    {
        data_[LEVEL] = v1;
        init_impl<LEVEL+1>(v2...);
    }

    template <int LEVEL>
    void init_impl(VALUETYPE v1)
    {
        data_[LEVEL] = v1;
    }

    template <class ITERATOR,
              XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
    void init_impl(ITERATOR i)
    {
        for(index_t k=0; k < static_size; ++k, ++i)
            data_[k] = static_cast<VALUETYPE>(*i);
    }

  public:

    template <class NEW_VALUETYPE>
    using as_type = tiny_array<NEW_VALUETYPE, N...>;

    using value_type             = VALUETYPE;
    using const_value_type       = typename std::add_const<VALUETYPE>::type;
    using reference              = value_type &;
    using const_reference        = const_value_type &;
    using pointer                = value_type *;
    using const_pointer          = const_value_type *;
    using iterator               = value_type *;
    using const_iterator         = const_value_type *;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using index_type             = tiny_array<index_t, sizeof...(N)>;

    static const index_t static_ndim  = sizeof...(N);
    static const index_t static_size  = shape_helper::total_size;
    static const bool may_use_uninitialized_memory =
                                   detail::may_use_uninitialized_memory<VALUETYPE>::value;

    // constructors

    constexpr tiny_array_base(tiny_array_base const &) = default;

  protected:

    tiny_array_base(skip_initialization_tag)
    {}

    // constructors to be used by tiny_array

    template <class OTHER, class OTHER_DERIVED>
    tiny_array_base(tiny_array_base<OTHER, OTHER_DERIVED, N...> const & other)
    {
        xtensor_precondition(size() == other.size(),
                      "tiny_array_base(): shape mismatch.");
        for(int i=0; i<static_size; ++i)
            data_[i] = static_cast<value_type>(other[i]);
    }

    // constructor for zero or one argument
    explicit tiny_array_base(value_type v = value_type())
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = v;
    }

    // // constructor for two or more arguments
    // template <class ... V>
    // constexpr tiny_array_base(value_type v0, value_type v1, V ... v)
    // : data_{VALUETYPE(v0), VALUETYPE(v1), VALUETYPE(v)...}
    // {
        // static_assert(sizeof...(V)+2 == static_size,
                      // "tiny_array_base(): number of constructor arguments contradicts size().");
    // }

    template <class U>
    explicit tiny_array_base(U const * u)
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = static_cast<value_type>(u[i]);
    }

    template <class U>
    tiny_array_base(U const * u, reverse_copy_tag)
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = static_cast<value_type>(u[static_size-1-i]);
    }

        // for compatibility with tiny_array_base<..., runtime_size>
    template <class U>
    tiny_array_base(U const * u, U const * /* end */, reverse_copy_tag)
    : tiny_array_base(u, copy_reversed)
    {}

  public:

    // assignment

    tiny_array_base & operator=(tiny_array_base const &) = default;

    tiny_array_base & operator=(value_type v)
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = v;
        return *this;
    }

    tiny_array_base & operator=(value_type const (&v)[static_size])
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = v[i];
        return *this;
    }

    template <class OTHER, class OTHER_DERIVED>
    tiny_array_base & operator=(tiny_array_base<OTHER, OTHER_DERIVED, N...> const & other)
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = static_cast<value_type>(other[i]);
        return *this;
    }

    template <class OTHER, class OTHER_DERIVED>
    tiny_array_base & operator=(tiny_array_base<OTHER, OTHER_DERIVED, runtime_size> const & other)
    {
        xtensor_precondition(size() == other.size(),
            "tiny_array_base::operator=(): size mismatch.");
        for(int i=0; i<size(); ++i)
            data_[i] = static_cast<value_type>(other[i]);
        return *this;
    }

    template <class OTHER, class OTHER_DERIVED, int ... M>
    constexpr bool
    is_same_shape(tiny_array_base<OTHER, OTHER_DERIVED, M...> const &) const
    {
        return false;
    }

    template <class OTHER, class OTHER_DERIVED>
    constexpr bool
    is_same_shape(tiny_array_base<OTHER, OTHER_DERIVED, N...> const &) const
    {
        return true;
    }

    template <class OTHER, class OTHER_DERIVED>
    bool
    is_same_shape(tiny_array_base<OTHER, OTHER_DERIVED, runtime_size> const & other) const
    {
        return sizeof...(N) == 1 && size() == other.size();
    }

    DERIVED & init(value_type v = value_type())
    {
        for(int i=0; i<static_size; ++i)
            data_[i] = v;
        return static_cast<DERIVED &>(*this);
    }

    template <class ... V>
    DERIVED & init(value_type v0, value_type v1, V... v)
    {
        static_assert(sizeof...(V)+2 == static_size,
                      "tiny_array_base::init(): wrong number of arguments.");
        init_impl<0>(v0, v1, v...);
        return static_cast<DERIVED &>(*this);
    }

    template <class Iterator>
    DERIVED & init(Iterator first, Iterator end)
    {
        index_t range = std::distance(first, end);
        if(static_size < range)
            range = static_size;
        for(index_t i=0; i<range; ++i, ++first)
            data_[i] = static_cast<value_type>(*first);
        return static_cast<DERIVED &>(*this);
    }

    // index access

    reference operator[](index_t i)
    {
        return data_[i];
    }

    constexpr const_reference operator[](index_t i) const
    {
        return data_[i];
    }

    reference at(index_t i)
    {
        if(i < 0 || i >= static_size)
            throw std::out_of_range("tiny_array_base::at()");
        return data_[i];
    }

    const_reference at(index_t i) const
    {
        if(i < 0 || i >= static_size)
            throw std::out_of_range("tiny_array_base::at()");
        return data_[i];
    }

    reference operator[](index_t const (&i)[static_ndim])
    {
        return data_[shape_helper::offset(i)];
    }

    constexpr const_reference operator[](index_t const (&i)[static_ndim]) const
    {
        return data_[shape_helper::offset(i)];
    }

    reference at(index_t const (&i)[static_ndim])
    {
        return at(shape_helper::offset(i));
    }

    const_reference at(index_t const (&i)[static_ndim]) const
    {
        return at(shape_helper::offset(i));
    }

    reference operator[](index_type const & i)
    {
        return data_[shape_helper::offset(i.data())];
    }

    constexpr const_reference operator[](index_type const & i) const
    {
        return data_[shape_helper::offset(i.data())];
    }

    reference at(index_type const & i)
    {
        return at(shape_helper::offset(i.data()));
    }

    const_reference at(index_type const & i) const
    {
        return at(shape_helper::offset(i.data()));
    }

    template <class ... V>
    reference operator()(V...v)
    {
        static_assert(sizeof...(V) == static_ndim,
                      "tiny_array_base::operator(): wrong number of arguments.");
        return data_[shape_helper::offset(v...)];
    }

    template <class ... V>
    constexpr const_reference operator()(V...v) const
    {
        static_assert(sizeof...(V) == static_ndim,
                      "tiny_array_base::operator(): wrong number of arguments.");
        return data_[shape_helper::offset(v...)];
    }

        /** Get a view to the subarray with length <tt>(TO-FROM)</tt> starting at <tt>FROM</tt>.
            The bounds must fullfill <tt>0 <= FROM < TO <= static_size</tt>.
            Only available if <tt>static_ndim == 1</tt>.
        */
    template <int FROM, int TO>
    tiny_array_view<value_type, TO-FROM>
    subarray() const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_base::subarray(): array must be 1-dimensional.");
        static_assert(FROM >= 0 && FROM < TO && TO <= static_size,
            "tiny_array_base::subarray(): range out of bounds.");
        return tiny_array_view<value_type, TO-FROM>(const_cast<VALUETYPE*>(data_)+FROM);
    }

    tiny_array_view<value_type, runtime_size>
    subarray(index_t FROM, index_t TO) const
    {
        xtensor_precondition(FROM >= 0 && FROM < TO && TO <= static_size,
                      "tiny_array_base::subarray(): range out of bounds.");
        return tiny_array_view<value_type, runtime_size>(TO-FROM, const_cast<VALUETYPE*>(data_)+FROM);
    }

    template<int M = static_ndim>
    tiny_array<value_type, static_size-1>
    erase(index_t m) const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_base::erase(): array must be 1-dimensional.");
        xtensor_precondition(m >= 0 && m < static_size, "tiny_array::erase(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+").");
        tiny_array<value_type, static_size-1> res(static_size-1, dont_init);
        for(int k=0; k<m; ++k)
            res[k] = data_[k];
        for(index_t k=m; k<static_size-1; ++k)
            res[k] = data_[k+1];
        return res;
    }

    template<int M = static_ndim>
    tiny_array<value_type, static_size-1>
    pop_front() const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_base::pop_front(): array must be 1-dimensional.");
        return erase(0);
    }

    template<int M = static_ndim>
    tiny_array<value_type, static_size-1>
    pop_back() const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_base::pop_back(): array must be 1-dimensional.");
        return erase(size()-1);
    }

    template<int M = static_ndim>
    tiny_array<value_type, static_size+1>
    insert(index_t m, value_type v) const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array_base::insert(): array must be 1-dimensional.");
        xtensor_precondition(m >= 0 && m <= static_size, "tiny_array::insert(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+"].");
        tiny_array<value_type, static_size+1> res(dont_init);
        for(int k=0; k<m; ++k)
            res[k] = data_[k];
        res[m] = v;
        for(index_t k=m; k<static_size; ++k)
            res[k+1] = data_[k];
        return res;
    }

    template <class V, class D, int M>
    inline
    tiny_array<value_type, static_size>
    transpose(tiny_array_base<V, D, M> const & permutation) const
    {
        static_assert(sizeof...(N) == 1,
            "tiny_array::transpose(): only allowed for 1-dimensional arrays.");
        static_assert(M == static_size || M == runtime_size,
            "tiny_array::transpose(): size mismatch.");
        XTENSOR_ASSERT_RUNTIME_SIZE(M, size() == 0 || size() == permutation.size(),
            "tiny_array::transpose(): size mismatch.");
        tiny_array<value_type, static_size> res(dont_init);
        for(int k=0; k < size(); ++k)
        {
            XTENSOR_ASSERT_MSG(permutation[k] >= 0 && permutation[k] < size(),
                "transpose():  Permutation index out of bounds");
            res[k] = (*this)[permutation[k]];
        }
        return res;
    }

    // boiler plate

    iterator begin() { return data_; }
    iterator end()   { return data_ + static_size; }
    const_iterator begin() const { return data_; }
    const_iterator end()   const { return data_ + static_size; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend()   const { return data_ + static_size; }

    reverse_iterator rbegin() { return reverse_iterator(data_ + static_size); }
    reverse_iterator rend()   { return reverse_iterator(data_); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(data_ + static_size); }
    const_reverse_iterator rend()   const { return const_reverse_iterator(data_); }
    const_reverse_iterator crbegin() const { return const_reverse_iterator(data_ + static_size); }
    const_reverse_iterator crend()   const { return const_reverse_iterator(data_); }

    pointer data() { return data_; }
    const_pointer data() const { return data_; }

    reference front() { return data_[0]; }
    reference back()  { return data_[static_size-1]; }
    constexpr const_reference front() const { return data_[0]; }
    constexpr const_reference back()  const { return data_[static_size-1]; }

    constexpr bool       empty() const { return static_size == 0; }
    constexpr index_t size()  const { return static_size; }
    constexpr index_t max_size()  const { return static_size; }
    constexpr index_type shape() const { return index_type{ N... }; }
    constexpr index_t ndim()  const { return static_ndim; }

    tiny_array_base & reverse()
    {
        using std::swap;
        index_t i=0, j=size()-1;
        while(i < j)
             swap(data_[i++], data_[j--]);
        return *this;
    }

    void swap(tiny_array_base & other)
    {
        using std::swap;
        for(int k=0; k<static_size; ++k)
        {
            swap(data_[k], other[k]);
        }
    }

    template <class OTHER, class OTHER_DERIVED>
    void swap(tiny_array_base<OTHER, OTHER_DERIVED, N...> & other)
    {
        for(int k=0; k<static_size; ++k)
        {
            promote_t<value_type, OTHER> t = data_[k];
            data_[k] = static_cast<value_type>(other[k]);
            other[k] = static_cast<OTHER>(t);
        }
    }

        /// factory function for fixed-size unit matrix
    template <int SIZE>
    static inline
    tiny_array<value_type, SIZE, SIZE>
    eye()
    {
        tiny_array<value_type, SIZE, SIZE> res;
        for(int k=0; k<SIZE; ++k)
            res(k,k) = 1;
        return res;
    }

        /// factory function for the fixed-size k-th unit vector
    template <int SIZE=static_size>
    static inline
    tiny_array<value_type, SIZE>
    unit_vector(index_t k)
    {
        tiny_array<value_type, SIZE> res;
        res(k) = 1;
        return res;
    }

        /// factory function for the k-th unit vector
        // (for compatibility with tiny_array<..., runtime_size>)
    static inline
    tiny_array<value_type, static_size>
    unit_vector(tags::size_proxy const & size, index_t k)
    {
        XTENSOR_ASSERT_MSG(size.value == static_size,
            "tiny_array::unit_vector(): size mismatch.");
        tiny_array<value_type, static_size> res;
        res(k) = 1;
        return res;
    }

        /// factory function for fixed-size linear sequence starting at <tt>start</tt> with stepsize <tt>step</tt>
    static inline
    tiny_array<value_type, N...>
    linear_sequence(value_type start = value_type(), value_type step = value_type(1))
    {
        tiny_array<value_type, N...> res(dont_init);
        for(index_t k=0; k < static_size; ++k, start += step)
            res[k] = start;
        return res;
    }

        /// factory function for fixed-size linear sequence ending at <tt>end-1</tt>
    static inline
    tiny_array<value_type, N...>
    range(value_type end)
    {
        value_type start = end - static_cast<value_type>(static_size);
        tiny_array<value_type, N...> res(dont_init);
        for(index_t k=0; k < static_size; ++k, ++start)
            res[k] = start;
        return res;
    }

  protected:
    data_array_type data_;
};

/********************************************************/
/*                                                      */
/*                tiny_array_base output                */
/*                                                      */
/********************************************************/

template <class T, class DERIVED, int ... N>
std::ostream & operator<<(std::ostream & o, tiny_array_base<T, DERIVED, N...> const & v)
{
    o << "{";
    if(v.size() > 0)
        o << promote_t<T>(v[0]);
    for(int i=1; i < v.size(); ++i)
        o << ", " << promote_t<T>(v[i]);
    o << "}";
    return o;
}

template <class T, class DERIVED, int N1, int N2>
std::ostream & operator<<(std::ostream & o, tiny_array_base<T, DERIVED, N1, N2> const & v)
{
    o << "{";
    for(int i=0; N2>0 && i<N1; ++i)
    {
        if(i > 0)
            o << ",\n ";
        o << promote_t<T>(v(i,0));
        for(int j=1; j<N2; ++j)
        {
            o << ", " << promote_t<T>(v(i, j));
        }
    }
    o << "}";
    return o;
}

/********************************************************/
/*                                                      */
/*         tiny_array_base<..., runtime_size>           */
/*                                                      */
/********************************************************/

/** \brief Specialization of tiny_array_base for dynamic arrays.

    This class contains functionality shared by
    \ref tiny_array and \ref tiny_array_view, and enables these classes
    to be freely mixed within expressions. It is typically not used directly.

    <b>\#include</b> \<vigra/tinyarray.hxx\><br>
    Namespace: vigra
**/
template <class VALUETYPE, class DERIVED>
class tiny_array_base<VALUETYPE, DERIVED, runtime_size>
: public tiny_array_tag
{
  public:

    template <class NEW_VALUETYPE>
    using as_type = tiny_array<NEW_VALUETYPE, runtime_size>;

    using value_type             = VALUETYPE;
    using const_value_type       = typename std::add_const<VALUETYPE>::type;
    using reference              = value_type &;
    using const_reference        = const_value_type &;
    using pointer                = value_type *;
    using const_pointer          = const_value_type *;
    using iterator               = value_type *;
    using const_iterator         = const_value_type *;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using index_type             = index_t;

    static const index_t static_size  = runtime_size;
    static const index_t static_ndim  = 1;
    static const bool may_use_uninitialized_memory =
                                   detail::may_use_uninitialized_memory<VALUETYPE>::value;

  protected:

    template <int LEVEL, class ... V2>
    void init_impl(VALUETYPE v1, V2... v2)
    {
        data_[LEVEL] = v1;
        init_impl<LEVEL+1>(v2...);
    }

    template <int LEVEL>
    void init_impl(VALUETYPE v1)
    {
        data_[LEVEL] = v1;
    }

  public:

    tiny_array_base(index_t size=0, pointer data=0)
    : size_(size)
    , data_(data)
    {}

    tiny_array_base(tiny_array_base const &) = default;

    // assignment

    tiny_array_base & operator=(value_type v)
    {
        for(int i=0; i<size_; ++i)
            data_[i] = v;
        return *this;
    }

    tiny_array_base & operator=(tiny_array_base const & rhs)
    {
        xtensor_precondition(size_ == rhs.size(),
            "tiny_array_base::operator=(): size mismatch.");
        for(int i=0; i<size_; ++i)
            data_[i] = rhs[i];
        return *this;
    }

    template <class OTHER, class OTHER_DERIVED, int N>
    tiny_array_base & operator=(tiny_array_base<OTHER, OTHER_DERIVED, N> const & other)
    {
        xtensor_precondition(size_ == other.size(),
            "tiny_array_base::operator=(): size mismatch.");
        for(int i=0; i<size_; ++i)
            data_[i] = static_cast<value_type>(other[i]);
        return *this;
    }

    template <class OTHER, class OTHER_DERIVED, int ... M>
    bool
    is_same_shape(tiny_array_base<OTHER, OTHER_DERIVED, M...> const & other) const
    {
        return sizeof...(M) == 1 && size() == other.size();;
    }

    template <class OTHER, class OTHER_DERIVED>
    bool
    is_same_shape(tiny_array_base<OTHER, OTHER_DERIVED, runtime_size> const & other) const
    {
        return size() == other.size();
    }

    DERIVED & init(value_type v = value_type())
    {
        for(int i=0; i<size_; ++i)
            data_[i] = v;
        return static_cast<DERIVED &>(*this);
    }

    template <class ... V>
    DERIVED & init(value_type v0, value_type v1, V... v)
    {
        xtensor_precondition(sizeof...(V)+2 == size_,
                      "tiny_array_base::init(): wrong number of arguments.");
        init_impl<0>(v0, v1, v...);
        return static_cast<DERIVED &>(*this);
    }

    template <class Iterator>
    DERIVED & init(Iterator first, Iterator end)
    {
        index_t range = std::distance(first, end);
        if(size_ < range)
            range = size_;
        for(index_t i=0; i<range; ++i, ++first)
            data_[i] = static_cast<value_type>(*first);
        return static_cast<DERIVED &>(*this);
    }

    template <class V>
    DERIVED & init(std::initializer_list<V> l)
    {
        return init(l.begin(), l.end());
    }

    // index access

    reference operator[](index_t i)
    {
        return data_[i];
    }

    constexpr const_reference operator[](index_t i) const
    {
        return data_[i];
    }

    reference at(index_t i)
    {
        if(i < 0 || i >= size_)
            throw std::out_of_range("tiny_array_base::at()");
        return data_[i];
    }

    const_reference at(index_t i) const
    {
        if(i < 0 || i >= size_)
            throw std::out_of_range("tiny_array_base::at()");
        return data_[i];
    }

        /** Get a view to the subarray with length <tt>(TO-FROM)</tt> starting at <tt>FROM</tt>.
            The bounds must fullfill <tt>0 <= FROM < TO <= size_</tt>.
        */
    template <int FROM, int TO>
    tiny_array_view<value_type, TO-FROM>
    subarray() const
    {
        static_assert(FROM >= 0 && FROM < TO,
            "tiny_array_base::subarray(): range out of bounds.");
        xtensor_precondition(TO <= size_,
            "tiny_array_base::subarray(): range out of bounds.");
        return tiny_array_view<value_type, TO-FROM>(data_+FROM);
    }

        /** Get a view to the subarray with length <tt>(TO-FROM)</tt> starting at <tt>FROM</tt>.
            The bounds must fullfill <tt>0 <= FROM < TO <= size_</tt>.
        */
    tiny_array_view<value_type, runtime_size>
    subarray(index_t FROM, index_t TO) const
    {
        xtensor_precondition(FROM >= 0 && FROM < TO && TO <= size_,
            "tiny_array_base::subarray(): range out of bounds.");
        return tiny_array_view<value_type, runtime_size>(TO-FROM, const_cast<pointer>(data_)+FROM);
    }


    tiny_array<value_type, runtime_size>
    erase(index_t m) const
    {
        xtensor_precondition(m >= 0 && m < size(), "tiny_array::erase(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+").");
        tiny_array<value_type, runtime_size> res(size()-1, dont_init);
        for(int k=0; k<m; ++k)
            res[k] = data_[k];
        for(index_t k=m+1; k<size(); ++k)
            res[k-1] = data_[k];
        return res;
    }

    tiny_array<value_type, runtime_size>
    pop_front() const
    {
        return erase(0);
    }

    tiny_array<value_type, runtime_size>
    pop_back() const
    {
        return erase(size()-1);
    }

    tiny_array<value_type, runtime_size>
    insert(index_t m, value_type v) const
    {
        xtensor_precondition(m >= 0 && m <= size(), "tiny_array::insert(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+"].");
        tiny_array<value_type, runtime_size> res(size()+1, dont_init);
        for(int k=0; k<m; ++k)
            res[k] = data_[k];
        res[m] = v;
        for(index_t k=m; k<size(); ++k)
            res[k+1] = data_[k];
        return res;
    }

    template <class V, class D, int M>
    inline
    tiny_array<value_type, runtime_size>
    transpose(tiny_array_base<V, D, M> const & permutation) const
    {
        xtensor_precondition(size() == 0 || size() == permutation.size(),
            "tiny_array::transpose(): size mismatch.");
        tiny_array<value_type, runtime_size> res(size(), dont_init);
        for(index_t k=0; k < size(); ++k)
        {
            XTENSOR_ASSERT_MSG(permutation[k] >= 0 && permutation[k] < size(),
                "transpose():  Permutation index out of bounds");
            res[k] = (*this)[permutation[k]];
        }
        return res;
    }

    // boiler plate

    iterator begin() { return data_; }
    iterator end()   { return data_ + size_; }
    const_iterator begin() const { return data_; }
    const_iterator end()   const { return data_ + size_; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend()   const { return data_ + size_; }

    reverse_iterator rbegin() { return reverse_iterator(data_ + size_); }
    reverse_iterator rend()   { return reverse_iterator(data_); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator(data_ + size_); }
    const_reverse_iterator rend()   const { return const_reverse_iterator(data_); }
    const_reverse_iterator crbegin() const { return const_reverse_iterator(data_ + size_); }
    const_reverse_iterator crend()   const { return const_reverse_iterator(data_); }

    pointer data() { return data_; }
    const_pointer data() const { return data_; }

    reference front() { return data_[0]; }
    reference back()  { return data_[size_-1]; }
    const_reference front() const { return data_[0]; }
    const_reference back()  const { return data_[size_-1]; }

    bool       empty() const { return size_ == 0; }
    index_t size()  const { return size_; }
    index_t max_size()  const { return size_; }
    index_t ndim()  const { return static_ndim; }

    tiny_array_base & reverse()
    {
        using std::swap;
        index_t i=0, j=size_-1;
        while(i < j)
             swap(data_[i++], data_[j--]);
        return *this;
    }

    void swap(tiny_array_base & other)
    {
        using std::swap;
        swap(size_, other.size_);
        swap(data_, other.data_);
    }

        /// factory function for the fixed-size k-th unit vector
    static inline
    tiny_array<value_type, runtime_size>
    unit_vector(tags::size_proxy const & size, index_t k)
    {
        tiny_array<value_type, runtime_size> res(size.value);
        res[k] = 1;
        return res;
    }

        /// factory function for a linear sequence from <tt>begin</tt> to <tt>end</tt>
        /// (exclusive) with stepsize <tt>step</tt>
    static inline
    tiny_array<value_type, runtime_size>
    range(value_type begin,
          value_type end,
          value_type step = value_type(1))
    {
        using namespace cmath;
        xtensor_precondition(step != 0,
            "tiny_array::range(): step must be non-zero.");
        xtensor_precondition((step > 0 && begin <= end) || (step < 0 && begin >= end),
            "tiny_array::range(): sign mismatch between step and (end-begin).");
        index_t size = floor((abs(end-begin+step)-1)/abs(step));
        tiny_array<value_type, runtime_size> res(size, dont_init);
        for(index_t k=0; k < size; ++k, begin += step)
            res[k] = begin;
        return res;
    }

        /// factory function for a linear sequence from 0 to <tt>end</tt>
        /// (exclusive) with stepsize 1
    static inline
    tiny_array<value_type, runtime_size>
    range(value_type end)
    {
        xtensor_precondition(end >= 0,
            "tiny_array::range(): end must be non-negative.");
        tiny_array<value_type, runtime_size> res(end, dont_init);
        auto begin = value_type();
        for(index_t k=0; k < res.size(); ++k, ++begin)
            res[k] = begin;
        return res;
    }

  protected:
    index_t size_;
    pointer data_;
};

/********************************************************/
/*                                                      */
/*                       tiny_array                     */
/*                                                      */
/********************************************************/

/** \brief Class for fixed size arrays.
    \ingroup RangesAndPoints

    This class contains an array of the specified VALUETYPE with
    (possibly multi-dimensional) shape given by the sequence <tt>index_t ... N</tt>.
    The interface conforms to STL vector, except that there are no functions
    that change the size of a tiny_array.

    \ref TinyArrayOperators "Arithmetic operations"
    on TinyArrays are defined as component-wise applications of these
    operations.

    See also:<br>
    <UL style="list-style-image:url(documents/bullet.gif)">
        <LI> \ref vigra::tiny_array_base
        <LI> \ref vigra::tiny_array_view
        <LI> \ref TinyArrayOperators
    </UL>

    <b>\#include</b> \<vigra/tiny_array.hxx\><br>
    Namespace: vigra
**/
template <class VALUETYPE, int M, int ... N>
class tiny_array
: public tiny_array_base<VALUETYPE, tiny_array<VALUETYPE, M, N...>, M, N...>
{
  public:
    using base_type = tiny_array_base<VALUETYPE, tiny_array<VALUETYPE, M, N...>, M, N...>;
    using value_type = VALUETYPE;
    static const int static_size = base_type::static_size;

    explicit constexpr
    tiny_array()
    : base_type(value_type())
    {}

    explicit
    tiny_array(skip_initialization_tag)
    : base_type(dont_init)
    {}

        // This constructor would allow construction with round brackets, e.g.:
        //     tiny_array<int, 1> a(2);
        // However, this may lead to bugs when fixed-size arrays are mixed with
        // runtime_size arrays, where
        //     tiny_array<int, runtime_size> a(2);
        // constructs an array of length 2 with initial value 0. To avoid such bugs,
        // construction is restricted to curly braces:
        //     tiny_array<int, 1> a{2};
        //
    // template <class ... V>
    // constexpr tiny_array(value_type v0, V... v)
    // : base_type(v0, v...)
    // {}

    template <class V>
    tiny_array(std::initializer_list<V> v)
    : base_type(dont_init)
    {
        if(v.size() == 1)
            base_type::init(static_cast<value_type>(*v.begin()));
        else if(v.size() == static_size)
            base_type::init_impl(v.begin());
        else
            xtensor_precondition(false,
                "tiny_array(std::initializer_list<V>): wrong initialization size.");
    }

        // for compatibility with tiny_array<VALUETYPE, runtime_size>
    explicit
    tiny_array(tags::size_proxy const & size,
              value_type const & v = value_type())
    : base_type(v)
    {
        XTENSOR_ASSERT_MSG(size.value == static_size,
            "tiny_array(size): size argument conflicts with array length.");
    }

        // for compatibility with tiny_array<VALUETYPE, runtime_size>
    tiny_array(tags::size_proxy const & size, skip_initialization_tag)
    : base_type(dont_init)
    {
        XTENSOR_ASSERT_MSG(size.value == static_size,
            "tiny_array(size): size argument conflicts with array length.");
    }

        // for compatibility with tiny_array<VALUETYPE, runtime_size>
    tiny_array(index_t size, skip_initialization_tag)
    : base_type(dont_init)
    {
        XTENSOR_ASSERT_MSG(size == static_size,
            "tiny_array(size): size argument conflicts with array length.");
    }

    constexpr tiny_array(tiny_array const &) = default;

    template <class OTHER, class DERIVED>
    tiny_array(tiny_array_base<OTHER, DERIVED, M, N...> const & other)
    : base_type(other)
    {}

    template <class OTHER, class DERIVED>
    tiny_array(tiny_array_base<OTHER, DERIVED, runtime_size> const & other)
    : base_type(dont_init)
    {
        if(other.size() == 0)
        {
            this->init(value_type());
        }
        else if(this->size() != 0)
        {
            xtensor_precondition(this->size() == other.size(),
                "tiny_array(): shape mismatch.");
            this->init_impl(other.begin());
        }
    }

    explicit tiny_array(value_type const (&u)[1])
    : base_type(*u)
    {}

    template <class U>
    explicit tiny_array(U const (&u)[1])
    : base_type(static_cast<value_type>(*u))
    {}

    template <class U, int S=static_size,
              XTENSOR_REQUIRE<S != 1>>
    explicit tiny_array(U const (&u)[static_size])
    : base_type(u)
    {}

    template <class U,
              XTENSOR_REQUIRE<iterator_concept<U>::value> >
    explicit tiny_array(U u, U /* end */ = U())
    : base_type(u)
    {}

    template <class U,
              XTENSOR_REQUIRE<iterator_concept<U>::value> >
    tiny_array(U u, reverse_copy_tag)
    : base_type(u, copy_reversed)
    {}

        // for compatibility with tiny_array<..., runtime_size>
    template <class U,
              XTENSOR_REQUIRE<iterator_concept<U>::value> >
    tiny_array(U u, U end, reverse_copy_tag)
    : base_type(u, copy_reversed)
    {}

    tiny_array & operator=(tiny_array const &) = default;

    tiny_array & operator=(value_type v)
    {
        base_type::operator=(v);
        return *this;
    }

    tiny_array & operator=(value_type const (&v)[static_size])
    {
        base_type::operator=(v);
        return *this;
    }

    template <class OTHER, class OTHER_DERIVED>
    tiny_array & operator=(tiny_array_base<OTHER, OTHER_DERIVED, M, N...> const & other)
    {
        base_type::operator=(other);
        return *this;
    }
};

/********************************************************/
/*                                                      */
/*            tiny_array<..., runtime_size>             */
/*                                                      */
/********************************************************/

/** \brief Specialization of tiny_array for dynamic arrays.
    \ingroup RangesAndPoints

    This class contains an array of the specified VALUETYPE with
    size specified at runtim.
    The interface conforms to STL vector, except that there are no functions
    that change the size of a tiny_array.

    \ref TinyArrayOperators "Arithmetic operations"
    on TinyArrays are defined as component-wise applications of these
    operations.

    See also:<br>
    <UL style="list-style-image:url(documents/bullet.gif)">
        <LI> \ref vigra::tiny_array_base
        <LI> \ref vigra::tiny_array_view
        <LI> \ref TinyArrayOperators
    </UL>

    <b>\#include</b> \<vigra/tiny_array.hxx\><br>
    Namespace: vigra
**/
template <class VALUETYPE>
class tiny_array<VALUETYPE, runtime_size>
: public tiny_array_base<VALUETYPE, tiny_array<VALUETYPE, runtime_size>, runtime_size>
{
  public:
    using base_type = tiny_array_base<VALUETYPE, tiny_array<VALUETYPE, runtime_size>, runtime_size>;
    using value_type = VALUETYPE;

    tiny_array()
    : base_type()
    {}

    explicit
    tiny_array(index_t size,
               value_type const & initial = value_type())
    : base_type(size)
    {
        this->data_ = alloc_.allocate(this->size_);
        std::uninitialized_fill(this->begin(), this->end(), initial);
    }

    explicit
    tiny_array(tags::size_proxy const & size,
               value_type const & initial = value_type())
    : tiny_array(size.value, initial)
    {}

    tiny_array(index_t size, skip_initialization_tag)
    : base_type(size)
    {
        this->data_ = alloc_.allocate(this->size_);
        if(!base_type::may_use_uninitialized_memory)
            std::uninitialized_fill(this->begin(), this->end(), value_type());
    }

    tiny_array(tiny_array const & rhs )
    : base_type(rhs.size())
    {
        this->data_ = alloc_.allocate(this->size_);
        std::uninitialized_copy(rhs.begin(), rhs.end(), this->begin());
    }

    tiny_array(tiny_array && rhs)
    : base_type()
    {
        this->swap(rhs);
    }

    template <class U, class D, int ... N>
    tiny_array(tiny_array_base<U, D, N...> const & other)
    : tiny_array(other.begin(), other.end())
    {}

    template <class U,
              XTENSOR_REQUIRE<iterator_concept<U>::value> >
    tiny_array(U begin, U end)
    : base_type(std::distance(begin, end))
    {
        this->data_ = alloc_.allocate(this->size_);
        for(int i=0; i<this->size_; ++i, ++begin)
            new(this->data_+i) value_type(static_cast<value_type>(*begin));
    }

    template <class U,
              XTENSOR_REQUIRE<iterator_concept<U>::value> >
    tiny_array(U begin, U end, reverse_copy_tag)
    : base_type(std::distance(begin, end))
    {
        this->data_ = alloc_.allocate(this->size_);
        for(int i=0; i<this->size_; ++i, --end)
            new(this->data_+i) value_type(static_cast<value_type>(*(end-1)));
    }

    template <class U, size_t SIZE>
    tiny_array(const U (&u)[SIZE])
    : tiny_array(u, u+SIZE)
    {}

    template <class U>
    tiny_array(std::initializer_list<U> rhs)
    : tiny_array(rhs.begin(), rhs.end())
    {}

    tiny_array & operator=(value_type const & v)
    {
        base_type::operator=(v);
        return *this;
    }

    tiny_array & operator=(tiny_array && rhs)
    {
        if(this->size_ != rhs.size())
            rhs.swap(*this);
        else
            base_type::operator=(rhs);
        return *this;
    }

    tiny_array & operator=(tiny_array const & rhs)
    {
        if(this == &rhs)
            return *this;
        if(this->size_ != rhs.size())
            tiny_array(rhs).swap(*this);
        else
            base_type::operator=(rhs);
        return *this;
    }

    template <class U, class D, int ... N>
    tiny_array & operator=(tiny_array_base<U, D, N...> const & rhs)
    {
        if(this->size_ == 0 || rhs.size() == 0)
            tiny_array(rhs).swap(*this);
        else
            base_type::operator=(rhs);
        return *this;
    }

    ~tiny_array()
    {
        if(!base_type::may_use_uninitialized_memory)
        {
            for(index_t i=0; i<this->size_; ++i)
                (this->data_+i)->~value_type();
        }
        alloc_.deallocate(this->data_, this->size_);
    }

#if 0
    // FIXME: hacks to use tiny_array as shape in xtensor
    using base_type::erase;

    void erase(base_type::pointer p)
    {
        --this->size_;
        for(; p < this->data_ + this->size_; ++p)
            *p = p[1];
    }

    void resize(size_t s)
    {
        if(s > this->size_)
        {
            alloc_.deallocate(this->data_, this->size_);
            this->data_ = alloc_.allocate(s);
        }
        this->size_ = s;
    }

    template <class U>
    tiny_array(std::vector<U> const & rhs)
    : tiny_array(rhs.cbegin(), rhs.cend())
    {}

    template <class U>
    tiny_array & operator=(std::vector<U> const & rhs)
    {
        resize(rhs.size());
        for(int k=0; k < this->size_; ++k)
            this->data_[k] = rhs[k];
        return *this;
    }

    template <class U>
    bool operator==(std::vector<U> const & rhs) const
    {
        if(this->size_ != (int)rhs.size())
            return false;
        for(int k=0; k < this->size_; ++k)
            if(this->data_[k] != rhs[k])
                return false;
        return true;
    }
#endif

  private:
    // FIXME: implement an optimized allocator
    // FIXME: (look at Alexandrescu's Loki library or Kolmogorov's code)
    std::allocator<value_type> alloc_;
};


template <class U>
bool operator==(std::vector<U> const & l, tiny_array<U, -1> const & r)
{
    if((int)l.size() != r.size())
        return false;
    for(int k=0; k < (int)l.size(); ++k)
        if(l[k] != r[k])
            return false;
    return true;
}

/********************************************************/
/*                                                      */
/*                    tiny_array_view                   */
/*                                                      */
/********************************************************/

/** \brief Wrapper for fixed size arrays.

    This class wraps the memory of an array of the specified VALUETYPE
    with (possibly multi-dimensional) shape given by <tt>index_t....N</tt>.
    Thus, the array can be accessed with an interface similar to
    that of std::vector (except that there are no functions
    that change the size of a tiny_array_view). The tiny_array_view
    does <em>not</em> assume ownership of the given memory.

    \ref TinyArrayOperators "Arithmetic operations"
    on TinyArrayViews are defined as component-wise applications of these
    operations.

    <b>See also:</b>
    <ul>
        <li> \ref vigra::tiny_array_base
        <li> \ref vigra::tiny_array
        <li> \ref vigra::tiny_symmetric_view
        <li> \ref TinyArrayOperators
    </ul>

    <b>\#include</b> \<vigra/tinyarray.hxx\><br>
    Namespace: vigra
**/
template <class VALUETYPE, int M, int ... N>
class tiny_array_view
: public tiny_array_base<VALUETYPE, tiny_array_view<VALUETYPE, M, N...>, M, N...>
{
  public:
    using base_type     = tiny_array_base<VALUETYPE, tiny_array_view<VALUETYPE, M, N...>, M, N...>;
    using value_type    = VALUETYPE;
    using pointer       = typename base_type::pointer;
    using const_pointer = typename base_type::const_pointer;

    static const index_t static_size  = base_type::static_size;
    static const index_t static_ndim  = base_type::static_ndim;

    tiny_array_view()
    : base_type(dont_init)
    {
        base_type::data_ = nullptr;
    }

        /** Construct view for given data array
        */
    tiny_array_view(const_pointer data)
    : base_type(dont_init)
    {
        base_type::data_ = const_cast<pointer>(data);
    }

        /** Copy constructor (shallow copy).
        */
    tiny_array_view(tiny_array_view const & other)
    : base_type(dont_init)
    {
        base_type::data_ = const_cast<pointer>(other.data());
    }

        /** Construct view from other tiny_array.
        */
    template <class OTHER_DERIVED>
    tiny_array_view(tiny_array_base<value_type, OTHER_DERIVED, M, N...> const & other)
    : base_type(dont_init)
    {
        base_type::data_ = const_cast<pointer>(other.data());
    }

        /** Reset to the other array's pointer.
        */
    template <class OTHER_DERIVED>
    void reset(tiny_array_base<value_type, OTHER_DERIVED, M, N...> const & other)
    {
        base_type::data_ = const_cast<pointer>(other.data());
    }

        /** Copy the data (not the pointer) of the rhs.
        */
    tiny_array_view & operator=(tiny_array_view const & r)
    {
        for(int k=0; k<static_size; ++k)
            base_type::data_[k] = static_cast<value_type>(r[k]);
        return *this;
    }

        /** Copy the data of the rhs with cast.
        */
    template <class U, class OTHER_DERIVED>
    tiny_array_view & operator=(tiny_array_base<U, OTHER_DERIVED, M, N...> const & r)
    {
        for(int k=0; k<static_size; ++k)
            base_type::data_[k] = static_cast<value_type>(r[k]);
        return *this;
    }
};

template <class VALUETYPE>
class tiny_array_view<VALUETYPE, runtime_size>
: public tiny_array_base<VALUETYPE, tiny_array_view<VALUETYPE, runtime_size>, runtime_size>
{
  public:
    using base_type = tiny_array_base<VALUETYPE, tiny_array_view<VALUETYPE, runtime_size>, runtime_size>;

    using base_type::base_type;
    using base_type::operator=;
};

/********************************************************/
/*                                                      */
/*                  tiny_symmetric_view                 */
/*                                                      */
/********************************************************/

/** \brief Wrapper for fixed size arrays.

    This class wraps the memory of an 1D array of the specified VALUETYPE
    with size <tt>N*(N+1)/2</tt> and interprets this array as a symmetric
    matrix. Specifically, the data are interpreted as the row-wise
    representation of the upper triangular part of the symmetric matrix.
    All index access operations are overloaded such that the view appears
    as if it were a full matrix. The tiny_symmetric_view
    does <em>not</em> assume ownership of the given memory.

    \ref TinyArrayOperators "Arithmetic operations"
    on tiny_symmetric_view are defined as component-wise applications of these
    operations.

    <b>See also:</b>
    <ul>
        <li> \ref vigra::tiny_array_base
        <li> \ref vigra::tiny_array
        <li> \ref vigra::tiny_array_view
        <li> \ref TinyArrayOperators
    </ul>

    <b>\#include</b> \<vigra/tinyarray.hxx\><br>
    Namespace: vigra
**/
template <class VALUETYPE, int N>
class tiny_symmetric_view
: public tiny_array_base<VALUETYPE, tiny_symmetric_view<VALUETYPE, N>, N*(N+1)/2>
{
  public:
    using base_type       = tiny_array_base<VALUETYPE, tiny_symmetric_view<VALUETYPE, N>, N*(N+1)/2>;
    using value_type      = VALUETYPE;
    using reference       = typename base_type::reference;
    using const_reference = typename base_type::const_reference;
    using pointer         = typename base_type::pointer;
    using const_pointer   = typename base_type::const_pointer;
    using index_type      = tiny_array<index_t, 2>;

    static const index_t static_size  = base_type::static_size;
    static const index_t static_ndim  = 2;

    tiny_symmetric_view()
    : base_type(dont_init)
    {
        base_type::data_ = nullptr;
    }

        /** Construct view for given data array
        */
    tiny_symmetric_view(const_pointer data)
    : base_type(dont_init)
    {
        base_type::data_ = const_cast<pointer>(data);
    }

        /** Copy constructor (shallow copy).
        */
    tiny_symmetric_view(tiny_symmetric_view const & other)
    : base_type(dont_init)
    {
        base_type::data_ = const_cast<pointer>(other.data());
    }

        /** Construct view from other tiny_array.
        */
    template <class OTHER_DERIVED>
    tiny_symmetric_view(tiny_array_base<value_type, OTHER_DERIVED, N*(N+1)/2> const & other)
    : base_type(dont_init)
    {
        base_type::data_ = const_cast<pointer>(other.data());
    }

        /** Reset to the other array's pointer.
        */
    template <class OTHER_DERIVED>
    void reset(tiny_array_base<value_type, OTHER_DERIVED, N*(N+1)/2> const & other)
    {
        base_type::data_ = const_cast<pointer>(other.data());
    }

        /** Copy the data (not the pointer) of the rhs.
        */
    tiny_symmetric_view & operator=(tiny_symmetric_view const & r)
    {
        for(int k=0; k<static_size; ++k)
            base_type::data_[k] = static_cast<value_type>(r[k]);
        return *this;
    }

        /** Copy the data of the rhs with cast.
        */
    template <class U, class OTHER_DERIVED>
    tiny_symmetric_view & operator=(tiny_array_base<U, OTHER_DERIVED, N*(N+1)/2> const & r)
    {
        for(int k=0; k<static_size; ++k)
            base_type::data_[k] = static_cast<value_type>(r[k]);
        return *this;
    }

    // index access

    reference operator[](index_t i)
    {
        return base_type::operator[](i);
    }

    constexpr const_reference operator[](index_t i) const
    {
        return base_type::operator[](i);
    }

    reference at(index_t i)
    {
        return base_type::at(i);
    }

    const_reference at(index_t i) const
    {
        return base_type::at(i);
    }

    reference operator[](index_t const (&i)[2])
    {
        return this->operator()(i[0], i[1]);
    }

    constexpr const_reference operator[](index_t const (&i)[2]) const
    {
        return this->operator()(i[0], i[1]);
    }

    reference at(index_t const (&i)[static_ndim])
    {
        return this->at(i[0], i[1]);
    }

    const_reference at(index_t const (&i)[static_ndim]) const
    {
        return this->at(i[0], i[1]);
    }

    reference operator[](index_type const & i)
    {
        return this->operator()(i[0], i[1]);
    }

    constexpr const_reference operator[](index_type const & i) const
    {
        return this->operator()(i[0], i[1]);
    }

    reference at(index_type const & i)
    {
        return this->at(i[0], i[1]);
    }

    const_reference at(index_type const & i) const
    {
        return this->at(i[0], i[1]);
    }

    reference operator()(index_t i, index_t j)
    {
        return (i > j)
            ? base_type::data_[i + N*j - (j*(j+1) >> 1)]
            : base_type::data_[N*i + j - (i*(i+1) >> 1)];
    }

    constexpr const_reference operator()(index_t const i, index_t const j) const
    {
        return (i > j)
            ? base_type::data_[i + (j*((2 * N - 1) - j) >> 1)]
            : base_type::data_[j + (i*((2 * N - 1) - i) >> 1)];
    }

    reference at(index_t i, index_t j)
    {
        index_t k = (i > j)
                           ? i + (j*((2*N-1) - j) >> 1)
                           : j + (i*((2*N-1) - i) >> 1);
        if(k < 0 || k >= static_size)
            throw std::out_of_range("tiny_symmetric_view::at()");
        return base_type::data_[k];
    }

    const_reference at(index_t i, index_t j) const
    {
        index_t k = (i > j)
                           ? i + N*j - (j*(j+1) >> 1)
                           : N*i + j - (i*(i+1) >> 1);
        if(k < 0 || k >= static_size)
            throw std::out_of_range("tiny_symmetric_view::at()");
        return base_type::data_[k];
    }

    constexpr index_type shape() const { return index_type{N, N}; }
    constexpr index_t ndim () const { return static_ndim; }
};

/********************************************************/
/*                                                      */
/*              tiny_symmetric_view output              */
/*                                                      */
/********************************************************/

template <class T, int N>
std::ostream & operator<<(std::ostream & o, tiny_symmetric_view<T, N> const & v)
{
    o << "{";
    for(int i=0; i<N; ++i)
    {
        if(i > 0)
            o << ",\n ";
        o << promote_t<T>(v(i,0));
        for(int j=1; j<N; ++j)
        {
            o << ", " << promote_t<T>(v(i, j));
        }
    }
    o << "}";
    return o;
}


/********************************************************/
/*                                                      */
/*                tiny_array Comparison                 */
/*                                                      */
/********************************************************/

/** \addtogroup TinyArrayOperators Functions for tiny_array

    \brief Implement basic arithmetic and equality for tiny_array.

    These functions fulfill the requirements of a Linear Space (vector space).
    Return types are determined according to \ref promote_t or \ref real_promote_t.

    <b>\#include</b> \<vigra/tiny_array.hxx\><br>
    Namespace: vigra
*/
//@{

    /// element-wise equal
template <class V1, class D1, class V2, class D2, int ...M, int ... N>
inline bool
operator==(tiny_array_base<V1, D1, M...> const & l,
           tiny_array_base<V2, D2, N...> const & r)
{
    if(!l.is_same_shape(r))
        return false;
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r[k])
            return false;
    return true;
}

    /// element-wise equal to a constant
template <class V1, class D1, class V2, int ...M,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
operator==(tiny_array_base<V1, D1, M...> const & l,
           V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r)
            return false;
    return true;
}

    /// element-wise equal to a constant
template <class V1, class V2, class D2, int ...M,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
operator==(V1 const & l,
           tiny_array_base<V2, D2, M...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if(l != r[k])
            return false;
    return true;
}

    /// element-wise not equal
template <class V1, class D1, class V2, class D2, int ... M, int ... N>
inline bool
operator!=(tiny_array_base<V1, D1, M...> const & l,
           tiny_array_base<V2, D2, N...> const & r)
{
    if(!l.is_same_shape(r))
        return true;
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r[k])
            return true;
    return false;
}

    /// element-wise not equal to a constant
template <class V1, class D1, class V2, int ... M,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
operator!=(tiny_array_base<V1, D1, M...> const & l,
           V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if(l[k] != r)
            return true;
    return false;
}

    /// element-wise not equal to a constant
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
operator!=(V1 const & l,
           tiny_array_base<V2, D2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if(l != r[k])
            return true;
    return false;
}

    /// lexicographical comparison
template <class V1, class D1, class V2, class D2, int ... N>
inline bool
operator<(tiny_array_base<V1, D1, N...> const & l,
          tiny_array_base<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "tiny_array_base::operator<(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
    {
        if(l[k] < r[k])
            return true;
        if(r[k] < l[k])
            return false;
    }
    return false;
}

    /// lexicographical comparison
template <class V1, class D, int N, class V2, class A>
inline bool
operator<(tiny_array_base<V1, D, N> const & l,
          std::vector<V2, A> const & r)
{
    index_t size = r.size();
    bool equal_res  = false;
    if(l.size() < size)
    {
        size      = l.size();
        equal_res = true;
    }
    for(int k=0; k < size; ++k)
    {
        if(l[k] < r[k])
            return true;
        if(r[k] < l[k])
            return false;
    }
    return equal_res;
}

template <class V1, class A, class V2, class D, int N>
inline bool
operator<(std::vector<V1, A> const & l,
          tiny_array_base<V2, D, N> const & r)
{
    index_t size = r.size();
    bool equal_res  = false;
    if(l.size() < size)
    {
        size      = l.size();
        equal_res = true;
    }
    for(int k=0; k < size; ++k)
    {
        if(l[k] < r[k])
            return true;
        if(r[k] < l[k])
            return false;
    }
    return equal_res;
}

template <class V1, class D, int N, class V2, class A>
inline bool
operator>(tiny_array_base<V1, D, N> const & l,
          std::vector<V2, A> const & r)
{
    return r < l;
}

template <class V1, class A, class V2, class D, int N>
inline bool
operator>(std::vector<V1, A> const & l,
          tiny_array_base<V2, D, N> const & r)
{
    return r < l;
}

    /// check if all elements are non-zero (or 'true' if V is bool)
template <class V, class D, int ... N>
inline bool
all(tiny_array_base<V, D, N...> const & t)
{
    for(int i=0; i<t.size(); ++i)
        if(t[i] == V())
            return false;
    return true;
}

    /// check if at least one element is non-zero (or 'true' if V is bool)
template <class V, class D, int ... N>
inline bool
any(tiny_array_base<V, D, N...> const & t)
{
    for(int i=0; i<t.size(); ++i)
        if(t[i] != V())
            return true;
    return false;
}

    /// check if all elements are zero (or 'false' if V is bool)
template <class V, class D, int ... N>
inline bool
all_zero(tiny_array_base<V, D, N...> const & t)
{
    for(int i=0; i<t.size(); ++i)
        if(t[i] != V())
            return false;
    return true;
}

    /// pointwise less-than
template <class V1, class D1, class V2, class D2, int ... N>
inline bool
all_less(tiny_array_base<V1, D1, N...> const & l,
        tiny_array_base<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "tiny_array_base::all_less(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if (l[k] >= r[k])
            return false;
    return true;
}

    /// pointwise less than a constant
    /// (typically used to check negativity with `r = 0`)
template <class V1, class D1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_less(tiny_array_base<V1, D1, N...> const & l,
        V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] >= r)
            return false;
    return true;
}

    /// constant pointwise less than the vector
    /// (typically used to check positivity with `l = 0`)
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_less(V1 const & l,
        tiny_array_base<V2, D2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l >= r[k])
            return false;
    return true;
}

    /// pointwise less-equal
template <class V1, class D1, class V2, class D2, int ... N>
inline bool
all_less_equal(tiny_array_base<V1, D1, N...> const & l,
             tiny_array_base<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "tiny_array_base::all_less_equal(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if (l[k] > r[k])
            return false;
    return true;
}

    /// pointwise less-equal with a constant
    /// (typically used to check non-positivity with `r = 0`)
template <class V1, class D1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_less_equal(tiny_array_base<V1, D1, N...> const & l,
             V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] > r)
            return false;
    return true;
}

    /// pointwise less-equal with a constant
    /// (typically used to check non-negativity with `l = 0`)
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_less_equal(V1 const & l,
             tiny_array_base<V2, D2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l > r[k])
            return false;
    return true;
}

    /// pointwise greater-than
template <class V1, class D1, class V2, class D2, int ... N>
inline bool
all_greater(tiny_array_base<V1, D1, N...> const & l,
           tiny_array_base<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "tiny_array_base::all_greater(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if (l[k] <= r[k])
            return false;
    return true;
}

    /// pointwise greater-than with a constant
    /// (typically used to check positivity with `r = 0`)
template <class V1, class D1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_greater(tiny_array_base<V1, D1, N...> const & l,
           V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] <= r)
            return false;
    return true;
}

    /// constant pointwise greater-than a vector
    /// (typically used to check negativity with `l = 0`)
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_greater(V1 const & l,
           tiny_array_base<V2, D2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l <= r[k])
            return false;
    return true;
}

    /// pointwise greater-equal
template <class V1, class D1, class V2, class D2, int ... N>
inline bool
all_greater_equal(tiny_array_base<V1, D1, N...> const & l,
                tiny_array_base<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "tiny_array_base::all_greater_equal(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if (l[k] < r[k])
            return false;
    return true;
}

    /// pointwise greater-equal with a constant
    /// (typically used to check non-negativity with `r = 0`)
template <class V1, class D1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_greater_equal(tiny_array_base<V1, D1, N...> const & l,
                V2 const & r)
{
    for(int k=0; k < l.size(); ++k)
        if (l[k] < r)
            return false;
    return true;
}

    /// pointwise greater-equal with a constant
    /// (typically used to check non-positivity with `l = 0`)
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                          std::is_convertible<V2, V1>::value> >
inline bool
all_greater_equal(V1 const & l,
                tiny_array_base<V2, D2, N...> const & r)
{
    for(int k=0; k < r.size(); ++k)
        if (l < r[k])
            return false;
    return true;
}

template <class V1, class D1, class V2, class D2, int ... N>
inline bool
isclose(tiny_array_base<V1, D1, N...> const & l,
        tiny_array_base<V2, D2, N...> const & r,
        promote_t<V1, V2> epsilon = 2.0*std::numeric_limits<promote_t<V1, V2> >::epsilon())
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "tiny_array_base::isclose(): size mismatch.");
    for(int k=0; k < l.size(); ++k)
        if(!isclose(l[k], r[k], epsilon, epsilon))
            return false;
    return true;
}

/********************************************************/
/*                                                      */
/*                 tiny_array-Arithmetic                */
/*                                                      */
/********************************************************/

/** \addtogroup TinyArrayOperators
 */
//@{

#ifdef DOXYGEN
// Declare arithmetic functions for documentation,
// the implementations are provided by a macro below.

    /// scalar add-assignment
template <class V1, class DERIVED, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
DERIVED &
operator+=(tiny_array_base<V1, DERIVED, N...> & l,
           V2 r);

    /// element-wise add-assignment
template <class V1, class DERIVED, class V2, class OTHER_DERIVED, int ... N>
DERIVED &
operator+=(tiny_array_base<V1, DERIVED, N...> & l,
           tiny_array_base<V2, OTHER_DERIVED, N...> const & r);

    /// element-wise addition
template <class V1, class D1, class V2, class D2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator+(tiny_array_base<V1, D1, N...> const & l,
          tiny_array_base<V2, D2, N...> const & r);

    /// element-wise scalar addition
template <class V1, class D1, class V2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator+(tiny_array_base<V1, D1, N...> const & l,
          V2 r);

    /// element-wise left scalar addition
template <class V1, class V2, class D2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator+(V1 l,
          tiny_array_base<V2, D2, N...> const & r);

    /// scalar subtract-assignment
template <class V1, class DERIVED, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
DERIVED &
operator-=(tiny_array_base<V1, DERIVED, N...> & l,
           V2 r);

    /// element-wise subtract-assignment
template <class V1, class DERIVED, class V2, class OTHER_DERIVED, int ... N>
DERIVED &
operator-=(tiny_array_base<V1, DERIVED, N...> & l,
           tiny_array_base<V2, OTHER_DERIVED, N...> const & r);

    /// element-wise subtraction
template <class V1, class D1, class V2, class D2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator-(tiny_array_base<V1, D1, N...> const & l,
          tiny_array_base<V2, D2, N...> const & r);

    /// element-wise scalar subtraction
template <class V1, class D1, class V2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator-(tiny_array_base<V1, D1, N...> const & l,
          V2 r);

    /// element-wise left scalar subtraction
template <class V1, class V2, class D2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator-(V1 l,
          tiny_array_base<V2, D2, N...> const & r);

    /// scalar multiply-assignment
template <class V1, class DERIVED, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
DERIVED &
operator*=(tiny_array_base<V1, DERIVED, N...> & l,
           V2 r);

    /// element-wise multiply-assignment
template <class V1, class DERIVED, class V2, class OTHER_DERIVED, int ... N>
DERIVED &
operator*=(tiny_array_base<V1, DERIVED, N...> & l,
           tiny_array_base<V2, OTHER_DERIVED, N...> const & r);

    /// element-wise multiplication
template <class V1, class D1, class V2, class D2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator*(tiny_array_base<V1, D1, N...> const & l,
          tiny_array_base<V2, D2, N...> const & r);

    /// element-wise scalar multiplication
template <class V1, class D1, class V2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator*(tiny_array_base<V1, D1, N...> const & l,
          V2 r);

    /// element-wise left scalar multiplication
template <class V1, class V2, class D2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator*(V1 l,
          tiny_array_base<V2, D2, N...> const & r);

    /// scalar divide-assignment
template <class V1, class DERIVED, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
DERIVED &
operator/=(tiny_array_base<V1, DERIVED, N...> & l,
           V2 r);

    /// element-wise divide-assignment
template <class V1, class DERIVED, class V2, class OTHER_DERIVED, int ... N>
DERIVED &
operator/=(tiny_array_base<V1, DERIVED, N...> & l,
           tiny_array_base<V2, OTHER_DERIVED, N...> const & r);

    /// element-wise division
template <class V1, class D1, class V2, class D2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator/(tiny_array_base<V1, D1, N...> const & l,
          tiny_array_base<V2, D2, N...> const & r);

    /// element-wise scalar division
template <class V1, class D1, class V2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator/(tiny_array_base<V1, D1, N...> const & l,
          V2 r);

    /// element-wise left scalar division
template <class V1, class V2, class D2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator/(V1 l,
          tiny_array_base<V2, D2, N...> const & r);

    /// scalar modulo-assignment
template <class V1, class DERIVED, int ... N, class V2,
          XTENSOR_REQUIRE<std::is_convertible<V2, V1>::value> >
DERIVED &
operator%=(tiny_array_base<V1, DERIVED, N...> & l,
           V2 r);

    /// element-wise modulo-assignment
template <class V1, class DERIVED, class V2, class OTHER_DERIVED, int ... N>
DERIVED &
operator%=(tiny_array_base<V1, DERIVED, N...> & l,
           tiny_array_base<V2, OTHER_DERIVED, N...> const & r);

    /// element-wise modulo
template <class V1, class D1, class V2, class D2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator%(tiny_array_base<V1, D1, N...> const & l,
          tiny_array_base<V2, D2, N...> const & r);

    /// element-wise scalar modulo
template <class V1, class D1, class V2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator%(tiny_array_base<V1, D1, N...> const & l,
          V2 r);

    /// element-wise left scalar modulo
template <class V1, class V2, class D2, int ... N>
tiny_array<promote_t<V1, V2>, N...>
operator%(V1 l,
          tiny_array_base<V2, D2, N...> const & r);

#else

#define XTENSOR_TINYARRAY_OPERATORS(OP) \
template <class V1, class DERIVED, int ... N, class V2, \
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value && \
                          std::is_convertible<V2, V1>::value> > \
inline DERIVED & \
operator OP##=(tiny_array_base<V1, DERIVED, N...> & l, \
               V2 r) \
{ \
    for(int i=0; i<l.size(); ++i) \
        l[i] OP##= r; \
    return static_cast<DERIVED &>(l); \
} \
 \
template <class V1, class DERIVED, class V2, class OTHER_DERIVED, int ... M, int ... N> \
inline DERIVED &  \
operator OP##=(tiny_array_base<V1, DERIVED, M...> & l, \
               tiny_array_base<V2, OTHER_DERIVED, N...> const & r) \
{ \
    XTENSOR_ASSERT_MSG(l.size() == r.size(), \
        "tiny_array_base::operator" #OP "=(): size mismatch."); \
    for(int i=0; i<l.size(); ++i) \
        l[i] OP##= r[i]; \
    return static_cast<DERIVED &>(l); \
} \
template <class V1, class D1, class V2, class D2, int ... N> \
inline \
tiny_array<decltype((*(V1*)0) OP (*(V2*)0)), N...> \
operator OP(tiny_array_base<V1, D1, N...> const & l, \
            tiny_array_base<V2, D2, N...> const & r) \
{ \
    tiny_array<decltype((*(V1*)0) OP (*(V2*)0)), N...> res(l); \
    return res OP##= r; \
} \
 \
template <class V1, class D1, class V2, int ... N, \
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value> >\
inline \
tiny_array<decltype((*(V1*)0) OP (*(V2*)0)), N...> \
operator OP(tiny_array_base<V1, D1, N...> const & l, \
            V2 r) \
{ \
    tiny_array<decltype((*(V1*)0) OP (*(V2*)0)), N...> res(l); \
    return res OP##= r; \
} \
 \
template <class V1, class V2, class D2, int ... N, \
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value && \
                          !std::is_base_of<std::ios_base, V1>::value> >\
inline \
tiny_array<decltype((*(V1*)0) OP (*(V2*)0)), N...> \
operator OP(V1 l, \
             tiny_array_base<V2, D2, N...> const & r) \
{ \
    tiny_array<decltype((*(V1*)0) OP (*(V2*)0)), N...> res{l}; \
    return res OP##= r; \
} \
 \
template <class V1, class V2, class D2, \
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value && \
                          !std::is_base_of<std::ios_base, V1>::value> >\
inline \
tiny_array<decltype((*(V1*)0) OP (*(V2*)0)), runtime_size> \
operator OP(V1 l, \
             tiny_array_base<V2, D2, runtime_size> const & r) \
{ \
    tiny_array<decltype((*(V1*)0) OP (*(V2*)0)), runtime_size> res(tags::size=r.size(), l); \
    return res OP##= r; \
}

XTENSOR_TINYARRAY_OPERATORS(+)
XTENSOR_TINYARRAY_OPERATORS(-)
XTENSOR_TINYARRAY_OPERATORS(*)
XTENSOR_TINYARRAY_OPERATORS(/)
XTENSOR_TINYARRAY_OPERATORS(%)
XTENSOR_TINYARRAY_OPERATORS(&)
XTENSOR_TINYARRAY_OPERATORS(|)
XTENSOR_TINYARRAY_OPERATORS(^)
XTENSOR_TINYARRAY_OPERATORS(<<)
XTENSOR_TINYARRAY_OPERATORS(>>)

#undef XTENSOR_TINYARRAY_OPERATORS

#endif // DOXYGEN

#define XTENSOR_TINYARRAY_UNARY_FUNCTION(FCT) \
template <class V, class D, int ... N> \
inline auto \
FCT(tiny_array_base<V, D, N...> const & v) \
{ \
    using namespace cmath; \
    tiny_array<bool_promote_t<decltype(FCT(*(V*)0))>, N...> res(v.size(), dont_init); \
    for(int k=0; k < v.size(); ++k) \
        res[k] = FCT(v[k]); \
    return res; \
}

XTENSOR_TINYARRAY_UNARY_FUNCTION(abs)
XTENSOR_TINYARRAY_UNARY_FUNCTION(fabs)

XTENSOR_TINYARRAY_UNARY_FUNCTION(cos)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sin)
XTENSOR_TINYARRAY_UNARY_FUNCTION(tan)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sin_pi)
XTENSOR_TINYARRAY_UNARY_FUNCTION(cos_pi)
XTENSOR_TINYARRAY_UNARY_FUNCTION(acos)
XTENSOR_TINYARRAY_UNARY_FUNCTION(asin)
XTENSOR_TINYARRAY_UNARY_FUNCTION(atan)

XTENSOR_TINYARRAY_UNARY_FUNCTION(cosh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sinh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(tanh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(acosh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(asinh)
XTENSOR_TINYARRAY_UNARY_FUNCTION(atanh)

XTENSOR_TINYARRAY_UNARY_FUNCTION(sqrt)
XTENSOR_TINYARRAY_UNARY_FUNCTION(cbrt)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sq)
XTENSOR_TINYARRAY_UNARY_FUNCTION(elementwise_norm)
XTENSOR_TINYARRAY_UNARY_FUNCTION(elementwise_squared_norm)

XTENSOR_TINYARRAY_UNARY_FUNCTION(exp)
XTENSOR_TINYARRAY_UNARY_FUNCTION(exp2)
XTENSOR_TINYARRAY_UNARY_FUNCTION(expm1)
XTENSOR_TINYARRAY_UNARY_FUNCTION(log)
XTENSOR_TINYARRAY_UNARY_FUNCTION(log2)
XTENSOR_TINYARRAY_UNARY_FUNCTION(log10)
XTENSOR_TINYARRAY_UNARY_FUNCTION(log1p)
XTENSOR_TINYARRAY_UNARY_FUNCTION(logb)
XTENSOR_TINYARRAY_UNARY_FUNCTION(ilogb)

XTENSOR_TINYARRAY_UNARY_FUNCTION(ceil)
XTENSOR_TINYARRAY_UNARY_FUNCTION(floor)
XTENSOR_TINYARRAY_UNARY_FUNCTION(trunc)
XTENSOR_TINYARRAY_UNARY_FUNCTION(round)
XTENSOR_TINYARRAY_UNARY_FUNCTION(lround)
XTENSOR_TINYARRAY_UNARY_FUNCTION(llround)
XTENSOR_TINYARRAY_UNARY_FUNCTION(even)
XTENSOR_TINYARRAY_UNARY_FUNCTION(odd)
XTENSOR_TINYARRAY_UNARY_FUNCTION(sign)
XTENSOR_TINYARRAY_UNARY_FUNCTION(signi)

XTENSOR_TINYARRAY_UNARY_FUNCTION(erf)
XTENSOR_TINYARRAY_UNARY_FUNCTION(erfc)
XTENSOR_TINYARRAY_UNARY_FUNCTION(tgamma)
XTENSOR_TINYARRAY_UNARY_FUNCTION(lgamma)

XTENSOR_TINYARRAY_UNARY_FUNCTION(conj)
XTENSOR_TINYARRAY_UNARY_FUNCTION(real)
XTENSOR_TINYARRAY_UNARY_FUNCTION(imag)
XTENSOR_TINYARRAY_UNARY_FUNCTION(arg)

#undef XTENSOR_TINYARRAY_UNARY_FUNCTION

    /// Arithmetic negation
template <class V, class D, int ... N>
inline
tiny_array<V, N...>
operator-(tiny_array_base<V, D, N...> const & v)
{
    tiny_array<V, N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = -v[k];
    return res;
}

    /// Boolean negation
template <class V, class D, int ... N>
inline
tiny_array<V, N...>
operator!(tiny_array_base<V, D, N...> const & v)
{
    tiny_array<V, N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = !v[k];
    return res;
}

    /// Bitwise negation
template <class V, class D, int ... N>
inline
tiny_array<V, N...>
operator~(tiny_array_base<V, D, N...> const & v)
{
    tiny_array<V, N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = ~v[k];
    return res;
}

#define XTENSOR_TINYARRAY_BINARY_FUNCTION(FCT) \
template <class V1, class D1, class V2, class D2, int ... N> \
inline auto \
FCT(tiny_array_base<V1, D1, N...> const & l, \
    tiny_array_base<V2, D2, N...> const & r) \
{ \
    XTENSOR_ASSERT_MSG(l.size() == r.size(), #FCT "(tiny_array, tiny_array): size mismatch."); \
    using namespace cmath; \
    tiny_array<decltype(FCT(*(V1*)0, *(V2*)0)), N...> res(l.size(), dont_init); \
    for(int k=0; k < l.size(); ++k) \
        res[k] = FCT(l[k], r[k]); \
    return res; \
}

XTENSOR_TINYARRAY_BINARY_FUNCTION(atan2)
XTENSOR_TINYARRAY_BINARY_FUNCTION(copysign)
XTENSOR_TINYARRAY_BINARY_FUNCTION(fdim)
XTENSOR_TINYARRAY_BINARY_FUNCTION(fmax)
XTENSOR_TINYARRAY_BINARY_FUNCTION(fmin)
XTENSOR_TINYARRAY_BINARY_FUNCTION(fmod)
XTENSOR_TINYARRAY_BINARY_FUNCTION(hypot)

#undef XTENSOR_TINYARRAY_BINARY_FUNCTION

    /** Apply pow() function to each vector component.
    */
template <class V, class D, class E, int ... N>
inline auto
pow(tiny_array_base<V, D, N...> const & v, E exponent)
{
    using namespace cmath;
    tiny_array<decltype(pow(v[0], exponent)), N...> res(v.size(), dont_init);
    for(int k=0; k < v.size(); ++k)
        res[k] = pow(v[k], exponent);
    return res;
}

    /// cross product
template <class V1, class D1, class V2, class D2, int N,
          XTENSOR_REQUIRE<N == 3 || N == runtime_size> >
inline
tiny_array<promote_t<V1, V2>, N>
cross(tiny_array_base<V1, D1, N> const & r1,
      tiny_array_base<V2, D2, N> const & r2)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N, r1.size() == 3 && r2.size() == 3,
        "cross(): cross product requires size() == 3.");
    typedef tiny_array<promote_t<V1, V2>, N> Res;
    return  Res{r1[1]*r2[2] - r1[2]*r2[1],
                r1[2]*r2[0] - r1[0]*r2[2],
                r1[0]*r2[1] - r1[1]*r2[0]};
}

    /// dot product of two vectors
template <class V1, class D1, class V2, class D2, int N, int M>
inline
promote_t<V1, V2>
dot(tiny_array_base<V1, D1, N> const & l,
    tiny_array_base<V2, D2, M> const & r)
{
    XTENSOR_ASSERT_MSG(l.size() == r.size(), "dot(): size mismatch.");
    promote_t<V1, V2> res = promote_t<V1, V2>();
    for(int k=0; k < l.size(); ++k)
        res += l[k] * r[k];
    return res;
}

    /// sum of the vector's elements
template <class V, class D, int ... N>
inline
promote_t<V>
sum(tiny_array_base<V, D, N...> const & l)
{
    promote_t<V> res = promote_t<V>();
    for(int k=0; k < l.size(); ++k)
        res += l[k];
    return res;
}

    /// mean of the vector's elements
template <class V, class D, int ... N>
inline real_promote_t<V>
mean(tiny_array_base<V, D, N...> const & t)
{
    using Promote = real_promote_t<V>;
    const Promote sumVal = static_cast<Promote>(sum(t));
    if(t.size() > 0)
        return sumVal / t.size();
    else
        return sumVal;
}

    /// cumulative sum of the vector's elements
template <class V, class D, int ... N>
inline
tiny_array<promote_t<V>, N...>
cumsum(tiny_array_base<V, D, N...> const & l)
{
    tiny_array<promote_t<V>, N...> res(l);
    for(int k=1; k < l.size(); ++k)
        res[k] += res[k-1];
    return res;
}

    /// product of the vector's elements
template <class V, class D, int ... N>
inline
promote_t<V>
prod(tiny_array_base<V, D, N...> const & l)
{
    using Promote = promote_t<V>;
    if(l.size() == 0)
        return Promote();
    Promote res = Promote(1);
    for(int k=0; k < l.size(); ++k)
        res *= l[k];
    return res;
}

    /// cumulative product of the vector's elements
template <class V, class D, int ... N>
inline
tiny_array<promote_t<V>, N...>
cumprod(tiny_array_base<V, D, N...> const & l)
{
    tiny_array<promote_t<V>, N...> res(l);
    for(int k=1; k < l.size(); ++k)
        res[k] *= res[k-1];
    return res;
}

    /// element-wise minimum
template <class V1, class D1, class V2, class D2, int ... N>
inline
tiny_array<promote_t<V1, V2>, N...>
min(tiny_array_base<V1, D1, N...> const & l,
    tiny_array_base<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "min(): size mismatch.");
    tiny_array<promote_t<V1, V2>, N...> res(l.size(), dont_init);
    for(int k=0; k < l.size(); ++k)
        res[k] =  min(l[k], r[k]);
    return res;
}

    /// element-wise minimum with a constant
template <class V1, class D1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value>>
inline
tiny_array<std::common_type_t<V1, V2>, N...>
min(tiny_array_base<V1, D1, N...> const & l,
    V2 const & r)
{
    tiny_array<std::common_type_t<V1, V2>, N...> res(l.size(), dont_init);
    for(int k=0; k < l.size(); ++k)
        res[k] =  min(l[k], r);
    return res;
}

    /// element-wise minimum with a constant
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value>>
inline
tiny_array<std::common_type_t<V1, V2>, N...>
min(V1 const & l,
    tiny_array_base<V2, D2, N...> const & r)
{
    tiny_array<std::common_type_t<V1, V2>, N...> res(r.size(), dont_init);
    for(int k=0; k < r.size(); ++k)
        res[k] =  min(l, r[k]);
    return res;
}

    /** Index of minimal element.

        Returns -1 for an empty array.
    */
template <class V, class D, int ... N>
inline int
min_element(tiny_array_base<V, D, N...> const & l)
{
    if(l.size() == 0)
        return -1;
    int m = 0;
    for(int i=1; i<l.size(); ++i)
        if(l[i] < l[m])
            m = i;
    return m;
}

    /// minimal element
template <class V, class D, int ... N>
inline
V const &
min(tiny_array_base<V, D, N...> const & l)
{
    int m = min_element(l);
    xtensor_precondition(m >= 0, "min() on empty tiny_array.");
    return l[m];
}

    /// element-wise maximum
template <class V1, class D1, class V2, class D2, int ... N>
inline
tiny_array<std::common_type_t<V1, V2>, N...>
max(tiny_array_base<V1, D1, N...> const & l,
    tiny_array_base<V2, D2, N...> const & r)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., l.size() == r.size(),
        "max(): size mismatch.");
    tiny_array<std::common_type_t<V1, V2>, N...> res(l.size(), dont_init);
    for(int k=0; k < l.size(); ++k)
        res[k] =  max(l[k], r[k]);
    return res;
}

    /// element-wise maximum with a constant
template <class V1, class D1, class V2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V2>::value>>
inline
tiny_array<std::common_type_t<V1, V2>, N...>
max(tiny_array_base<V1, D1, N...> const & l,
    V2 const & r)
{
    tiny_array<std::common_type_t<V1, V2>, N...> res(l.size(), dont_init);
    for(int k=0; k < l.size(); ++k)
        res[k] =  max(l[k], r);
    return res;
}

    /// element-wise maximum with a constant
template <class V1, class V2, class D2, int ... N,
          XTENSOR_REQUIRE<!tiny_array_concept<V1>::value>>
inline
tiny_array<std::common_type_t<V1, V2>, N...>
max(V1 const & l,
    tiny_array_base<V2, D2, N...> const & r)
{
    tiny_array<std::common_type_t<V1, V2>, N...> res(r.size(), dont_init);
    for(int k=0; k < r.size(); ++k)
        res[k] =  max(l, r[k]);
    return res;
}

    /** Index of maximal element.

        Returns -1 for an empty array.
    */
template <class V, class D, int ... N>
inline int
max_element(tiny_array_base<V, D, N...> const & l)
{
    if(l.size() == 0)
        return -1;
    int m = 0;
    for(int i=1; i<l.size(); ++i)
        if(l[m] < l[i])
            m = i;
    return m;
}

    /// maximal element
template <class V, class D, int ... N>
inline V const &
max(tiny_array_base<V, D, N...> const & l)
{
    int m = max_element(l);
    xtensor_precondition(m >= 0, "max() on empty tiny_array.");
    return l[m];
}

/// squared norm
template <class V, class D, int ... N>
inline squared_norm_t<tiny_array_base<V, D, N...> >
squared_norm(tiny_array_base<V, D, N...> const & t)
{
    using Type = squared_norm_t<tiny_array_base<V, D, N...> >;
    Type result = Type();
    for(int i=0; i<t.size(); ++i)
        result += squared_norm(t[i]);
    return result;
}

template <class V, int N>
inline squared_norm_t<tiny_symmetric_view<V, N> >
squared_norm(tiny_symmetric_view<V, N> const & t)
{
    using Type = squared_norm_t<tiny_symmetric_view<V, N> >;
    Type result = Type();
    for (int i = 0; i < N; ++i)
    {
        result += squared_norm(t(i, i));
        for (int j = i + 1; j < N; ++j)
        {
            auto c = squared_norm(t(i, j));
            result += c + c;
        }
    }

    return result;
}

template <class V, class D, int ... N>
inline
norm_t<V>
mean_square(tiny_array_base<V, D, N...> const & t)
{
    return norm_t<V>(squared_norm(t)) / t.size();
}

    /// reversed copy
template <class V, class D, int ... N>
inline
tiny_array<V, N...>
reversed(tiny_array_base<V, D, N...> const & t)
{
    return tiny_array<V, N...>(t.begin(), t.end(), copy_reversed);
}

    /** \brief transposed copy

        Elements are arranged such that <tt>res[k] = t[permutation[k]]</tt>.
    */
template <class V1, class D1, class V2, class D2, int N, int M>
inline
tiny_array<V1, N>
transpose(tiny_array_base<V1, D1, N> const & v,
          tiny_array_base<V2, D2, M> const & permutation)
{
    return v.transpose(permutation);
}

template <class V1, class D1, int N>
inline
tiny_array<V1, N>
transpose(tiny_array_base<V1, D1, N> const & v)
{
    return reversed(v);
}

template <class V1, class D1, int N1, int N2>
inline
tiny_array<V1, N2, N1>
transpose(tiny_array_base<V1, D1, N1, N2> const & v)
{
    tiny_array<V1, N2, N1> res(dont_init);
    for(int i=0; i < N1; ++i)
    {
        for(int j=0; j < N2; ++j)
        {
            res(j,i) = v(i,j);
        }
    }
    return res;
}

template <class V, int N>
inline
tiny_symmetric_view<V, N>
transpose(tiny_symmetric_view<V, N> const & v)
{
    return v;
}

    /** \brief Clip negative values.

        All elements smaller than 0 are set to zero.
    */
template <class V, class D, int ... N>
inline
tiny_array<V, N...>
clip_lower(tiny_array_base<V, D, N...> const & t)
{
    return clip_lower(t, V());
}

    /** \brief Clip values below a threshold.

        All elements smaller than \a val are set to \a val.
    */
template <class V, class D, int ... N>
inline
tiny_array<V, N...>
clip_lower(tiny_array_base<V, D, N...> const & t, const V val)
{
    tiny_array<V, N...> res(t.size(), dont_init);
    for(int k=0; k < t.size(); ++k)
    {
        res[k] = t[k] < val ? val :  t[k];
    }
    return res;
}

    /** \brief Clip values above a threshold.

        All elements bigger than \a val are set to \a val.
    */
template <class V, class D, int ... N>
inline
tiny_array<V, N...>
clip_upper(tiny_array_base<V, D, N...> const & t, const V val)
{
    tiny_array<V, N...> res(t.size(), dont_init);
    for(int k=0; k < t.size(); ++k)
    {
        res[k] = t[k] > val ? val :  t[k];
    }
    return res;
}

    /** \brief Clip values to an interval.

        All elements less than \a valLower are set to \a valLower, all elements
        bigger than \a valUpper are set to \a valUpper.
    */
template <class V, class D, int ... N>
inline
tiny_array<V, N...>
clip(tiny_array_base<V, D, N...> const & t,
     const V valLower, const V valUpper)
{
    tiny_array<V, N...> res(t.size(), dont_init);
    for(int k=0; k < t.size(); ++k)
    {
        res[k] =  (t[k] < valLower)
                       ? valLower
                       : (t[k] > valUpper)
                             ? valUpper
                             : t[k];
    }
    return res;
}

    /** \brief Clip values to a vector of intervals.

        All elements less than \a valLower are set to \a valLower, all elements
        bigger than \a valUpper are set to \a valUpper.
    */
template <class V, class D1, class D2, class D3, int ... N>
inline
tiny_array<V, N...>
clip(tiny_array_base<V, D1, N...> const & t,
     tiny_array_base<V, D2, N...> const & valLower,
     tiny_array_base<V, D3, N...> const & valUpper)
{
    XTENSOR_ASSERT_RUNTIME_SIZE(N..., t.size() == valLower.size() && t.size() == valUpper.size(),
        "clip(): size mismatch.");
    tiny_array<V, N...> res(t.size(), dont_init);
    for(int k=0; k < t.size(); ++k)
    {
        res[k] =  (t[k] < valLower[k])
                       ? valLower[k]
                       : (t[k] > valUpper[k])
                             ? valUpper[k]
                             : t[k];
    }
    return res;
}

template <class T1, class D1, class T2, class D2, int ... N>
inline void
swap(tiny_array_base<T1, D1, N...> & l,
     tiny_array_base<T2, D2, N...> & r)
{
    l.swap(r);
}

} // namespace xt

#undef XTENSOR_ASSERT_INSIDE

#endif // XTENSOR_XTINY_HPP
