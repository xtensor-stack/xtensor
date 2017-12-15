/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_XTINY_HPP
#define XTENSOR_XTINY_HPP

#include <array>
#include <type_traits>
#include <iosfwd>
#include <algorithm>
#include <memory>
#include <iterator>
#include <utility>
#include <tuple>  // std::ignore

#include "xconcepts.hpp"
#include "xexception.hpp"
#include "xutils.hpp"
#include "xbuffer_adaptor.hpp"
#include "xstorage.hpp"

namespace xt
{
    /*****************/
    /* prerequisites */
    /*****************/

    using index_t = std::ptrdiff_t;

    static const index_t runtime_size  = -1;

    namespace tags
    {
        struct xtiny_tag {};

        struct skip_initialization_tag {};
    }

    namespace
    {
        tags::skip_initialization_tag  dont_init;
    }

    template <class T>
    struct xtiny_concept
    : public std::integral_constant<bool,
                                    std::is_base_of<tags::xtiny_tag, std::decay_t<T>>::value>
    {
    };

    /****************/
    /* declarations */
    /****************/

    template <class VALUETYPE, index_t N, class REPRESENTATION>
    class xtiny_impl;

    template <class VALUETYPE, index_t N=runtime_size, class REPRESENTATION=void>
    class xtiny;

    /*********/
    /* xtiny */
    /*********/

    /* Adds common functionality to the respective xtiny_impl */
    template <class VALUETYPE, index_t N, class REPRESENTATION>
    class xtiny
    : public xtiny_impl<VALUETYPE, N, REPRESENTATION>
    {
      public:

        using self_type = xtiny<VALUETYPE, N, REPRESENTATION>;
        using base_type = xtiny_impl<VALUETYPE, N, REPRESENTATION>;
        using value_type = typename base_type::value_type;
        using const_value_type = typename base_type::const_value_type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;
        using reverse_iterator = typename base_type::reverse_iterator;
        using const_reverse_iterator = typename base_type::const_reverse_iterator;
        using size_type = typename base_type::size_type;
        using difference_type = typename base_type::difference_type;
        using index_type = typename base_type::index_type;

        using base_type::owns_memory;
        using base_type::has_fixed_size;
        using base_type::static_size;

        using base_type::base_type;

        xtiny();
        xtiny(xtiny const & rhs);
        xtiny(xtiny && rhs);

        template <class T, index_t M, class R>
        xtiny(xtiny<T, M, R> const & rhs);

        template <class T, class A>
        xtiny(std::vector<T, A> const & v);

        template <class T, std::size_t M>
        xtiny(std::array<T, M> const & v);

        xtiny & operator=(xtiny const & rhs);
        xtiny & operator=(xtiny && rhs);

        xtiny & operator=(value_type const & v);

        template <class T, class A>
        xtiny & operator=(std::vector<T, A> const & v);

        template <class T, std::size_t M>
        xtiny & operator=(std::array<T, M> const & v);

        template <class U, index_t M, class R>
        xtiny & operator=(xtiny<U, M, R> const & rhs);

        using base_type::assign;

        template <class T>
        void assign(std::initializer_list<T> v);

        using base_type::data;

        using base_type::operator[];
        reference at(size_type i);
        constexpr const_reference at(size_type i) const;

        reference front();
        reference back();
        constexpr const_reference front() const;
        constexpr const_reference back()  const;

        template <index_t FROM, index_t TO>
        auto subarray();
        template <index_t FROM, index_t TO>
        auto subarray() const;
        auto subarray(size_type FROM, size_type TO);
        auto subarray(size_type FROM, size_type TO) const;

        auto erase(size_type m) const;
        auto pop_front() const;
        auto pop_back() const;

        auto insert(size_type m, value_type v) const;
        auto push_front(value_type v) const;
        auto push_back(value_type v) const;

        using base_type::begin;
        constexpr const_iterator cbegin() const;
        iterator end();
        constexpr const_iterator end() const;
        constexpr const_iterator cend() const;

        using base_type::rbegin;
        constexpr const_reverse_iterator crbegin() const;
        reverse_iterator rend();
        constexpr const_reverse_iterator rend() const;
        constexpr const_reverse_iterator crend() const;

        using base_type::size;
        using base_type::max_size;
        using base_type::shape;
        constexpr bool empty() const;

        using base_type::swap;
    };

    /******************************/
    /* default dynamic xtiny_impl */
    /******************************/

    template <class VALUETYPE>
    class xtiny_impl<VALUETYPE, runtime_size, void>
    : public xtiny_impl<VALUETYPE, runtime_size, VALUETYPE[4]>
    {
        using base_type = xtiny_impl<VALUETYPE, runtime_size, VALUETYPE[4]>;
      public:
        using base_type::base_type;
    };

    /******************************************/
    /* xtiny_impl: dynamic shape, owns memory */
    /******************************************/

    template <class VALUETYPE, index_t BUFFER_SIZE>
    class xtiny_impl<VALUETYPE, runtime_size, VALUETYPE[BUFFER_SIZE]>
    : public tags::xtiny_tag
    {
      public:
        using self_type = xtiny_impl<VALUETYPE, runtime_size, VALUETYPE[BUFFER_SIZE]>;
        using representation_type = VALUETYPE *;
        using buffer_type = VALUETYPE[BUFFER_SIZE < 1 ? 1 : BUFFER_SIZE];
        using allocator_type = std::allocator<VALUETYPE>;

        using value_type = VALUETYPE;
        using const_value_type = typename std::add_const<value_type>::type;
        using reference = value_type &;
        using const_reference = const_value_type &;
        using pointer = value_type *;
        using const_pointer = const_value_type *;
        using iterator = value_type *;
        using const_iterator = const_value_type *;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using index_type = std::array<size_type, 1>;

        static const bool owns_memory = true;
        static const bool has_fixed_size = false;
        static const index_t static_size = runtime_size;
        static const index_t buffer_size = BUFFER_SIZE;

        template <class NEW_VALUETYPE>
        using rebind = xtiny<NEW_VALUETYPE, runtime_size, NEW_VALUETYPE[BUFFER_SIZE]>;

        template <index_t NEW_SIZE>
        using rebind_size = xtiny<value_type, NEW_SIZE < runtime_size ? runtime_size : NEW_SIZE>;

        xtiny_impl();
        ~xtiny_impl();

        explicit xtiny_impl(size_type n);
        xtiny_impl(size_type n, const value_type& v);
        xtiny_impl(size_type n, tags::skip_initialization_tag);

        template <class IT,
                  class = detail::require_input_iter<IT>>
        xtiny_impl(IT begin, IT end);

        template <class T>
        xtiny_impl(std::initializer_list<T> const & v);

        xtiny_impl(xtiny_impl const & v);
        xtiny_impl(xtiny_impl && v);

        xtiny_impl & operator=(xtiny_impl const & v);
        xtiny_impl & operator=(xtiny_impl && v);

        void assign(size_type n, const value_type& v);

        template <class IT,
                  class = detail::require_input_iter<IT>>
        void assign(IT other_begin, IT other_end);

        reference operator[](size_type i);
        constexpr const_reference operator[](size_type i) const;

        pointer data();
        constexpr const_pointer data() const;

        void resize(size_type n);

        size_type capacity() const;
        size_type size() const;
        size_type max_size() const;
        index_type shape() const;
        bool on_stack() const;

        iterator begin();
        const_iterator begin() const;

        reverse_iterator rbegin();
        const_reverse_iterator rbegin() const;

        void swap(xtiny_impl & other);

      protected:
        static const bool may_use_uninitialized_memory = xtrivially_default_constructible<value_type>::value;

        /* allocate() assumes that m_size is already set,
           but no memory has been allocated yet */
        void allocate(value_type const & v = value_type());
        void allocate(tags::skip_initialization_tag);
        template <class IT,
                  class = detail::require_input_iter<IT>>
        void allocate(IT other_begin);

        void deallocate();

        allocator_type m_allocator;
        size_type m_size;
        representation_type m_data;
        buffer_type m_buffer;
    };

    /**********************************/
    /* default fixed shape xtiny_impl */
    /**********************************/

    template <class VALUETYPE, index_t N>
    class xtiny_impl<VALUETYPE, N, void>
    : public xtiny_impl<VALUETYPE, N, std::array<VALUETYPE, (std::size_t)N>>
    {
        using base_type = xtiny_impl<VALUETYPE, N, std::array<VALUETYPE, (std::size_t)N>>;
      public:
        using base_type::base_type;
    };

    /****************************************/
    /* xtiny_impl: fixed shape, owns memory */
    /****************************************/

    template <class VALUETYPE, index_t N>
    class xtiny_impl<VALUETYPE, N, std::array<VALUETYPE, (size_t)N>>
    : public std::array<VALUETYPE, (size_t)N>
    , public tags::xtiny_tag
    {
      public:
        using base_type = std::array<VALUETYPE, (size_t)N>;
        using self_type = xtiny_impl<VALUETYPE, N, base_type>;

        using value_type = VALUETYPE;
        using const_value_type = typename std::add_const<value_type>::type;
        using reference = typename base_type::reference;
        using const_reference = typename base_type::const_reference;
        using pointer = typename base_type::pointer;
        using const_pointer = typename base_type::const_pointer;
        using iterator = typename base_type::iterator;
        using const_iterator = typename base_type::const_iterator;
        using reverse_iterator = typename base_type::reverse_iterator;
        using const_reverse_iterator = typename base_type::const_reverse_iterator;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using index_type = std::array<size_type, 1>;

        static const bool owns_memory = true;
        static const bool has_fixed_size = true;
        static const index_t static_size = N;

        template <class NEW_VALUETYPE, index_t NEW_SIZE=N>
        using rebind = xtiny<NEW_VALUETYPE, NEW_SIZE < runtime_size ? runtime_size : NEW_SIZE>;

        template <index_t NEW_SIZE>
        using rebind_size = xtiny<value_type, NEW_SIZE < runtime_size ? runtime_size : NEW_SIZE>;

        xtiny_impl();

        explicit xtiny_impl(size_type n);
        xtiny_impl(size_type n, const value_type& v);
        xtiny_impl(size_type n, tags::skip_initialization_tag);

        template <class IT,
                  class = detail::require_input_iter<IT>>
        xtiny_impl(IT begin);

        template <class IT,
                  class = detail::require_input_iter<IT>>
        xtiny_impl(IT begin, IT end);

        template <class T>
        xtiny_impl(std::initializer_list<T> const & v);

        xtiny_impl(xtiny_impl const & v);
        xtiny_impl(xtiny_impl && v);

        xtiny_impl & operator=(xtiny_impl const & v);
        xtiny_impl & operator=(xtiny_impl && v);

        void assign(size_type n, const value_type& v);

        template <class IT,
                  class = detail::require_input_iter<IT>>
        void assign(IT other_begin, IT other_end);

        using base_type::operator[];
        using base_type::data;

        using base_type::size;
        using base_type::max_size;
        constexpr size_type capacity() const;
        constexpr index_type shape() const;

        using base_type::begin;
        using base_type::cbegin;
        using base_type::rbegin;
        using base_type::crbegin;
        using base_type::end;
        using base_type::cend;
        using base_type::rend;
        using base_type::crend;

        using base_type::swap;
    };

    /******************************/
    /* representation type traits */
    /******************************/

    namespace xtiny_detail
    {

        template <class T>
        struct test_value_type
        {
            static void test(...);

            template <class U>
            static typename U::value_type test(U *, typename U::value_type * = 0);

            static const bool value = !std::is_same<decltype(test((T*)0)), void>::value;
        };

        template <class T,
                  bool has_embedded_types=test_value_type<T>::value,
                  bool is_iterator=iterator_concept<T>::value>
        struct representation_type_traits;

        template <class T>
        struct representation_type_traits<T, true, false> // T is a container
        {
            using value_type = typename T::value_type;
            using iterator = typename T::iterator;
            using const_iterator = typename T::const_iterator;
            using reverse_iterator = typename T::reverse_iterator;
            using const_reverse_iterator = typename T::const_reverse_iterator;
        };

        template <class T>
        struct representation_type_traits<T, true, true> // T is an iterator
        {
            using value_type = typename T::value_type;
            using iterator = T;
            using const_iterator = T;
            using reverse_iterator = std::reverse_iterator<T>;
            using const_reverse_iterator = std::reverse_iterator<T>;
        };

        template <class T>
        struct representation_type_traits<T *, false, true>
        {
            using value_type             = T;
            using iterator               = value_type *;
            using const_iterator         = value_type const *;
            using reverse_iterator       = std::reverse_iterator<iterator>;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        };

        template <class T>
        struct representation_type_traits<T const *, false, true>
        {
            using value_type             = T const;
            using iterator               = value_type *;
            using const_iterator         = value_type *;
            using reverse_iterator       = std::reverse_iterator<iterator>;
            using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        };
    }

    /********************************************/
    /* xtiny_impl: fixed shape, borrowed memory */
    /********************************************/

    template <class VALUETYPE, index_t N, class REPRESENTATION>
    class xtiny_impl
    : public tags::xtiny_tag
    {
        using traits = xtiny_detail::representation_type_traits<REPRESENTATION>;
        using deduced_value_type = std::remove_const_t<typename traits::value_type>;
        static_assert(std::is_same<std::remove_const_t<VALUETYPE>, deduced_value_type>::value,
                      "xtiny_impl: type mismatch between VALUETYPE and REPRESENTATION.");

      public:
        using representation_type = REPRESENTATION;
        using self_type = xtiny_impl<VALUETYPE, N, representation_type>;

        using value_type = VALUETYPE;
        using const_value_type = typename std::add_const<value_type>::type;
        using reference              = value_type &;
        using const_reference        = const_value_type &;
        using pointer                = value_type *;
        using const_pointer          = const_value_type *;
        using iterator               = typename traits::iterator;
        using const_iterator         = typename traits::const_iterator;
        using reverse_iterator       = typename traits::reverse_iterator;
        using const_reverse_iterator = typename traits::const_reverse_iterator;
        using size_type              = std::size_t;
        using difference_type        = std::ptrdiff_t;
        using index_type             = std::array<index_t, 1>;

        static const bool owns_memory = false;
        static const bool has_fixed_size = true;
        static const index_t static_size = N;

        xtiny_impl();

        explicit xtiny_impl(representation_type const & begin);
        xtiny_impl(representation_type const & begin, representation_type const & end);

        template <class IT,
                  class = detail::require_input_iter<IT>>
        xtiny_impl(IT begin, IT end);

        xtiny_impl(xtiny_impl const & v) = default;
        xtiny_impl(xtiny_impl && v) = default;

        xtiny_impl & operator=(xtiny_impl const & v) = default;
        xtiny_impl & operator=(xtiny_impl && v) = default;

        void reset(representation_type const & begin);

        void assign(size_type n, const value_type& v);

        template <class IT,
                  class = detail::require_input_iter<IT>>
        void assign(IT other_begin, IT other_end);

        reference operator[](size_type i);
        constexpr const_reference operator[](size_type i) const;

        pointer data();
        constexpr const_pointer data() const;

        constexpr size_type size() const;
        constexpr size_type max_size() const;
        constexpr size_type capacity() const;
        constexpr index_type shape() const;

        iterator begin();
        constexpr const_iterator begin() const;

        reverse_iterator rbegin();
        constexpr const_reverse_iterator rbegin() const;

        void swap(xtiny_impl &);

      protected:

        representation_type m_data;
    };

    /**********************************************/
    /* xtiny_impl: dynamic shape, borrowed memory */
    /**********************************************/

    template <class VALUETYPE, class REPRESENTATION>
    class xtiny_impl<VALUETYPE, runtime_size, REPRESENTATION>
    : public tags::xtiny_tag
    {
        using traits = xtiny_detail::representation_type_traits<REPRESENTATION>;
        using deduced_value_type = std::remove_const_t<typename traits::value_type>;
        static_assert(std::is_same<std::remove_const_t<VALUETYPE>, deduced_value_type>::value,
                      "xtiny_impl: type mismatch between VALUETYPE and REPRESENTATION.");

      public:
        using representation_type = REPRESENTATION;
        using self_type = xtiny_impl<VALUETYPE, runtime_size, representation_type>;

        using value_type = VALUETYPE;
        using const_value_type = typename std::add_const<value_type>::type;
        using reference              = value_type &;
        using const_reference        = const_value_type &;
        using pointer                = value_type *;
        using const_pointer          = const_value_type *;
        using iterator               = typename traits::iterator;
        using const_iterator         = typename traits::const_iterator;
        using reverse_iterator       = typename traits::reverse_iterator;
        using const_reverse_iterator = typename traits::const_reverse_iterator;
        using size_type              = std::size_t;
        using difference_type        = std::ptrdiff_t;
        using index_type             = std::array<index_t, 1>;

        static const bool owns_memory = false;
        static const bool has_fixed_size = false;
        static const index_t static_size = runtime_size;

        xtiny_impl();

        xtiny_impl(representation_type const & begin, representation_type const & end);

        template <class IT,
                  class = detail::require_input_iter<IT>>
        xtiny_impl(IT begin, IT end);

        xtiny_impl(xtiny_impl const & v) = default;
        xtiny_impl(xtiny_impl && v) = default;

        xtiny_impl & operator=(xtiny_impl const & v) = default;
        xtiny_impl & operator=(xtiny_impl && v) = default;

        void reset(representation_type const & begin, representation_type const & end);

        void assign(size_type n, const value_type& v);

        template <class IT,
                  class = detail::require_input_iter<IT>>
        void assign(IT other_begin, IT other_end);

        reference operator[](size_type i);
        constexpr const_reference operator[](size_type i) const;

        pointer data();
        constexpr const_pointer data() const;

        constexpr size_type size() const;
        constexpr size_type max_size() const;
        constexpr size_type capacity() const;
        constexpr index_type shape() const;

        iterator begin();
        constexpr const_iterator begin() const;

        reverse_iterator rbegin();
        constexpr const_reverse_iterator rbegin() const;

        void swap(xtiny_impl &);

      protected:

        size_type m_size;
        representation_type m_data;
    };

    /**********************************************/
    /* xtiny_impl: dynamic shape, xbuffer_adaptor */
    /**********************************************/

    template <class VALUETYPE, class CP, class O, class A>
    class xtiny_impl<VALUETYPE, runtime_size, xbuffer_adaptor<CP, O, A>>
    : public xbuffer_adaptor<CP, O, A>
    , public tags::xtiny_tag
    {
        using deduced_value_type = typename xbuffer_adaptor<CP, O, A>::value_type;
        static_assert(std::is_same<VALUETYPE, deduced_value_type>::value,
                      "tiny_array_base: type mismatch between VALUETYPE and REPRESENTATION.");
      public:
        using base_type              = xbuffer_adaptor<CP, O, A>;
        using self_type              = xtiny_impl<VALUETYPE, runtime_size, base_type>;
        using value_type             = VALUETYPE;
        using const_value_type       = typename std::add_const<value_type>::type;
        using reference              = value_type &;
        using const_reference        = const_value_type &;
        using pointer                = value_type *;
        using const_pointer          = const_value_type *;
        using iterator               = typename base_type::iterator;
        using const_iterator         = typename base_type::const_iterator;
        using reverse_iterator       = typename base_type::reverse_iterator;
        using const_reverse_iterator = typename base_type::const_reverse_iterator;
        using size_type              = std::size_t;
        using difference_type        = std::ptrdiff_t;
        using index_type             = std::array<index_t, 1>;

        static const bool owns_memory = false;
        static const bool has_fixed_size = false;
        static const index_t static_size = runtime_size;

        using base_type::base_type;

        void assign(size_type n, const value_type& v);

        template <class IT,
                  class = detail::require_input_iter<IT>>
        void assign(IT other_begin, IT other_end);

        using base_type::operator[];
        using base_type::data;
        using base_type::size;

        constexpr size_type max_size() const;
        constexpr size_type capacity() const;
        constexpr index_type shape() const;

        using base_type::begin;
        using base_type::cbegin;
        using base_type::rbegin;
        using base_type::crbegin;
        using base_type::end;
        using base_type::cend;
        using base_type::rend;
        using base_type::crend;

        using base_type::swap;
        void swap(xtiny_impl &);
    };

    template <class V, index_t N, class R>
    inline void
    swap(xtiny<V, N, R> & l, xtiny<V, N, R> & r)
    {
        l.swap(r);
    }

    /****************/
    /* xtiny output */
    /****************/

    template <class T, index_t N, class R>
    std::ostream & operator<<(std::ostream & o, xtiny<T, N, R> const & v)
    {
        o << "{";
        if(v.size() > 0)
            o << promote_type_t<T>(v[0]);
        for(index_t i=1; i < (index_t)v.size(); ++i)
            o << ", " << promote_type_t<T>(v[i]);
        o << "}";
        return o;
    }

    /********************/
    /* xtiny comparison */
    /********************/

    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    operator==(xtiny<V1, N1, R1> const & l,
               xtiny<V2, N2, R2> const & r)
    {
        if(l.size() != r.size())
            return false;
        for(index_t k=0; k < (index_t)l.size(); ++k)
            if(l[k] != r[k])
                return false;
        return true;
    }

    template <class V1, index_t N1, class R1, class V2,
              XTENSOR_REQUIRE<!xtiny_concept<V2>::value &&
                              std::is_convertible<V2, V1>::value> >
    inline bool
    operator==(xtiny<V1, N1, R1> const & l,
               V2 const & r)
    {
        for(index_t k=0; k < (index_t)l.size(); ++k)
            if(l[k] != r)
                return false;
        return true;
    }

    template <class V1, class V2, index_t N2, class R2,
              XTENSOR_REQUIRE<!xtiny_concept<V1>::value &&
                              std::is_convertible<V2, V1>::value> >
    inline bool
    operator==(V1 const & l,
               xtiny<V2, N2, R2> const & r)
    {
        for(index_t k=0; k < (index_t)r.size(); ++k)
            if(l != r[k])
                return false;
        return true;
    }

    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    operator!=(xtiny<V1, N1, R1> const & l,
               xtiny<V2, N2, R2> const & r)
    {
        if(l.size() != r.size())
            return true;
        for(index_t k=0; k < (index_t)l.size(); ++k)
            if(l[k] != r[k])
                return true;
        return false;
    }

    template <class V1, index_t N1, class R1, class V2,
              XTENSOR_REQUIRE<!xtiny_concept<V2>::value &&
                              std::is_convertible<V2, V1>::value> >
    inline bool
    operator!=(xtiny<V1, N1, R1> const & l,
               V2 const & r)
    {
        for(index_t k=0; k < (index_t)l.size(); ++k)
            if(l[k] != r)
                return true;
        return false;
    }

    template <class V1, class V2, index_t N2, class R2,
              XTENSOR_REQUIRE<!xtiny_concept<V1>::value &&
                              std::is_convertible<V2, V1>::value> >
    inline bool
    operator!=(V1 const & l,
               xtiny<V2, N2, R2> const & r)
    {
        for(index_t k=0; k < (index_t)r.size(); ++k)
            if(l != r[k])
                return true;
        return false;
    }

    /************************/
    /* xtiny implementation */
    /************************/

    template <class V, index_t N, class R>
    inline
    xtiny<V, N, R>::xtiny()
    : base_type()
    {
    }

    template <class V, index_t N, class R>
    inline
    xtiny<V, N, R>::xtiny(xtiny const & v)
    : base_type(v)
    {
    }

    template <class V, index_t N, class R>
    inline
    xtiny<V, N, R>::xtiny(xtiny && v)
    : base_type(std::forward<base_type>(v))
    {
    }

    template <class V, index_t N, class R>
    template <class T, index_t M, class Q>
    inline
    xtiny<V, N, R>::xtiny(xtiny<T, M, Q> const & v)
    : base_type(v.begin(), v.end())
    {
    }

    template <class V, index_t N, class R>
    template <class T, class A>
    inline
    xtiny<V, N, R>::xtiny(std::vector<T, A> const & v)
    : base_type(v.begin(), v.end())
    {
    }

    template <class V, index_t N, class R>
    template <class T, std::size_t M>
    inline
    xtiny<V, N, R>::xtiny(std::array<T, M> const & v)
    : base_type(v.begin(), v.end())
    {
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::operator=(xtiny const & v) -> xtiny &
    {
        base_type::operator=(v);
        return *this;
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::operator=(xtiny && v) -> xtiny &
    {
        base_type::operator=(v);
        return *this;
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::operator=(value_type const & v) -> xtiny &
    {
        base_type::assign(size(), v);
        return *this;
    }

    template <class V, index_t N, class R>
    template <class T, class A>
    inline auto
    xtiny<V, N, R>::operator=(std::vector<T, A> const & v) -> xtiny &
    {
        base_type::assign(v.begin(), v.end());
        return *this;
    }

    template <class V, index_t N, class R>
    template <class T, std::size_t M>
    inline auto
    xtiny<V, N, R>::operator=(std::array<T, M> const & v) -> xtiny &
    {
        base_type::assign(v.begin(), v.end());
        return *this;
    }

    template <class V, index_t N, class R>
    template <class U, index_t M, class Q>
    inline auto
    xtiny<V, N, R>::operator=(xtiny<U, M, Q> const & v) -> xtiny &
    {
        base_type::assign(v.begin(), v.end());
        return *this;
    }

    template <class V, index_t N, class R>
    template <class T>
    inline void
    xtiny<V, N, R>::assign(std::initializer_list<T> v)
    {
        base_type::assign(v.begin(), v.end());
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::at(size_type i) -> reference
    {
        if(i < 0 || i >= size())
            throw std::out_of_range("xtiny::at()");
        return (*this)[i];
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny<V, N, R>::at(size_type i) const -> const_reference
    {
        if(i < 0 || i >= size())
            throw std::out_of_range("xtiny::at()");
        return (*this)[i];
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::front() -> reference
    {
        return (*this)[0];
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::back() -> reference
    {
        return (*this)[size()-1];
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny<V, N, R>::front() const -> const_reference
    {
        return (*this)[0];
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny<V, N, R>::back() const -> const_reference
    {
        return (*this)[size()-1];
    }

    template <class V, index_t N, class R>
    template <index_t FROM, index_t TO>
    inline auto
    xtiny<V, N, R>::subarray()
    {
        static_assert(FROM >= 0 && FROM < TO,
            "xtiny::subarray(): range out of bounds.");
        XTENSOR_PRECONDITION(TO <= size(),
            "xtiny::subarray(): range out of bounds.");
        return xtiny<value_type, TO-FROM, iterator>(begin()+FROM);
    }

    template <class V, index_t N, class R>
    template <index_t FROM, index_t TO>
    inline auto
    xtiny<V, N, R>::subarray() const
    {
        static_assert(FROM >= 0 && FROM < TO,
            "xtiny::subarray(): range out of bounds.");
        XTENSOR_PRECONDITION(TO <= size(),
            "xtiny::subarray(): range out of bounds.");
        return xtiny<const_value_type, TO-FROM, const_iterator>(begin()+FROM);
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::subarray(size_type FROM, size_type TO)
    {
        XTENSOR_PRECONDITION(FROM >= 0 && FROM < TO && TO <= size(),
            "xtiny::subarray(): range out of bounds.");
        return xtiny<value_type, runtime_size, iterator>(begin()+FROM, begin()+TO);
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::subarray(size_type FROM, size_type TO) const
    {
        XTENSOR_PRECONDITION(FROM >= 0 && FROM < TO && TO <= size(),
            "xtiny::subarray(): range out of bounds.");
        return xtiny<const_value_type, runtime_size, const_iterator>(begin()+FROM, begin()+TO);
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::erase(size_type m) const
    {
        XTENSOR_PRECONDITION(m >= 0 && m < size(), "xtiny::erase(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+").");
        static const index_t res_size = has_fixed_size
                                            ? static_size-1
                                            : runtime_size;
        xtiny<value_type, res_size> res(size()-1, dont_init);
        for(size_type k=0; k<m; ++k)
            res[k] = (*this)[k];
        for(size_type k=m+1; k<size(); ++k)
            res[k-1] = (*this)[k];
        return res;
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::pop_front() const
    {
        return erase(0);
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::pop_back() const
    {
        return erase(size()-1);
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::insert(size_type m, value_type v) const
    {
        XTENSOR_PRECONDITION(m >= 0 && m <= size(), "xtiny::insert(): "
            "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(size())+"].");
        static const index_t res_size = has_fixed_size
                                            ? static_size+1
                                            : runtime_size;
        xtiny<value_type, res_size> res(size()+1, dont_init);
        for(size_type k=0; k<m; ++k)
            res[k] = (*this)[k];
        res[m] = v;
        for(size_type k=m; k<size(); ++k)
            res[k+1] = (*this)[k];
        return res;
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::push_front(value_type v) const
    {
        return insert(0, v);
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::push_back(value_type v) const
    {
        return insert(size(), v);
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny<V, N, R>::cbegin() const -> const_iterator
    {
        return base_type::begin();
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::end() -> iterator
    {
        return begin() + size();
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny<V, N, R>::end() const -> const_iterator
    {
        return begin() + size();
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny<V, N, R>::cend() const -> const_iterator
    {
        return cbegin() + size();
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny<V, N, R>::crbegin() const -> const_reverse_iterator
    {
        return base_type::rbegin();
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny<V, N, R>::rend() -> reverse_iterator
    {
        return rbegin() + size();
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny<V, N, R>::rend() const -> const_reverse_iterator
    {
        return rbegin() + size();
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny<V, N, R>::crend() const -> const_reverse_iterator
    {
        return crbegin() + size();
    }

    template <class V, index_t N, class R>
    constexpr bool
    xtiny<V, N, R>::empty() const
    {
        return size() == 0;
    }

    /*******************************************/
    /* xtiny_impl dynamic shape implementation */
    /*******************************************/

    template <class V, index_t B>
    inline
    xtiny_impl<V, runtime_size, V[B]>::xtiny_impl()
    : m_size(0)
    , m_data(m_buffer)
    {
    }

    template <class V, index_t B>
    inline
    xtiny_impl<V, runtime_size, V[B]>::~xtiny_impl()
    {
        deallocate();
    }

    template <class V, index_t B>
    inline
    xtiny_impl<V, runtime_size, V[B]>::xtiny_impl(size_type n)
    : m_size(n)
    , m_data(m_buffer)
    {
        allocate();
    }

    template <class V, index_t B>
    inline
    xtiny_impl<V, runtime_size, V[B]>::xtiny_impl(size_type n, const value_type& v)
    : m_size(n)
    , m_data(m_buffer)
    {
        allocate(v);
    }

    template <class V, index_t B>
    inline
    xtiny_impl<V, runtime_size, V[B]>::xtiny_impl(size_type n, tags::skip_initialization_tag)
    : m_size(n)
    , m_data(m_buffer)
    {
        allocate(dont_init);
    }

    template <class V, index_t B>
    template <class IT, class>
    inline
    xtiny_impl<V, runtime_size, V[B]>::xtiny_impl(IT begin, IT end)
    : m_size(0)
    , m_data(m_buffer)
    {
        assign(begin, end);
    }

    template <class V, index_t B>
    template <class T>
    inline
    xtiny_impl<V, runtime_size, V[B]>::xtiny_impl(std::initializer_list<T> const & v)
    : m_size(0)
    , m_data(m_buffer)
    {
        assign(v.begin(), v.end());
    }

    template <class V, index_t B>
    inline
    xtiny_impl<V, runtime_size, V[B]>::xtiny_impl(xtiny_impl const & v)
    : xtiny_impl(v.begin(), v.begin()+v.size())
    {
    }

    template <class V, index_t B>
    inline
    xtiny_impl<V, runtime_size, V[B]>::xtiny_impl(xtiny_impl && v)
    : m_size(0)
    , m_data(m_buffer)
    {
        v.swap(*this);
    }

    template <class V, index_t B>
    inline auto
    xtiny_impl<V, runtime_size, V[B]>::operator=(xtiny_impl const & v) -> xtiny_impl &
    {
        if(this != &v)
        {
            assign(v.begin(), v.begin()+v.size());
        }
        return *this;
    }

    template <class V, index_t B>
    inline auto
    xtiny_impl<V, runtime_size, V[B]>::operator=(xtiny_impl && v) -> xtiny_impl &
    {
        if(this != &v)
        {
            assign(v.begin(), v.begin() + v.size());
        }
        return *this;
    }

    template <class V, index_t B>
    inline void
    xtiny_impl<V, runtime_size, V[B]>::assign(size_type n, const value_type& v)
    {
        if(m_size == n)
        {
            std::fill(begin(), begin()+size(), v);
        }
        else
        {
            deallocate();
            m_size = n;
            allocate(v);
        }
    }

    template <class V, index_t B>
    template <class IT, class>
    inline void
    xtiny_impl<V, runtime_size, V[B]>::assign(IT begin, IT end)
    {
        size_type n = std::distance(begin, end);
        if(m_size == n)
        {
            for (size_type k = 0; k < m_size; ++k, ++begin)
            {
                m_data[k] = static_cast<value_type>(*begin);
            }
        }
        else
        {
            deallocate();
            m_size = n;
            allocate(begin);
        }
    }

    template <class V, index_t B>
    inline auto
    xtiny_impl<V, runtime_size, V[B]>::data() -> pointer
    {
        return m_data;
    }

    template <class V, index_t B>
    constexpr inline auto
    xtiny_impl<V, runtime_size, V[B]>::data() const -> const_pointer
    {
        return m_data;
    }

    template <class V, index_t B>
    inline auto
    xtiny_impl<V, runtime_size, V[B]>::operator[](size_type i) -> reference
    {
        return m_data[i];
    }

    template <class V, index_t B>
    constexpr inline auto
    xtiny_impl<V, runtime_size, V[B]>::operator[](size_type i) const -> const_reference
    {
        return m_data[i];
    }

    template <class V, index_t B>
    inline void
    xtiny_impl<V, runtime_size, V[B]>::resize(size_type n)
    {
        if(n != m_size)
        {
            deallocate();
            m_size = n;
            allocate();
        }
    }

    template <class V, index_t B>
    inline auto
    xtiny_impl<V, runtime_size, V[B]>::capacity() const -> size_type
    {
        return std::max<std::size_t>(m_size, buffer_size);
    }

    template <class V, index_t B>
    inline auto
    xtiny_impl<V, runtime_size, V[B]>::size() const -> size_type
    {
        return m_size;
    }

    template <class V, index_t B>
    inline auto
    xtiny_impl<V, runtime_size, V[B]>::max_size() const -> size_type
    {
        return m_allocator.max_size();
    }

    template <class V, index_t B>
    inline auto
    xtiny_impl<V, runtime_size, V[B]>::shape() const -> index_type
    {
        return {m_size};
    }

    template <class V, index_t B>
    inline bool
    xtiny_impl<V, runtime_size, V[B]>::on_stack() const
    {
        return m_data == m_buffer;
    }

    template <class V, index_t B>
    inline auto
    xtiny_impl<V, runtime_size, V[B]>::begin() -> iterator
    {
        return m_data;
    }

    template <class V, index_t B>
    inline auto
    xtiny_impl<V, runtime_size, V[B]>::begin() const -> const_iterator
    {
        return m_data;
    }

    template <class V, index_t B>
    inline auto
    xtiny_impl<V, runtime_size, V[B]>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(m_data + m_size);
    }

    template <class V, index_t B>
    inline auto
    xtiny_impl<V, runtime_size, V[B]>::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_data + m_size);
    }

    template <class V, index_t B>
    inline void
    xtiny_impl<V, runtime_size, V[B]>::swap(xtiny_impl & other)
    {
        using std::swap;
        if(this == &other)
        {
            return;
        }
        if(m_size == 0 || m_size > buffer_size)
        {
            if(other.m_size == 0 || other.m_size > buffer_size)
            {
                // both use allocated memory (or no memory at all)
                swap(m_data, other.m_data);
            }
            else
            {
                // self uses allocated memory, other the buffer
                for(size_type k=0; k<other.m_size; ++k)
                {
                    m_buffer[k] = other.m_data[k];
                }
                other.m_data = m_data;
                m_data = m_buffer;
            }
        }
        else
        {
            if(other.m_size > buffer_size)
            {
                // self uses the buffer, other allocated memory
                for(size_type k=0; k<m_size; ++k)
                {
                    other.m_buffer[k] = m_data[k];
                }
                m_data = other.m_data;
                other.m_data = other.m_buffer;
            }
            else
            {
                // both use the buffer
                if(m_size < other.m_size)
                {
                    for(size_type k=0; k<m_size; ++k)
                    {
                        swap(m_data[k], other.m_data[k]);
                    }
                    for(size_type k=m_size; k<other.m_size; ++k)
                    {
                        m_data[k] = other.m_data[k];
                    }
                }
                else
                {
                    for(size_type k=0; k<other.m_size; ++k)
                    {
                        swap(m_data[k], other.m_data[k]);
                    }
                    for(size_type k=other.m_size; k<m_size; ++k)
                    {
                        other.m_data[k] = m_data[k];
                    }
                }
            }
        }
        swap(m_size, other.m_size);
    }

    template <class V, index_t B>
    inline void
    xtiny_impl<V, runtime_size, V[B]>::allocate(value_type const & v)
    {
        if(m_size > buffer_size)
        {
            m_data = m_allocator.allocate(m_size);
            std::uninitialized_fill(m_data, m_data+m_size, v);
        }
        else
        {
            std::fill(m_data, m_data+m_size, v);
        }
    }

    template <class V, index_t B>
    inline void
    xtiny_impl<V, runtime_size, V[B]>::allocate(tags::skip_initialization_tag)
    {
        if(m_size > buffer_size)
        {
            m_data = m_allocator.allocate(m_size);
            if(!may_use_uninitialized_memory)
                std::uninitialized_fill(m_data, m_data+m_size, value_type());
        }
    }

    template <class V, index_t B>
    template <class IT, class>
    inline void
    xtiny_impl<V, runtime_size, V[B]>::allocate(IT begin)
    {
        if(m_size > buffer_size)
        {
            m_data = m_allocator.allocate(m_size);
            for(size_type k=0; k<m_size; ++k, ++begin)
                m_allocator.construct(m_data+k, static_cast<value_type>(*begin));
        }
        else
        {
            for(size_type k=0; k<m_size; ++k, ++begin)
                m_data[k] = static_cast<value_type>(*begin);
        }
    }

    template <class V, index_t B>
    inline void
    xtiny_impl<V, runtime_size, V[B]>::deallocate()
    {
        if(m_size > buffer_size)
        {
            if(!may_use_uninitialized_memory)
            {
                for(size_type k=0; k<m_size; ++k)
                    m_allocator.destroy(m_data+k);
            }
            m_allocator.deallocate(m_data, m_size);
            m_data = m_buffer;
        }
        m_size = 0;
    }

    /*****************************************/
    /* xtiny_impl fixed shape implementation */
    /*****************************************/

    template <class V, index_t N>
    inline
    xtiny_impl<V, N, std::array<V, (size_t)N>>::xtiny_impl()
    : base_type{}
    {
    }

    template <class V, index_t N>
    inline
    xtiny_impl<V, N, std::array<V, (size_t)N>>::xtiny_impl(size_type n)
    : xtiny_impl()
    {
        std::ignore = n;
        XTENSOR_ASSERT_MSG(n == size(), "xtiny_impl(n): size mismatch");
    }

    template <class V, index_t N>
    inline
    xtiny_impl<V, N, std::array<V, (size_t)N>>::xtiny_impl(size_type n, const value_type& v)
    {
        std::ignore = n;
        XTENSOR_ASSERT_MSG(n == size(), "xtiny_impl(n): size mismatch");
        base_type::fill(v);
    }

    template <class V, index_t N>
    inline
    xtiny_impl<V, N, std::array<V, (size_t)N>>::xtiny_impl(size_type n, tags::skip_initialization_tag)
    {
        std::ignore = n;
        XTENSOR_ASSERT_MSG(n == size(), "xtiny_impl(n): size mismatch.");
    }

    template <class V, index_t N>
    template <class IT, class>
    inline
    xtiny_impl<V, N, std::array<V, (size_t)N>>::xtiny_impl(IT begin)
    {
        assign(begin, begin+N);
    }

    template <class V, index_t N>
    template <class IT, class>
    inline
    xtiny_impl<V, N, std::array<V, (size_t)N>>::xtiny_impl(IT begin, IT end)
    {
        assign(begin, end);
    }

    template <class V, index_t N>
    template <class T>
    inline
    xtiny_impl<V, N, std::array<V, (size_t)N>>::xtiny_impl(std::initializer_list<T> const & v)
    {
        const size_t n = v.size();
        if(n == 1)
        {
            assign(N, static_cast<value_type>(*v.begin()));
        }
        else if(n == N)
        {
            assign(v.begin(), v.end());
        }
        else
        {
            XTENSOR_ASSERT_MSG(false, "xtiny_impl::xtiny_impl(std::initializer_list<T>): size mismatch.");
        }
    }

    template <class V, index_t N>
    inline
    xtiny_impl<V, N, std::array<V, (size_t)N>>::xtiny_impl(xtiny_impl const & v)
    : base_type(v)
    {
    }

    template <class V, index_t N>
    inline
    xtiny_impl<V, N, std::array<V, (size_t)N>>::xtiny_impl(xtiny_impl && v)
    : base_type(std::forward<base_type>(v))
    {
    }

    template <class V, index_t N>
    inline auto
    xtiny_impl<V, N, std::array<V, (size_t)N>>::operator=(xtiny_impl const & v) -> xtiny_impl &
    {
        base_type::operator=(v);
        return *this;
    }

    template <class V, index_t N>
    inline auto
    xtiny_impl<V, N, std::array<V, (size_t)N>>::operator=(xtiny_impl && v) -> xtiny_impl &
    {
        base_type::operator=(std::forward<base_type>(v));
        return *this;
    }

    template <class V, index_t N>
    inline void
    xtiny_impl<V, N, std::array<V, (size_t)N>>::assign(size_type n, const value_type& v)
    {
        std::ignore = n;
        XTENSOR_ASSERT_MSG(n == size(), "xtiny_impl::assign(n, v): size mismatch.");
        base_type::fill(v);
    }

    template <class V, index_t N>
    template <class IT, class>
    inline void
    xtiny_impl<V, N, std::array<V, (size_t)N>>::assign(IT begin, IT end)
    {
        std::ignore = end;
        XTENSOR_ASSERT_MSG(std::distance(begin, end) == size(), "xtiny_impl::assign(begin, end): size mismatch.");
        for(size_type k=0; k<N; ++k, ++begin)
        {
            (*this)[k] = static_cast<value_type>(*begin);
        }
    }

    template <class V, index_t N>
    constexpr inline auto
    xtiny_impl<V, N, std::array<V, (size_t)N>>::capacity() const -> size_type
    {
        return N;
    }

    template <class V, index_t N>
    constexpr inline auto
    xtiny_impl<V, N, std::array<V, (size_t)N>>::shape() const -> index_type
    {
        return {N};
    }

    /**********************************************/
    /* xtiny_impl fixed shape view implementation */
    /**********************************************/

    template <class V, index_t N, class R>
    inline
    xtiny_impl<V, N, R>::xtiny_impl()
    : m_data()
    {
    }

    template <class V, index_t N, class R>
    inline
    xtiny_impl<V, N, R>::xtiny_impl(representation_type const & begin)
    : m_data(begin)
    {
    }

    template <class V, index_t N, class R>
    inline
    xtiny_impl<V, N, R>::xtiny_impl(representation_type const & begin, representation_type const & end)
    : m_data(begin)
    {
        std::ignore = end;
        XTENSOR_ASSERT_MSG(std::distance(begin, end) == size(), "xtiny_impl(begin, end): size mismatch");
    }

    template <class V, index_t N, class R>
    template <class IT, class>
    inline
    xtiny_impl<V, N, R>::xtiny_impl(IT begin, IT end)
    : m_data(const_cast<representation_type>(&*begin))
    {
        std::ignore = end;
        XTENSOR_ASSERT_MSG(std::distance(begin, end) == size(), "xtiny_impl::assign(begin, end): size mismatch.");
    }

    template <class V, index_t N, class R>
    inline void
    xtiny_impl<V, N, R>::reset(representation_type const & begin)
    {
        m_data = begin;
    }

    template <class V, index_t N, class R>
    inline void
    xtiny_impl<V, N, R>::assign(size_type n, const value_type& v)
    {
        std::ignore = n;
        XTENSOR_ASSERT_MSG(n == size(), "xtiny_impl::assign(n, v): size mismatch.");
        for(size_type k=0; k<N; ++k)
        {
            (*this)[k] = v;
        }
    }

    template <class V, index_t N, class R>
    template <class IT, class>
    inline void
    xtiny_impl<V, N, R>::assign(IT begin, IT end)
    {
        std::ignore = end;
        XTENSOR_ASSERT_MSG(std::distance(begin, end) == size(), "xtiny_impl::assign(begin, end): size mismatch.");
        for(size_type k=0; k<N; ++k, ++begin)
        {
            (*this)[k] = static_cast<value_type>(*begin);
        }
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny_impl<V, N, R>::operator[](size_type i) -> reference
    {
        return m_data[i];
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny_impl<V, N, R>::operator[](size_type i) const -> const_reference
    {
        return m_data[i];
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny_impl<V, N, R>::data() -> pointer
    {
        return &m_data[0];
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny_impl<V, N, R>::data() const -> const_pointer
    {
        return &m_data[0];
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny_impl<V, N, R>::size() const -> size_type
    {
        return N;
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny_impl<V, N, R>::max_size() const -> size_type
    {
        return N;
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny_impl<V, N, R>::capacity() const -> size_type
    {
        return N;
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny_impl<V, N, R>::shape() const -> index_type
    {
        return {N};
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny_impl<V, N, R>::begin() -> iterator
    {
        return m_data;
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny_impl<V, N, R>::begin() const -> const_iterator
    {
        return m_data;
    }

    template <class V, index_t N, class R>
    inline auto
    xtiny_impl<V, N, R>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(m_data+N);
    }

    template <class V, index_t N, class R>
    constexpr inline auto
    xtiny_impl<V, N, R>::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_data+N);
    }

    template <class V, index_t N, class R>
    inline void
    xtiny_impl<V, N, R>::swap(xtiny_impl & other)
    {
        using std::swap;
        swap(m_data, other.m_data);
    }

    /************************************************/
    /* xtiny_impl dynamic shape view implementation */
    /************************************************/

    template <class V, class R>
    inline
    xtiny_impl<V, runtime_size, R>::xtiny_impl()
    : m_size(0)
    , m_data()
    {
    }

    template <class V, class R>
    inline
    xtiny_impl<V, runtime_size, R>::xtiny_impl(representation_type const & begin, representation_type const & end)
    : m_size(std::distance(begin, end))
    , m_data(begin)
    {
    }

    template <class V, class R>
    template <class IT, class>
    inline
    xtiny_impl<V, runtime_size, R>::xtiny_impl(IT begin, IT end)
    : m_size(std::distance(begin, end))
    , m_data(const_cast<representation_type>(&*begin))
    {
    }

    template <class V, class R>
    inline void
    xtiny_impl<V, runtime_size, R>::reset(representation_type const & begin, representation_type const & end)
    {
        m_size = std::distance(begin, end);
        m_data = begin;
    }

    template <class V, class R>
    inline void
    xtiny_impl<V, runtime_size, R>::assign(size_type n, const value_type& v)
    {
        XTENSOR_ASSERT_MSG(n == size(), "xtiny_impl::assign(n, v): size mismatch.");
        for(size_type k=0; k<size(); ++k)
        {
            (*this)[k] = v;
        }
    }

    template <class V, class R>
    template <class IT, class>
    inline void
    xtiny_impl<V, runtime_size, R>::assign(IT begin, IT end)
    {
        std::ignore = end;
        XTENSOR_ASSERT_MSG(std::distance(begin, end) == size(), "xtiny_impl::assign(begin, end): size mismatch.");
        for(size_type k=0; k<size(); ++k, ++begin)
        {
            (*this)[k] = static_cast<value_type>(*begin);
        }
    }

    template <class V, class R>
    inline auto
    xtiny_impl<V, runtime_size, R>::operator[](size_type i) -> reference
    {
        return m_data[i];
    }

    template <class V, class R>
    constexpr inline auto
    xtiny_impl<V, runtime_size, R>::operator[](size_type i) const -> const_reference
    {
        return m_data[i];
    }

    template <class V, class R>
    inline auto
    xtiny_impl<V, runtime_size, R>::data() -> pointer
    {
        return &m_data[0];
    }

    template <class V, class R>
    constexpr inline auto
    xtiny_impl<V, runtime_size, R>::data() const -> const_pointer
    {
        return &m_data[0];
    }

    template <class V, class R>
    constexpr inline auto
    xtiny_impl<V, runtime_size, R>::size() const -> size_type
    {
        return m_size;
    }

    template <class V, class R>
    constexpr inline auto
    xtiny_impl<V, runtime_size, R>::max_size() const -> size_type
    {
        return m_size;
    }

    template <class V, class R>
    constexpr inline auto
    xtiny_impl<V, runtime_size, R>::capacity() const -> size_type
    {
        return m_size;
    }

    template <class V, class R>
    constexpr inline auto
    xtiny_impl<V, runtime_size, R>::shape() const -> index_type
    {
        return {m_size};
    }

    template <class V, class R>
    inline auto
    xtiny_impl<V, runtime_size, R>::begin() -> iterator
    {
        return m_data;
    }

    template <class V, class R>
    constexpr inline auto
    xtiny_impl<V, runtime_size, R>::begin() const -> const_iterator
    {
        return m_data;
    }

    template <class V, class R>
    inline auto
    xtiny_impl<V, runtime_size, R>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(m_data+m_size);
    }

    template <class V, class R>
    constexpr inline auto
    xtiny_impl<V, runtime_size, R>::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_data+m_size);
    }

    template <class V, class R>
    inline void
    xtiny_impl<V, runtime_size, R>::swap(xtiny_impl & other)
    {
        using std::swap;
        swap(m_size, other.m_size);
        swap(m_data, other.m_data);
    }


    /**************************************************/
    /* xtiny_impl xbuffer_adaptor view implementation */
    /**************************************************/

    template <class V, class CP, class O, class A>
    inline void
    xtiny_impl<V, runtime_size, xbuffer_adaptor<CP, O, A>>::assign(size_type n, const value_type& v)
    {
        XTENSOR_ASSERT_MSG(n == size(), "xtiny_impl::assign(n, v): size mismatch.");
        for(size_type k=0; k<n; ++k)
        {
            (*this)[k] = v;
        }
    }

    template <class V, class CP, class O, class A>
    template <class IT, class>
    inline void
    xtiny_impl<V, runtime_size, xbuffer_adaptor<CP, O, A>>::assign(IT begin, IT end)
    {
        std::ignore = end;
        XTENSOR_ASSERT_MSG(std::distance(begin, end) == size(), "xtiny_impl::assign(begin, end): size mismatch.");
        for(size_type k=0; k<size(); ++k, ++begin)
        {
            (*this)[k] = static_cast<value_type>(*begin);
        }
    }

    template <class V, class CP, class O, class A>
    constexpr inline auto
    xtiny_impl<V, runtime_size, xbuffer_adaptor<CP, O, A>>::max_size() const -> size_type
    {
        return size();
    }

    template <class V, class CP, class O, class A>
    constexpr inline auto
    xtiny_impl<V, runtime_size, xbuffer_adaptor<CP, O, A>>::capacity() const -> size_type
    {
        return size();
    }

    template <class V, class CP, class O, class A>
    constexpr inline auto
    xtiny_impl<V, runtime_size, xbuffer_adaptor<CP, O, A>>::shape() const -> index_type
    {
        return {size()};
    }

} // namespace xt

#endif // XTENSOR_XTINY_HPP
