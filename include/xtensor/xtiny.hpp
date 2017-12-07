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
#include "xmath.hpp"
#include "xnorm.hpp"

namespace xt
{
    namespace tags
    {
        struct tiny_array_tag {};

        enum skip_initialization_tag { dont_init };
    }

    using tags::dont_init;

    using index_t = std::ptrdiff_t;

        /// Determine size of an array type at runtime.
    static const index_t runtime_size  = -1;

    template <class T>
    struct tiny_array_concept
    {
        static const bool value = std::is_base_of<tags::tiny_array_tag, std::decay_t<T>>::value;
    };

    template <class VALUETYPE, index_t N, class REPRESENTATION>
    class tiny_array_base;

    template <class VALUETYPE, index_t N=runtime_size, class REPRESENTATION=void>
    class tiny_array;

    namespace tiny_detail
    {
        template <class T>
        struct value_type_traits
        {
            void test(...);

            template <class U>
            typename U::value_type test(U *, typename U::value_type * = 0);

            using type = decltype(test((T*)0));
        };

        template <class T>
        struct value_type_traits<T *>
        {
            using type = T;
        };

        template <class T>
        struct value_type_traits<T const *>
        {
            using type = T const;
        };
    }

    /**************************************************/
    /* tiny_array_base: static shape, borrowed memory */
    /**************************************************/

    template <class VALUETYPE, index_t N, class REPRESENTATION>
    class tiny_array_base
    : public tags::tiny_array_tag
    {
        using deduced_value_type = typename tiny_detail::value_type_traits<REPRESENTATION>::type;
        static_assert(std::is_same<VALUETYPE, deduced_value_type>::value,
                      "tiny_array_base: type mismatch between VALUETYPE and REPRESENTATION.");
      public:
        using representation_type    = REPRESENTATION;
        using value_type             = VALUETYPE;
        using const_value_type       = typename std::add_const<value_type>::type;
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
        using index_type             = std::array<index_t, 1>;

        static const bool    owns_memory = false;
        static const bool    is_static   = true;
        static const index_t static_size = N;

        constexpr tiny_array_base(tiny_array_base const &) = default;

        explicit tiny_array_base()
        : data_(nullptr)
        {}

        template <class R>
        explicit
        tiny_array_base(tiny_array<value_type, N, R> const & other)
        : data_(other.begin())
        {
            XTENSOR_PRECONDITION(size() == other.size(),
                "tiny_array_base(tiny_array): size mismatch.");
        }

        explicit tiny_array_base(representation_type const & begin)
        : data_(begin)
        {}

        explicit tiny_array_base(representation_type const & begin, representation_type const & end)
        : data_(begin)
        {
            std::ignore = end;
            XTENSOR_ASSERT_MSG(end - data_ == static_size,
                "tiny_array_base(representation_type begin, representation_type end): size mismatch.");
        }

        void reset(representation_type const & begin)
        {
            data_ = begin;
        }

        void init(value_type const & v = value_type())
        {
            for(index_t i=0; i<static_size; ++i)
                data_[i] = v;
        }

        template <class ITERATOR,
                  XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
        void init(ITERATOR begin, ITERATOR end)
        {
            const index_t range = std::distance(begin, end);
            if(range == 1)
            {
                init(static_cast<value_type>(*begin));
            }
            else
            {
                XTENSOR_PRECONDITION(range == static_size,
                    "tiny_array_base::init(): size mismatch.");
                init_impl(begin);
            }
        }

        void swap(tiny_array_base & other)
        {
            using std::swap;
            swap(data_, other.data_);
        }

        constexpr index_t size() const { return static_size; }
        constexpr index_t max_size() const { return static_size; }
        constexpr index_type shape() const { return { N }; }

      protected:

        template <class ITERATOR,
                  XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
        void init_impl(ITERATOR u)
        {
            for(index_t k=0; k < static_size; ++k, ++u)
                data_[k] = static_cast<value_type>(*u);
        }

        representation_type data_;
    };

    /**********************************************/
    /* tiny_array_base: static shape, owns memory */
    /**********************************************/

    template <class VALUETYPE, index_t N>
    class tiny_array_base<VALUETYPE, N, void>
    : public tags::tiny_array_tag
    {
      public:
        using representation_type    = VALUETYPE[N];
        using value_type             = VALUETYPE;
        using const_value_type       = typename std::add_const<value_type>::type;
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
        using index_type             = std::array<index_t, 1>;

        static const bool    owns_memory = true;
        static const bool    is_static   = true;
        static const index_t static_size = N;

        template <class NEW_VALUETYPE, index_t NEW_SIZE=N>
        using rebind = tiny_array<NEW_VALUETYPE, NEW_SIZE>;

        template <index_t NEW_SIZE>
        using rebind_size = tiny_array<value_type, NEW_SIZE>;

        tiny_array_base(tiny_array_base const &) = default;
        tiny_array_base(tiny_array_base &&) = default;

        explicit tiny_array_base()
        {
            init();
        }

        explicit tiny_array_base(tags::skip_initialization_tag)
        {}

        tiny_array_base(std::size_t size, value_type const & v)
        {
            std::ignore = size;
            XTENSOR_ASSERT_MSG(size == static_size,
                "tiny_array_base(size, value): size mismatch.");
            init(v);
        }

        tiny_array_base(std::size_t size, tags::skip_initialization_tag)
        {
            std::ignore = size;
            XTENSOR_ASSERT_MSG(size == static_size,
                "tiny_array_base(size, dont_init): size mismatch.");
        }

        template <class V, index_t M, class R>
        tiny_array_base(tiny_array<V, M, R> const & v)
        {
            init(v.begin(), v.end());
        }

        template <class V>
        tiny_array_base(std::initializer_list<V> v)
        {
            if(v.size() == 1)
            {
                init(static_cast<value_type>(*v.begin()));
            }
            else
            {
                // FIXME: use static_assert when C++11 is no longer supported
                XTENSOR_ASSERT_MSG(v.size() == static_size,
                    "tiny_array_base(std::initializer_list<V>): size mismatch.");
                init_impl(v.begin());
            }
        }

        template <class V>
        tiny_array_base(std::array<V, N> const & v)
        {
            init_impl(v.begin());
        }

        template <class V>
        tiny_array_base(std::vector<V> const & v)
        {
            XTENSOR_PRECONDITION(v.size() == static_size,
                "tiny_array_base(std::vector<V>): size mismatch.");
            init_impl(v.begin());
        }

        template <class V>
        explicit tiny_array_base(V const (&v)[N])
        {
            init_impl(v);
        }

        template <class ITERATOR,
                  XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
        explicit tiny_array_base(ITERATOR begin)
        {
            init_impl(begin);
        }

        template <class ITERATOR,
                  XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
        explicit tiny_array_base(ITERATOR begin, ITERATOR end)
        {
            init(begin, end);
        }

        tiny_array_base & operator=(tiny_array_base const &) = default;
        tiny_array_base & operator=(tiny_array_base &&) = default;

        void init(value_type const & v = value_type())
        {
            for(index_t k=0; k<static_size; ++k)
                data_[k] = v;
        }

        template <class ITERATOR,
                  XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
        void init(ITERATOR begin, ITERATOR end)
        {
            const index_t range = std::distance(begin, end);
            if(range == 0)
            {
                init();
            }
            else if(range == 1)
            {
                init(static_cast<value_type>(*begin));
            }
            else
            {
                XTENSOR_PRECONDITION(range == static_size,
                    "tiny_array_base::init(ITERATOR begin, ITERATOR end): size mismatch.");
                init_impl(begin);
            }
        }

        void swap(tiny_array_base & other)
        {
            using std::swap;
            for(index_t k=0; k<static_size; ++k)
            {
                swap(data_[k], other.data_[k]);
            }
        }

        constexpr index_t size() const { return static_size; }
        constexpr index_t max_size() const { return static_size; }
        constexpr index_type shape() const { return { N }; }

      protected:

        template <class ITERATOR,
                  XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
        void init_impl(ITERATOR u)
        {
            for(index_t k=0; k < static_size; ++k, ++u)
                data_[k] = static_cast<value_type>(*u);
        }

        representation_type data_;
    };

    /***************************************************/
    /* tiny_array_base: dynamic shape, borrowed memory */
    /***************************************************/

    template <class VALUETYPE, class REPRESENTATION>
    class tiny_array_base<VALUETYPE, runtime_size, REPRESENTATION>
    : public tags::tiny_array_tag
    {
        using deduced_value_type = typename tiny_detail::value_type_traits<REPRESENTATION>::type;
        static_assert(std::is_same<VALUETYPE, deduced_value_type>::value,
                      "tiny_array_base: type mismatch between VALUETYPE and REPRESENTATION.");
      public:
        using representation_type    = REPRESENTATION;
        using value_type             = VALUETYPE;
        using const_value_type       = typename std::add_const<value_type>::type;
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
        using index_type             = std::array<index_t, 1>;

        static const bool    owns_memory = false;
        static const bool    is_static   = false;
        static const index_t static_size = runtime_size;

        tiny_array_base()
        : size_(0)
        , data_(representation_type())
        {}

        tiny_array_base(tiny_array_base const & rhs )
        : size_(rhs.size_)
        , data_(const_cast<representation_type>(rhs.data_))
        {}

        tiny_array_base(representation_type begin, representation_type end)
        : size_(std::distance(begin, end))
        , data_(begin)
        {}

        void reset(representation_type begin, representation_type end)
        {
            size_ = std::distance(begin, end);
            data_ = begin;
        }

        void init(value_type const & v = value_type())
        {
            for(index_t i=0; i<size_; ++i)
                data_[i] = v;
        }

        template <class ITERATOR,
                  XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
        void init(ITERATOR begin, ITERATOR end)
        {
            index_t range = std::distance(begin, end);
            if(range == 1)
                init(static_cast<value_type>(*begin));
            else if(range == size_)
                init_impl(begin);
            else
                XTENSOR_PRECONDITION(false,
                    "tiny_array_base::init(): size mismatch.");
        }

        void swap(tiny_array_base & other)
        {
            using std::swap;
            swap(this->size_, other.size_);
            swap(this->data_, other.data_);
        }

        index_t size() const { return size_; }
        index_t max_size() const { return size_; }
        index_type shape() const { return { size_ }; }

      protected:

        template <class ITERATOR,
                  XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
        void init_impl(ITERATOR u)
        {
            for(index_t k=0; k < size_; ++k, ++u)
                data_[k] = static_cast<value_type>(*u);
        }

        index_t size_;
        representation_type data_;
    };

    /***********************************************/
    /* tiny_array_base: dynamic shape, owns memory */
    /***********************************************/

    template <class VALUETYPE>
    class tiny_array_base<VALUETYPE, runtime_size, void>
    : public tags::tiny_array_tag
    {
      public:
        using representation_type    = VALUETYPE *;
        using value_type             = VALUETYPE;
        using const_value_type       = typename std::add_const<value_type>::type;
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
        using index_type             = std::array<index_t, 1>;

        static const bool    owns_memory = true;
        static const bool    is_static   = false;
        static const index_t static_size = runtime_size;

        template <class NEW_VALUETYPE>
        using rebind = tiny_array<NEW_VALUETYPE, runtime_size>;

        tiny_array_base()
        : size_(0)
        , data_(nullptr)
        {}

        tiny_array_base(tiny_array_base && v)
        : tiny_array_base()
        {
            v.swap(*this);
        }

        tiny_array_base(tiny_array_base const & v)
        : tiny_array_base(v.begin(), v.end())
        {}

        tiny_array_base(std::size_t size, value_type const & initial)
        : size_(size)
        , data_(nullptr)
        {
            allocate(initial);
        }

        tiny_array_base(index_t size, tags::skip_initialization_tag)
        : size_(size)
        , data_(nullptr)
        {
            if(size_ > buffer_size)
            {
                data_ = alloc_.allocate(size_);
                if(!may_use_uninitialized_memory)
                    std::uninitialized_fill(data_, data_+size_, value_type());
            }
            else
            {
                data_ = buffer_;
            }
        }

        template <class V, index_t M, class R>
        tiny_array_base(tiny_array<V, M, R> const & v)
        : tiny_array_base(v.begin(), v.end())
        {}

        template <class ITERATOR,
                  XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value> >
        tiny_array_base(ITERATOR begin, ITERATOR end)
        : size_(std::distance(begin, end))
        , data_(nullptr)
        {
            if(size_ > buffer_size)
            {
                data_ = alloc_.allocate(size_);
                for(index_t k=0; k<size_; ++k, ++begin)
                    new(data_+k) value_type(static_cast<value_type>(*begin));
            }
            else
            {
                data_ = buffer_;
                for(index_t k=0; k<size_; ++k, ++begin)
                    data_[k] = static_cast<value_type>(*begin);
            }
        }

        template <class V, size_t SIZE>
        tiny_array_base(const V (&v)[SIZE])
        : tiny_array_base(v, v+SIZE)
        {}

        template <class V>
        tiny_array_base(std::initializer_list<V> v)
        : tiny_array_base(v.begin(), v.end())
        {}

        template <class V, index_t M>
        tiny_array_base(std::array<V, M> const & v)
        : tiny_array_base(v.begin(), v.end())
        {}

        template <class V>
        tiny_array_base(std::vector<V> const & v)
        : tiny_array_base(v.begin(), v.end())
        {}

        ~tiny_array_base()
        {
            deallocate();
        }

        void resize(size_t new_size)
        {
            if(new_size != size_)
            {
                deallocate();
                size_ = new_size;
                allocate();
            }
        }

        void init(value_type const & v = value_type())
        {
            for(index_t i=0; i<size_; ++i)
            {
                data_[i] = v;
            }
        }

        template <class ITERATOR,
                  XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
        void init(ITERATOR begin, ITERATOR end)
        {
            index_t range = std::distance(begin, end);
            if(range == 1)
            {
                init(static_cast<value_type>(*begin));
            }
            else if(range == size_)
            {
                for(index_t k=0; k<size_; ++k, ++begin)
                    data_[k] = static_cast<value_type>(*begin);
            }
            else
            {
                XTENSOR_PRECONDITION(false,
                    "tiny_array_base::init(): size mismatch.");
            }
        }

        void swap(tiny_array_base & other)
        {
            using std::swap;
            if(size_ == 0 || size_ > buffer_size)
            {
                if(other.size_ == 0 || other.size_ > buffer_size)
                {
                    // both use allocated memory (or no memory at all)
                    swap(data_, other.data_);
                }
                else
                {
                    // self uses allocated memory, other the buffer
                    for(index_t k=0; k<other.size_; ++k)
                    {
                        buffer_[k] = other.data_[k];
                    }
                    other.data_ = data_;
                    data_ = buffer_;
                }
            }
            else
            {
                if(other.size_ > buffer_size)
                {
                    // self uses the buffer, other allocated memory
                    for(index_t k=0; k<size_; ++k)
                    {
                        other.buffer_[k] = data_[k];
                    }
                    data_ = other.data_;
                    other.data_ = other.buffer_;
                }
                else
                {
                    // both use the buffer
                    if(size_ < other.size_)
                    {
                        for(index_t k=0; k<size_; ++k)
                        {
                            swap(data_[k], other.data_[k]);
                        }
                        for(index_t k=size_; k<other.size_; ++k)
                        {
                            data_[k] = other.data_[k];
                        }
                    }
                    else
                    {
                        for(index_t k=0; k<other.size_; ++k)
                        {
                            swap(data_[k], other.data_[k]);
                        }
                        for(index_t k=other.size_; k<size_; ++k)
                        {
                            other.data_[k] = data_[k];
                        }
                    }
                }
            }
            swap(size_, other.size_);
        }

        index_t size() const { return size_; }
        index_t max_size() const { return std::max_size(alloc_); }
        index_type shape() const { return { size_ }; }

      protected:
        void allocate(value_type const & initial = value_type())
        {
            if(size_ > buffer_size)
            {
                data_ = alloc_.allocate(size_);
                std::uninitialized_fill(data_, data_+size_, initial);
            }
            else
            {
                data_ = buffer_;
                std::fill(data_, data_+size_, initial);
            }
        }

        void deallocate()
        {
            if(size_ > buffer_size)
            {
                if(!may_use_uninitialized_memory)
                {
                    for(index_t k=0; k<size_; ++k)
                        (data_+k)->~value_type();
                }
                alloc_.deallocate(data_, size_);
            }
            size_ = 0;
            data_ = nullptr;
        }

        template <class ITERATOR,
                  XTENSOR_REQUIRE<iterator_concept<ITERATOR>::value>>
        void init_impl(ITERATOR u)
        {
            for(index_t k=0; k < size_; ++k, ++u)
            {
                data_[k] = static_cast<value_type>(*u);
            }
        }

        static const bool may_use_uninitialized_memory = std::is_scalar<value_type>::value ||
                                                         std::is_pod<value_type>::value;
        static const index_t buffer_size = 4;

        std::allocator<value_type> alloc_;
        index_t      size_;
        representation_type data_;
        value_type   buffer_[buffer_size < 1 ? 1 : buffer_size];
    };

    /**************/
    /* tiny_array */
    /**************/

    /** \brief Class for small, number-like arrays.
    **/
    template <class VALUETYPE, index_t N, class REPRESENTATION>
    class tiny_array
    : public tiny_array_base<VALUETYPE, N, REPRESENTATION>
    {
      public:

        using base_type = tiny_array_base<VALUETYPE, N, REPRESENTATION>;
        using typename base_type::value_type;
        using typename base_type::const_value_type;
        using typename base_type::reference;
        using typename base_type::const_reference;
        using typename base_type::pointer;
        using typename base_type::const_pointer;
        using typename base_type::iterator;
        using typename base_type::const_iterator;
        using typename base_type::reverse_iterator;
        using typename base_type::const_reverse_iterator;
        using typename base_type::size_type;
        using typename base_type::difference_type;
        using typename base_type::index_type;

        using base_type::owns_memory;
        using base_type::is_static;
        using base_type::static_size;

        using base_type::base_type;

        tiny_array()
        : base_type()
        {}

        tiny_array(tiny_array const & rhs)
        : base_type(rhs)
        {}

        tiny_array(tiny_array && rhs)
        : base_type(std::forward<base_type>(rhs))
        {}

        // assignment

        tiny_array & operator=(value_type const & v)
        {
            this->init(v);
            return *this;
        }

        tiny_array & operator=(tiny_array const & rhs)
        {
            if(this == &rhs)
                return *this;
            if(this->size() != rhs.size())
            {
                // can only happen if N == runtime_size
                tiny_array(rhs).swap(*this);
            }
            else
            {
                this->init_impl(rhs.begin());
            }
            return *this;
        }

        tiny_array & operator=(tiny_array && rhs)
        {
            if(this->size() != rhs.size())
                rhs.swap(*this);
            else
                this->init_impl(rhs.begin());
            return *this;
        }

        template<index_t M>
        tiny_array & operator=(value_type const (&v)[M])
        {
            if(this->size() != M)
            {
                XTENSOR_PRECONDITION(!is_static && owns_memory,
                    "tiny_array::operator=(): size mismatch.");
                tiny_array(v, v+M).swap(*this);
            }
            else
            {
                this->init_impl(v);
            }
            return *this;
        }

        template <class U, index_t M, class R>
        tiny_array & operator=(tiny_array<U, M, R> const & rhs)
        {
            if(this->size() != rhs.size())
            {
                XTENSOR_PRECONDITION(!is_static && owns_memory,
                    "tiny_array::operator=(): size mismatch.");
                tiny_array(rhs).swap(*this);
            }
            else
            {
                this->init_impl(rhs.begin());
            }
            return *this;
        }

        using base_type::init;

        template <class V>
        void init(std::initializer_list<V> v)
        {
            init(v.begin(), v.end());
        }

        // index access

        reference operator[](index_t i)
        {
            return this->data_[i];
        }

        constexpr const_reference operator[](index_t i) const
        {
            return this->data_[i];
        }

        reference at(index_t i)
        {
            if(i < 0 || i >= this->size())
                throw std::out_of_range("tiny_array::at()");
            return this->data_[i];
        }

        const_reference at(index_t i) const
        {
            if(i < 0 || i >= this->size())
                throw std::out_of_range("tiny_array::at()");
            return this->data_[i];
        }

            /** Get a view to the subarray with length <tt>(TO-FROM)</tt> starting at <tt>FROM</tt>.
                The bounds must fullfill <tt>0 <= FROM < TO <= size()</tt>.
                Only available if this array is 1-dimensional, i.e. <tt>static_ndim == 1</tt>.
            */
        template <index_t FROM, index_t TO>
        tiny_array<value_type, TO-FROM, iterator>
        subarray()
        {
            static_assert(FROM >= 0 && FROM < TO,
                "tiny_array::subarray(): range out of bounds.");
            XTENSOR_PRECONDITION(TO <= this->size(),
                "tiny_array::subarray(): range out of bounds.");
            return tiny_array<value_type, TO-FROM, iterator>(this->data_+FROM);
        }

        template <index_t FROM, index_t TO>
        tiny_array<const_value_type, TO-FROM, const_iterator>
        subarray() const
        {
            static_assert(FROM >= 0 && FROM < TO,
                "tiny_array::subarray(): range out of bounds.");
            XTENSOR_PRECONDITION(TO <= this->size(),
                "tiny_array::subarray(): range out of bounds.");
            return tiny_array<const_value_type, TO-FROM, const_iterator>(this->data_+FROM);
        }

            /** Get a view to the subarray with length <tt>(TO-FROM)</tt> starting at <tt>FROM</tt>.
                The bounds must fullfill <tt>0 <= FROM < TO <= size()</tt>.
                Only available if this array is 1-dimensional, i.e. <tt>static_ndim == 1</tt>.
            */
        tiny_array<value_type, runtime_size, iterator>
        subarray(index_t FROM, index_t TO)
        {
            XTENSOR_PRECONDITION(FROM >= 0 && FROM < TO && TO <= this->size(),
                "tiny_array::subarray(): range out of bounds.");
            return tiny_array<value_type, runtime_size, iterator>(this->data_+FROM, this->data_+TO);
        }

        tiny_array<const_value_type, runtime_size, const_iterator>
        subarray(index_t FROM, index_t TO) const
        {
            XTENSOR_PRECONDITION(FROM >= 0 && FROM < TO && TO <= this->size(),
                "tiny_array::subarray(): range out of bounds.");
            return tiny_array<const_value_type, runtime_size, const_iterator>(this->data_+FROM, this->data_+TO);
        }

        tiny_array<value_type, is_static ? static_size-1 : runtime_size>
        erase(index_t m) const
        {
            XTENSOR_PRECONDITION(m >= 0 && m < this->size(), "tiny_array::erase(): "
                "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(this->size())+").");
            static const index_t res_size = is_static
                                                ? static_size-1
                                                : runtime_size;
            tiny_array<value_type, res_size> res(this->size()-1, dont_init);
            for(index_t k=0; k<m; ++k)
                res[k] = this->data_[k];
            for(index_t k=m+1; k<this->size(); ++k)
                res[k-1] = this->data_[k];
            return res;
        }

        tiny_array<value_type, is_static ? static_size-1 : runtime_size>
        pop_front() const
        {
            return erase(0);
        }

        tiny_array<value_type, is_static ? static_size-1 : runtime_size>
        pop_back() const
        {
            return erase(this->size()-1);
        }

        tiny_array<value_type, is_static ? static_size+1 : runtime_size>
        insert(index_t m, value_type v) const
        {
            XTENSOR_PRECONDITION(m >= 0 && m <= this->size(), "tiny_array::insert(): "
                "Index "+std::to_string(m)+" out of bounds [0, "+std::to_string(this->size())+"].");
            static const index_t res_size = is_static
                                                ? static_size+1
                                                : runtime_size;
            tiny_array<value_type, res_size> res(this->size()+1, dont_init);
            for(index_t k=0; k<m; ++k)
                res[k] = this->data_[k];
            res[m] = v;
            for(index_t k=m; k<this->size(); ++k)
                res[k+1] = this->data_[k];
            return res;
        }

        template <class V, index_t M, class R>
        inline
        tiny_array<value_type, static_size>
        transpose(tiny_array<V, M, R> const & permutation) const
        {
            static_assert(M == static_size || M == runtime_size,
                "tiny_array::transpose(): size mismatch.");
            XTENSOR_PRECONDITION(this->size() == permutation.size(),
                "tiny_array::transpose(): size mismatch.");
            tiny_array<value_type, static_size> res(this->size(), dont_init);
            for(index_t k=0; k < this->size(); ++k)
            {
                XTENSOR_ASSERT_MSG(permutation[k] >= 0 && permutation[k] < this->size(),
                    "tiny_array::transpose():  Permutation index out of bounds");
                res[k] = (*this)[permutation[k]];
            }
            return res;
        }

        // boiler plate

        iterator begin() { return this->data_; }
        iterator end()   { return this->data_ + this->size(); }
        const_iterator begin() const { return this->data_; }
        const_iterator end()   const { return this->data_ + this->size(); }
        const_iterator cbegin() const { return this->data_; }
        const_iterator cend()   const { return this->data_ + this->size(); }

        reverse_iterator rbegin() { return reverse_iterator(this->data_ + this->size()); }
        reverse_iterator rend()   { return reverse_iterator(this->data_); }
        const_reverse_iterator rbegin() const { return const_reverse_iterator(this->data_ + this->size()); }
        const_reverse_iterator rend()   const { return const_reverse_iterator(this->data_); }
        const_reverse_iterator crbegin() const { return const_reverse_iterator(this->data_ + this->size()); }
        const_reverse_iterator crend()   const { return const_reverse_iterator(this->data_); }

        pointer data() { return this->data_; }
        const_pointer data() const { return this->data_; }

        reference front() { return this->data_[0]; }
        reference back()  { return this->data_[this->size()-1]; }
        constexpr const_reference front() const { return this->data_[0]; }
        constexpr const_reference back()  const { return this->data_[this->size()-1]; }

        constexpr bool    empty() const { return this->size() == 0; }

        tiny_array & reverse()
        {
            using std::swap;
            index_t i=0, j=this->size()-1;
            while(i < j)
                 swap(this->data_[i++], this->data_[j--]);
            return *this;
        }

            /// factory function for the fixed-size k-th unit vector
        template <index_t SIZE=static_size>
        static inline
        tiny_array<value_type, SIZE>
        unit_vector(index_t k)
        {
            static_assert(SIZE > 0,
                "tiny_array::unit_vector(): SIZE must be poisitive.");
            tiny_array<value_type, SIZE> res;
            res[k] = 1;
            return res;
        }

            /// factory function for the fixed-size k-th unit vector
        static inline
        tiny_array<value_type, runtime_size>
        unit_vector(index_t size, index_t k)
        {
            tiny_array<value_type, runtime_size> res(size, value_type());
            res[k] = 1;
            return res;
        }

            /// factory function for fixed-size linear sequence ending at <tt>end-1</tt>
        static inline
        tiny_array<value_type, N>
        range(value_type end)
        {
            XTENSOR_PRECONDITION(static_size != runtime_size || end >= 0,
                "tiny_array::range(): end must be non-negative.");
            value_type start = (static_size != runtime_size)
                                   ? end - static_cast<value_type>(static_size)
                                   : value_type();
            tiny_array<value_type, N> res(end-start, dont_init);
            for(index_t k=0; k < res.size(); ++k, ++start)
                res[k] = start;
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
            using namespace math;
            XTENSOR_PRECONDITION(step != 0,
                "tiny_array::range(): step must be non-zero.");
            XTENSOR_PRECONDITION((step > 0 && begin <= end) || (step < 0 && begin >= end),
                "tiny_array::range(): sign mismatch between step and (end-begin).");
            // use floor() here because value_type could be floating point
            index_t size = (index_t)floor((abs(end-begin+step)-1)/abs(step));
            tiny_array<value_type, runtime_size> res(size, dont_init);
            for(index_t k=0; k < size; ++k, begin += step)
                res[k] = begin;
            return res;
        }

            /// factory function for fixed-size linear sequence starting at <tt>start</tt> with stepsize <tt>step</tt>
        template <index_t SIZE=static_size>
        static inline
        tiny_array<value_type, SIZE>
        linear_sequence(value_type start = value_type(), value_type step = value_type(1))
        {
            static_assert(SIZE > 0,
                "tiny_array::linear_sequence(): SIZE must be poisitive.");
            tiny_array<value_type, SIZE> res(dont_init);
            for(index_t k=0; k < SIZE; ++k, start += step)
                res[k] = start;
            return res;
        }
    };

    /*********************/
    /* tiny_array output */
    /*********************/

    template <class T, index_t N, class R>
    std::ostream & operator<<(std::ostream & o, tiny_array<T, N, R> const & v)
    {
        o << "{";
        if(v.size() > 0)
            o << promote_type_t<T>(v[0]);
        for(index_t i=1; i < v.size(); ++i)
            o << ", " << promote_type_t<T>(v[i]);
        o << "}";
        return o;
    }

    /*************************/
    /* tiny_array comparison */
    /*************************/

    /** \addtogroup TinyArrayOperators Arithmetic, relational, and algebraic functions for tiny_array
    */
    //@{

        /// element-wise equal
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    operator==(tiny_array<V1, N1, R1> const & l,
               tiny_array<V2, N2, R2> const & r)
    {
        if(l.size() != r.size())
            return false;
        for(index_t k=0; k < l.size(); ++k)
            if(l[k] != r[k])
                return false;
        return true;
    }

        /// element-wise equal to a constant
    template <class V1, index_t N1, class R1, class V2,
              XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                              std::is_convertible<V2, V1>::value> >
    inline bool
    operator==(tiny_array<V1, N1, R1> const & l,
               V2 const & r)
    {
        for(index_t k=0; k < l.size(); ++k)
            if(l[k] != r)
                return false;
        return true;
    }

        /// element-wise equal to a constant
    template <class V1, class V2, index_t N2, class R2,
              XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                              std::is_convertible<V2, V1>::value> >
    inline bool
    operator==(V1 const & l,
               tiny_array<V2, N2, R2> const & r)
    {
        for(index_t k=0; k < r.size(); ++k)
            if(l != r[k])
                return false;
        return true;
    }

        /// element-wise not equal
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    operator!=(tiny_array<V1, N1, R1> const & l,
               tiny_array<V2, N2, R2> const & r)
    {
        if(l.size() != r.size())
            return true;
        for(index_t k=0; k < l.size(); ++k)
            if(l[k] != r[k])
                return true;
        return false;
    }

        /// element-wise not equal to a constant
    template <class V1, index_t N1, class R1, class V2,
              XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&
                              std::is_convertible<V2, V1>::value> >
    inline bool
    operator!=(tiny_array<V1, N1, R1> const & l,
               V2 const & r)
    {
        for(index_t k=0; k < l.size(); ++k)
            if(l[k] != r)
                return true;
        return false;
    }

        /// element-wise not equal to a constant
    template <class V1, class V2, index_t N2, class R2,
              XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&
                              std::is_convertible<V2, V1>::value> >
    inline bool
    operator!=(V1 const & l,
               tiny_array<V2, N2, R2> const & r)
    {
        for(index_t k=0; k < r.size(); ++k)
            if(l != r[k])
                return true;
        return false;
    }

        /// lexicographical less
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    operator<(tiny_array<V1, N1, R1> const & l,
              tiny_array<V2, N2, R2> const & r)
    {
        const index_t min_size = std::min(l.size(), r.size());
        for(index_t k = 0; k < min_size; ++k)
        {
            if(l[k] < r[k])
                return true;
            if(r[k] < l[k])
                return false;
        }
        return (l.size() < r.size());
    }

        /// lexicographical less-equal
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    operator<=(tiny_array<V1, N1, R1> const & l,
               tiny_array<V2, N2, R2> const & r)
    {
        return !(r < l);
    }

        /// lexicographical greater
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    operator>(tiny_array<V1, N1, R1> const & l,
              tiny_array<V2, N2, R2> const & r)
    {
        return r < l;
    }

        /// lexicographical greater-equal
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    operator>=(tiny_array<V1, N1, R1> const & l,
               tiny_array<V2, N2, R2> const & r)
    {
        return !(l < r);
    }

        /// check if all elements are non-zero (or 'true' if V is bool)
    template <class V, index_t N, class R>
    inline bool
    all(tiny_array<V, N, R> const & t)
    {
        for(index_t i=0; i<t.size(); ++i)
            if(t[i] == V())
                return false;
        return true;
    }

        /// check if at least one element is non-zero (or 'true' if V is bool)
    template <class V, index_t N, class R>
    inline bool
    any(tiny_array<V, N, R> const & t)
    {
        for(index_t i=0; i<t.size(); ++i)
            if(t[i] != V())
                return true;
        return false;
    }

    /**********************************/
    /* pointwise relational operators */
    /**********************************/

    #define XTENSOR_TINY_COMPARISON(NAME, OP)                                  \
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>  \
    inline bool                                                                \
    all_##NAME(tiny_array<V1, N1, R1> const & l,                               \
               tiny_array<V2, N2, R2> const & r)                               \
    {                                                                          \
        XTENSOR_ASSERT_MSG(l.size() == r.size(),                               \
            "tiny_array::all_" #NAME "(): size mismatch.");                    \
        for(index_t k=0; k < l.size(); ++k)                                    \
            if (l[k] OP r[k])                                                  \
                return false;                                                  \
        return true;                                                           \
    }                                                                          \
                                                                               \
    template <class V1, index_t N1, class R1, class V2,                        \
              XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&                \
                              std::is_convertible<V2, V1>::value> >            \
    inline bool                                                                \
    all_##NAME(tiny_array<V1, N1, R1> const & l,                               \
               V2 const & r)                                                   \
    {                                                                          \
        for(index_t k=0; k < l.size(); ++k)                                    \
            if (l[k] OP r)                                                     \
                return false;                                                  \
        return true;                                                           \
    }                                                                          \
                                                                               \
    template <class V1, class V2, index_t N2, class R2,                        \
              XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&                \
                              std::is_convertible<V2, V1>::value> >            \
    inline bool                                                                \
    all_##NAME(V1 const & l,                                                   \
               tiny_array<V2, N2, R2> const & r)                               \
    {                                                                          \
        for(index_t k=0; k < r.size(); ++k)                                    \
            if (l OP r[k])                                                     \
                return false;                                                  \
        return true;                                                           \
    }

    XTENSOR_TINY_COMPARISON(less, >=)
    XTENSOR_TINY_COMPARISON(less_equal, >)
    XTENSOR_TINY_COMPARISON(greater, <=)
    XTENSOR_TINY_COMPARISON(greater_equal, <)

    #undef XTENSOR_TINY_COMPARISON

    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline bool
    all_close(tiny_array<V1, N1, R1> const & l,
              tiny_array<V2, N2, R2> const & r,
              double rtol = 2.0*std::numeric_limits<double>::epsilon(),
              double atol = 2.0*std::numeric_limits<double>::epsilon(),
              bool equal_nan = false)
    {
        if(l.size() != r.size())
            return false;
        detail::isclose<promote_type_t<V1, V2>> isclose_fct{rtol, atol, equal_nan};
        for(index_t k=0; k < l.size(); ++k)
            if(!isclose_fct(l[k], r[k]))
                return false;
        return true;
    }

    /***************************/
    /* tiny_array manipulation */
    /***************************/

        /// reversed copy
    template <class V, index_t N, class R>
    inline
    tiny_array<V, N>
    reversed(tiny_array<V, N, R> const & v)
    {
        tiny_array<V, N> res(v.size(), dont_init);
        for(index_t k=0; k<v.size(); ++k)
            res[k] = v[v.size()-1-k];
        return res;
    }

        /** \brief transposed copy

            Elements are arranged such that <tt>res[k] = v[permutation[k]]</tt>.
        */
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline
    tiny_array<V1, N1>
    transpose(tiny_array<V1, N1, R1> const & v,
              tiny_array<V2, N2, R2> const & permutation)
    {
        return v.transpose(permutation);
    }

    template <class V, index_t N, class R>
    inline
    tiny_array<V, N>
    transpose(tiny_array<V, N, R> const & v)
    {
        return reversed(v);
    }

    /*************************/
    /* tiny_array arithmetic */
    /*************************/

    namespace tiny_detail
    {
        template <index_t N1, index_t N2>
        struct size_promote
        {
            static const index_t value  = N1;
            static const bool valid = (N1 == N2);
        };

        template <index_t N1>
        struct size_promote<N1, runtime_size>
        {
            static const index_t value  = runtime_size;
            static const bool valid = true;
        };

        template <index_t N2>
        struct size_promote<runtime_size, N2>
        {
            static const index_t value  = runtime_size;
            static const bool valid = true;
        };

        template <>
        struct size_promote<runtime_size, runtime_size>
        {
            static const index_t value  = runtime_size;
            static const bool valid = true;
        };
    }

    #define XTENSOR_TINYARRAY_OPERATORS(OP)                                                  \
    template <class V1, index_t N1, class R1, class V2,                                      \
              XTENSOR_REQUIRE<!tiny_array_concept<V2>::value &&                              \
                              std::is_convertible<V2, V1>::value> >                          \
    inline tiny_array<V1, N1, R1> &                                                          \
    operator OP##=(tiny_array<V1, N1, R1> & l,                                               \
                   V2 r)                                                                     \
    {                                                                                        \
        for(index_t i=0; i<l.size(); ++i)                                                    \
            l[i] OP##= r;                                                                    \
        return l;                                                                            \
    }                                                                                        \
                                                                                             \
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>                \
    inline tiny_array<V1, N1, R1> &                                                          \
    operator OP##=(tiny_array<V1, N1, R1> & l,                                               \
                   tiny_array<V2, N2, R2> const & r)                                         \
    {                                                                                        \
        XTENSOR_ASSERT_MSG(l.size() == r.size(),                                             \
            "tiny_array::operator" #OP "=(): size mismatch.");                               \
        for(index_t i=0; i<l.size(); ++i)                                                    \
            l[i] OP##= r[i];                                                                 \
        return l;                                                                            \
    }                                                                                        \
                                                                                             \
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>                \
    inline                                                                                   \
    tiny_array<decltype((*(V1*)0) OP (*(V2*)0)), tiny_detail::size_promote<N1, N2>::value>   \
    operator OP(tiny_array<V1, N1, R1> const & l,                                            \
                tiny_array<V2, N2, R2> const & r)                                            \
    {                                                                                        \
        static_assert(tiny_detail::size_promote<N1, N2>::valid,                              \
            "tiny_array::operator" #OP "(): size mismatch.");                                \
        XTENSOR_ASSERT_MSG(l.size() == r.size(),                                             \
            "tiny_array::operator" #OP "(): size mismatch.");                                \
        tiny_array<decltype((*(V1*)0) OP (*(V2*)0)),                                         \
                   tiny_detail::size_promote<N1, N2>::value> res(l);                         \
        res OP##= r;                                                                         \
        return res;                                                                          \
    }                                                                                        \
                                                                                             \
    template <class V1, index_t N1, class R1, class V2,                                      \
              XTENSOR_REQUIRE<!tiny_array_concept<V2>::value> >                              \
    inline                                                                                   \
    tiny_array<decltype((*(V1*)0) OP (*(V2*)0)), N1>                                         \
    operator OP(tiny_array<V1, N1, R1> const & l,                                            \
                V2 r)                                                                        \
    {                                                                                        \
        tiny_array<decltype((*(V1*)0) OP (*(V2*)0)), N1> res(l);                             \
        res OP##= r;                                                                         \
        return res;                                                                          \
    }                                                                                        \
                                                                                             \
    template <class V1, class V2, index_t N2, class R2,                                      \
              XTENSOR_REQUIRE<!tiny_array_concept<V1>::value &&                              \
                              !std::is_base_of<std::ios_base, V1>::value> >                  \
    inline                                                                                   \
    tiny_array<decltype((*(V1*)0) OP (*(V2*)0)), N2>                                         \
    operator OP(V1 l,                                                                        \
                tiny_array<V2, N2, R2> const & r)                                            \
    {                                                                                        \
        tiny_array<decltype((*(V1*)0) OP (*(V2*)0)), N2> res(r.size(), l);                   \
        res OP##= r;                                                                         \
        return res;                                                                          \
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

        /// Arithmetic identity
    template <class V, index_t N, class R>
    inline
    tiny_array<V, N, R> const &
    operator+(tiny_array<V, N, R> const & v)
    {
        return v;
    }

        /// Arithmetic negation
    template <class V, index_t N, class R>
    inline
    tiny_array<decltype(-(*(V*)0)), N>
    operator-(tiny_array<V, N, R> const & v)
    {
        tiny_array<decltype(-(*(V*)0)), N> res(v.size(), dont_init);
        for(index_t k=0; k < v.size(); ++k)
            res[k] = -v[k];
        return res;
    }

        /// Boolean negation
    template <class V, index_t N, class R>
    inline
    tiny_array<decltype(!(*(V*)0)), N>
    operator!(tiny_array<V, N, R> const & v)
    {
        tiny_array<V, decltype(!(*(V*)0)), N> res(v.size(), dont_init);
        for(index_t k=0; k < v.size(); ++k)
            res[k] = !v[k];
        return res;
    }

        /// Bitwise negation
    template <class V, index_t N, class R>
    inline
    tiny_array<decltype(~(*(V*)0)), N>
    operator~(tiny_array<V, N, R> const & v)
    {
        tiny_array<V, decltype(~(*(V*)0)), N> res(v.size(), dont_init);
        for(index_t k=0; k < v.size(); ++k)
            res[k] = ~v[k];
        return res;
    }

    #define XTENSOR_TINYARRAY_UNARY_FUNCTION(FCT)                           \
    template <class V, index_t N, class R>                                  \
    inline auto                                                             \
    FCT(tiny_array<V, N, R> const & v)                                      \
    {                                                                       \
        using namespace math;                                               \
        tiny_array<decltype(FCT(v[0])), N> res(v.size(), dont_init);        \
        for(index_t k=0; k < v.size(); ++k)                                 \
            res[k] = FCT(v[k]);                                             \
        return res;                                                         \
    }

    XTENSOR_TINYARRAY_UNARY_FUNCTION(abs)
    XTENSOR_TINYARRAY_UNARY_FUNCTION(fabs)

    XTENSOR_TINYARRAY_UNARY_FUNCTION(cos)
    XTENSOR_TINYARRAY_UNARY_FUNCTION(sin)
    XTENSOR_TINYARRAY_UNARY_FUNCTION(tan)
    // XTENSOR_TINYARRAY_UNARY_FUNCTION(sin_pi)
    // XTENSOR_TINYARRAY_UNARY_FUNCTION(cos_pi)
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
    // XTENSOR_TINYARRAY_UNARY_FUNCTION(sq)
    // XTENSOR_TINYARRAY_UNARY_FUNCTION(elementwise_norm)
    // XTENSOR_TINYARRAY_UNARY_FUNCTION(elementwise_squared_norm)

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
    // XTENSOR_TINYARRAY_UNARY_FUNCTION(even)
    // XTENSOR_TINYARRAY_UNARY_FUNCTION(odd)
    // XTENSOR_TINYARRAY_UNARY_FUNCTION(sign)
    // XTENSOR_TINYARRAY_UNARY_FUNCTION(signi)

    XTENSOR_TINYARRAY_UNARY_FUNCTION(erf)
    XTENSOR_TINYARRAY_UNARY_FUNCTION(erfc)
    XTENSOR_TINYARRAY_UNARY_FUNCTION(tgamma)
    XTENSOR_TINYARRAY_UNARY_FUNCTION(lgamma)

    XTENSOR_TINYARRAY_UNARY_FUNCTION(conj)
    XTENSOR_TINYARRAY_UNARY_FUNCTION(real)
    XTENSOR_TINYARRAY_UNARY_FUNCTION(imag)
    XTENSOR_TINYARRAY_UNARY_FUNCTION(arg)

    #undef XTENSOR_TINYARRAY_UNARY_FUNCTION

    #define XTENSOR_TINYARRAY_BINARY_FUNCTION(FCT)                                              \
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>                   \
    inline auto                                                                                 \
    FCT(tiny_array<V1, N1, R1> const & l,                                                       \
        tiny_array<V2, N2, R2> const & r)                                                       \
    {                                                                                           \
        using namespace math;                                                                   \
        static_assert(tiny_detail::size_promote<N1, N2>::valid,                                 \
            #FCT "(tiny_array, tiny_array): size mismatch.");                                   \
        XTENSOR_ASSERT_MSG(l.size() == r.size(),                                                \
            #FCT "(tiny_array, tiny_array): size mismatch.");                                   \
        tiny_array<decltype(FCT(l[0], r[0])),                                                   \
                   tiny_detail::size_promote<N1, N2>::value> res(l.size(), dont_init);          \
        for(index_t k=0; k < l.size(); ++k)                                                     \
            res[k] = FCT(l[k], r[k]);                                                           \
        return res;                                                                             \
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
    template <class V, index_t N, class R, class E>
    inline auto
    pow(tiny_array<V, N, R> const & v, E exponent)
    {
        using namespace math;
        tiny_array<decltype(pow(v[0], exponent)), N> res(v.size(), dont_init);
        for(index_t k=0; k < v.size(); ++k)
            res[k] = pow(v[k], exponent);
        return res;
    }

        /// sum of the vector's elements
    template <class V, index_t N, class R>
    inline auto
    sum(tiny_array<V, N, R> const & v)
    {
        using result_type = decltype(v[0] + v[0]);
        result_type res = result_type();
        for(index_t k=0; k < v.size(); ++k)
            res += v[k];
        return res;
    }

        /// mean of the vector's elements
    template <class V, index_t N, class R>
    inline auto
    mean(tiny_array<V, N, R> const & v)
    {
        using result_type = real_promote_type_t<decltype(sum(v))>;
        const result_type sumVal = static_cast<result_type>(sum(v));
        if(v.size() > 0)
            return sumVal / static_cast<result_type>(v.size());
        else
            return sumVal;
    }

        /// cumulative sum of the vector's elements
    template <class V, index_t N, class R>
    inline auto
    cumsum(tiny_array<V, N, R> const & v)
    {
        using promote_type = decltype(v[0] + v[0]);
        tiny_array<promote_type, N> res(v);
        for(index_t k=1; k < v.size(); ++k)
            res[k] += res[k-1];
        return res;
    }

        /// product of the vector's elements
    template <class V, index_t N, class R>
    inline auto
    prod(tiny_array<V, N, R> const & v)
    {
        using result_type = decltype(v[0] * v[0]);
        if(v.size() == 0)
            return result_type();
        result_type res = v[0];
        for(index_t k=1; k < v.size(); ++k)
            res *= v[k];
        return res;
    }

        /// cumulative product of the vector's elements
    template <class V, index_t N, class R>
    inline auto
    cumprod(tiny_array<V, N, R> const & v)
    {
        using promote_type = decltype(v[0] * v[0]);
        tiny_array<promote_type, N> res(v);
        for(index_t k=1; k < v.size(); ++k)
            res[k] *= res[k-1];
        return res;
    }

        /// element-wise minimum
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline auto
    min(tiny_array<V1, N1, R1> const & l,
        tiny_array<V2, N2, R2> const & r)
    {
        using std::min;
        using promote_type = promote_type_t<V1, V2>;
        static_assert(tiny_detail::size_promote<N1, N2>::valid,
            "min(tiny_array, tiny_array): size mismatch.");
        XTENSOR_ASSERT_MSG(l.size() == r.size(),
            "min(tiny_array, tiny_array): size mismatch.");
        tiny_array<promote_type, tiny_detail::size_promote<N1, N2>::value> res(l.size(), dont_init);
        for(index_t k=0; k < l.size(); ++k)
            res[k] = min(static_cast<promote_type>(l[k]), static_cast<promote_type>(r[k]));
        return res;
    }

        /// element-wise minimum with a constant
    template <class V1, index_t N1, class R1, class V2,
              XTENSOR_REQUIRE<!tiny_array_concept<V2>::value>>
    inline auto
    min(tiny_array<V1, N1, R1> const & l,
        V2 const & r)
    {
        using std::min;
        using promote_type = promote_type_t<V1, V2>;
        tiny_array<promote_type, N1> res(l.size(), dont_init);
        for(index_t k=0; k < l.size(); ++k)
            res[k] =  min(static_cast<promote_type>(l[k]), static_cast<promote_type>(r));
        return res;
    }

        /// element-wise minimum with a constant
    template <class V1, class V2, index_t N2, class R2,
              XTENSOR_REQUIRE<!tiny_array_concept<V1>::value>>
    inline auto
    min(V1 const & l, tiny_array<V2, N2, R2> const & r)
    {
        return min(r, l);
    }

        /// minimal element
    template <class V, index_t N, class R>
    inline V const &
    min(tiny_array<V, N, R> const & l)
    {
        index_t m = min_element(l);
        XTENSOR_PRECONDITION(m >= 0, "min() of an empty tiny_array is undefined.");
        return l[m];
    }

        /** Index of minimal element.

            Returns -1 for an empty array.
        */
    template <class V, index_t N, class R>
    inline index_t
    min_element(tiny_array<V, N, R> const & l)
    {
        if(l.size() == 0)
            return -1;
        index_t m = 0;
        for(index_t i=1; i<l.size(); ++i)
            if(l[i] < l[m])
                m = i;
        return m;
    }

        /// element-wise maximum
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline auto
    max(tiny_array<V1, N1, R1> const & l,
        tiny_array<V2, N2, R2> const & r)
    {
        using std::max;
        using promote_type = promote_type_t<V1, V2>;
        static_assert(tiny_detail::size_promote<N1, N2>::valid,
            "max(tiny_array, tiny_array): size mismatch.");
        XTENSOR_ASSERT_MSG(l.size() == r.size(),
            "max(tiny_array, tiny_array): size mismatch.");
        tiny_array<promote_type, tiny_detail::size_promote<N1, N2>::value> res(l.size(), dont_init);
        for(index_t k=0; k < l.size(); ++k)
            res[k] = max(static_cast<promote_type>(l[k]), static_cast<promote_type>(r[k]));
        return res;
    }

        /// element-wise maximum with a constant
    template <class V1, index_t N1, class R1, class V2,
              XTENSOR_REQUIRE<!tiny_array_concept<V2>::value>>
    inline auto
    max(tiny_array<V1, N1, R1> const & l,
        V2 const & r)
    {
        using std::max;
        using promote_type = promote_type_t<V1, V2>;
        tiny_array<promote_type, N1> res(l.size(), dont_init);
        for(index_t k=0; k < l.size(); ++k)
            res[k] =  max(static_cast<promote_type>(l[k]), static_cast<promote_type>(r));
        return res;
    }

        /// element-wise maximum with a constant
    template <class V1, class V2, index_t N2, class R2,
              XTENSOR_REQUIRE<!tiny_array_concept<V1>::value>>
    inline auto
    max(V1 const & l, tiny_array<V2, N2, R2> const & r)
    {
        return max(r, l);
    }

        /// maximal element
    template <class V, index_t N, class R>
    inline V const &
    max(tiny_array<V, N, R> const & l)
    {
        index_t m = max_element(l);
        XTENSOR_PRECONDITION(m >= 0, "max() of an empty tiny_array is undefined.");
        return l[m];
    }

        /** Index of maximal element.

            Returns -1 for an empty array.
        */
    template <class V, index_t N, class R>
    inline index_t
    max_element(tiny_array<V, N, R> const & l)
    {
        if(l.size() == 0)
            return -1;
        index_t m = 0;
        for(index_t i=1; i<l.size(); ++i)
            if(l[i] > l[m])
                m = i;
        return m;
    }

        /** \brief Clip values below a threshold.

            All elements smaller than \a val are set to \a val.
        */
    template <class V, index_t N, class R>
    inline auto
    clip_lower(tiny_array<V, N, R> const & t, const V val)
    {
        tiny_array<V, N> res(t.size(), dont_init);
        for(index_t k=0; k < t.size(); ++k)
        {
            res[k] = t[k] < val ? val :  t[k];
        }
        return res;
    }

        /** \brief Clip values above a threshold.

            All elements bigger than \a val are set to \a val.
        */
    template <class V, index_t N, class R>
    inline auto
    clip_upper(tiny_array<V, N, R> const & t, const V val)
    {
        tiny_array<V, N> res(t.size(), dont_init);
        for(index_t k=0; k < t.size(); ++k)
        {
            res[k] = t[k] > val ? val :  t[k];
        }
        return res;
    }

        /** \brief Clip values to an interval.

            All elements less than \a valLower are set to \a valLower, all elements
            bigger than \a valUpper are set to \a valUpper.
        */
    template <class V, index_t N, class R>
    inline auto
    clip(tiny_array<V, N, R> const & t,
         const V valLower, const V valUpper)
    {
        tiny_array<V, N> res(t.size(), dont_init);
        for(index_t k=0; k < t.size(); ++k)
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
    template <class V, index_t N1, class R1, index_t N2, class R2, index_t N3, class R3>
    inline auto
    clip(tiny_array<V, N1, R1> const & t,
         tiny_array<V, N2, R2> const & valLower,
         tiny_array<V, N3, R3> const & valUpper)
    {
        XTENSOR_ASSERT_MSG(t.size() == valLower.size() && t.size() == valUpper.size(),
            "clip(): size mismatch.");
        tiny_array<V, N1> res(t.size(), dont_init);
        for(index_t k=0; k < t.size(); ++k)
        {
            res[k] =  (t[k] < valLower[k])
                           ? valLower[k]
                           : (t[k] > valUpper[k])
                                 ? valUpper[k]
                                 : t[k];
        }
        return res;
    }

        /// dot product of two tiny_arrays
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline auto
    dot(tiny_array<V1, N1, R1> const & l,
        tiny_array<V2, N2, R2> const & r)
    {
        XTENSOR_ASSERT_MSG(l.size() == r.size(),
            "dot(tiny_array, tiny_array): size mismatch.");
        using result_type = decltype(l[0] * r[0]);
        result_type res = result_type();
        for(index_t k=0; k < l.size(); ++k)
            res += l[k] * r[k];
        return res;
    }

        /// cross product of two tiny_arrays
    template <class V1, index_t N1, class R1, class V2, index_t N2, class R2>
    inline auto
    cross(tiny_array<V1, N1, R1> const & r1,
          tiny_array<V2, N2, R2> const & r2)
    {
        XTENSOR_ASSERT_MSG(r1.size() == 3 && r2.size() == 3,
            "cross(tiny_array, tiny_array): cross product requires size() == 3.");
        using result_type = tiny_array<decltype(r1[0] * r2[0]), 3>;
        return  result_type{r1[1]*r2[2] - r1[2]*r2[1],
                            r1[2]*r2[0] - r1[0]*r2[2],
                            r1[0]*r2[1] - r1[1]*r2[0]};
    }

    template <class V, index_t N, class R>
    inline void
    swap(tiny_array<V, N, R> & l,
         tiny_array<V, N, R> & r)
    {
        l.swap(r);
    }

    /// squared norm
    template <class V, index_t N, class R>
    inline auto
    norm_sq(tiny_array<V, N, R> const & t)
    {
        using result_type = squared_norm_type_t<tiny_array<V, N, R>>;
        result_type result = result_type();
        for(index_t i=0; i<t.size(); ++i)
            result += norm_sq(t[i]);
        return result;
    }

    // template <class V, tags::memory_policy MP, int ... N>
    // inline
    // norm_type_t<V>
    // mean_square(tiny_array<V, MP, N...> const & t)
    // {
        // return norm_type_t<V>(squared_norm(t)) / t.size();
    // }

//@}

} // namespace xt

#endif // XTENSOR_XTINY_HPP
