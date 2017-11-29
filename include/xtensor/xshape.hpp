/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_XSHAPE_HPP
#define XTENSOR_XSHAPE_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>

#include "xexception.hpp"
#include "xstorage.hpp"

#if (__GNUC__ || __clang__)
    #define likely(x)       __builtin_expect((x), 1)
    #define unlikely(x)     __builtin_expect((x), 0)
#else
    #define likely(x)       x
    #define unlikely(x)     x
#endif

#ifndef XSHAPE_ALIGNMENT
#ifdef XTENSOR_USE_XSIMD
#include "xsimd/xsimd.hpp"
#define XSHAPE_ALIGNMENT XSIMD_DEFAULT_ALIGNMENT
#else
#define XSHAPE_ALIGNMENT T
#endif
#endif

namespace xt
{
    template <class T, std::size_t N, class A = std::allocator<T>, bool INIT = true>
    class small_vector;

    template <class T>
    using dynamic_shape = small_vector<T, 4>;

    template <class T, std::size_t N>
    using static_shape = std::array<T, N>;

    template <std::size_t... X>
    class fixed_shape;

    template <class T, std::size_t N, class A, bool INIT>
    class small_vector
    {
    public:

        using self_type = small_vector<T, N, A, INIT>;
        using allocator_type = A;
        using size_type = typename A::size_type;
        using value_type = typename A::value_type;
        using pointer = typename A::pointer;
        using const_pointer = typename A::const_pointer;
        using reference = typename A::reference;
        using const_reference = typename A::const_reference;
        using difference_type = typename A::difference_type;

        using iterator = pointer;
        using const_iterator = const_pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        small_vector() noexcept;
        ~small_vector();

        explicit small_vector(const allocator_type& alloc) noexcept;
        explicit small_vector(size_type n, const allocator_type& alloc = allocator_type());
        small_vector(size_type n, const value_type& v, const allocator_type& alloc = allocator_type());
        small_vector(std::initializer_list<T> il, const allocator_type& alloc = allocator_type());

        explicit small_vector(const std::vector<T>& vec);

        template <class IT, class = detail::require_input_iter<IT>>
        small_vector(IT begin, IT end, const allocator_type& alloc = allocator_type());

        small_vector& operator=(const small_vector& rhs);
        small_vector& operator=(small_vector&& rhs);
        small_vector& operator=(std::vector<T>& rhs);

        small_vector(const small_vector& other);
        small_vector(small_vector&& other);

        void assign(size_type n, const value_type& v);

        template <class V>
        void assign(std::initializer_list<V> il);

        template <class IT>
        void assign(IT other_begin, IT other_end);

        reference operator[](size_type idx);
        const_reference operator[](size_type idx) const;

        pointer data();
        const_pointer data() const;

        void resize(size_type n);

        size_type capacity() const;
        void push_back(const T& elt);

        void pop_back();

        iterator begin();
        const_iterator begin() const;
        const_iterator cbegin() const;
        iterator end();
        const_iterator end() const;
        const_iterator cend() const;

        reverse_iterator rbegin();
        const_reverse_iterator rbegin() const;
        const_reverse_iterator crbegin() const;
        reverse_iterator rend();
        const_reverse_iterator rend() const;
        const_reverse_iterator crend() const;

        size_type size() const;

        bool empty() const;

        reference front();
        const_reference front() const;
        reference back();
        const_reference back() const;

        bool on_stack();

        iterator erase(const_iterator cit);
        iterator erase(const_iterator cfirst, const_iterator clast);

        iterator insert(const_iterator it, const T& elt);

        template <std::size_t ON, class OA>
        void swap(small_vector<T, ON, OA>& rhs);

        allocator_type get_allocator() const noexcept;

    private:

        A m_allocator;

        T* m_begin = std::begin(m_data);
        T* m_end = std::begin(m_data);
        T* m_capacity = std::end(m_data);

        // stack allocated memory
        alignas(XSHAPE_ALIGNMENT) T m_data[N > 0 ? N : 1];

        void grow(size_type min_capacity = 0);
        void destroy_range(T* begin, T* end);
    };

    template <class T, std::size_t N, class A, bool INIT>
    inline small_vector<T, N, A, INIT>::~small_vector()
    {
        if (!on_stack())
        {
            detail::safe_destroy_deallocate(m_allocator, m_begin, static_cast<std::size_t>(m_capacity - m_begin));
        }
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline small_vector<T, N, A, INIT>::small_vector() noexcept
        : small_vector(allocator_type())
    {
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline small_vector<T, N, A, INIT>::small_vector(const allocator_type& alloc) noexcept
        : m_allocator(alloc)
    {
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline small_vector<T, N, A, INIT>::small_vector(size_type n, const allocator_type& alloc)
        : m_allocator(alloc)
    {
        if (INIT)
        {
            assign(n, T(0));
        }
        else
        {
            resize(n);
        }
    }

    template <class T, std::size_t N, class A, bool INIT>
    template <class IT, class>
    inline small_vector<T, N, A, INIT>::small_vector(IT begin, IT end, const allocator_type& alloc)
        : m_allocator(alloc)
    {
        assign(begin, end);
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline small_vector<T, N, A, INIT>::small_vector(const std::vector<T>& vec)
    {
        assign(vec.begin(), vec.end());
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline small_vector<T, N, A, INIT>::small_vector(size_type n, const value_type& v, const allocator_type& alloc)
        : m_allocator(alloc)
    {
        assign(n, v);
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline small_vector<T, N, A, INIT>::small_vector(std::initializer_list<T> il, const allocator_type& alloc)
        : m_allocator(alloc)
    {
        assign(il.begin(), il.end());
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline small_vector<T, N, A, INIT>& small_vector<T, N, A, INIT>::operator=(const small_vector& rhs)
    {
        assign(rhs.begin(), rhs.end());
        return *this;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline small_vector<T, N, A, INIT>& small_vector<T, N, A, INIT>::operator=(small_vector&& rhs)
    {
        assign(rhs.begin(), rhs.end());
        return *this;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline small_vector<T, N, A, INIT>& small_vector<T, N, A, INIT>::operator=(std::vector<T>& rhs)
    {
        if (this != &rhs)
        {
            m_allocator = std::allocator_traits<allocator_type>::select_on_container_copy_construction(rhs.get_allocator());
            assign(rhs.begin(), rhs.end());
        }
        return *this;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline small_vector<T, N, A, INIT>::small_vector(const small_vector& rhs)
        : m_allocator(std::allocator_traits<allocator_type>::select_on_container_copy_construction(rhs.get_allocator()))
    {
        assign(rhs.begin(), rhs.end());
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline small_vector<T, N, A, INIT>::small_vector(small_vector&& rhs)
    {
        this->swap(rhs);
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline void small_vector<T, N, A, INIT>::assign(size_type n, const value_type& v)
    {
        if (unlikely(n > N))
        {
            grow(n);
        }
        m_end = m_begin + n;
        std::fill(begin(), end(), v);
    }

    template <class T, std::size_t N, class A, bool INIT>
    template <class V>
    inline void small_vector<T, N, A, INIT>::assign(std::initializer_list<V> il)
    {
        assign(il.begin(), il.end());
    }

    template <class T, std::size_t N, class A, bool INIT>
    template <class IT>
    inline void small_vector<T, N, A, INIT>::assign(IT other_begin, IT other_end)
    {
        std::size_t size = static_cast<std::size_t>(other_end - other_begin);
        if (unlikely(size > N))
        {
            grow(size);
        }
        std::uninitialized_copy(other_begin, other_end, m_begin);
        m_end = m_begin + size;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::operator[](size_type idx) -> reference
    {
        return m_begin[idx];
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::operator[](size_type idx) const -> const_reference
    {
        return m_begin[idx];
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::data() -> pointer
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::data() const -> const_pointer
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool INIT>
    void small_vector<T, N, A, INIT>::resize(size_type n)
    {
        if (unlikely(n > N))
        {
            grow(n);
        }
        m_end = m_begin + n;
        if (INIT)
        {
            std::fill(begin(), end(), T());
        }
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::capacity() const -> size_type
    {
        return static_cast<std::size_t>(m_capacity - m_begin);
    }

    template <class T, std::size_t N, class A, bool INIT>
    void small_vector<T, N, A, INIT>::push_back(const T& elt)
    {
        if (m_end >= m_capacity)
        {
            grow();
        }
        *(m_end++) = elt;
    }

    template <class T, std::size_t N, class A, bool INIT>
    void small_vector<T, N, A, INIT>::pop_back()
    {
        --m_end;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::begin() -> iterator
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::begin() const -> const_iterator
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::cbegin() const -> const_iterator
    {
        return m_begin;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::end() -> iterator
    {
        return m_end;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::end() const -> const_iterator
    {
        return m_end;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::cend() const -> const_iterator
    {
        return m_end;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::rbegin() -> reverse_iterator
    {
        return reverse_iterator(m_end);
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::rbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_end);
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::crbegin() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_end);
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::rend() -> reverse_iterator
    {
        return reverse_iterator(m_begin);
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::rend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_begin);
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::crend() const -> const_reverse_iterator
    {
        return const_reverse_iterator(m_begin);
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::size() const -> size_type
    {
        return static_cast<size_type>(m_end - m_begin);
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::empty() const -> bool
    {
        return m_begin == m_end;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::front() -> reference
    {
        XTENSOR_ASSERT(!empty());
        return m_begin[0];
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::front() const -> const_reference
    {
        XTENSOR_ASSERT(!empty());
        return m_begin[0];
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::back() -> reference
    {
        XTENSOR_ASSERT(!empty());
        return m_end[-1];
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::back() const -> const_reference
    {
        XTENSOR_ASSERT(!empty());
        return m_end[-1];
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::on_stack() -> bool
    {
        return m_begin == &m_data[0];
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::get_allocator() const noexcept -> allocator_type
    {
        return m_allocator;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::erase(const_iterator cit) -> iterator
    {
        auto it = const_cast<pointer>(cit);
        iterator ret_val = it;
        std::move(it + 1, m_end, it);
        --m_end;
        return ret_val;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::erase(const_iterator cfirst, const_iterator clast) -> iterator
    {
        auto first = const_cast<pointer>(cfirst);
        auto last = const_cast<pointer>(clast);
        if (last == m_end)
        {
            m_end = first;
            return first;
        }

        iterator new_end = std::move(last, m_end, first);
        m_end = new_end;
        return first;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline auto small_vector<T, N, A, INIT>::insert(const_iterator cit, const T& elt) -> iterator
    {
        auto it = const_cast<pointer>(cit);
        if (it == m_end)
        {
            push_back(elt);
            return m_end - 1;
        }

        if (m_end >= m_capacity)
        {
            ptrdiff_t elt_no = it - m_begin;
            grow();
            it = m_begin + elt_no;
        }

        (*m_end) = back();
        std::move_backward(it, m_end - 1, m_end);
        ++m_end;

        // Update ref if element moved
        const T* elt_ptr = &elt;
        if (it <= elt_ptr && elt_ptr < m_end)
        {
            ++elt_ptr;
        }
        *it = *elt_ptr;
        return it;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline void small_vector<T, N, A, INIT>::destroy_range(T* begin, T* end)
    {
        if (!xtrivially_default_constructible<T>::value)
        {
            while (begin != end)
            {
                --end;
                end->~T();
            }
        }
    }

    template <class T, std::size_t N, class A, bool INIT>
    template <std::size_t ON, class OA>
    inline void small_vector<T, N, A, INIT>::swap(small_vector<T, ON, OA>& rhs)
    {
        if (this == &rhs)
        {
            return;
        }

        // We can only avoid copying elements if neither vector is small.
        if (!this->on_stack() && !rhs.on_stack()) {
            std::swap(this->m_begin, rhs.m_begin);
            std::swap(this->m_end, rhs.m_end);
            std::swap(this->m_capacity, rhs.m_capacity);
            return;
        }

        if (rhs.size() > this->capacity())
            this->resize(rhs.size());
        if (this->size() > rhs.capacity())
            rhs.resize(this->size());

        // Swap the shared elements.
        size_t num_shared = std::min(this->size(), rhs.size());

        for (size_type i = 0; i != num_shared; ++i)
        {
            std::swap((*this)[i], rhs[i]);
        }

        // Copy over the extra elts.
        if (this->size() > rhs.size())
        {
            size_t elements_diff = this->size() - rhs.size();
            std::copy(this->begin() + num_shared, this->end(), rhs.end());
            rhs.m_end = rhs.end() + elements_diff;
            this->destroy_range(this->begin() + num_shared, this->end());
            this->m_end = this->begin() + num_shared;
        }
        else if (rhs.size() > this->size())
        {
            size_t elements_diff = rhs.size() - this->size();
            std::uninitialized_copy(rhs.begin() + num_shared, rhs.end(), this->end());
            this->m_end = this->end() + elements_diff;
            this->destroy_range(rhs.begin() + num_shared, rhs.end());
            rhs.m_end = rhs.begin() + num_shared;
        }
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline void small_vector<T, N, A, INIT>::grow(size_type min_capacity)
    {
        size_type current_size = size();
        size_type new_capacity = 2 * current_size + 1; // Always grow.
        if (new_capacity < min_capacity)
        {
            new_capacity = min_capacity;
        }

        T* new_alloc;
        // is data stack allocated?
        if (m_begin == &m_data[0])
        {
            new_alloc = m_allocator.allocate(new_capacity);
            std::uninitialized_copy(m_begin, m_end, new_alloc);
        }
        else
        {
            // If this wasn't grown from the inline copy, grow the allocated space.
            new_alloc = reinterpret_cast<pointer>(realloc(this->m_begin, new_capacity * sizeof(T)));
        }
        assert(new_alloc && "Out of memory");

        m_end = new_alloc + current_size;
        m_begin = new_alloc;
        m_capacity = new_alloc + new_capacity;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline bool operator==(const std::vector<T>& lhs, const small_vector<T, N, A, INIT>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline bool operator==(const small_vector<T, N, A, INIT>& lhs, const std::vector<T>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline bool operator==(const small_vector<T, N, A, INIT>& lhs, const small_vector<T, N, A, INIT>& rhs)
    {
        return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline bool operator!=(const small_vector<T, N, A, INIT>& lhs, const small_vector<T, N, A, INIT>& rhs)
    {
        return !(lhs == rhs);
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline bool operator<(const small_vector<T, N, A, INIT>& lhs, const small_vector<T, N, A, INIT>& rhs)
    {
        return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                            rhs.begin(), rhs.end());
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline bool operator<=(const small_vector<T, N, A, INIT>& lhs, const small_vector<T, N, A, INIT>& rhs)
    {
        return !(lhs > rhs);
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline bool operator>(const small_vector<T, N, A, INIT>& lhs, const small_vector<T, N, A, INIT>& rhs)
    {
        return rhs < lhs;
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline bool operator>=(const small_vector<T, N, A, INIT>& lhs, const small_vector<T, N, A, INIT>& rhs)
    {
        return !(lhs < rhs);
    }

    template <class T, std::size_t N, class A, bool INIT>
    inline void swap(small_vector<T, N, A, INIT>& lhs, small_vector<T, N, A, INIT>& rhs) noexcept
    {
        lhs.swap(rhs);
    }

}

#endif
