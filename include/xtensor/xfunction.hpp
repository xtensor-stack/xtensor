/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XFUNCTION_HPP
#define XFUNCTION_HPP

#include <cstddef>
#include <type_traits>
#include <utility>
#include <tuple>
#include <algorithm>
#include <iterator>

#include "xexpression.hpp"
#include "xiterator.hpp"
#include "xutils.hpp"

namespace xt
{

    namespace detail
    {

        /********************
         * common_size_type *
         ********************/

        template <class... Args>
        struct common_size_type_impl
        {
            using type = std::common_type_t<typename Args::size_type...>;
        };

        template <>
        struct common_size_type_impl<>
        {
            using type = std::size_t;
        };

        template<class... Args>
        using common_size_type = typename common_size_type_impl<Args...>::type;

        /**************************
         * common_difference type *
         **************************/

        template <class... Args>
        struct common_difference_type_impl
        {
            using type = std::common_type_t<typename Args::difference_type...>;
        };

        template <>
        struct common_difference_type_impl<>
        {
            using type = std::size_t;
        };

        template<class... Args>
        using common_difference_type = typename common_difference_type_impl<Args...>::type;

        /*********************
         * common_value_type *
         *********************/

        template <class... Args>
        using common_value_type = std::common_type_t<get_value_type<Args>...>;
    }

    template <class F, class R, class... CT>
    class xf_storage_iterator;

    template <class F, class R, class... CT>
    class xfunction_stepper;

    /*************
     * xfunction *
     *************/

    /**
     * @class xfunction
     * @brief Multidimensional function operating on xexpression.
     *
     * Th xfunction class implements a multidimensional function
     * operating on xexpression.
     *
     * @tparam F the function type
     * @tparam R the return type of the function
     * @tparam CT the closure types for arguments of the function
     */
    template <class F, class R, class... CT>
    class xfunction : public xexpression<xfunction<F, R, CT...>>
    {

    public:

        using self_type = xfunction<F, R, CT...>;
        using functor_type = typename std::remove_reference<F>::type;

        using value_type = R;
        using reference = value_type;
        using const_reference = value_type;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = detail::common_size_type<typename std::decay<typename CT::type>::type...>;
        using difference_type = detail::common_difference_type<typename std::decay<typename CT::type>::type...>;

        using shape_type = promote_shape_t<typename std::decay<typename CT::type>::type::shape_type...>;

        using const_stepper = xfunction_stepper<F, R, CT...>;
        using const_iterator = xiterator<const_stepper, shape_type>;
        using const_storage_iterator = xf_storage_iterator<F, R, CT...>;

        template <class Func>
        xfunction(Func&& f, typename CT::type... e) noexcept;

        size_type dimension() const noexcept;
        const shape_type& shape() const;

        template <class... Args>
        const_reference operator()(Args... args) const;

        const_reference operator[](const xindex& index) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool broadcast_shape(S& shape) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const noexcept;

        const_iterator begin() const noexcept;
        const_iterator end() const noexcept;
        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;

        template <class S>
        xiterator<const_stepper, S> xbegin(const S& shape) const noexcept;
        template <class S>
        xiterator<const_stepper, S> xend(const S& shape) const noexcept;
        template <class S>
        xiterator<const_stepper, S> cxbegin(const S& shape) const noexcept;
        template <class S>
        xiterator<const_stepper, S> cxend(const S& shape) const noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;

        template <class S>
        const_stepper stepper_end(const S& shape) const noexcept;

        const_storage_iterator storage_begin() const noexcept;
        const_storage_iterator storage_end() const noexcept;

        const_storage_iterator storage_cbegin() const noexcept;
        const_storage_iterator storage_cend() const noexcept;

    private:

        template <std::size_t... I, class... Args>
        const_reference access_impl(std::index_sequence<I...>, Args... args) const;

        template <std::size_t... I, class It>
        const_reference element_access_impl(std::index_sequence<I...>, It first, It last) const;

        template <class Func, std::size_t... I>
        const_stepper build_stepper(Func&& f, std::index_sequence<I...>) const noexcept;

        template <class Func, std::size_t... I>
        const_storage_iterator build_storage_iterator(Func&& f, std::index_sequence<I...>) const noexcept;

        std::tuple<typename CT::type...> m_e;
        functor_type m_f;
        shape_type m_shape;

        friend class xf_storage_iterator<F, R, CT...>;
        friend class xfunction_stepper<F, R, CT...>;
    };

    /***********************
     * xf_storage_iterator *
     ***********************/

    template <class F, class R, class... CT>
    class xf_storage_iterator
    {

    public:

        using self_type = xf_storage_iterator<F, R, CT...>;
        using functor_type = typename std::remove_reference<F>::type;
        using xfunction_type = xfunction<F, R, CT...>;

        using value_type = typename xfunction_type::value_type;
        using reference = typename xfunction_type::value_type;
        using pointer = typename xfunction_type::const_pointer;
        using difference_type = typename xfunction_type::difference_type;
        using iterator_category = std::forward_iterator_tag;

        template <class... It>
        xf_storage_iterator(const xfunction_type* func, It&&... it) noexcept;

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;

        bool equal(const self_type& rhs) const;

    private:

        template <std::size_t... I>
        reference deref_impl(std::index_sequence<I...>) const;

        const xfunction_type* p_f;
        std::tuple<typename std::decay<typename CT::type>::type::const_storage_iterator...> m_it;

    };

    template <class F, class R, class... CT>
    bool operator==(const xf_storage_iterator<F, R, CT...>& it1,
                    const xf_storage_iterator<F, R, CT...>& it2);

    template <class F, class R, class... CT>
    bool operator!=(const xf_storage_iterator<F, R, CT...>& it1,
                    const xf_storage_iterator<F, R, CT...>& it2);

    /*********************
     * xfunction_stepper *
     *********************/

    template <class F, class R, class... CT>
    class xfunction_stepper
    {

    public:

        using self_type = xfunction_stepper<F, R, CT...>;
        using functor_type = typename std::remove_reference<F>::type;
        using xfunction_type = xfunction<F, R, CT...>;

        using value_type = typename xfunction_type::value_type;
        using reference = typename xfunction_type::value_type;
        using pointer = typename xfunction_type::const_pointer;
        using size_type = typename xfunction_type::size_type;
        using difference_type = typename xfunction_type::difference_type;
        using iterator_category = std::input_iterator_tag;

        using shape_type = typename xfunction_type::shape_type;

        template <class... It>
        xfunction_stepper(const xfunction_type* func, It&&... it) noexcept;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);

        void to_end();

        reference operator*() const;

        bool equal(const self_type& rhs) const;

    private:

        template <std::size_t... I>
        reference deref_impl(std::index_sequence<I...>) const;

        const xfunction_type* p_f;
        std::tuple<typename std::decay<typename CT::type>::type::const_stepper...> m_it;
    };

    template <class F, class R, class... CT>
    bool operator==(const xfunction_stepper<F, R, CT...>& it1,
                    const xfunction_stepper<F, R, CT...>& it2);

    template <class F, class R, class... CT>
    bool operator!=(const xfunction_stepper<F, R, CT...>& it1,
                    const xfunction_stepper<F, R, CT...>& it2);

    /****************************
     * xfunction implementation *
     ****************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xfunction applying the specified function to the given
     * arguments.
     * @param f the function to apply
     * @param e the \ref xexpression arguments
     */
    template <class F, class R, class... CT>
    template <class Func>
    inline xfunction<F, R, CT...>::xfunction(Func&& f, typename CT::type... e) noexcept
        : m_e(e...), m_f(std::forward<Func>(f)), m_shape(make_sequence<shape_type>(dimension(), size_type(1)))
    {
        broadcast_shape(m_shape);
    }
    //@}

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the number of dimensions of the function.
     */
    template <class F, class R, class... CT>
    inline auto xfunction<F, R, CT...>::dimension() const noexcept -> size_type
    {
        auto func = [](size_type d, auto&& e) noexcept { return std::max(d, e.dimension()); };
        return accumulate(func, size_type(0), m_e);
    }

    /**
     * Returns the shape of the xfunction.
     */
    template <class F, class R, class... CT>
    inline auto xfunction<F, R, CT...>::shape() const -> const shape_type&
    {
        return m_shape;
    }
    //@}

    /**
     * @name Data
     */
    /**
     * Returns a constant reference to the element at the specified position in the function.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the function.
     */
    template <class F, class R, class... CT>
    template <class... Args>
    inline auto xfunction<F, R, CT...>::operator()(Args... args) const -> const_reference
    {
        return access_impl(std::make_index_sequence<sizeof...(CT)>(), args...);
    }

    template <class F, class R, class... CT>
    inline auto xfunction<F, R, CT...>::operator[](const xindex& index) const -> const_reference
    {
        return element(index.cbegin(), index.cend());
    }
    
    /**
     * Returns a constant reference to the element at the specified position in the function.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the squence should be equal to or greater
     * than the number of dimensions of the container.
     */
    template <class F, class R, class... CT>
    template <class It>
    inline auto xfunction<F, R, CT...>::element(It first, It last) const -> const_reference
    {
        return element_access_impl(std::make_index_sequence<sizeof...(CT)>(), first, last);
    }
    //@}
    
    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the function to the specified parameter.
     * @param shape the result shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class R, class... CT>
    template <class S>
    inline bool xfunction<F, R, CT...>::broadcast_shape(S& shape) const
    {
        // e.broadcast_shape must be evaluated even if b is false
        auto func = [&shape](bool b, auto&& e) { return e.broadcast_shape(shape) && b; };
        return accumulate(func, true, m_e);
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class R, class... CT>
    template <class S>
    inline bool xfunction<F, R, CT...>::is_trivial_broadcast(const S& strides) const noexcept
    {
        auto func = [&strides](bool b, auto&& e) { return b && e.is_trivial_broadcast(strides); };
        return accumulate(func, true, m_e);
    }
    //@}

    /**
     * @name Iterators
     */
    //@{
    /**
     * Returns a constant iterator to the first element of the function.
     */
    template <class F, class R, class... CT>
    inline auto xfunction<F, R, CT...>::begin() const noexcept -> const_iterator
    {
        return xbegin(shape());
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the function.
     */
    template <class F, class R, class... CT>
    inline auto xfunction<F, R, CT...>::end() const noexcept -> const_iterator
    {
        return xend(shape());
    }

    /**
     * Returns a constant iterator to the first element of the function.
     */
    template <class F, class R, class... CT>
    inline auto xfunction<F, R, CT...>::cbegin() const noexcept -> const_iterator
    {
        return begin();
    }

    /**
     * Returns a constant iterator to the element following the last element
     * of the function.
     */
    template <class F, class R, class... CT>
    inline auto xfunction<F, R, CT...>::cend() const noexcept -> const_iterator
    {
        return end();
    }

    /**
     * Returns a constant iterator to the first element of the function. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class F, class R, class... CT>
    template <class S>
    inline auto xfunction<F, R, CT...>::xbegin(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xiterator<const_stepper, S>(stepper_begin(shape), shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * function. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class F, class R, class... CT>
    template <class S>
    inline auto xfunction<F, R, CT...>::xend(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xiterator<const_stepper, S>(stepper_end(shape), shape);
    }

    /**
     * Returns a constant iterator to the first element of the function. The
     * iteration is broadcasted to the specified shape.
     * @param shape the shape used for braodcasting
     */
    template <class F, class R, class... CT>
    template <class S>
    inline auto xfunction<F, R, CT...>::cxbegin(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xbegin(shape);
    }

    /**
     * Returns a constant iterator to the element following the last element of the
     * function. The iteration is broadcasted to the specified shape.
     * @param shape the shape used for broadcasting
     */
    template <class F, class R, class... CT>
    template <class S>
    inline auto xfunction<F, R, CT...>::cxend(const S& shape) const noexcept -> xiterator<const_stepper, S>
    {
        return xend(shape);
    }
    //@}

    template <class F, class R, class... CT>
    template <class S>
    inline auto xfunction<F, R, CT...>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        auto f = [&shape](const auto& e) noexcept { return e.stepper_begin(shape); };
        return build_stepper(f, std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class R, class... CT>
    template <class S>
    inline auto xfunction<F, R, CT...>::stepper_end(const S& shape) const noexcept -> const_stepper
    {
        auto f = [&shape](const auto& e) noexcept { return e.stepper_end(shape); };
        return build_stepper(f, std::make_index_sequence<sizeof...(CT)>());
    }

    /**
     * @name Storage iterators
     */
    /**
     * Returns an iterator to the first element of the buffer
     * containing the elements of the function.
     */
    template <class F, class R, class... CT>
    inline auto xfunction<F, R, CT...>::storage_begin() const noexcept -> const_storage_iterator
    {
        auto f = [](const auto& e) noexcept { return e.storage_cbegin(); };
        return build_storage_iterator(f, std::make_index_sequence<sizeof...(CT)>());
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the function.
     */
    template <class F, class R, class... CT>
    inline auto xfunction<F, R, CT...>::storage_end() const noexcept -> const_storage_iterator
    {
        auto f = [](const auto& e) noexcept { return e.storage_cend(); };
        return build_storage_iterator(f, std::make_index_sequence<sizeof...(CT)>());
    }

    /**
     * Returns a constant iterator to the first element of the buffer
     * containing the elements of the function.
     */
    template <class F, class R, class... CT>
    inline auto xfunction<F, R, CT...>::storage_cbegin() const noexcept -> const_storage_iterator
    {
        return storage_begin();
    }

    /**
     * Returns a constant iterator to the element following the last
     * element of the buffer containing the elements of the function.
     */
    template <class F, class R, class... CT>
    inline auto xfunction<F, R, CT...>::storage_cend() const noexcept -> const_storage_iterator
    {
        return storage_end();
    }
    //@}

    template <class F, class R, class... CT>
    template <std::size_t... I, class... Args>
    inline auto xfunction<F, R, CT...>::access_impl(std::index_sequence<I...>, Args... args) const -> const_reference
    {
        return m_f(detail::get_element(std::get<I>(m_e), args...)...);
    }

    template <class F, class R, class... CT>
    template <std::size_t... I, class It>
    inline auto xfunction<F, R, CT...>::element_access_impl(std::index_sequence<I...>, It first, It last) const -> const_reference
    {
        return m_f((std::get<I>(m_e).element(first, last))...);
    }

    template <class F, class R, class... CT>
    template <class Func, std::size_t... I>
    inline auto xfunction<F, R, CT...>::build_stepper(Func&& f, std::index_sequence<I...>) const noexcept -> const_stepper
    {
        return const_stepper(this, f(std::get<I>(m_e))...);
    }

    template <class F, class R, class... CT>
    template <class Func, std::size_t... I>
    inline auto xfunction<F, R, CT...>::build_storage_iterator(Func&& f, std::index_sequence<I...>) const noexcept -> const_storage_iterator
    {
        return const_storage_iterator(this, f(std::get<I>(m_e))...);
    }

    /**************************************
     * xf_storage_iterator implementation *
     **************************************/

    template <class F, class R, class... CT>
    template <class... It>
    inline xf_storage_iterator<F, R, CT...>::xf_storage_iterator(const xfunction_type* func, It&&... it) noexcept
        : p_f(func), m_it(std::forward<It>(it)...)
    {
    }

    template <class F, class R, class... CT>
    inline auto xf_storage_iterator<F, R, CT...>::operator++() -> self_type&
    {
        auto f = [](auto& it) { ++it; };
        for_each(f, m_it);
        return *this;
    }

    template <class F, class R, class... CT>
    inline auto xf_storage_iterator<F, R, CT...>::operator++(int) -> self_type
    {
        self_type tmp(*this);
        auto f = [](auto& it) { ++it; };
        for_each(f, m_it);
        return tmp;
    }

    template <class F, class R, class... CT>
    inline auto xf_storage_iterator<F, R, CT...>::operator*() const -> reference
    {
        return deref_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class R, class... CT>
    inline bool xf_storage_iterator<F, R, CT...>::equal(const self_type& rhs) const
    {
        return p_f == rhs.p_f && m_it == rhs.m_it;
    }

    template <class F, class R, class... CT>
    template <std::size_t... I>
    inline auto xf_storage_iterator<F, R, CT...>::deref_impl(std::index_sequence<I...>) const -> reference
    {
        return (p_f->m_f)(*std::get<I>(m_it)...);
    }

    template <class F, class R, class... CT>
    inline bool operator==(const xf_storage_iterator<F, R, CT...>& it1,
                           const xf_storage_iterator<F, R, CT...>& it2)
    {
        return it1.equal(it2);
    }

    template <class F, class R, class... CT>
    inline bool operator!=(const xf_storage_iterator<F, R, CT...>& it1,
                           const xf_storage_iterator<F, R, CT...>& it2)
    {
        return !(it1.equal(it2));
    }

    /************************************
     * xfunction_stepper implementation *
     ************************************/

    template <class F, class R, class... CT>
    template <class... It>
    inline xfunction_stepper<F, R, CT...>::xfunction_stepper(const xfunction_type* func, It&&... it) noexcept
        : p_f(func), m_it(std::forward<It>(it)...)
    {
    }

    template <class F, class R, class... CT>
    inline void xfunction_stepper<F, R, CT...>::step(size_type dim, size_type n)
    {
        auto f = [dim, n](auto& it) { it.step(dim, n); };
        for_each(f, m_it);
    }

    template <class F, class R, class... CT>
    inline void xfunction_stepper<F, R, CT...>::step_back(size_type dim, size_type n)
    {
        auto f = [dim, n](auto& it) { it.step_back(dim, n); };
        for_each(f, m_it);
    }

    template <class F, class R, class... CT>
    inline void xfunction_stepper<F, R, CT...>::reset(size_type dim)
    {
        auto f = [dim](auto& it) { it.reset(dim); };
        for_each(f, m_it);
    }

    template <class F, class R, class... CT>
    inline void xfunction_stepper<F, R, CT...>::to_end()
    {
        auto f = [](auto& it) { it.to_end(); };
        for_each(f, m_it);
    }

    template <class F, class R, class... CT>
    inline auto xfunction_stepper<F, R, CT...>::operator*() const -> reference
    {
        return deref_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class R, class... CT>
    inline bool xfunction_stepper<F, R, CT...>::equal(const self_type& rhs) const
    {
        return p_f == rhs.p_f && m_it == rhs.m_it;
    }

    template <class F, class R, class... CT>
    template <std::size_t... I>
    inline auto xfunction_stepper<F, R, CT...>::deref_impl(std::index_sequence<I...>) const -> reference
    {
        return (p_f->m_f)(*std::get<I>(m_it)...);
    }

    template <class F, class R, class... CT>
    inline bool operator==(const xfunction_stepper<F, R, CT...>& it1,
                           const xfunction_stepper<F, R, CT...>& it2)
    {
        return it1.equal(it2);
    }

    template <class F, class R, class... CT>
    inline bool operator!=(const xfunction_stepper<F, R, CT...>& it1,
                           const xfunction_stepper<F, R, CT...>& it2)
    {
        return !(it1.equal(it2));
    }
}

#endif

