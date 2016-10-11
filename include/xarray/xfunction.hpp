#ifndef XFUNCTION_HPP
#define XFUNCTION_HPP

#include <cstddef>
#include <type_traits>
#include <utility>
#include <tuple>
#include <algorithm>

#include "xexpression.hpp"
#include "xiterator.hpp"
#include "xutils.hpp"

namespace qs
{

    namespace detail
    {
        template <class... Args>
        using common_size_type = std::common_type_t<typename Args::size_type...>;

        template <class... Args>
        using common_difference_type = std::common_type_t<typename Args::difference_type...>;

        template <class... Args>
        using common_value_type = std::common_type_t<get_value_type<Args>...>;
    }

    template <class F, class R, class... E>
    class xf_storage_iterator;

    template <class F, class R, class... E>
    class xfunction_stepper;


    /***************
     * xfunction
     ***************/

    template <class F, class R, class... E>
    class xfunction : public xexpression<xfunction<F, R, E...>>
    {

    public:

        using self_type = xfunction<F, E...>;
        using functor_type = F;

        using value_type = R;
        using const_reference = value_type;
        using const_pointer = const value_type*;
        using size_type = detail::common_size_type<E...>;
        using difference_type = detail::common_difference_type<E...>;

        using shape_type = xshape<size_type>;
        using strides_type = xstrides<size_type>;
        using closure_type = const self_type;

        using const_stepper = xfunction_stepper<F, R, E...>;
        using const_iterator = xiterator<const_stepper>;
        using const_storage_iterator = xf_storage_iterator<F, R, E...>;

        template <class Func>
        xfunction(Func&& f, const E&...e) noexcept;

        size_type dimension() const;
        bool broadcast_shape(shape_type& shape) const;
        bool is_trivial_broadcast(const strides_type& strides) const;

        template <class... Args>
        const_reference operator()(Args... args) const;

        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const;
        const_iterator cend() const;

        const_iterator xbegin(const shape_type& shape) const;
        const_iterator xend(const shape_type& shape) const;
        const_iterator cxbegin(const shape_type& shape) const;
        const_iterator cxend(const shape_type& shape) const;

        const_stepper stepper_begin(const shape_type& shape) const;
        const_stepper stepper_end(const shape_type& shape) const;

        const_storage_iterator storage_begin() const;
        const_storage_iterator storage_end() const;

    private:

        template <std::size_t... I, class... Args>
        const_reference access_impl(std::index_sequence<I...>, Args... args) const;

        template <class Func, std::size_t... I>
        const_stepper build_stepper(Func&& f, std::index_sequence<I...>) const;

        template <class Func, std::size_t... I>
        const_storage_iterator build_storage_iterator(Func&& f, std::index_sequence<I...>) const;

        std::tuple<typename E::closure_type...> m_e;
        typename std::remove_reference<F>::type m_f;

        friend class xf_storage_iterator<F, R, E...>;
        friend class xfunction_stepper<F, R, E...>;
    };


    /*************************
     * xf_storage_iterator
     *************************/

    template <class F, class R, class... E>
    class xf_storage_iterator
    {

    public:

        using self_type = xf_storage_iterator<F, R, E...>;
        using xfunction_type = xfunction<F, R, E...>;

        using value_type = typename xfunction_type::value_type;
        using reference = typename xfunction_type::value_type;
        using pointer = typename xfunction_type::const_pointer;
        using difference_type = typename xfunction_type::difference_type;
        using iterator_category = std::input_iterator_tag;

        template <class... It>
        xf_storage_iterator(const xfunction_type* func, It&&... it);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;

        bool equal(const self_type& rhs) const;

    private:

        template <std::size_t... I>
        reference deref_impl(std::index_sequence<I...>) const;

        const xfunction_type* p_f;
        std::tuple<typename E::const_storage_iterator...> m_it;

    };

    template <class F, class R, class... E>
    bool operator==(const xf_storage_iterator<F, R, E...>& it1,
                    const xf_storage_iterator<F, R, E...>& it2);

    template <class F, class R, class... E>
    bool operator!=(const xf_storage_iterator<F, R, E...>& it1,
                    const xf_storage_iterator<F, R, E...>& it2);


    /******************************
     * xfunction_stepper
     ******************************/

    template <class F, class R, class... E>
    class xfunction_stepper
    {

    public:

        using self_type = xfunction_stepper<F, R, E...>;
        using xfunction_type = xfunction<F, R, E...>;

        using value_type = typename xfunction_type::value_type;
        using reference = typename xfunction_type::value_type;
        using pointer = typename xfunction_type::const_pointer;
        using size_type = typename xfunction_type::size_type;
        using difference_type = typename xfunction_type::difference_type;
        using iterator_category = std::input_iterator_tag;

        template <class... It>
        xfunction_stepper(const xfunction_type* func, It&&... it);

        void step(size_type i);
        void reset(size_type i);

        void to_end();

        reference operator*() const;

        bool equal(const self_type& rhs) const;

    private:

        template <std::size_t... I>
        reference deref_impl(std::index_sequence<I...>) const;

        const xfunction_type* p_f;
        std::tuple<typename E::const_stepper...> m_it;
    };

    template <class F, class R, class... E>
    bool operator==(const xfunction_stepper<F, R, E...>& it1,
                    const xfunction_stepper<F, R, E...>& it2);

    template <class F, class R, class... E>
    bool operator!=(const xfunction_stepper<F, R, E...>& it1,
                    const xfunction_stepper<F, R, E...>& it2);


    /******************************
     * xfunction implementation
     ******************************/

    namespace detail
    {
        template <class E, class S>
        inline typename E::const_reference get_element(const E& e, S i)
        {
            return e(i);
        }

        template <class E, class S, class... Args>
        inline typename E::const_reference get_element(const E& e, S i, Args... args)
        {
            if(sizeof...(Args) >= e.dimension())
                return get_element(e, args...);
            return e(i, args...);
        }
    }

    template <class F, class R, class... E>
    template <class Func>
    inline xfunction<F, R, E...>::xfunction(Func&& f, const E&... e) noexcept
        : m_f(std::forward<Func>(f)), m_e(e...)
    {
    }

    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::dimension() const -> size_type
    {
        auto func = [](size_type d, auto&& e) { return std::max(d, e.dimension()); };
        return accumulate(func, size_type(0), m_e);
    }

    template <class F, class R, class... E>
    inline bool xfunction<F, R, E...>::broadcast_shape(shape_type& shape) const
    {
        // e.broadcast_shape must be evaluated even if b is false
        auto func = [&shape](bool b, auto&& e) { return e.broadcast_shape(shape) && b; };
        return accumulate(func, true, m_e);
    }
    
    template <class F, class R, class... E>
    inline bool xfunction<F, R, E...>::is_trivial_broadcast(const strides_type& strides) const
    {
        auto func = [&strides](bool b, auto&& e) { return b && e.is_trivial_broadcast(strides); };
        return accumulate(func, true, m_e);
    }

    template <class F, class R, class... E>
    template <class... Args>
    inline auto xfunction<F, R, E...>::operator()(Args... args) const -> const_reference
    {
        return access_impl(std::make_index_sequence<sizeof...(E)>(), args...);
    }

    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::begin() const -> const_iterator
    {
        shape_type shape(dimension(), size_type(1));
        broadcast_shape(shape);
        return xbegin(shape);
    }

    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::end() const -> const_iterator
    {
        shape_type shape(dimension(), size_type(1));
        broadcast_shape(shape);
        return xend(shape);
    }

    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::cbegin() const -> const_iterator
    {
        return begin();
    }

    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::cend() const -> const_iterator
    {
        return end();
    }

    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::xbegin(const shape_type& shape) const -> const_iterator
    {
        return const_iterator(stepper_begin(shape), shape);
    }

    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::xend(const shape_type& shape) const -> const_iterator
    {
        return const_iterator(stepper_end(shape), shape);
    }

    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::cxbegin(const shape_type& shape) const -> const_iterator
    {
        return xbegin(shape);
    }

    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::cxend(const shape_type& shape) const -> const_iterator
    {
        return xend(shape);
    }

    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::stepper_begin(const shape_type& shape) const -> const_stepper 
    {
        auto f = [&shape](const auto& e) { return e.stepper_begin(shape); };
        return build_stepper(f, std::make_index_sequence<sizeof...(E)>());
    }

    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::stepper_end(const shape_type& shape) const -> const_stepper
    {
        auto f = [&shape](const auto& e) { return e.stepper_end(shape); };
        return build_stepper(f, std::make_index_sequence<sizeof...(E)>());
    }
    
    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::storage_begin() const -> const_storage_iterator
    {
        auto f = [](const auto& e) { return e.storage_begin(); };
        return build_storage_iterator(f, std::make_index_sequence<sizeof...(E)>());
    }
    
    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::storage_end() const -> const_storage_iterator
    {
        auto f = [](const auto& e) { return e.storage_end(); };
        return build_storage_iterator(f, std::make_index_sequence<sizeof...(E)>());
    }

    template <class F, class R, class... E>
    template <std::size_t... I, class... Args>
    inline auto xfunction<F, R, E...>::access_impl(std::index_sequence<I...>, Args... args) const -> const_reference
    {
        return m_f(detail::get_element(std::get<I>(m_e), args...)...);
    }

    template <class F, class R, class... E>
    template <class Func, std::size_t... I>
    inline auto xfunction<F, R, E...>::build_stepper(Func&& f, std::index_sequence<I...>) const -> const_stepper
    {
        return const_stepper(this, f(std::get<I>(m_e))...);
    }

    template <class F, class R, class... E>
    template <class Func, std::size_t... I>
    inline auto xfunction<F, R, E...>::build_storage_iterator(Func&& f, std::index_sequence<I...>) const -> const_storage_iterator
    {
        return const_storage_iterator(this, f(std::get<I>(m_e))...);
    }


    /****************************************
     * xf_storage_iterator implementation
     ****************************************/

    template <class F, class R, class... E>
    template <class... It>
    inline xf_storage_iterator<F, R, E...>::xf_storage_iterator(const xfunction_type* func, It&&... it)
        : p_f(func), m_it(std::forward<It>(it)...)
    {
    }

    template <class F, class R, class... E>
    inline auto xf_storage_iterator<F, R, E...>::operator++() -> self_type&
    {
        auto f = [](auto& it) { ++it; };
        for_each(f, m_it);
        return *this;
    }
    
    template <class F, class R, class... E>
    inline auto xf_storage_iterator<F, R, E...>::operator++(int) -> self_type
    {
        self_type tmp(*this);
        auto f = [](auto& it) { ++it; };
        for_each(f, m_it);
        return tmp;
    }

    template <class F, class R, class... E>
    inline auto xf_storage_iterator<F, R, E...>::operator*() const -> reference
    {
        return deref_impl(std::make_index_sequence<sizeof...(E)>());
    }

    template <class F, class R, class... E>
    inline bool xf_storage_iterator<F, R, E...>::equal(const xf_storage_iterator& rhs) const
    {
        return p_f == rhs.p_f && m_it == rhs.m_it;
    }

    template <class F, class R, class... E>
    template <std::size_t... I>
    inline auto xf_storage_iterator<F, R, E...>::deref_impl(std::index_sequence<I...>) const -> reference
    {
        return (p_f->m_f)(*std::get<I>(m_it)...);
    }

    template <class F, class R, class... E>
    inline bool operator==(const xf_storage_iterator<F, R, E...>& it1,
                           const xf_storage_iterator<F, R, E...>& it2)
    {
        return it1.equal(it2);
    }

    template <class F, class R, class... E>
    inline bool operator!=(const xf_storage_iterator<F, R, E...>& it1,
                           const xf_storage_iterator<F, R, E...>& it2)
    {
        return !(it1.equal(it2));
    }


    /**************************************
     * xfunction_stepper implementation
     **************************************/

    template <class F, class R, class... E>
    template <class... It>
    inline xfunction_stepper<F, R, E...>::xfunction_stepper(const xfunction_type* func, It&&... it)
        : p_f(func), m_it(std::forward<It>(it)...)
    {
    }

    template <class F, class R, class... E>
    inline void xfunction_stepper<F, R, E...>::step(size_type i)
    {
        auto f = [i](auto& it) { it.step(i); };
        for_each(f, m_it);
    }

    template <class F, class R, class... E>
    inline void xfunction_stepper<F, R, E...>::reset(size_type i)
    {
        auto f = [i](auto& it) { it.reset(i); };
        for_each(f, m_it);
    }

    template <class F, class R, class... E>
    inline void xfunction_stepper<F, R, E...>::to_end()
    {
        auto f = [](auto& it) { it.to_end(); };
        for_each(f, m_it);
    }

    template <class F, class R, class... E>
    inline auto xfunction_stepper<F, R, E...>::operator*() const -> reference
    {
        return deref_impl(std::make_index_sequence<sizeof...(E)>());
    }

    template <class F, class R, class... E>
    inline bool xfunction_stepper<F, R, E...>::equal(const self_type& rhs) const
    {
        return p_f == rhs.p_f && m_it == rhs.m_it;
    }

    template <class F, class R, class... E>
    template <std::size_t... I>
    inline auto xfunction_stepper<F, R, E...>::deref_impl(std::index_sequence<I...>) const -> reference
    {
        return (p_f->m_f)(*std::get<I>(m_it)...);
    }

    template <class F, class R, class... E>
    inline bool operator==(const xfunction_stepper<F, R, E...>& it1,
                           const xfunction_stepper<F, R, E...>& it2)
    {
        return it1.equal(it2);
    }

    template <class F, class R, class... E>
    inline bool operator!=(const xfunction_stepper<F, R, E...>& it1,
                           const xfunction_stepper<F, R, E...>& it2)
    {
        return !(it1.equal(it2));
    }

}

#endif

