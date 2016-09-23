#ifndef XFUNCTION_HPP
#define XFUNCTION_HPP

#include <type_traits>
#include <utility>
#include <tuple>
#include <algorithm>

#include "utils.hpp"
#include "xexpression.hpp"
#include "broadcast.hpp"

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
    class xfunction_iterator;

    template <class F, class R, class... E>
    class xf_broadcasting_iterator;


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

        using shape_type = array_shape<size_type>;
        using closure_type = const self_type;

        using const_iterator = xfunction_iterator<F, R, E...>;
        using broadcasting_iterator = xf_broadcasting_iterator<F, R, E...>;

        template <class Func>
        xfunction(Func&& f, const E&...e) noexcept;

        size_type dimension() const;
        bool broadcast_shape(shape_type& shape) const;

        template <class... Args>
        const_reference operator()(Args... args) const;

        const_iterator begin() const;
        const_iterator end() const;

        const_iterator cbegin() const;
        const_iterator cend() const;

        broadcasting_iterator begin(const shape_type& shape) const;
        broadcasting_iterator end(const shape_type& shape) const;

    private:

        template <size_t... I, class... Args>
        const_reference access_impl(std::index_sequence<I...>, Args... args) const;

        template <class Func, size_t... I>
        const_iterator build_const_iterator(Func&& f, std::index_sequence<I...>) const;

        template <class Func, size_t... I>
        broadcasting_iterator build_broadcasting_iterator(Func&& f, std::index_sequence<I...>) const;

        std::tuple<typename E::closure_type...> m_e;
        F m_f;

        friend class xfunction_iterator<F, R, E...>;
        friend class xf_broadcasting_iterator<F, R, E...>;
    };


    /************************
     * xfunction_iterator
     ************************/

    template <class F, class R, class... E>
    class xfunction_iterator
    {

    public:

        using self_type = xfunction_iterator<F, R, E...>;
        using xfunction_type = xfunction<F, R, E...>;

        using value_type = typename xfunction_type::value_type;
        using reference = typename xfunction_type::value_type;
        using pointer = typename xfunction_type::const_pointer;
        using difference_type = typename xfunction_type::difference_type;
        using iterator_category = std::input_iterator_tag;

        template <class... It>
        xfunction_iterator(const xfunction_type* func, It&&... it);

        self_type& operator++();
        self_type operator++(int);

        reference operator*() const;

        bool equal(const self_type& rhs) const;

    private:

        template <size_t... I>
        reference deref_impl(std::index_sequence<I...>) const;

        const xfunction_type* p_f;
        std::tuple<typename E::const_iterator...> m_it;

    };

    template <class F, class R, class... E>
    bool operator==(const xfunction_iterator<F, R, E...>& it1,
                    const xfunction_iterator<F, R, E...>& it2);

    template <class F, class R, class... E>
    bool operator!=(const xfunction_iterator<F, R, E...>& it1,
                    const xfunction_iterator<F, R, E...>& it2);


    /*************************************
     * xfunction_broadcasting_iterator
     *************************************/

    template <class F, class R, class... E>
    class xf_broadcasting_iterator
    {

    public:

        using self_type = xf_broadcasting_iterator<F, R, E...>;
        using xfunction_type = xfunction<F, R, E...>;

        using value_type = typename xfunction_type::value_type;
        using reference = typename xfunction_type::value_type;
        using pointer = typename xfunction_type::const_pointer;
        using size_type = typename xfunction_type::size_type;
        using difference_type = typename xfunction_type::difference_type;
        using iterator_category = std::input_iterator_tag;

        template <class... It>
        xf_broadcasting_iterator(const xfunction_type* func, It&&... it);

        void increment(size_type i);
        void reset(size_type i);

        reference operator*() const;

        bool equal(const self_type& rhs) const;

    private:

        template <size_t... I>
        reference deref_impl(std::index_sequence<I...>) const;

        const xfunction_type* p_f;
        std::tuple<typename E::broadcasting_iterator...> m_it;
    };

    template <class F, class R, class... E>
    bool operator==(const xf_broadcasting_iterator<F, R, E...>& it1,
                    const xf_broadcasting_iterator<F, R, E...>& it2);

    template <class F, class R, class... E>
    bool operator!=(const xf_broadcasting_iterator<F, R, E...>& it1,
                    const xf_broadcasting_iterator<F, R, E...>& it2);


    /******************************
     * xfunction implementation
     ******************************/

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
        auto func = [&shape](bool b, auto&& e) { return b && e.broadcast_shape(shape); };
        return accumulate(func, true, m_e);
    }
    
    template <class F, class R, class... E>
    template <class... Args>
    inline auto xfunction<F, R, E...>::operator()(Args... args) const -> const_reference
    {
        return access_impl(std::make_index_sequence<sizeof...(E)>(), args...);
    }

    template <class F, class R, class... E>
    template <size_t... I, class... Args>
    inline auto xfunction<F, R, E...>::access_impl(std::index_sequence<I...>, Args... args) const -> const_reference
    {
        return m_f(std::get<I>(m_e)(args...)...);
    }

    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::begin() const -> const_iterator
    {
        auto f = [](const auto& e) { return e.begin(); };
        return build_const_iterator(f, std::make_index_sequence<sizeof...(E)>());
    }
    
    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::end() const -> const_iterator
    {
        auto f = [](const auto& e) { return e.end(); };
        return build_const_iterator(f, std::make_index_sequence<sizeof...(E)>());
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
    inline auto xfunction<F, R, E...>::begin(const shape_type& shape) const -> broadcasting_iterator
    {
        auto f = [&shape](const auto& e) { return e.begin(shape); };
        return build_broadcasting_iterator(f, std::make_index_sequence<sizeof...(E)>());
    }

    template <class F, class R, class... E>
    inline auto xfunction<F, R, E...>::end(const shape_type& shape) const -> broadcasting_iterator
    {
        auto f = [&shape](const auto& e) { return e.end(shape); };
        return build_broadcasting_iterator(f, std::make_index_sequence<sizeof...(E)>());
    }

    template <class F, class R, class... E>
    template <class Func, size_t... I>
    inline auto xfunction<F, R, E...>::build_const_iterator(Func&& f, std::index_sequence<I...>) const -> const_iterator
    {
        return const_iterator(this, f(std::get<I>(m_e))...);
    }

    template <class F, class R, class... E>
    template <class Func, size_t... I>
    inline auto xfunction<F, R, E...>::build_broadcasting_iterator(Func&& f, std::index_sequence<I...>) const -> broadcasting_iterator
    {
        return broadcasting_iterator(this, f(std::get<I>(m_e))...);
    }


    /***************************************
     * xfunction_iterator implementation
     ***************************************/

    template <class F, class R, class... E>
    template <class... It>
    inline xfunction_iterator<F, R, E...>::xfunction_iterator(const xfunction_type* func, It&&... it)
        : p_f(func), m_it(std::forward<It>(it)...)
    {
    }

    template <class F, class R, class... E>
    inline auto xfunction_iterator<F, R, E...>::operator++() -> self_type&
    {
        auto f = [](auto& it) { ++it; };
        for_each(f, m_it);
        return *this;
    }
    
    template <class F, class R, class... E>
    inline auto xfunction_iterator<F, R, E...>::operator++(int) -> self_type
    {
        self_type tmp(*this);
        auto f = [](auto& it) { ++it; };
        for_each(f, m_it);
        return tmp;
    }

    template <class F, class R, class... E>
    inline auto xfunction_iterator<F, R, E...>::operator*() const -> reference
    {
        return deref_impl(std::make_index_sequence<sizeof...(E)>());
    }

    template <class F, class R, class... E>
    inline bool xfunction_iterator<F, R, E...>::equal(const xfunction_iterator& rhs) const
    {
        return p_f == rhs.p_f && m_it = rhs.m_it;
    }

    template <class F, class R, class... E>
    template <size_t... I>
    inline auto xfunction_iterator<F, R, E...>::deref_impl(std::index_sequence<I...>) const -> reference
    {
        return (p_f->m_f)(*std::get<I>(m_it)...);
    }

    template <class F, class R, class... E>
    inline bool operator==(const xfunction_iterator<F, R, E...>& it1,
                           const xfunction_iterator<F, R, E...>& it2)
    {
        return it1.equal(it2);
    }

    template <class F, class R, class... E>
    inline bool operator!=(const xfunction_iterator<F, R, E...>& it1,
                           const xfunction_iterator<F, R, E...>& it2)
    {
        return !(it1.equal(it2));
    }


    /*********************************************
     * xf_broadcasting_iterator implementation
     *********************************************/

    template <class F, class R, class... E>
    template <class... It>
    inline xf_broadcasting_iterator<F, R, E...>::xf_broadcasting_iterator(const xfunction_type* func, It&&... it)
        : p_f(func), m_it(std::forward<It>(it)...)
    {
    }

    template <class F, class R, class... E>
    inline void xf_broadcasting_iterator<F, R, E...>::increment(size_type i)
    {
        auto f = [i](auto& it) { it.increment(i); };
        for_each(f, m_it);
    }

    template <class F, class R, class... E>
    inline void xf_broadcasting_iterator<F, R, E...>::reset(size_type i)
    {
        auto f = [i](auto& it) { it.reset(i); };
        for_each(f, m_it);
    }

    template <class F, class R, class... E>
    inline auto xf_broadcasting_iterator<F, R, E...>::operator*() const -> reference
    {
        return deref_impl(std::make_index_sequence<sizeof...(E)>());
    }

    template <class F, class R, class... E>
    inline bool xf_broadcasting_iterator<F, R, E...>::equal(const self_type& rhs) const
    {
        return p_f == rhs.pf && m_it == rhs.m_it;
    }

    template <class F, class R, class... E>
    template <size_t... I>
    inline auto xf_broadcasting_iterator<F, R, E...>::deref_impl(std::index_sequence<I...>) const -> reference
    {
        return (p_f->m_f)(*std::get<I>(m_it)...);
    }

    template <class F, class R, class... E>
    inline bool operator==(const xf_broadcasting_iterator<F, R, E...>& it1,
                           const xf_broadcasting_iterator<F, R, E...>& it2)
    {
        return it1.equal(it2);
    }

    template <class F, class R, class... E>
    inline bool operator!=(const xf_broadcasting_iterator<F, R, E...>& it1,
                           const xf_broadcasting_iterator<F, R, E...>& it2)
    {
        return !(it1.equal(it2));
    }

}

#endif

