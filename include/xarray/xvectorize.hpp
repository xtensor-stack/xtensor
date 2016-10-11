#ifndef XVECTORIZE_HPP
#define XVECTORIZE_HPP

#include "xutils.hpp"

namespace qs
{

    /***************
     * xvectorizer *
     ***************/

    template <class F, class R>
    class xvectorizer
    {

    public:


        template <class... E>
        using get_xfunction_type = xfunction<F, R, get_xexpression_type<E>...>;

        template <class Func>
        explicit xvectorizer(Func&& f);

        template <class... E>
        get_xfunction_type<E...> operator()(const E&... e) const;

    private:

        typename std::remove_reference<F>::type m_f;
    };

    namespace detail
    {
        template <class F>
        using get_function_type = remove_class_t<decltype(&std::remove_reference_t<F>::operator())>;
    }

    template <class R, class... Args>
    xvectorizer<R (*) (Args...), R> vectorize(R (*f) (Args...));
    
    template <class F, class R, class... Args>
    xvectorizer<F, R> vectorize(F&& f, R (*) (Args...));

    template <class F>
    auto vectorize(F&& f) -> decltype(vectorize(std::forward<F>(f), (detail::get_function_type<F>*)nullptr));

    /******************************
     * xvectorizer implementation *
     ******************************/

    template <class F, class R>
    template <class Func>
    inline xvectorizer<F, R>::xvectorizer(Func&& f)
        : m_f(std::forward<Func>(f))
    {
    }

    template <class F, class R>
    template <class... E>
    inline auto xvectorizer<F, R>::operator()(const E&... e) const -> get_xfunction_type<E...>
    {
        return get_xfunction_type<E...>(m_f, e...);
    }

    template <class R, class... Args>
    inline xvectorizer<R (*)(Args...), R> vectorize(R (*f) (Args...))
    {
        return xvectorizer<R (*) (Args...), R>(f);
    }

    template <class F, class R, class... Args>
    inline xvectorizer<F, R> vectorize(F&& f, R (*) (Args...))
    {
        return xvectorizer<F, R>(std::forward<F>(f));
    }

    template <class F>
    inline auto vectorize(F&& f) -> decltype(vectorize(std::forward<F>(f), (detail::get_function_type<F>*)nullptr))
    {
        return vectorize(std::forward<F>(f),(detail::get_function_type<F>*)nullptr);
    }
}

#endif

