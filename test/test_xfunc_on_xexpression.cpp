/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"

#include <cstddef>

#include "xtensor/xexpression.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

    struct field_expression_tag
    {
    };

    template<class D>
    class field_expression : public xt::xexpression<D> {
    public:
        using expression_tag = field_expression_tag;
    };

    template<class F, class... CT>
    class field_function : public field_expression<field_function<F, CT...>> {
    public:
        using self_type = field_function<F, CT...>;
        using functor_type = std::remove_reference_t<F>;

        using expression_tag = field_expression_tag;

        template<class Func, class... CTA,
                class U = std::enable_if<!std::is_base_of<Func, self_type>::value>>
        field_function(Func &&f, CTA &&... e) noexcept
            : m_e(std::forward<CTA>(e)...), m_f(std::forward<Func>(f))
        {}

        template<class... T>
        auto operator()(const std::size_t begin, const std::size_t end) const
        {
            return evaluate(std::make_index_sequence<sizeof...(CT)>(), begin, end);
        }

        template<std::size_t... I, class... T>
        auto evaluate(std::index_sequence<I...>, T &&... t) const
        {
            return m_f(
                std::get<I>(m_e).template operator()(std::forward<T>(t)...)...);
        }

    private:
        std::tuple<CT...> m_e;
        functor_type m_f;
    };

    namespace xt
    {
        namespace detail
        {
            template<class F, class... E>
            struct select_xfunction_expression<field_expression_tag, F, E...>
            {
                using type = field_function<F, E...>;
            };
        }
    }

    // using xt::operator+;
    // using xt::operator-;
    // using xt::operator*;
    // using xt::operator/;
    // using xt::operator%;

    struct Field : public field_expression<Field>
    {
        Field() : m_data(std::array<std::size_t, 1>{10})
        {}

        auto operator()(const std::size_t begin, const std::size_t end) const
        {
            return xt::view(m_data, xt::range(begin, end));
        }

        auto operator()(const std::size_t begin, const std::size_t end)
        {
            return xt::view(m_data, xt::range(begin, end));
        }

        template<class E>
        Field &operator=(const field_expression<E> &e)
        {
            (*this)(0, 5) = e.derived_cast()(0, 5);
            return *this;
        }

        xt::xtensor<double, 1> m_data;
    };

    TEST(xfunc_on_xexpression, field_expression)
    {
        Field x, y;
        x.m_data.fill(10);

        y = x + x;

        std::cout << y.m_data << "\n";
    }
