/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_IO_HPP
#define XTENSOR_IO_HPP

#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#include "xexpression.hpp"
#include "xmath.hpp"
#include "xstrided_view.hpp"

namespace xt
{

    template <class E>
    inline std::ostream& operator<<(std::ostream& out, const xexpression<E>& e);

    namespace print_options
    {
        struct print_options_impl
        {
            std::size_t edgeitems = 3;
            std::size_t line_width = 75;
            std::size_t threshold = 1000;
            std::streamsize precision = -1;  // default precision
        };

        inline print_options_impl& print_options()
        {
            static print_options_impl po;
            return po;
        }

        /**
         * @brief Sets the line width. After \a line_width chars,
         *        a new line is added.
         *
         * @param line_width The line width
         */
        inline void set_line_width(std::size_t line_width)
        {
            print_options().line_width = line_width;
        }

        /**
         * @brief Sets the threshold after which summarization is triggered (default: 1000).
         *
         * @param threshold The number of elements in the xexpression that triggers
         *                  summarization in the output
         */
        inline void set_threshold(std::size_t threshold)
        {
            print_options().threshold = threshold;
        }

        /**
         * @brief Sets the number of edge items. If the summarization is
         *        triggered, this value defines how many items of each dimension
         *        are printed.
         *
         * @param edgeitems The number of edge items
         */
        inline void set_edgeitems(std::size_t edgeitems)
        {
            print_options().edgeitems = edgeitems;
        }

        /**
         * @brief Sets the precision for printing floating point values.
         *
         * @param precision The number of digits for floating point output
         */
        inline void set_precision(std::streamsize precision)
        {
            print_options().precision = precision;
        }
    }

    /**************************************
     * xexpression ostream implementation *
     **************************************/

    namespace detail
    {
        template <class E, class F>
        std::ostream& xoutput(std::ostream& out, const E& e,
                              xstrided_slice_vector& slices, F& printer, std::size_t blanks,
                              std::streamsize element_width, std::size_t edgeitems, std::size_t line_width)
        {
            using size_type = typename E::size_type;

            const auto view = xt::strided_view(e, slices);
            if (view.dimension() == 0)
            {
                printer.print_next(out);
            }
            else
            {
                std::string indents(blanks, ' ');

                size_type i = 0;
                size_type elems_on_line = 0;
                size_type ewp2 = static_cast<size_type>(element_width) + size_type(2);
                size_type line_lim = static_cast<size_type>(std::floor(line_width / ewp2));

                out << '{';
                for (; i != size_type(view.shape()[0] - 1); ++i)
                {
                    if (edgeitems && size_type(view.shape()[0]) > (edgeitems * 2) && i == edgeitems)
                    {
                        out << "..., ";
                        if (view.dimension() > 1)
                        {
                            elems_on_line = 0;
                            out << std::endl
                                << indents;
                        }
                        i = size_type(view.shape()[0]) - edgeitems;
                    }
                    if (view.dimension() == 1 && line_lim != 0 && elems_on_line >= line_lim)
                    {
                        out << std::endl
                            << indents;
                        elems_on_line = 0;
                    }
                    slices.push_back(static_cast<int>(i));
                    xoutput(out, e, slices, printer, blanks + 1, element_width, edgeitems, line_width) << ',';
                    slices.pop_back();
                    elems_on_line++;

                    if (view.dimension() == 1)
                    {
                        out << ' ';
                    }
                    else
                    {
                        out << std::endl
                            << indents;
                    }
                }
                if (view.dimension() == 1 && line_lim != 0 && elems_on_line >= line_lim)
                {
                    out << std::endl
                        << indents;
                }
                slices.push_back(static_cast<int>(i));
                xoutput(out, e, slices, printer, blanks + 1, element_width, edgeitems, line_width) << '}';
                slices.pop_back();
            }
            return out;
        }

        template <class F, class E>
        static void recurser_run(F& fn, const E& e, xstrided_slice_vector& slices, std::size_t lim = 0)
        {
            using size_type = typename E::size_type;
            const auto view = strided_view(e, slices);
            if (view.dimension() == 0)
            {
                fn.update(view());
            }
            else
            {
                size_type i = 0;
                for (; i != static_cast<size_type>(view.shape()[0] - 1); ++i)
                {
                    if (lim && size_type(view.shape()[0]) > (lim * 2) && i == lim)
                    {
                        i = static_cast<size_type>(view.shape()[0]) - lim;
                    }
                    slices.push_back(static_cast<int>(i));
                    recurser_run(fn, e, slices, lim);
                    slices.pop_back();
                }
                slices.push_back(static_cast<int>(i));
                recurser_run(fn, e, slices, lim);
                slices.pop_back();
            }
        }

        template <class T, class E = void>
        struct printer;

        template <class T>
        struct printer<T, std::enable_if_t<std::is_floating_point<typename T::value_type>::value>>
        {
            using value_type = std::decay_t<typename T::value_type>;
            using cache_type = std::vector<value_type>;
            using cache_iterator = typename cache_type::const_iterator;

            explicit printer(std::streamsize precision)
                : m_precision(precision)
            {
            }

            void init()
            {
                m_precision = m_required_precision < m_precision ? m_required_precision : m_precision;
                m_it = m_cache.cbegin();
                if (m_scientific)
                {
                    // 3 = sign, number and dot and 4 = "e+00"
                    m_width = m_precision + 7;
                    if (m_large_exponent)
                    {
                        // = e+000 (additional number)
                        m_width += 1;
                    }
                }
                else
                {
                    std::streamsize decimals = 1;  // print a leading 0
                    if (std::floor(m_max) != 0)
                    {
                        decimals += std::streamsize(std::log10(std::floor(m_max)));
                    }
                    // 2 => sign and dot
                    m_width = 2 + decimals + m_precision;
                }
                if (!m_required_precision)
                {
                    --m_width;
                }
            }

            std::ostream& print_next(std::ostream& out)
            {
                if (!m_scientific)
                {
                    std::stringstream buf;
                    buf.width(m_width);
                    buf << std::fixed;
                    buf.precision(m_precision);
                    buf << (*m_it);
                    if (!m_required_precision)
                    {
                        buf << '.';
                    }
                    std::string res = buf.str();
                    auto sit = res.rbegin();
                    while (*sit == '0')
                    {
                        *sit = ' ';
                        ++sit;
                    }
                    out << res;
                }
                else
                {
                    if (!m_large_exponent)
                    {
                        out << std::scientific;
                        out.width(m_width);
                        out << (*m_it);
                    }
                    else
                    {
                        std::stringstream buf;
                        buf.width(m_width);
                        buf << std::scientific;
                        buf.precision(m_precision);
                        buf << (*m_it);
                        std::string res = buf.str();

                        if (res[res.size() - 4] == 'e')
                        {
                            res.erase(0, 1);
                            res.insert(res.size() - 2, "0");
                        }
                        out << res;
                    }
                }
                ++m_it;
                return out;
            }

            void update(const value_type& val)
            {
                if (val != 0 && !std::isinf(val) && !std::isnan(val))
                {
                    if (!m_scientific || !m_large_exponent)
                    {
                        int exponent = 1 + int(std::log10(math::abs(val)));
                        if (exponent <= -5 || exponent > 7)
                        {
                            m_scientific = true;
                            m_required_precision = m_precision;
                            if (exponent <= -100 || exponent >= 100)
                            {
                                m_large_exponent = true;
                            }
                        }
                    }
                    if (math::abs(val) > m_max)
                    {
                        m_max = math::abs(val);
                    }
                    if (m_required_precision < m_precision)
                    {
                        while (std::floor(val * std::pow(10, m_required_precision)) != val * std::pow(10, m_required_precision))
                        {
                            m_required_precision++;
                        }
                    }
                }
                m_cache.push_back(val);
            }

            std::streamsize width()
            {
                return m_width;
            }

        private:

            bool m_large_exponent = false;
            bool m_scientific = false;
            std::streamsize m_width = 9;
            std::streamsize m_precision;
            std::streamsize m_required_precision = 0;
            value_type m_max = 0;

            cache_type m_cache;
            cache_iterator m_it;
        };

        template <class T>
        struct printer<T, std::enable_if_t<std::is_integral<typename T::value_type>::value && !std::is_same<typename T::value_type, bool>::value>>
        {
            using value_type = std::decay_t<typename T::value_type>;
            using cache_type = std::vector<value_type>;
            using cache_iterator = typename cache_type::const_iterator;

            explicit printer(std::streamsize)
            {
            }

            void init()
            {
                m_it = m_cache.cbegin();
                m_width = 1 + std::streamsize(std::log10(m_max)) + m_sign;
            }

            std::ostream& print_next(std::ostream& out)
            {
                // + enables printing of chars etc. as numbers
                // TODO should chars be printed as numbers?
                out.width(m_width);
                out << +(*m_it);
                ++m_it;
                return out;
            }

            void update(const value_type& val)
            {
                if (math::abs(val) > m_max)
                {
                    m_max = math::abs(val);
                }
                if (std::is_signed<value_type>::value && val < 0)
                {
                    m_sign = true;
                }
                m_cache.push_back(val);
            }

            std::streamsize width()
            {
                return m_width;
            }

        private:

            std::streamsize m_width;
            bool m_sign = false;
            value_type m_max = 0;

            cache_type m_cache;
            cache_iterator m_it;
        };

        template <class T>
        struct printer<T, std::enable_if_t<std::is_same<typename T::value_type, bool>::value>>
        {
            using value_type = bool;
            using cache_type = std::vector<bool>;
            using cache_iterator = typename cache_type::const_iterator;

            explicit printer(std::streamsize)
            {
            }

            void init()
            {
                m_it = m_cache.cbegin();
            }

            std::ostream& print_next(std::ostream& out)
            {
                if (*m_it)
                {
                    out << " true";
                }
                else
                {
                    out << "false";
                }
                // TODO: the following std::setw(5) isn't working correctly on OSX.
                //out << std::boolalpha << std::setw(m_width) << (*m_it);
                ++m_it;
                return out;
            }

            void update(const value_type& val)
            {
                m_cache.push_back(val);
            }

            std::streamsize width()
            {
                return m_width;
            }

        private:

            std::streamsize m_width = 5;

            cache_type m_cache;
            cache_iterator m_it;
        };

        template <class T>
        struct printer<T, std::enable_if_t<xtl::is_complex<typename T::value_type>::value>>
        {
            using value_type = std::decay_t<typename T::value_type>;
            using cache_type = std::vector<bool>;
            using cache_iterator = typename cache_type::const_iterator;

            explicit printer(std::streamsize precision)
                : real_printer(precision), imag_printer(precision)
            {
            }

            void init()
            {
                real_printer.init();
                imag_printer.init();
                m_it = m_signs.cbegin();
            }

            std::ostream& print_next(std::ostream& out)
            {
                real_printer.print_next(out);
                if (*m_it)
                {
                    out << "-";
                }
                else
                {
                    out << "+";
                }
                std::stringstream buf;
                imag_printer.print_next(buf);
                std::string s = buf.str();
                if (s[0] == ' ')
                {
                    s.erase(0, 1);  // erase space for +/-
                }
                // insert j at end of number
                std::size_t idx = s.find_last_not_of(" ");
                s.insert(idx + 1, "i");
                out << s;
                ++m_it;
                return out;
            }

            void update(const value_type& val)
            {
                real_printer.update(val.real());
                imag_printer.update(std::abs(val.imag()));
                m_signs.push_back(std::signbit(val.imag()));
            }

            std::streamsize width()
            {
                return real_printer.width() + imag_printer.width() + 2;
            }

        private:

            printer<value_type> real_printer, imag_printer;
            cache_type m_signs;
            cache_iterator m_it;
        };

        template <class T>
        struct printer<T, std::enable_if_t<!std::is_fundamental<typename T::value_type>::value && !xtl::is_complex<typename T::value_type>::value>>
        {
            using value_type = std::decay_t<typename T::value_type>;
            using cache_type = std::vector<std::string>;
            using cache_iterator = typename cache_type::const_iterator;

            explicit printer(std::streamsize)
            {
            }

            void init()
            {
                m_it = m_cache.cbegin();
                if (m_width > 20)
                {
                    m_width = 0;
                }
            }

            std::ostream& print_next(std::ostream& out)
            {
                out.width(m_width);
                out << *m_it;
                ++m_it;
                return out;
            }

            void update(const value_type& val)
            {
                std::stringstream buf;
                buf << val;
                std::string s = buf.str();
                if (int(s.size()) > m_width)
                {
                    m_width = std::streamsize(s.size());
                }
                m_cache.push_back(s);
            }

            std::streamsize width()
            {
                return m_width;
            }

        private:

            std::streamsize m_width = 0;
            cache_type m_cache;
            cache_iterator m_it;
        };

        template <class E>
        struct custom_formatter
        {
            using value_type = std::decay_t<typename E::value_type>;

            template <class F>
            custom_formatter(F&& func)
                : m_func(func)
            {
            }

            std::string operator()(const value_type& val) const
            {
                return m_func(val);
            }

        private:

            std::function<std::string(const value_type&)> m_func;
        };
    }

    template <class E, class F>
    std::ostream& pretty_print(const xexpression<E>& e, F&& func, std::ostream& out = std::cout)
    {
        xfunction<detail::custom_formatter<E>, std::string, const_xclosure_t<E>> print_fun(detail::custom_formatter<E>(std::forward<F>(func)), e);
        return pretty_print(print_fun, out);
    }

    template <class E>
    std::ostream& pretty_print(const xexpression<E>& e, std::ostream& out = std::cout)
    {
        const E& d = e.derived_cast();

        std::size_t lim = 0;
        std::size_t sz = compute_size(d.shape());
        if (sz > print_options::print_options().threshold)
        {
            lim = print_options::print_options().edgeitems;
        }
        if (sz == 0)
        {
            out << "{}";
            return out;
        }

        auto temp_precision = out.precision();
        auto precision = temp_precision;
        if (print_options::print_options().precision != -1)
        {
            out.precision(print_options::print_options().precision);
            precision = print_options::print_options().precision;
        }

        detail::printer<E> p(precision);

        xstrided_slice_vector sv;
        detail::recurser_run(p, d, sv, lim);
        p.init();
        sv.clear();
        xoutput(out, d, sv, p, 1, p.width(), lim, print_options::print_options().line_width);

        out.precision(temp_precision);  // restore precision

        return out;
    }

    template <class E>
    inline std::ostream& operator<<(std::ostream& out, const xexpression<E>& e)
    {
        return pretty_print(e, out);
    }

#ifdef __CLING__

    template <class P>
    void compute_1d_row(std::stringstream& out, P& printer, const std::size_t& row_idx)
    {
        out << "<tr><td style='font-family:monospace;' title='" << row_idx << "'><pre>";
        printer.print_next(out);
        out << "</pre></td></tr>";
    }

    template <class P, class T>
    void compute_1d_table(std::stringstream& out, P& printer, const T& expr,
                          const std::size_t& edgeitems)
    {
        const auto& dim = expr.shape()[0];

        out << "<table style='border-style:solid;border-width:1px;'><tbody>";
        if (edgeitems == 0 || 2 * edgeitems >= dim)
        {
            for (std::size_t row_idx = 0; row_idx < dim; ++row_idx)
            {
                compute_1d_row(out, printer, row_idx);
            }
        }
        else
        {
            for (std::size_t row_idx = 0; row_idx < edgeitems; ++row_idx)
            {
                compute_1d_row(out, printer, row_idx);
            }
            out << "<tr><td><center>...</center></td></tr>";
            for (std::size_t row_idx = dim - edgeitems; row_idx < dim; ++row_idx)
            {
                compute_1d_row(out, printer, row_idx);
            }
        }
        out << "</tbody></table>";
    }

    template <class P>
    void compute_2d_element(std::stringstream& out, P& printer, const std::string& idx_str,
                            const std::size_t& row_idx, const std::size_t& column_idx)
    {
        out << "<td style='font-family:monospace;' title='("
            << idx_str << row_idx << ", " << column_idx << ")'><pre>";
        printer.print_next(out);
        out << "</pre></td>";
    }

    template <class P, class T>
    void compute_2d_row(std::stringstream& out, P& printer, const T& expr,
                        const std::size_t& edgeitems, const std::string& idx_str,
                        const std::size_t& row_idx)
    {
        const auto& dim = expr.shape()[expr.dimension() - 1];

        out << "<tr>";
        if (edgeitems == 0 || 2 * edgeitems >= dim)
        {
            for (std::size_t column_idx = 0; column_idx < dim; ++column_idx)
            {
                compute_2d_element(out, printer, idx_str, row_idx, column_idx);
            }
        }
        else
        {
            for (std::size_t column_idx = 0; column_idx < edgeitems; ++column_idx)
            {
                compute_2d_element(out, printer, idx_str, row_idx, column_idx);
            }
            out << "<td><center>...</center></td>";
            for (std::size_t column_idx = dim - edgeitems; column_idx < dim; ++column_idx)
            {
                compute_2d_element(out, printer, idx_str, row_idx, column_idx);
            }
        }
        out << "</tr>";
    }

    template <class P, class T, class I>
    void compute_2d_table(std::stringstream& out, P& printer, const T& expr,
                               const std::size_t& edgeitems, const std::vector<I>& idx)
    {
        const auto& dim = expr.shape()[expr.dimension() - 2];
        std::string idx_str;
        std::for_each(idx.cbegin(), idx.cend(), [&idx_str](const auto& i) {
            idx_str += std::to_string(i) + ", ";
        });

        out << "<table style='border-style:solid;border-width:1px;'><tbody>";
        if (edgeitems == 0 || 2 * edgeitems >= dim)
        {
            for (std::size_t row_idx = 0; row_idx < dim; ++row_idx)
            {
                compute_2d_row(out, printer, expr, edgeitems, idx_str, row_idx);
            }
        }
        else
        {
            for (std::size_t row_idx = 0; row_idx < edgeitems; ++row_idx)
            {
                compute_2d_row(out, printer, expr, edgeitems, idx_str, row_idx);
            }
            out << "<tr>";
            for (std::size_t column_idx = 0; column_idx < 2 * edgeitems + 1; ++column_idx)
            {
                out << "<td><center>...</center></td>";
            }
            out << "</tr>";
            for (std::size_t row_idx = dim - edgeitems; row_idx < dim; ++row_idx)
            {
                compute_2d_row(out, printer, expr, edgeitems, idx_str, row_idx);
            }
        }
        out << "</tbody></table>";
    }

    template <class P, class T, class I>
    void compute_nd_row(std::stringstream& out, P& printer, const T& expr,
                         const std::size_t& edgeitems, const std::vector<I>& idx)
    {
        out << "<tr><td>";
        compute_nd_table_impl(out, printer, expr, edgeitems, idx);
        out << "</td></tr>";
    }

    template <class P, class T, class I>
    void compute_nd_table_impl(std::stringstream& out, P& printer, const T& expr,
                               const std::size_t& edgeitems, const std::vector<I>& idx)
    {
        const auto& displayed_dimension = idx.size();
        const auto& expr_dim = expr.dimension();
        const auto& dim = expr.shape()[displayed_dimension];

        if (expr_dim - displayed_dimension == 2)
        {
            return compute_2d_table(out, printer, expr, edgeitems, idx);
        }

        std::vector<I> idx2 = idx;
        idx2.resize(displayed_dimension + 1);

        out << "<table style='border-style:solid;border-width:1px;'>";
        if (edgeitems == 0 || 2 * edgeitems >= dim)
        {
            for (std::size_t i = 0; i < dim; ++i)
            {
                idx2[displayed_dimension] = i;
                compute_nd_row(out, printer, expr, edgeitems, idx2);
            }
        }
        else
        {
            for (std::size_t i = 0; i < edgeitems; ++i)
            {
                idx2[displayed_dimension] = i;
                compute_nd_row(out, printer, expr, edgeitems, idx2);
            }
            out << "<tr><td><center>...</center></td></tr>";
            for (std::size_t i = dim - edgeitems; i < dim; ++i)
            {
                idx2[displayed_dimension] = i;
                compute_nd_row(out, printer, expr, edgeitems, idx2);
            }
        }
        out << "</table>";
    }

    template <class P, class T>
    void compute_nd_table(std::stringstream& out, P& printer, const T& expr,
                          const std::size_t& edgeitems)
    {
        if (expr.dimension() == 1)
        {
            compute_1d_table(out, printer, expr, edgeitems);
        }
        else
        {
            std::vector<std::size_t> empty_vector;
            compute_nd_table_impl(out, printer, expr, edgeitems, empty_vector);
        }
    }

    template <class E>
    xeus::xjson mime_bundle_repr_impl(const E& expr)
    {
        std::stringstream out;

        std::size_t edgeitems = 0;
        std::size_t size = compute_size(expr.shape());
        if (size > print_options::print_options().threshold)
        {
            edgeitems = print_options::print_options().edgeitems;
        }

        if (print_options::print_options().precision != -1)
        {
            out.precision(print_options::print_options().precision);
        }

        detail::printer<E> printer(out.precision());

        xstrided_slice_vector slice_vector;
        detail::recurser_run(printer, expr, slice_vector, edgeitems);
        printer.init();

        compute_nd_table(out, printer, expr, edgeitems);

        auto bundle = xeus::xjson::object();
        bundle["text/html"] = out.str();
        return bundle;
    }

    template <class F, class CT>
    class xfunctor_view;

    template <class F, class CT>
    xeus::xjson mime_bundle_repr(const xfunctor_view<F, CT>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class F, class R, class... CT>
    class xfunction;

    template <class F, class R, class... CT>
    xeus::xjson mime_bundle_repr(const xfunction<F, R, CT...>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class EC, layout_type L, class SC, class Tag>
    class xarray_container;

    template <class EC, layout_type L, class SC, class Tag>
    xeus::xjson mime_bundle_repr(const xarray_container<EC, L, SC, Tag>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class EC, std::size_t N, layout_type L, class Tag>
    class xtensor_container;

    template <class EC, std::size_t N, layout_type L, class Tag>
    xeus::xjson mime_bundle_repr(const xtensor_container<EC, N, L, Tag>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class ET, class S, layout_type L, class Tag>
    class xfixed_container;

    template <class ET, class S, layout_type L, class Tag>
    xeus::xjson mime_bundle_repr(const xfixed_container<ET, S, L, Tag>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class F, class CT, class X>
    class xreducer;

    template <class F, class CT, class X>
    xeus::xjson mime_bundle_repr(const xreducer<F, CT, X>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class VE, class FE>
    class xoptional_assembly;

    template <class VE, class FE>
    xeus::xjson mime_bundle_repr(const xoptional_assembly<VE, FE>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class VEC, class FEC>
    class xoptional_assembly_adaptor;

    template <class VEC, class FEC>
    xeus::xjson mime_bundle_repr(const xoptional_assembly_adaptor<VEC, FEC>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class CT>
    class xscalar;

    template <class CT>
    xeus::xjson mime_bundle_repr(const xscalar<CT>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class CT, class X>
    class xbroadcast;

    template <class CT, class X>
    xeus::xjson mime_bundle_repr(const xbroadcast<CT, X>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class F, class R, class S>
    class xgenerator;

    template <class F, class R, class S>
    xeus::xjson mime_bundle_repr(const xgenerator<F, R, S>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class CT, class... S>
    class xview;

    template <class CT, class... S>
    xeus::xjson mime_bundle_repr(const xview<CT, S...>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }

    template <class CT, class S, layout_type L, class FST>
    class xstrided_view;

    template <class CT, class S, layout_type L, class FST>
    xeus::xjson mime_bundle_repr(const xstrided_view<CT, S, L, FST>& expr)
    {
        return mime_bundle_repr_impl(expr);
    }
#endif
}

#endif
