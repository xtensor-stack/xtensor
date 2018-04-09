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

#ifdef _WIN32
    using precision_type = typename std::streamsize;
#else
    using precision_type = int;
#endif

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
            precision_type precision = -1;  // default precision
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
        inline void set_precision(precision_type precision)
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
        std::ostream& xoutput(std::ostream& out, const E& e, slice_vector& slices, F& printer, std::size_t blanks,
                              precision_type element_width, std::size_t edgeitems, std::size_t line_width)
        {
            using size_type = typename E::size_type;

            const auto view = xt::dynamic_view(e, slices);
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
                for (; i != view.shape()[0] - 1; ++i)
                {
                    if (edgeitems && view.shape()[0] > (edgeitems * 2) && i == edgeitems)
                    {
                        out << "..., ";
                        if (view.dimension() > 1)
                        {
                            elems_on_line = 0;
                            out << std::endl
                                << indents;
                        }
                        i = view.shape()[0] - edgeitems;
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
        static void recurser_run(F& fn, const E& e, slice_vector& slices, std::size_t lim = 0)
        {
            using size_type = typename E::size_type;
            const auto view = dynamic_view(e, slices);
            if (view.dimension() == 0)
            {
                fn.update(view());
            }
            else
            {
                size_type i = 0;
                for (; i != view.shape()[0] - 1; ++i)
                {
                    if (lim && view.shape()[0] > (lim * 2) && i == lim)
                    {
                        i = view.shape()[0] - lim;
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

            printer(precision_type precision)
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
                    precision_type decimals = 1;  // print a leading 0
                    if (std::floor(m_max) != 0)
                    {
                        decimals += precision_type(std::log10(std::floor(m_max)));
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
                    buf << std::setw(m_width) << std::fixed << std::setprecision(m_precision) << (*m_it);
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
                        out << std::scientific << std::setw(m_width) << (*m_it);
                    }
                    else
                    {
                        std::stringstream buf;
                        buf << std::setw(m_width) << std::scientific << std::setprecision(m_precision) << (*m_it);
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

            precision_type width()
            {
                return m_width;
            }

        private:

            bool m_large_exponent = false;
            bool m_scientific = false;
            precision_type m_width = 9;
            precision_type m_precision;
            precision_type m_required_precision = 0;
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

            printer(precision_type)
            {
            }

            void init()
            {
                m_it = m_cache.cbegin();
                m_width = 1 + precision_type(std::log10(m_max)) + m_sign;
            }

            std::ostream& print_next(std::ostream& out)
            {
                // + enables printing of chars etc. as numbers
                // TODO should chars be printed as numbers?
                out << std::setw(m_width) << +(*m_it);
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

            precision_type width()
            {
                return m_width;
            }

        private:

            precision_type m_width;
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

            printer(precision_type)
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
                // the following std::setw(5) isn't working correctly on OSX.
                // out << std::boolalpha << std::setw(m_width) << (*m_it);
                ++m_it;
                return out;
            }

            void update(const value_type& val)
            {
                m_cache.push_back(val);
            }

            precision_type width()
            {
                return m_width;
            }

        private:

            precision_type m_width = 5;

            cache_type m_cache;
            cache_iterator m_it;
        };

        template <class T>
        struct printer<T, std::enable_if_t<xtl::is_complex<typename T::value_type>::value>>
        {
            using value_type = std::decay_t<typename T::value_type>;
            using cache_type = std::vector<bool>;
            using cache_iterator = typename cache_type::const_iterator;

            printer(precision_type precision)
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

            precision_type width()
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

            printer(precision_type)
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
                out << std::setw(m_width) << *m_it;
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
                    m_width = int(s.size());
                }
                m_cache.push_back(s);
            }

            precision_type width()
            {
                return m_width;
            }

        private:

            precision_type m_width = 0;
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

        size_t lim = 0;
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

        precision_type temp_precision = precision_type(out.precision());
        precision_type precision = temp_precision;
        if (print_options::print_options().precision != -1)
        {
            out << std::setprecision(print_options::print_options().precision);
            precision = print_options::print_options().precision;
        }

        detail::printer<E> p(precision);

        slice_vector sv;
        detail::recurser_run(p, d, sv, lim);
        p.init();
        sv.clear();
        xoutput(out, d, sv, p, 1, p.width(), lim, print_options::print_options().line_width);

        out << std::setprecision(temp_precision);  // restore precision

        return out;
    }

    template <class E>
    inline std::ostream& operator<<(std::ostream& out, const xexpression<E>& e)
    {
        return pretty_print(e, out);
    }
}

#endif
