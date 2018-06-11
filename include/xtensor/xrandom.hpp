/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

/**
 * @brief functions to obtain xgenerators generating random numbers with given shape
 */

#ifndef XTENSOR_RANDOM_HPP
#define XTENSOR_RANDOM_HPP

#include <functional>
#include <random>
#include <utility>

#include "xbuilder.hpp"
#include "xgenerator.hpp"
#include "xtensor.hpp"
#include "xview.hpp"

namespace xt
{
    /*********************
     * Random generators *
     *********************/

    namespace random
    {
        using default_engine_type = std::mt19937;
        using seed_type = default_engine_type::result_type;

        default_engine_type& get_default_random_engine();
        void seed(seed_type seed);

        template <class T, class S, class E = random::default_engine_type>
        auto rand(const S& shape, T lower = 0, T upper = 1,
                  E& engine = random::get_default_random_engine());

        template <class T, class S, class E = random::default_engine_type>
        auto randint(const S& shape, T lower = 0, T upper = (std::numeric_limits<T>::max)(),
                     E& engine = random::get_default_random_engine());

        template <class T, class S, class E = random::default_engine_type>
        auto randn(const S& shape, T mean = 0, T std_dev = 1,
                   E& engine = random::get_default_random_engine());

#ifdef X_OLD_CLANG
        template <class T, class I, class E = random::default_engine_type>
        auto rand(std::initializer_list<I> shape, T lower = 0, T upper = 1,
                  E& engine = random::get_default_random_engine());

        template <class T, class I, class E = random::default_engine_type>
        auto randint(std::initializer_list<I> shape,
                     T lower = 0, T upper = (std::numeric_limits<T>::max)(),
                     E& engine = random::get_default_random_engine());

        template <class T, class I, class E = random::default_engine_type>
        auto randn(std::initializer_list<I>, T mean = 0, T std_dev = 1,
                   E& engine = random::get_default_random_engine());
#else
        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto rand(const I (&shape)[L], T lower = 0, T upper = 1,
                  E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto randint(const I (&shape)[L], T lower = 0, T upper = (std::numeric_limits<T>::max)(),
                     E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto randn(const I (&shape)[L], T mean = 0, T std_dev = 1,
                   E& engine = random::get_default_random_engine());
#endif

        template <class T, class E = random::default_engine_type>
        void shuffle(xexpression<T>& e, E& engine = random::get_default_random_engine());

        template <class T, class E = random::default_engine_type>
        xtensor<typename T::value_type, 1> choice(const xexpression<T>& e, std::size_t n,
                                                  E& engine = random::get_default_random_engine());
    }

    namespace detail
    {
        template <class T, class E, class D>
        struct random_impl
        {
            using value_type = T;

            random_impl(E& engine, D&& dist)
                : m_engine(engine), m_dist(std::move(dist))
            {
            }

            template <class... Args>
            inline value_type operator()(Args...) const
            {
                return m_dist(m_engine);
            }

            template <class It>
            inline value_type element(It, It) const
            {
                return m_dist(m_engine);
            }

            template <class EX>
            inline void assign_to(xexpression<EX>& e) const noexcept
            {
                auto& ed = e.derived_cast();
                for (auto& el : ed.storage())
                {
                    el = m_dist(m_engine);
                }
            }

        private:

            E& m_engine;
            mutable D m_dist;
        };
    }

    namespace random
    {
        /**
         * Returns a reference to the default random number engine
         */
        inline default_engine_type& get_default_random_engine()
        {
            static default_engine_type mt;
            return mt;
        }

        /**
         * Seeds the default random number generator with @p seed
         * @param seed The seed
         */
        inline void seed(seed_type seed)
        {
            get_default_random_engine().seed(seed);
        }

        /**
         * xexpression with specified @p shape containing uniformly distributed random numbers
         * in the interval from @p lower to @p upper, excluding upper.
         *
         * Numbers are drawn from @c std::uniform_real_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param lower lower bound
         * @param upper upper bound
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto rand(const S& shape, T lower, T upper, E& engine)
        {
            std::uniform_real_distribution<T> dist(lower, upper);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        /**
         * xexpression with specified @p shape containing uniformly distributed
         * random integers in the interval from @p lower to @p upper, excluding upper.
         *
         * Numbers are drawn from @c std::uniform_int_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param lower lower bound
         * @param upper upper bound
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto randint(const S& shape, T lower, T upper, E& engine)
        {
            std::uniform_int_distribution<T> dist(lower, upper - 1);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * the Normal (Gaussian) random number distribution with mean @p mean and
         * standard deviation @p std_dev.
         *
         * Numbers are drawn from @c std::normal_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param mean mean of normal distribution
         * @param std_dev standard deviation of normal distribution
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto randn(const S& shape, T mean, T std_dev, E& engine)
        {
            std::normal_distribution<T> dist(mean, std_dev);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

#ifdef X_OLD_CLANG
        template <class T, class I, class E>
        inline auto rand(std::initializer_list<I> shape, T lower, T upper, E& engine)
        {
            std::uniform_real_distribution<T> dist(lower, upper);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        template <class T, class I, class E>
        inline auto randint(std::initializer_list<I> shape, T lower, T upper, E& engine)
        {
            std::uniform_int_distribution<T> dist(lower, upper - 1);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        template <class T, class I, class E>
        inline auto randn(std::initializer_list<I> shape, T mean, T std_dev, E& engine)
        {
            std::normal_distribution<T> dist(mean, std_dev);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }
#else
        template <class T, class I, std::size_t L, class E>
        inline auto rand(const I (&shape)[L], T lower, T upper, E& engine)
        {
            std::uniform_real_distribution<T> dist(lower, upper);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        template <class T, class I, std::size_t L, class E>
        inline auto randint(const I (&shape)[L], T lower, T upper, E& engine)
        {
            std::uniform_int_distribution<T> dist(lower, upper - 1);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        template <class T, class I, std::size_t L, class E>
        inline auto randn(const I (&shape)[L], T mean, T std_dev, E& engine)
        {
            std::normal_distribution<T> dist(mean, std_dev);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }
#endif

        /**
         * Randomly shuffle elements inplace in xcontainer along first axis.
         * The order of sub-arrays is changed but their contents remain the same.
         *
         * @param e xcontainer to shuffle inplace
         * @param engine random number engine
         */
        template <class T, class E>
        void shuffle(xexpression<T>& e, E& engine)
        {
            T& de = e.derived_cast();

            if (de.dimension() == 1)
            {
                std::shuffle(de.storage().begin(), de.storage().end(), engine);
            }
            else
            {
                using size_type = typename T::size_type;
                decltype(auto) buf = empty_like(view(de, 0));

                for (std::size_t i = de.shape()[0] - 1; i > 0; --i)
                {
                    std::uniform_int_distribution<size_type> dist(0, i);
                    size_type j = dist(engine);

                    buf = view(de, j);
                    view(de, j) = view(de, i);
                    view(de, i) = buf;
                }
            }
        }

        /**
         * Randomly select n unique elements from xexpression e.
         * Note: this function makes a copy of your data, and only 1D data is accepted.
         *
         * @param e expression to sample from
         * @param n number of elements to sample
         * @param engine random number engine
         *
         * @return xtensor containing 1D container of sampled elements
         */
        template <class T, class E>
        xtensor<typename T::value_type, 1> choice(const xexpression<T>& e, std::size_t n, E& engine)
        {
            const auto& de = e.derived_cast();
            XTENSOR_ASSERT(de.dimension() == 1);
            XTENSOR_ASSERT(de.size() >= n);
            xtensor<typename T::value_type, 1> result;
            result.resize({n});

            xtensor<typename T::value_type, 1> shuffled = de;
            std::shuffle(shuffled.storage().begin(), shuffled.storage().end(), engine);

            std::copy(shuffled.storage().begin(), shuffled.storage().begin() + n, result.begin());

            return result;

            // Doesn't exist yet but would be much nicer as it probably prevent copies
            // std::experimental::sample(de.begin(), de.end(), result.begin(), std::ref(engine));
        }
    }
}

#endif
