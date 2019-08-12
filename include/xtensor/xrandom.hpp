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

        template <class T, class S, class D = double, class E = random::default_engine_type>
        auto binomial(const S& shape, T trials = 1, D prob = 0.5,
                      E& engine = random::get_default_random_engine());

        template <class T, class S, class D = double, class E = random::default_engine_type>
        auto geometric(const S& shape, D prob = 0.5, 
                       E& engine = random::get_default_random_engine());

        template <class T, class S, class D = double, class E = random::default_engine_type>
        auto negative_binomial(const S& shape, T k = 1, D prob = 0.5,
                               E& engine = random::get_default_random_engine()); 

        template <class T, class S, class D = double, class E = random::default_engine_type>
        auto poisson(const S& shape, D rate = 1.0,
                     E& engine = random::get_default_random_engine());

        template <class T, class S, class E = random::default_engine_type>
        auto exponential(const S& shape, T rate = 1.0,
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
        auto randn(std::initializer_list<I> shape, 
                   T mean = 0, T std_dev = 1,
                   E& engine = random::get_default_random_engine());

        template <class T, class I, class D = double, class E = random::default_engine_type>
        auto binomial(std::initializer_list<I> shape, 
                      T trials = 1, D prob = 0.5,
                      E& engine = random::get_default_random_engine());

        template <class T, class I, class E = random::default_engine_type>
        auto geometric(std::initializer_list<I> shape, D prob = 0.5, 
                      E& engine = random::get_default_random_engine());

        template <class T, class I, class D = double, class E = random::default_engine_type>
        auto negative_binomial(std::initializer_list<I> shape, T k = 1, D prob = 0.5,
                               E& engine = random::get_default_random_engine()); 

        template <class T, class I, class D = double, class E = random::default_engine_type>
        auto poisson(std::initializer_list<I> shape, D rate = 1.0, 
                     E& engine = random::get_default_random_engine());

        template <class T, class I, class E = random::default_engine_type>
        auto exponential(std::initializer_list<I> shape, T rate = 1.0, 
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

        template <class T, class I, std::size_t L, class D = double, class E = random::default_engine_type>
        auto binomial(const I (&shape)[L], T trials = 1, D prob = 0.5,
                      E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class D = double, class E = random::default_engine_type>
        auto geometric(const I (&shape)[L], D prob=0.5, 
                       E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class D = double, class E = random::default_engine_type>
        auto negative_binomial(const I (&shape)[L], T k = 1, D prob = 0.5,
                               E& engine = random::get_default_random_engine()); 
        
        template <class T, class I, std::size_t L, class D = double, class E = random::default_engine_type>
        auto poisson(const I (&shape)[L], D rate = 1.0, 
                     E& engine = random::get_default_random_engine());

        template <class T, class I, std::size_t L, class E = random::default_engine_type>
        auto exponential(const I (&shape)[L], T rate = 1.0, 
                         E& engine = random::get_default_random_engine());
#endif

        template <class T, class E = random::default_engine_type>
        void shuffle(xexpression<T>& e, E& engine = random::get_default_random_engine());

        template <class T, class E = random::default_engine_type>
        std::enable_if_t<std::is_integral<T>::value, xtensor<T, 1>>
        permutation(T e, E& engine = random::get_default_random_engine());

        template <class T, class E = random::default_engine_type>
        std::enable_if_t<is_xexpression<std::decay_t<T>>::value, std::decay_t<T>>
        permutation(T&& e, E& engine = random::get_default_random_engine());

        template <class T, class E = random::default_engine_type>
        xtensor<typename T::value_type, 1> choice(const xexpression<T>& e, std::size_t n, bool replace = true,
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
                // Note: we're not going row/col major here
                auto& ed = e.derived_cast();
                for (auto&& el : ed.storage())
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
            std::uniform_int_distribution<T> dist(lower, T(upper - 1));
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

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * the binomial random number distribution for @p trials trials with
         * probability of success equal to @p prob.
         *
         * Numbers are drawn from @c std::binomial_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param trials number of Bernoulli trials
         * @param prob probability of success of each trial
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class D, class E>
        inline auto binomial(const S& shape, T trials, D prob, E& engine)
        {
            std::binomial_distribution<T> dist(trials, prob);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

         /**
         * xexpression with specified @p shape containing numbers sampled from
         * a gemoetric random number distribution with
         * probability of success equal to @p prob for each of the Bernoulli trials.
         *
         * Numbers are drawn from @c std::geometric_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param prob probability of success of each trial
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class D, class E>
        inline auto geometric(const S& shape, D prob, E& engine)
        {
            std::geometric_distribution<T> dist(prob);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * a negative binomial random number distribution (also known as Pascal distribution)
         * that returns the number of successes before @p k trials with probability of success 
         * equal to @p prob for each of the Bernoulli trials.
         * 
         * Numbers are drawn from @c std::negative_binomial_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param k number of unsuccessful trials
         * @param prob probability of success of each trial
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class D, class E>
        inline auto negative_binomial(const S& shape, T k, D prob, E& engine)
        {
            std::negative_binomial_distribution<T> dist(k, prob);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);   
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * a Poisson random number distribution with rate @p rate
         * 
         * Numbers are drawn from @c std::poisson_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param rate rate of Poisson distribution
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class D, class E>
        inline auto poisson(const S& shape, D rate, E& engine)
        {
            std::poisson_distribution<T> dist(rate);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);   
        }

        /**
         * xexpression with specified @p shape containing numbers sampled from
         * a exponential random number distribution with rate @p rate
         * 
         * Numbers are drawn from @c std::exponential_distribution.
         *
         * @param shape shape of resulting xexpression
         * @param rate rate of exponential distribution
         * @param engine random number engine
         * @tparam T number type to use
         */
        template <class T, class S, class E>
        inline auto exponential(const S& shape, T rate, E& engine)
        {
            std::exponential_distribution<T> dist(rate);
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
            std::uniform_int_distribution<T> dist(lower, T(upper - 1));
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        template <class T, class I, class E>
        inline auto randn(std::initializer_list<I> shape, T mean, T std_dev, E& engine)
        {
            std::normal_distribution<T> dist(mean, std_dev);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        template <class T, class I, class D, class E>
        inline auto binomial(std::initializer_list<I> shape, T trials, D prob, E& engine)
        {
            std::binomial_distribution<T> dist(trials, prob);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        template <class T, class I, class D, class E>
        inline auto geometric(std::initializer_list<I> shape, D prob, E& engine)
        {
            std::geometric_distribution<T> dist(prob);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }
        
        template <class T, class I, class D, class E>
        inline auto negative_binomial(std::initializer_list<I> shape, T k, D prob, E& engine)
        {
            std::negative_binomial_distribution<T> dist(k, prob);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);   
        }

        template <class T, class I, class D, class E>
        inline auto poisson(std::initializer_list<I> shape, D rate, E& engine)
        {
            std::poisson_distribution<T> dist(rate);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);   
        }

        template <class T, class I, class E>
        inline auto exponential(std::initializer_list<I> shape, T rate, E& engine)
        {
            std::exponential_distribution<T> dist(rate);
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
            std::uniform_int_distribution<T> dist(lower, T(upper - 1));
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        template <class T, class I, std::size_t L, class E>
        inline auto randn(const I (&shape)[L], T mean, T std_dev, E& engine)
        {
            std::normal_distribution<T> dist(mean, std_dev);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        template <class T, class I, std::size_t L, class D, class E>
        inline auto binomial(const I (&shape)[L], T trials, D prob, E& engine)
        {
            std::binomial_distribution<T> dist(trials, prob);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        template <class T, class I, std::size_t L, class D, class E>
        inline auto geometric(const I (&shape)[L], D prob, E& engine)
        {
            std::geometric_distribution<T> dist(prob);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        template <class T, class I, std::size_t L, class D , class E>
        inline auto negative_binomial(const I (&shape)[L], T k, D prob, E& engine)
        {
            std::negative_binomial_distribution<T> dist(k, prob);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);
        }

        template <class T, class I, std::size_t L, class D, class E>
        inline auto poisson(const I (&shape)[L], D rate, E& engine)
        {
            std::poisson_distribution<T> dist(rate);
            return detail::make_xgenerator(detail::random_impl<T, E, decltype(dist)>(engine, std::move(dist)), shape);   
        }

        template <class T, class I, std::size_t L, class E>
        inline auto exponential(const I (&shape)[L], T rate, E& engine)
        {
            std::exponential_distribution<T> dist(rate);
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
                using size_type = typename T::size_type;
                auto first = de.begin();
                auto last = de.end();

                for (size_type i = std::size_t((last - first) - 1); i > 0; --i)
                {
                    std::uniform_int_distribution<size_type> dist(0, i);
                    auto j = dist(engine);
                    using std::swap;
                    swap(first[i], first[j]);
                }
            }
            else
            {
                using size_type = typename T::size_type;
                decltype(auto) buf = empty_like(view(de, 0));

                for (size_type i = de.shape()[0] - 1; i > 0; --i)
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
         * Randomly permute a sequence, or return a permuted range.
         *
         * If the first parameter is an integer, this function creates a new
         * ``arange(e)`` and returns it randomly permuted. Otherwise, this
         * function creates a copy of the input, passes it to @sa shuffle and
         * returns the result.
         *
         * @param e input xexpression or integer
         * @param engine random number engine to use (optional)
         *
         * @return randomly permuted copy of container or arange.
         */
        template <class T, class E>
        std::enable_if_t<std::is_integral<T>::value, xtensor<T, 1>>
        permutation(T e, E& engine)
        {
            xt::xtensor<T, 1> res = xt::arange<T>(e);
            shuffle(res, engine);
            return res;
        }

        /// @cond DOXYGEN_INCLUDE_SFINAE
        template <class T, class E>
        std::enable_if_t<is_xexpression<std::decay_t<T>>::value, std::decay_t<T>>
        permutation(T&& e, E& engine)
        {
            using copy_type = std::decay_t<T>;
            copy_type res = e;
            shuffle(res, engine);
            return res;
        }
        /// @endcond

        /**
         * Randomly select n unique elements from xexpression e.
         * Note: this function makes a copy of your data, and only 1D data is accepted.
         *
         * @param e expression to sample from
         * @param n number of elements to sample
         * @param replace whether to sample with or without replacement
         * @param engine random number engine
         *
         * @return xtensor containing 1D container of sampled elements
         */
        template <class T, class E>
        xtensor<typename T::value_type, 1> choice(const xexpression<T>& e, std::size_t n, bool replace, E& engine)
        {
            const auto& de = e.derived_cast();
            if (de.dimension() != 1)
            {
                throw std::runtime_error("Sample expression must be 1 dimensional");
            }
            if (de.size() < n && !replace)
            {
                throw std::runtime_error("If replace is false, then the sample expression's size must be > n");
            }
            using result_type = xtensor<typename T::value_type, 1>;
            using size_type = typename result_type::size_type;
            result_type result;
            result.resize({n});

            if (replace)
            {
                auto dist = std::uniform_int_distribution<size_type>(0, de.size() - 1);
                for (size_type i = 0; i < n; ++i)
                {
                    result[i] = de.storage()[dist(engine)];
                }
            }
            else
            {
                // Naive resevoir sampling without weighting:
                std::copy(de.storage().begin(), de.storage().begin() + n, result.begin());
                size_type i = n;
                for(auto it = de.storage().begin() + n; it != de.storage().end(); ++it, ++i)
                {
                    auto idx = std::uniform_int_distribution<size_type>(0, i)(engine);
                    if (idx < n)
                    {
                        result.storage()[idx] = *it;
                    }
                }
            }
            return result;
        }
    }
}

#endif
