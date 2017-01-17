/***************************************************************************
* Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

/**
 * @brief functions to obtain xgenerators generating random numbers with given shape
 */

#ifndef XRANDOM_HPP
#define XRANDOM_HPP

#include <utility>
#include <random>
#include <functional>

#include "xgenerator.hpp"

namespace xt
{
    namespace detail
    {
        template <class T>
        struct random_impl
        {
            using value_type = T;

            random_impl(std::function<value_type()>&& generator) :
                m_generator(std::move(generator))
            {
            }

            template <class... Args>
            inline value_type operator()(Args... /*args*/) const
            {
                return m_generator();
            }

            inline value_type operator[](const xindex& /*idx*/) const
            {
                return m_generator();
            }

            template <class It>
            inline value_type element(It /*first*/, It /*last*/) const
            {
                return m_generator();
            }

        private:
            std::function<value_type()> m_generator;
        };
    }
    
    namespace random
    {
        using default_engine_type = std::mt19937;
        using seed_type = default_engine_type::result_type;

        inline default_engine_type& get_default_random_engine()
        {
            static default_engine_type mt;
            return mt;
        }
        
        inline void set_seed(seed_type seed)
        {
            get_default_random_engine().seed(seed);
        }
    }

    /**
     * @function rand
     * @brief xexpression with specified @p shape containing uniformly distributed random numbers 
     *        in the interval from @p lower to @p upper, excluding upper.
     * 
     * Numbers are drawn from @c std::uniform_real_distribution.
     * 
     * @param shape shape of resulting xexpression
     * @param lower lower bound
     * @param upper upper bound
     * @param engine random number engine
     *
     * @tparam T number type to use
     */ 
    template <class T, class E = random::default_engine_type>
    inline auto rand(std::vector<std::size_t> shape, T lower = 0, T upper = 1,
                     E& engine = random::get_default_random_engine())
    {
        std::uniform_real_distribution<T> dist(lower, upper);
        return detail::make_xgenerator(detail::random_impl<T>(std::bind(dist, std::ref(engine))), shape);
    }
    
    /**
     * @function randint
     * @brief xexpression with specified @p shape containing uniformly distributed 
     *        random integers in the interval from @p lower to @p upper, excluding upper.
     * 
     * Numbers are drawn from @c std::uniform_int_distribution.
     * 
     * @param shape shape of resulting xexpression
     * @param lower lower bound
     * @param upper upper bound
     * @param engine random number engine
     *
     * @tparam T number type to use
     */
    template <class T, class E = random::default_engine_type>
    inline auto randint(std::vector<std::size_t> shape,
                        T lower = 0, T upper = std::numeric_limits<T>::max(), 
                        E& engine = random::get_default_random_engine())
    {
        std::uniform_int_distribution<T> dist(lower, upper - 1);
        return detail::make_xgenerator(detail::random_impl<T>(std::bind(dist, std::ref(engine))), shape);
    }

    /**
     * @function randn
     * @brief xexpression with specified @p shape containing numbers sampled from 
     *        the Normal (Gaussian) random number distribution with mean @p mean and 
     *        standard deviation @p std_dev.
     * 
     * Numbers are drawn from @c std::normal_distribution.
     * 
     * @param shape shape of resulting xexpression
     * @param mean mean of normal distribution
     * @param std_dev standard deviation of normal distribution
     * @param engine random number engine
     * 
     * @tparam T number type to use
     */
    template <class T, class E = random::default_engine_type>
    inline auto randn(std::vector<std::size_t> shape,
                      T mean = 0, T std_dev = 1,
                      E& engine = random::get_default_random_engine())
    {
        std::normal_distribution<T> dist(mean, std_dev);
        return detail::make_xgenerator(detail::random_impl<T>(std::bind(dist, std::ref(engine))), shape);
    }
}

#endif
