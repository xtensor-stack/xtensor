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
#include <memory>
#include <sstream>

#include "xgenerator.hpp"

namespace xt
{
    namespace detail
    {
        template <class D, class S, class V, class E = std::mt19937>
        struct random_impl
        {
            using value_type = V;
            using size_type = std::size_t;

            random_impl(E engine, D dist, S shape) :
                m_engine(engine), m_dist(dist), m_strides(shape.size()), 
                m_engine_state(std::make_shared<std::stringstream>())
            {
                std::size_t data_size = 1;
                for(size_type i = m_strides.size(); i != 0; --i)
                {
                    m_strides[i - 1] = data_size;
                    data_size = m_strides[i - 1] * shape[i - 1];
                }
                // save initial engine state
                (*m_engine_state) << m_engine; 
            }

            template <class... Args>
            inline value_type operator()(Args... args) const
            {
                std::array<size_type, sizeof...(Args)> args_arr({static_cast<size_type>(args)...});
                return access_impl(args_arr);
            }

            inline value_type operator[](const xindex& idx) const
            {
                return access_impl(idx);
            }

            template <class It>
            inline value_type element(It first, It /*last*/) const
            {
                return access_impl(*first);
            }

        private:
            mutable E m_engine;
            mutable D m_dist;
            mutable S m_strides;
            // std::stringstream is not copyable but has to be copied in
            // case of being used in an xfunction
            std::shared_ptr<std::stringstream> m_engine_state;
            mutable long m_advance_state = -1;

            template <class A>
            value_type access_impl(const A& idx) const
            {
                long advance_state = std::inner_product(idx.begin(), idx.end(), m_strides.begin(), 0);
                m_advance_state++;
                if (m_advance_state == advance_state)
                {
                    return m_dist(m_engine);
                }
                else
                {
                    if (advance_state > m_advance_state)
                    {
                        m_engine.discard(advance_state - m_advance_state);
                        m_advance_state = advance_state;
                        return m_dist(m_engine);
                    }
                    // TODO(BUG): this doesn't reset correctly after 2nd iteration...
                    (*m_engine_state) >> m_engine;
                    m_engine.discard(advance_state);
                    m_advance_state = advance_state;

                    return m_dist(m_engine);
                }
            }
        };

        template <class T>
        inline auto make_random_xgenerator(auto dist, auto& engine, auto shape) {
            auto f = detail::make_xgenerator(
                detail::random_impl<decltype(dist), decltype(shape), T>(engine, dist, shape), 
                shape
            );
            // Discard N random numbers from random number engine 
            std::size_t n_discard = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
            engine.discard(n_discard);
            return f;   
        }
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
     * @brief xexpression with specified @shape containing random numbers in the 
     *        interval from @lower to @upper, excluding upper. Numbers are 
     *        drawn from std::uniform_real_distribution.
     * 
     * @param shape shape of resulting xexpression
     * @param lower lower bound
     * @param upper upper bound
     * @param engine random number engine (defaults to xt::random::get_default_random_engine())
     *
     * @tparam T number type to use
     */ 
    template <class T, class E = random::default_engine_type>
    inline auto rand(std::vector<std::size_t> shape, T lower = 0, T upper = 1,
                     E& engine = random::get_default_random_engine())
    {
        std::uniform_real_distribution<T> dist(lower, upper);
        return detail::make_random_xgenerator<T>(dist, engine, shape);
    }
    
    /**
     * @function randint
     * @brief xexpression with specified @shape containing uniformly distributed 
     *        random integers in the interval from @lower to @upper, excluding upper.
     * 
     * @param shape shape of resulting xexpression
     * @param lower lower bound (defaults to 0)
     * @param upper upper bound (defaults to std::numeric_limits<T>::max())
     * @param engine random number engine (defaults to xt::random::get_default_random_engine())
     *
     * @tparam T number type to use
     */
    template <class T, class E = random::default_engine_type>
    inline auto randint(std::vector<std::size_t> shape,
                        T lower = 0, T upper = std::numeric_limits<T>::max(), 
                        E& engine = random::get_default_random_engine())
    {
        std::uniform_int_distribution<T> dist(lower, upper - 1);
        return detail::make_random_xgenerator<T>(dist, engine, shape);
    }

    /**
     * @function randn
     * @brief xexpression with specified @shape containing normal distributed 
     *        random numbers in the interval from @lower to @upper, excluding upper.
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
        return detail::make_random_xgenerator<T>(dist, engine, shape);
    }
}

#endif