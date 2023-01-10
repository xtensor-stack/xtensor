#include <sstream>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

#include "xtensor/xblockwise_reducer.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xnorm.hpp"

#include "test_common.hpp"

#define XTENSOR_REDUCER_TESTER(FNAME)                                           \
    struct FNAME##_tester                                                       \
    {                                                                           \
        template <class T = void, class... ARGS>                                \
        static auto op(ARGS&&... args)                                          \
        {                                                                       \
            return xt::blockwise::FNAME<T>(std::forward<ARGS>(args)...);        \
        }                                                                       \
        template <class T = void, class... ARGS>                                \
        static auto should_op(ARGS&&... args)                                   \
        {                                                                       \
            return xt::FNAME<T>(std::forward<ARGS>(args)...);                   \
        }                                                                       \
    };                                                                          \
    TYPE_TO_STRING(FNAME##_tester);                                             \
    TYPE_TO_STRING(std::tuple<FNAME##_tester, std::tuple<xt::keep_dims_type>>); \
    TYPE_TO_STRING(std::tuple<FNAME##_tester, std::tuple<xt::evaluation_strategy::immediate_type>>)

XTENSOR_REDUCER_TESTER(sum);
XTENSOR_REDUCER_TESTER(prod);
XTENSOR_REDUCER_TESTER(amin);
XTENSOR_REDUCER_TESTER(amax);
XTENSOR_REDUCER_TESTER(mean);
XTENSOR_REDUCER_TESTER(variance);
XTENSOR_REDUCER_TESTER(stddev);

#undef XTENSOR_REDUCER_TESTER

#define XTENSOR_NORM_REDUCER_TESTER(FNAME)                                      \
    struct FNAME##_tester                                                       \
    {                                                                           \
        template <class T = void, class... ARGS>                                \
        static auto op(ARGS&&... args)                                          \
        {                                                                       \
            return xt::blockwise::FNAME(std::forward<ARGS>(args)...);           \
        }                                                                       \
        template <class T = void, class... ARGS>                                \
        static auto should_op(ARGS&&... args)                                   \
        {                                                                       \
            return xt::FNAME(std::forward<ARGS>(args)...);                      \
        }                                                                       \
    };                                                                          \
    TYPE_TO_STRING(FNAME##_tester);                                             \
    TYPE_TO_STRING(std::tuple<FNAME##_tester, std::tuple<xt::keep_dims_type>>); \
    TYPE_TO_STRING(std::tuple<FNAME##_tester, std::tuple<xt::evaluation_strategy::immediate_type>>)

XTENSOR_NORM_REDUCER_TESTER(norm_l0);
XTENSOR_NORM_REDUCER_TESTER(norm_l1);
XTENSOR_NORM_REDUCER_TESTER(norm_l2);
XTENSOR_NORM_REDUCER_TESTER(norm_sq);
XTENSOR_NORM_REDUCER_TESTER(norm_linf);
XTENSOR_NORM_REDUCER_TESTER(norm_lp_to_p);
XTENSOR_NORM_REDUCER_TESTER(norm_lp);
#undef XTENSOR_NORM_REDUCER_TESTER

namespace xt
{

    using test_values_test_types = std::tuple<
        std::tuple<sum_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<sum_tester, std::tuple<xt::evaluation_strategy::immediate_type>>,
        std::tuple<prod_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<prod_tester, std::tuple<xt::evaluation_strategy::immediate_type>>,
        std::tuple<amin_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<amin_tester, std::tuple<xt::evaluation_strategy::immediate_type>>,
        std::tuple<amax_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<amax_tester, std::tuple<xt::evaluation_strategy::immediate_type>>,
        std::tuple<mean_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<mean_tester, std::tuple<xt::evaluation_strategy::immediate_type>>,
        std::tuple<variance_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<variance_tester, std::tuple<xt::evaluation_strategy::immediate_type>>,
        std::tuple<stddev_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<stddev_tester, std::tuple<xt::evaluation_strategy::immediate_type>>,
        std::tuple<norm_l0_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<norm_l0_tester, std::tuple<xt::evaluation_strategy::immediate_type>>,
        std::tuple<norm_l1_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<norm_l1_tester, std::tuple<xt::evaluation_strategy::immediate_type>>,
        std::tuple<norm_l2_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<norm_l2_tester, std::tuple<xt::evaluation_strategy::immediate_type>>,
        std::tuple<norm_sq_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<norm_sq_tester, std::tuple<xt::evaluation_strategy::immediate_type>>,
        std::tuple<norm_linf_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<norm_linf_tester, std::tuple<xt::evaluation_strategy::immediate_type>>>;

    using test_p_norm_values_test_types = std::tuple<
        std::tuple<norm_lp_to_p_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<norm_lp_to_p_tester, std::tuple<xt::evaluation_strategy::immediate_type>>,
        std::tuple<norm_lp_tester, std::tuple<xt::keep_dims_type>>,
        std::tuple<norm_lp_tester, std::tuple<xt::evaluation_strategy::immediate_type>>>;

    TEST_SUITE("xblockwise_reducer")
    {
        TEST_CASE_TEMPLATE_DEFINE("test_values", TesterTuple, test_values_id)
        {
            using tester_type = std::tuple_element_t<0, TesterTuple>;
            using options_type = std::tuple_element_t<1, TesterTuple>;

            dynamic_shape<std::size_t> shape({21, 10, 5});
            dynamic_shape<std::size_t> chunk_shape({5, 4, 2});
            xarray<int> input_exp(shape);

            // just iota is a bit boring since it will
            // lead to an uniform variance
            std::iota(input_exp.begin(), input_exp.end(), -5);
            for (std::size_t i = 0; i < input_exp.size(); ++i)
            {
                if (i % 2)
                {
                    input_exp.flat(i) += 10;
                }
            }

            std::vector<dynamic_shape<std::size_t>> axes_vec = {
                dynamic_shape<std::size_t>({0}),
                dynamic_shape<std::size_t>({1}),
                dynamic_shape<std::size_t>({2}),
                dynamic_shape<std::size_t>({0, 1}),
                dynamic_shape<std::size_t>({0, 2}),
                dynamic_shape<std::size_t>({0, 1, 2})};

            for (const auto& axes : axes_vec)
            {
                SUBCASE((std::string("axes = ") + stringify(axes)).c_str())
                {
                    auto reducer = tester_type::op(input_exp, chunk_shape, axes, options_type{});
                    auto should_reducer = tester_type::should_op(input_exp, axes, options_type{});

                    using should_result_value_type = typename std::decay_t<decltype(should_reducer)>::value_type;
                    using result_value_type = typename std::decay_t<decltype(reducer)>::value_type;

                    SUBCASE("result_value_type")
                    {
                        CHECK_UNARY(std::is_same<result_value_type, should_result_value_type>::value);
                    }

                    SUBCASE("shape")
                    {
                        CHECK_EQ(reducer.dimension(), should_reducer.dimension());
                        CHECK_EQ(reducer.shape(), should_reducer.shape());
                    }

                    SUBCASE("assign")
                    {
                        auto result = xarray<result_value_type>::from_shape(reducer.shape());
                        reducer.assign_to(result);

                        auto should_result = xt::eval(should_reducer);
                        if (std::is_same<tester_type, variance_tester>::value
                            || std::is_same<tester_type, stddev_tester>::value)
                        {
                            CHECK_UNARY(xt::allclose(result, should_result));
                        }
                        else
                        {
                            CHECK_EQ(result, should_result);
                        }
                    }
                }
            }
        }
        TEST_CASE_TEMPLATE_APPLY(test_values_id, test_values_test_types);


        TEST_CASE_TEMPLATE_DEFINE("test_p_norm_values", TesterTuple, test_p_norm_values_id)
        {
            using tester_type = std::tuple_element_t<0, TesterTuple>;
            using options_type = std::tuple_element_t<1, TesterTuple>;

            dynamic_shape<std::size_t> shape({21, 10, 5});
            dynamic_shape<std::size_t> chunk_shape({5, 4, 2});
            xarray<int> input_exp(shape);

            // just iota is a bit boring since it will
            // lead to an uniform variance
            std::iota(input_exp.begin(), input_exp.end(), -5);
            for (std::size_t i = 0; i < input_exp.size(); ++i)
            {
                if (i % 2)
                {
                    input_exp.flat(i) += 10;
                }
            }

            std::vector<dynamic_shape<std::size_t>> axes_vec = {
                dynamic_shape<std::size_t>({0}),
                dynamic_shape<std::size_t>({1}),
                dynamic_shape<std::size_t>({2}),
                dynamic_shape<std::size_t>({0, 1}),
                dynamic_shape<std::size_t>({0, 2}),
                dynamic_shape<std::size_t>({0, 1, 2})};

            for (const auto& axes : axes_vec)
            {
                SUBCASE((std::string("axes = ") + stringify(axes)).c_str())
                {
                    auto reducer = tester_type::op(input_exp, chunk_shape, 2.0, axes, options_type{});
                    auto should_reducer = tester_type::should_op(input_exp, 2.0, axes, options_type{});
                    auto should_result = xt::eval(should_reducer);

                    using should_result_value_type = typename std::decay_t<decltype(should_reducer)>::value_type;
                    using result_value_type = typename std::decay_t<decltype(reducer)>::value_type;

                    SUBCASE("result_value_type")
                    {
                        CHECK_UNARY(std::is_same<result_value_type, should_result_value_type>::value);
                    }


                    SUBCASE("shape")
                    {
                        CHECK_EQ(reducer.dimension(), should_reducer.dimension());
                        CHECK_EQ(reducer.shape(), should_reducer.shape());
                    }

                    SUBCASE("assign")
                    {
                        auto result = xarray<result_value_type>::from_shape(reducer.shape());
                        reducer.assign_to(result);
                        CHECK_UNARY(xt::allclose(result, should_result));
                    }
                }
            }
        }
        TEST_CASE_TEMPLATE_APPLY(test_p_norm_values_id, test_p_norm_values_test_types);

        TEST_CASE("test_api")
        {
            SUBCASE("sum")
            {
                dynamic_shape<std::size_t> shape({21, 10, 5});
                dynamic_shape<std::size_t> chunk_shape({5, 4, 2});
                xarray<int> input_exp(shape);

                SUBCASE("integral_axis")
                {
                    xt::blockwise::sum(input_exp, chunk_shape, 1);
                }
                SUBCASE("initalizer_list_axis")
                {
                    xt::blockwise::sum(input_exp, chunk_shape, {1});
                }
                SUBCASE("array")
                {
                    const int axes[2] = {0, 1};
                    xt::blockwise::sum(input_exp, chunk_shape, axes);
                }
                SUBCASE("no options")
                {
                    xt::blockwise::sum(input_exp, chunk_shape);
                }
                SUBCASE("templated")
                {
                    SUBCASE("integral_axis")
                    {
                        xt::blockwise::sum<double>(input_exp, chunk_shape, 1);
                    }
                    SUBCASE("initalizer_list_axis")
                    {
                        xt::blockwise::sum<double>(input_exp, chunk_shape, {1});
                    }
                    SUBCASE("array")
                    {
                        const int axes[2] = {0, 1};
                        xt::blockwise::sum<double>(input_exp, chunk_shape, axes);
                    }
                    SUBCASE("no options")
                    {
                        xt::blockwise::sum<double>(input_exp, chunk_shape);
                    }
                }
            }
            SUBCASE("l2_norm")
            {
                dynamic_shape<std::size_t> shape({21, 10, 5});
                dynamic_shape<std::size_t> chunk_shape({5, 4, 2});
                xarray<int> input_exp(shape);

                SUBCASE("integral_axis")
                {
                    xt::blockwise::norm_l2(input_exp, chunk_shape, 1);
                }
                SUBCASE("initalizer_list_axis")
                {
                    xt::blockwise::norm_l2(input_exp, chunk_shape, {1});
                }
                SUBCASE("array")
                {
                    const int axes[2] = {0, 1};
                    xt::blockwise::norm_l2(input_exp, chunk_shape, axes);
                }
            }
        }
    }

}
