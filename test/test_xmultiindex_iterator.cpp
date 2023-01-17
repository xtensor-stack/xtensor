#include <vector>

#include "xtensor/xmultiindex_iterator.hpp"

#include "test_common.hpp"

namespace xt
{

    TEST_SUITE("xmultiindex_iterator")
    {
        TEST_CASE("sum")
        {
            using shape_type = std::vector<std::size_t>;
            using iter_type = xmultiindex_iterator<shape_type>;

            shape_type roi_begin{2, 3, 4};
            shape_type roi_end{3, 5, 6};
            shape_type current{2, 3, 4};
            iter_type iter(roi_begin, roi_end, current, 0);


            iter_type end(roi_begin, roi_end, roi_end, 4);

            shape_type should(3);
            for (should[0] = roi_begin[0]; should[0] < roi_end[0]; ++should[0])
            {
                for (should[1] = roi_begin[1]; should[1] < roi_end[1]; ++should[1])
                {
                    for (should[2] = roi_begin[2]; should[2] < roi_end[2]; ++should[2])
                    {
                        EXPECT_EQ(*iter, should);
                        ++iter;
                    }
                }
            }
            EXPECT_TRUE(iter == end);
        }
    }

}
