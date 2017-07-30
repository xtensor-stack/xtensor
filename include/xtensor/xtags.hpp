/***************************************************************************
* Copyright (c) 2017, Ullrich Koethe                                       *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTAGS_HPP
#define XTAGS_HPP

/**********************************************************/
/*                                                        */
/*     helper for keyword arguments and related stuff     */
/*                                                        */
/**********************************************************/

namespace xt
{

    /// Determine size of an array type at runtime.
static const int runtime_size  = -1;

    /// Don't initialize memory that gets overwritten anyway.
enum skip_initialization_tag { dont_init };

    /// Copy-construct array in reversed order.
enum reverse_copy_tag { copy_reversed };

namespace tags {

/********************************************************/
/*                                                      */
/*                       tags::size                     */
/*                                                      */
/********************************************************/

    // Support for tags::size keyword argument
    // to disambiguate array sizes from initial values.
struct size_proxy
{
    size_t value;
};

struct size_tag
{
    size_proxy operator=(size_t s) const
    {
        return {s};
    }

    size_proxy operator()(size_t s) const
    {
        return {s};
    }
};

namespace {

size_tag size;

}

} // namespace tags


} // namespace xt

#endif // XTAGS_HPP
