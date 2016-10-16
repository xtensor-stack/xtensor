#ifndef XEXCEPTION_HPP
#define XEXCEPTION_HPP

#include <exception>
#include <sstream>

#include "xindex.hpp"

namespace xt
{

    /*******************
     * broadcast_error *
     *******************/

    template <class S>
    class broadcast_error : public std::exception
    {

    public:

        broadcast_error(const xshape<S>& lhs, const xshape<S>& rhs);

        virtual const char* what() const noexcept;

    private:

        std::string m_message;
    };

    /**********************************
     * broadcast_error implementation *
     **********************************/

    template <class S>
    inline broadcast_error<S>::broadcast_error(const xshape<S>& lhs,
                                               const xshape<S>& rhs)
    {
        std::ostringstream buf("Incompatible dimension of arrays:", std::ios_base::ate);
        buf << "\n LHS shape = (";

        std::ostream_iterator<S> iter1(buf, ", ");
        std::copy(lhs.begin(), lhs.end(), iter1);

        buf << ")\n RHS shape = (";
        std::ostream_iterator<S> iter2(buf, ", ");
        std::copy(rhs.begin(), rhs.end(), iter2);
        buf << ")";

        m_message = buf.str();
    }

    template <class S>
    const char* broadcast_error<S>::what() const noexcept
    {
        return m_message.c_str();
    }
}

#endif

