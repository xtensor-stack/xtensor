#ifndef XEXCEPTION_HPP
#define XEXCEPTION_HPP

#include <exception>
#include <string>
#include <sstream>

namespace xt
{

    /*******************
     * broadcast_error *
     *******************/

    class broadcast_error : public std::exception
    {

    public:

        template <class S1, class S2>
        broadcast_error(const S1& lhs, const S2& rhs);

        virtual const char* what() const noexcept;

    private:

        std::string m_message;
    };

    /**********************************
     * broadcast_error implementation *
     **********************************/

    template <class S1, class S2>
    inline broadcast_error::broadcast_error(const S1& lhs,
                                            const S2& rhs)
    {
        std::ostringstream buf("Incompatible dimension of arrays:", std::ios_base::ate);
        
        buf << "\n LHS shape = (";
        using size_type1 = typename S1::value_type;
        std::ostream_iterator<size_type1> iter1(buf, ", ");
        std::copy(lhs.cbegin(), lhs.cend(), iter1);

        buf << ")\n RHS shape = (";
        using size_type2 = typename S2::value_type;
        std::ostream_iterator<size_type2> iter2(buf, ", ");
        std::copy(rhs.cbegin(), rhs.cend(), iter2);
        buf << ")";

        m_message = buf.str();
    }

    inline const char* broadcast_error::what() const noexcept
    {
        return m_message.c_str();
    }

    /*******************
     * transpose_error *
     *******************/

    class transpose_error : public std::exception
    {

    public:

        transpose_error(const std::string& msg);

        virtual const char* what() const noexcept;

    private:

        std::string m_message;
    };

    /**********************************
     * transpose_error implementation *
     **********************************/

    inline transpose_error::transpose_error(const std::string& msg) : m_message(msg) {};

    inline const char* transpose_error::what() const noexcept
    {
        return m_message.c_str();
    }

    /****************************
     * unnormalized slice error *
     ****************************/

    class unnormalized_slice_error : public std::exception
    {

    public:

        unnormalized_slice_error(const std::string& msg);

        virtual const char* what() const noexcept;

    private:

        std::string m_message;
    };

    /*******************************************
     * unnormalized_slice_error implementation *
     *******************************************/

    inline unnormalized_slice_error::unnormalized_slice_error(const std::string& msg) : m_message(msg) {};

    inline const char* unnormalized_slice_error::what() const noexcept
    {
        return m_message.c_str();
    }
}

#endif
