#ifndef XTENSOR_DISK_ARRAY_HPP
#define XTENSOR_DISK_ARRAY_HPP

#include "xarray.hpp"
#include "xcsv.hpp"

namespace xt
{
    template <class T>
    class xdisk_reference
    {
    public:

        xdisk_reference(T& value, xarray<T>* array, std::ifstream* in_file, std::ofstream* out_file)
        {
            m_pvalue = &value;
            m_array = array;
            m_in_file = in_file;
            m_out_file = out_file;
        }

        operator const T() const
        {
            return *m_pvalue;
        }

        operator T()
        {
            return *m_pvalue;
        }

        T operator=(const T value)
        {
            *m_pvalue = value;
            dump_csv(*m_out_file, *m_array);
            return *m_pvalue;
        }

        friend std::ostream& operator<<(std::ostream& os, const xdisk_reference<T>& obj)
        {
            return os << *obj.m_pvalue;
        }

        std::ifstream* m_in_file;
        std::ofstream* m_out_file;
        xarray<T>* m_array;

    private:

        T* m_pvalue;
    };

    template <class T>
    class xdisk_array;

    template <class T>
    struct xcontainer_inner_types<xdisk_array<T>>
    {
        using const_reference = const T&;
        using reference = T&;
        using size_type = std::size_t;
        using storage_type = T;
    };

    template <class EC>
    struct xiterable_inner_types<xdisk_array<EC>>
    {
        using inner_shape_type = std::vector<size_t>;
        using const_stepper = xindexed_stepper<xdisk_array<EC>, true>;
        using stepper = xindexed_stepper<xdisk_array<EC>, false>;
    };

    template <class T>
    class xdisk_array: public xaccessible<xdisk_array<T>>,
                       public xiterable<xdisk_array<T>>
    {
    public:

        using const_reference = const T&;
        using reference = xdisk_reference<T>;
        using shape_type = std::vector<size_t>;
        using self_type = xdisk_array<T>;
        using inner_types = xcontainer_inner_types<self_type>;
        using size_type = typename inner_types::size_type;
        using storage_type = typename inner_types::storage_type;
        using value_type = storage_type;

        xdisk_array() {}

        xdisk_array(xarray<T>& arr, std::ifstream& in_file, std::ofstream& out_file)
        {
            m_array = &arr;
            m_in_file = &in_file;
            m_out_file = &out_file;
        }

        template <class S>
        void resize(S& shape)
        {
            m_array->resize(shape);
            *m_array = broadcast(0, shape);
        }

        template <class... Idxs>
        inline reference operator()(Idxs... idxs)
        {
            auto index = get_indexes(idxs...);
            return reference(m_array->element(index.cbegin(), index.cend()), m_array, m_in_file, m_out_file);
        }

        template <class... Idxs>
        inline const_reference operator()(Idxs... idxs) const
        {
            auto index = get_indexes(idxs...);
            return m_array->element(index.cbegin(), index.cend());
        }

        reference operator[](const xindex& index)
        {
            return reference(m_array->element(index.cbegin(), index.cend()), m_array, m_in_file, m_out_file);
        }

        const_reference operator[](const xindex& index) const
        {
            return m_array->element(index.cbegin(), index.cend());
        }

        template <class It>
        inline reference element(It first, It last)
        {
            return reference(m_array->element(first, last), m_array, m_in_file, m_out_file);
        }

        template <class It>
        inline const_reference element(It first, It last) const
        {
            return m_array->element(first, last);
        }

    private:

        xarray<T>* m_array;
        std::ifstream* m_in_file;
        std::ofstream* m_out_file;

        template <class... Idxs>
        inline std::array<size_t, sizeof...(Idxs)> get_indexes(Idxs... idxs) const
        {
            std::array<size_t, sizeof...(Idxs)> indexes = {{idxs...}};
            return indexes;
        }
    };
}

#endif
