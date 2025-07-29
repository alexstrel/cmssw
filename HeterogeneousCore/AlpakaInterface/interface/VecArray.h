#ifndef HeterogeneousCore_AlpakaInterface_interface_VecArray_h
#define HeterogeneousCore_AlpakaInterface_interface_VecArray_h

//
// Author: Felice Pantaleo, CERN
//

#include <utility>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/CMSUnrollLoop.h"

namespace cms::alpakatools {

  template <class T, int maxSize>
  class VecArray {
  public:
    using self = VecArray<T, maxSize>;
    using value_t = T;

    // same notations as std::vector/array
    using value_type = T;
    static constexpr int N = maxSize;

    inline constexpr int push_back_unsafe(const T &element) {
      auto previousSize = m_size;
      m_size++;
      if (previousSize < maxSize) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        --m_size;
        return -1;
      }
    }

    template <class... Ts>
    constexpr int emplace_back_unsafe(Ts &&...args) {
      auto previousSize = m_size;
      m_size++;
      if (previousSize < maxSize) {
        (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
        return previousSize;
      } else {
        --m_size;
        return -1;
      }
    }

    inline constexpr T const &back() const {
      if (m_size > 0) {
        return m_data[m_size - 1];
      } else
        return T();  //undefined behaviour
    }

    inline constexpr T &back() {
      if (m_size > 0) {
        return m_data[m_size - 1];
      } else
        return T();  //undefined behaviour
    }

    // thread-safe version of the vector, when used in a kernel
    template <typename TAcc>
    ALPAKA_FN_ACC int push_back(const TAcc &acc, const T &element) {
      auto previousSize = alpaka::atomicAdd(acc, &m_size, 1, alpaka::hierarchy::Blocks{});
      if (previousSize < maxSize) {
        m_data[previousSize] = element;
        return previousSize;
      } else {
        alpaka::atomicSub(acc, &m_size, 1, alpaka::hierarchy::Blocks{});
        return -1;
      }
    }

    template <typename TAcc, class... Ts>
    ALPAKA_FN_ACC int emplace_back(const TAcc &acc, Ts &&...args) {
      auto previousSize = alpaka::atomicAdd(acc, &m_size, 1, alpaka::hierarchy::Blocks{});
      if (previousSize < maxSize) {
        (new (&m_data[previousSize]) T(std::forward<Ts>(args)...));
        return previousSize;
      } else {
        alpaka::atomicSub(acc, &m_size, 1, alpaka::hierarchy::Blocks{});
        return -1;
      }
    }

    inline constexpr T pop_back() {
      if (m_size > 0) {
        auto previousSize = m_size--;
        return m_data[previousSize - 1];
      } else
        return T();
    }

    VecArray() = default;
    VecArray(const VecArray<T, maxSize> &) = default;
    VecArray(VecArray<T, maxSize> &&) = default;

    constexpr VecArray(const T &value) {
      CMS_UNROLL_LOOP
      for (int i = 0; i < m_size; i++) {
        m_data[i] = value;
      }
    }

    template<typename... U, typename = std::enable_if_t<(sizeof...(U) == maxSize) && (std::conjunction_v<std::is_same<T, U>...>)>>
    constexpr VecArray(U... args) : m_data{ args... } {}

    VecArray<T, maxSize> &operator=(const VecArray<T, maxSize> &) = default;
    VecArray<T, maxSize> &operator=(VecArray<T, maxSize> &&) = default;

    inline constexpr T const *begin() const { return m_data; }
    inline constexpr T const *end() const { return m_data + m_size; }
    inline constexpr T *begin() { return m_data; }
    inline constexpr T *end() { return m_data + m_size; }
    inline constexpr int size() const { return m_size; }
    inline constexpr T &operator[](int i) { return m_data[i]; }
    inline constexpr const T &operator[](int i) const { return m_data[i]; }
    inline constexpr void reset() { m_size = 0; }
    inline static constexpr int capacity() { return maxSize; }
    inline constexpr T const *data() const { return m_data; }
    inline constexpr void resize(int size) { m_size = size; }
    inline constexpr bool empty() const { return 0 == m_size; }
    inline constexpr bool full() const { return maxSize == m_size; }


    // Extra:
    inline constexpr T norm2() const {
      T res{0};	    
      CMS_UNROLL_LOOP
      for (int i = 0; i < m_size; i++) {
        res += m_data[i]*m_data[i];
      }
      return res;
    }


    template<unsigned int n>
    inline constexpr T partial_norm2() const {
      constexpr int reduced_size = n > maxSize ? maxSize : n;

      T res{0};
      CMS_UNROLL_LOOP
      for (int i = 0; i < reduced_size; i++) {
        res += m_data[i]*m_data[i];
      }

      return res;
    }
    
    inline constexpr VecArray<T, maxSize>& operator*=(const T& scale) {
      CMS_UNROLL_LOOP
      for (int i = 0; i < m_size; ++i) {
        m_data[i] *= scale;
      }
      return *this;
    }

    inline constexpr VecArray<T, maxSize>& operator+=(const VecArray<T, maxSize>& rhs) {
      CMS_UNROLL_LOOP
      for (int i = 0; i < m_size; ++i) {
        m_data[i] += rhs[i];
      }
      return *this;
    }    

    template <typename TAcc >
    ALPAKA_FN_ACC T norm( const TAcc &acc ) const {
      const T nrm2 = norm2();
      return alpaka::math::sqrt(acc, nrm2);
    }    

    template <typename TAcc, unsigned int n >
    ALPAKA_FN_ACC T partial_norm( const TAcc &acc) const {

      const T partial_nrm2 = partial_norm2<n>();
      return alpaka::math::sqrt(acc, partial_nrm2);
    }

    template <typename TAcc>
    ALPAKA_FN_ACC void normalize( const TAcc &acc) const {

      const T nrm = norm(acc);

      if (nrm == 0.) return;

      CMS_UNROLL_LOOP
      for (int i = 0; i < m_size; i++) {
        m_data[i] /= nrm;
      }
    }

  private:
    T m_data[maxSize];

    int m_size;
  };

  template <typename T>
  struct is_VecArray : std::false_type {};

  template <typename T, int N>
  struct is_VecArray<cms::alpakatools::VecArray<T, N>> : std::true_type {};

  template <typename T>
  inline constexpr bool is_VecArray_v = is_VecArray<T>::value;


  template <typename VecN, std::enable_if_t<is_VecArray_v<VecN>, int> = 0>
  inline constexpr VecN ax(const typename VecN::value_type a, const VecN& x){
    VecN res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < x.size(); i++) {
      res[i] = a*x[i];
    }

    return res;
  }  

  template <typename VecN, std::enable_if_t<is_VecArray_v<VecN>, int> = 0>
  inline constexpr VecN xpy( const VecN& x, const VecN& y ){
    VecN res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < x.size(); i++) {
      res[i] = x[i] + y[i];
    }

    return res;
  }

  template <typename VecN, std::enable_if_t<is_VecArray_v<VecN>, int> = 0>
  inline constexpr VecN xmy( const VecN& x, const VecN& y ){
    VecN res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < x.size(); i++) {
      res[i] = x[i] - y[i];
    }

    return res;
  }

  template <typename VecN, std::enable_if_t<is_VecArray_v<VecN>, int> = 0>
  inline constexpr VecN axpy( const typename VecN::value_type a, const VecN& x, const VecN& y ){
    VecN res;

    CMS_UNROLL_LOOP
    for (int i = 0; i < x.size(); i++) {
      res[i] = a * x[i] + y[i];
    }

    return res;
  }

  template <typename VecN, std::enable_if_t<is_VecArray_v<VecN>, int> = 0>
  inline constexpr VecN::value_type dot( const VecN& x, const VecN& y ){
    typename VecN::value_type res{0};

    CMS_UNROLL_LOOP
    for (int i = 0; i < x.size(); i++) {
      res += x[i] * y[i];
    }

    return res;
  }  

  template <typename VecN, std::enable_if_t<is_VecArray_v<VecN>, int> = 0>
  inline constexpr VecN::value_type diff2( const VecN& x, const VecN& y ){
    typename VecN::value_type res{0};

    CMS_UNROLL_LOOP
    for (int i = 0; i < x.size(); i++) {
      const typename VecN::value_type tmp = x[i] - y[i];
      res += tmp*tmp;
    }
    return res;
  }



}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_VecArray_h
