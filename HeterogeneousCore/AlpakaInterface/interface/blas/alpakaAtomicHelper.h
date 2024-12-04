#ifndef HeterogeneousCore_AlpakaInterface_interface_blas alpakaAtomicHelper_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaAtomicHelper_h

#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

namespace cms::alpakatools {

  template <typename T, typename SFINAE = void>
  struct atomic_type;

  template <>
  struct atomic_type<std::size_t> {
    using type = std::size_t;
  };

  template <>
  struct atomic_type<int> {
    using type = int;
  };

  template <>
  struct atomic_type<float> {
    using type = float;
  };

  template <>
  struct atomic_type<double> {
    using type = double;
  };

  template <typename T>
  struct atomic_type<T, std::enable_if_t<std::is_same_v<T, cms::alpakatools::VecArray<double, T::N>>>> {
    using type = double;
  };

  template <typename T>
  struct atomic_type<T, std::enable_if_t<std::is_same_v<T, cms::alpakatools::VecArray<float, T::N>>>> {
    using type = float;
  };

}  // namespace cms::alpakatools

#endif
