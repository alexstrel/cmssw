#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_AtomicType_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_AtomicType_h

#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

namespace cms::alpakatools {
  /**
   * The structure determines the atomic word size employed for a specific reduction type. 
   * This type must be lock-free to ensure correct behavior on platforms 
   * where memory coherence between the device (e.g., GPU) and host (CPU) isn't guaranteed.
   */

  template <typename T, typename SFINAE = void>
  struct AtomicType;

  template <>
  struct AtomicType<std::size_t> {
    using type = std::size_t;
  };

  template <>
  struct AtomicType<int> {
    using type = int;
  };

  template <>
  struct AtomicType<float> {
    using type = float;
  };

  template <>
  struct AtomicType<double> {
    using type = double;
  };

  template <typename T>
  struct AtomicType<T, std::enable_if_t<std::is_same_v<T, cms::alpakatools::VecArray<double, T::N>>>> {
    using type = double;
  };

  template <typename T>
  struct AtomicType<T, std::enable_if_t<std::is_same_v<T, cms::alpakatools::VecArray<float, T::N>>>> {
    using type = float;
  };

}  // namespace cms::alpakatools

#endif
