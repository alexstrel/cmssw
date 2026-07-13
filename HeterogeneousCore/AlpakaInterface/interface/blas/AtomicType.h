#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_AtomicType_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_AtomicType_h

#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"
#include <concepts>
#include <type_traits>

namespace cms::alpakatools {
  /**
   * The structure determines the atomic word size employed for a specific reduction type. 
   * This type must be lock-free to ensure correct behavior on platforms 
   * where memory coherence between the device (e.g., GPU) and host (CPU) isn't guaranteed.
   */



  template <typename T>
  concept AtomicScalar = (std::integral<std::remove_cvref_t<T>> || std::floating_point<std::remove_cvref_t<T>>) &&
                         (sizeof(std::remove_cvref_t<T>) == 4 || sizeof(std::remove_cvref_t<T>) == 8);

  template <typename T>
  struct AtomicTrait;

  template <AtomicScalar T>
  struct AtomicTrait<T> {
    using type = std::remove_cvref_t<T>;
  };

  template <typename T, int N>
  struct AtomicTrait<cms::alpakatools::VecArray<T, N>> {
    using type = typename AtomicTrait<std::remove_cvref_t<T>>::type;
  };

  template <typename T>
  using atomic_type_t = typename AtomicTrait<std::remove_cvref_t<T>>::type;

  template <typename T>
  concept AtomicType = requires { typename AtomicTrait<std::remove_cvref_t<T>>::type; };

  template <AtomicScalar T>
  constexpr inline std::size_t n_atomic_elements() {
    return 1;
  }

  template <VecArrayType T>
  constexpr inline std::size_t n_atomic_elements() {
    return T::N;
  }
}  // namespace cms::alpakatools

#endif
