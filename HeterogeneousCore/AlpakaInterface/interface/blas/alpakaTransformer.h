#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_alpakaTransformer_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaTransformer_h

/**
 *  Collection of the transform kernels
 */

namespace cms::alpakatools {
  namespace transform {

    template <typename T>
    struct xpy {
      //
      using data_t = T;

      ALPAKA_FN_HOST_ACC static inline T apply(T x, T y) { return x + y; }
      ALPAKA_FN_HOST_ACC inline T operator()(T x, T y) const { return apply(x, y); }
    };

    template <typename T>
    struct xmy {
      //
      using data_t = T;

      ALPAKA_FN_HOST_ACC static inline T apply(T x, T y) { return x - y; }
      ALPAKA_FN_HOST_ACC inline T operator()(T x, T y) const { return apply(x, y); }
    };

    template <typename T>
    struct axpy {
      //
      using data_t = T;

      ALPAKA_FN_HOST_ACC static inline T apply(T a, T x, T y) { return a * x + y; }
      ALPAKA_FN_HOST_ACC inline T operator()(T a, T x, T y) const { return apply(a, x, y); }
    };

    template <typename T>
    struct xpay {
      //
      using data_t = T;

      ALPAKA_FN_HOST_ACC static inline T apply(T a, T x, T y) { return x + a * y; }
      ALPAKA_FN_HOST_ACC inline T operator()(T a, T x, T y) const { return apply(a, x, y); }
    };

  }  // namespace transform
}  // namespace cms::alpakatools

#endif
