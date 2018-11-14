#include <immintrin.h>

// the following typedefs and templates should be included from converters/common.h
typedef int64_t i64;

template<typename T>
struct vec {
  T *ptr;
  i64 size;
};

template<typename T>
vec<T> make_vec(i64 size) {
  vec<T> t;
  t.ptr = (T *)malloc(size * sizeof(T));
  t.size = size;

  return t;
}

extern "C" void udf_add(vec<i64> *arr, i64 *scalar, vec<i64> *result) {
  *result = make_vec<i64>(arr->size);

  for (int i = 0; i < result->size; i++) {
    result->ptr[i] = arr->ptr[i] + *scalar;
  }
}
