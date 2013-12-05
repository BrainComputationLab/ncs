/*
	Various useful CUDA things.
*/

#pragma once

namespace warp {

inline __device__ bool leader() {
	return (threadIdx.x & 0x1F) == 0;
}

inline __device__ unsigned int index() {
	return threadIdx.x >> 5;
}

inline __device__ unsigned int thread() {
	return threadIdx.x & 0x1F;
}

inline __device__ void reduceOr(unsigned int* v, unsigned int warp_thread) {
	if (warp_thread < 16) v[warp_thread] |= v[warp_thread + 16];
	if (warp_thread <  8) v[warp_thread] |= v[warp_thread +  8];
	if (warp_thread <  4) v[warp_thread] |= v[warp_thread +  4];
	if (warp_thread <  2) v[warp_thread] |= v[warp_thread +  2];
	if (warp_thread <  1) v[warp_thread] |= v[warp_thread +  1];
}

} // namespace warp

namespace block {

inline __device__ unsigned int thread() {
	return threadIdx.x;
}

inline __device__ unsigned int stride() {
	return blockDim.x;
}

inline __device__ unsigned int size() {
	return blockDim.x;
}

inline __device__ bool leader() {
	return threadIdx.x == 0;
}

} // namespace block

namespace grid {

inline __device__ unsigned int thread() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

inline __device__ unsigned int stride() {
	return blockDim.x * gridDim.x;
}

} // namespace grid

namespace bit {

inline __device__ unsigned int mask(unsigned int position) {
	return 0x80000000 >> (position & 0x1F);
}

inline __device__ unsigned int word(unsigned int index) {
	return index >> 5;
}

inline __device__ unsigned int bit(unsigned int index) {
	return index & 0x1F;
}

inline __device__ unsigned int extract(unsigned int word, 
                                       unsigned int position) {
	return (word << position) & 0x80000000;
}

inline __device__ unsigned int rextract(unsigned int word,
                                        unsigned int position) {
	return (word << position) >> 31;
}

inline __device__ unsigned int extract(const unsigned int* array,
                                       unsigned int index) {
	return extract(array[word(index)], bit(index));
}

} // namespace bit

namespace math {

inline __device__ unsigned int ceiling(unsigned int v, unsigned int divisor) {
  return (v + divisor - 1) / divisor * divisor;
}

} // namespace math

