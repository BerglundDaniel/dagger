#include "kernel/internal/apply.cuh"

namespace dagger {
namespace kernal {
namespace internal {

//TODO macro this
template void applyInternal<int, 5>(int numBlocks, int threadsPerBlock,
				    const container::ContainerProperties prop,
				    const int* vecIn, int* vecOut);

} /* internal */
} /* kernel */
} /* dagger */
