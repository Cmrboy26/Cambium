#pragma once

#if defined(__CUDACC__)
#define MCSIM_HD __host__ __device__
#else
#define MCSIM_HD
#endif