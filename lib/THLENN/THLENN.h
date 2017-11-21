#ifndef THLENN_H
#define THLENN_H

#include <stdbool.h>
#include <TH.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define THLENN_(NAME) TH_CONCAT_3(THLENN_, Real, NAME)

#define THIndexTensor THLongTensor
#define THIndexTensor_(NAME) THLongTensor_ ## NAME

#define THIntegerTensor THIntTensor
#define THIntegerTensor_(NAME) THIntTensor_ ## NAME

typedef long THIndex_t;
typedef int THInteger_t;
typedef void THLENNState;

#define THLENN_resizeAs_indices(I1, I2)                    \
  THLongStorage *size2 = THIndexTensor_(newSizeOf)(I2);  \
  if (!THTensor_(isSize)(I1, size2))                     \
  { \
    THTensor_(resize)(I1, size2, NULL);                  \
  } \
  THLongStorage_free(size2);

#include "generic/THLENN.h"
#include <THGenerateFloatTypes.h>

#endif
