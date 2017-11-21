#include "TH.h"
#include "THLENN.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

#define THNN_CHECK_SHAPE(I1, I2)			\
  if (I1 != NULL && I2 != NULL && !THTensor_(isSameSizeAs)(I1, I2))	\
    {							\
       THDescBuff s1 = THTensor_(sizeDesc)(I1);		\
       THDescBuff s2 = THTensor_(sizeDesc)(I2);		\
       THError(#I1 " and " #I2 " shapes do not match: "	\
	       #I1 " %s, " #I2 " %s", s1.str, s2.str);	\
    }

#define THNN_CHECK_SHAPE_INDICES(I1, I2)             \
  THLongStorage *size2 = THLongTensor_newSizeOf(I2); \
  if (I1 != NULL && I2 != NULL && !THTensor_(isSize)(I1, size2)) \
    {             \
      THDescBuff s1 = THTensor_(sizeDesc)(I1);       \
      THDescBuff s2 = THLongTensor_sizeDesc(I2);     \
      THLongStorage_free(size2);                     \
      THError(#I1 " and " #I2 " shapes do not match: " \
        #I1 " %s, " #I2 " %s", s1.str, s2.str);      \
    } else {      \
      THLongStorage_free(size2);                     \
    }

#define THNN_CHECK_NELEMENT(I1, I2) \
  if (I1 != NULL && I2 != NULL ) {					\
    ptrdiff_t n1 = THTensor_(nElement)(I1);					\
    ptrdiff_t n2 = THTensor_(nElement)(I2);	                                \
    if (n1 != n2)							\
      {									\
	THDescBuff s1 = THTensor_(sizeDesc)(I1);			\
	THDescBuff s2 = THTensor_(sizeDesc)(I2);			\
	THError(#I1 " and " #I2 " have different number of elements: "	\
		#I1 "%s has %ld elements, while "			\
		#I2 "%s has %ld elements", s1.str, n1, s2.str, n2);	\
      }									\
  }

#define THNN_CHECK_DIM_SIZE(T, DIM, DIM_SIZE, SIZE)			\
  if (THTensor_(nDimension)(T) != DIM ||				\
      THTensor_(size)(T, DIM_SIZE) != SIZE) {				\
      THDescBuff s1 = THTensor_(sizeDesc)(T);				\
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d"	\
	      " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }

#define THNN_CHECK_DIM_SIZE_INDICES(T, DIM, DIM_SIZE, SIZE)			\
  if (THIndexTensor_(nDimension)(T) != DIM ||				\
      THIndexTensor_(size)(T, DIM_SIZE) != SIZE) {				\
      THDescBuff s1 = THIndexTensor_(sizeDesc)(T);				\
      THError("Need " #T " of dimension %d and " #T ".size[%d] == %d"	\
        " but got " #T " to be of shape: %s", DIM, DIM_SIZE, SIZE, s1.str); \
  }

#define THNN_ARGCHECK(COND, ARG, T, FORMAT)	\
  if (!(COND)) {				\
    THDescBuff s1 = THTensor_(sizeDesc)(T);	\
    THArgCheck(COND, ARG, FORMAT, s1.str);	\
  }


#include "generic/LenSoftMax.c"
#include "THGenerateFloatTypes.h"
