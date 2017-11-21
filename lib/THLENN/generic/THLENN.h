#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLENN.h"
#else

TH_API void THLENN_(LenSoftMax_updateOutput)(
          THLENNState *state,
          THTensor *input,
          THTensor *output,
          THIndexTensor *len);
TH_API void THLENN_(LenSoftMax_updateGradInput)(
          THLENNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          THIndexTensor *len);

#endif
