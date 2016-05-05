#include <TH.h>

#define BLASCONV
#define THNN_(NAME) TH_CONCAT_3(THNN_, Real, NAME)
typedef void THNNState;
		 
#include "generic/SpatialConvolutionMM.c"
#include <THGenerateFloatTypes.h>
