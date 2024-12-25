//==============================================================================
//
//    OPENROX   : File dynvec_struct_template.h
//
//    Contents  : API of dynvec_struct_template module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S.
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#ifndef __OPENROX_DYNVEC_Fpsm_Feature_STRUCT__
#define __OPENROX_DYNVEC_Fpsm_Feature_STRUCT__

#include <system/memory/datatypes.h>
#include <core/features/descriptors/fpsm/fpsm_feature_struct.h>
#include <core/features/descriptors/fpsm/fpsm_feature_struct.h>



//! Dynamic vector base structure
struct Rox_DynVec_Fpsm_Feature_Struct
{
   //! How many more blocks are allocated each time the allocated blocks are full
   Rox_Uint allocblocks;

   //! How many allocated cells are allocated
   Rox_Uint allocated;

   //! How many allocated cells are used
   Rox_Uint used;

   //! Real continuous memory storage
   Rox_Fpsm_Feature_Struct * data;
};

//! Alias to dynvec structure
typedef struct Rox_DynVec_Fpsm_Feature_Struct Rox_DynVec_Fpsm_Feature_Struct;

#endif
