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

#ifndef __OPENROX_DYNVEC_Float_STRUCT__
#define __OPENROX_DYNVEC_Float_STRUCT__

#include <system/memory/datatypes.h>
#include <system/memory/datatypes.h>
#include <system/memory/datatypes.h>



//! Dynamic vector base structure
struct Rox_DynVec_Float_Struct
{
   //! How many more blocks are allocated each time the allocated blocks are full
   Rox_Uint allocblocks;

   //! How many allocated cells are allocated
   Rox_Uint allocated;

   //! How many allocated cells are used
   Rox_Uint used;

   //! Real continuous memory storage
   Rox_Float * data;
};

//! Alias to dynvec structure
typedef struct Rox_DynVec_Float_Struct Rox_DynVec_Float_Struct;

#endif
