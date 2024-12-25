//==============================================================================
//
//    OPENROX   : File objset_template.h
//
//    Contents  : API of objset_template module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S.
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#ifndef __OPENROX_OBJSET_DynVec_Fpsm_Template_STRUCT__
#define __OPENROX_OBJSET_DynVec_Fpsm_Template_STRUCT__

#include <system/memory/datatypes.h>
#include <system/memory/memory.h>
#include <system/errors/errors.h>

#include <generated/dynvec_fpsm_template.h>
#include <generated/dynvec_fpsm_template.h>

//! \ingroup ObjSet
//! @defgroup ObjSet_DynVec_Fpsm_Template objset_DynVec_Fpsm_Template
//! @{

//! Dynamic vector base structure
struct Rox_ObjSet_DynVec_Fpsm_Template_Struct
{
   //! How many more blocks are allocated each time the allocated blocks are full
   Rox_Uint allocblocks;

   //! How many allocated cells are allocated
   Rox_Uint allocated;

   //! How many allocated cells are used
   Rox_Uint used;

   //! Real continuous memory storage
   Rox_DynVec_Fpsm_Template * data;
};

//! @}

#endif
