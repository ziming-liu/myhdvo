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

#ifndef __OPENROX_OBJSET_Edge_Point_STRUCT__
#define __OPENROX_OBJSET_Edge_Point_STRUCT__

#include <system/memory/datatypes.h>
#include <system/memory/memory.h>
#include <system/errors/errors.h>

#include <core/tracking/edge/edge_point.h>
#include <core/tracking/edge/edge_point.h>

//! \ingroup ObjSet
//! @defgroup ObjSet_Edge_Point objset_Edge_Point
//! @{

//! Dynamic vector base structure
struct Rox_ObjSet_Edge_Point_Struct
{
   //! How many more blocks are allocated each time the allocated blocks are full
   Rox_Uint allocblocks;

   //! How many allocated cells are allocated
   Rox_Uint allocated;

   //! How many allocated cells are used
   Rox_Uint used;

   //! Real continuous memory storage
   Rox_Edge_Point * data;
};

//! @}

#endif
