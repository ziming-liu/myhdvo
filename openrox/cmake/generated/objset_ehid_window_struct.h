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

#ifndef __OPENROX_OBJSET_Ehid_Window_STRUCT__
#define __OPENROX_OBJSET_Ehid_Window_STRUCT__

#include <system/memory/datatypes.h>
#include <system/memory/memory.h>
#include <system/errors/errors.h>

#include <core/features/descriptors/ehid/ehid_window.h>
#include <core/features/descriptors/ehid/ehid_window_struct.h>

//! \ingroup ObjSet
//! @defgroup ObjSet_Ehid_Window objset_Ehid_Window
//! @{

//! Dynamic vector base structure
struct Rox_ObjSet_Ehid_Window_Struct
{
   //! How many more blocks are allocated each time the allocated blocks are full
   Rox_Uint allocblocks;

   //! How many allocated cells are allocated
   Rox_Uint allocated;

   //! How many allocated cells are used
   Rox_Uint used;

   //! Real continuous memory storage
   Rox_Ehid_Window * data;
};

//! @}

#endif
