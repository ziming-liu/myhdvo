//==============================================================================
//
//    OPENROX   : File objset_matse3.h
//
//    Contents  : API of objset_matse3 module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S.
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#ifndef __OPENROX_OBJSET_MatSE3__
#define __OPENROX_OBJSET_MatSE3__

#include <system/memory/datatypes.h>
#include <system/memory/memory.h>
#include <system/errors/errors.h>

#include <baseproc/maths/linalg/matse3.h>
#include <core/tracking/edge/edge_cylinder.h>

//! \ingroup ObjSet
//! @defgroup ObjSet_MatSE3 objset_MatSE3
//! @{

//! A Objset pointer
typedef struct Rox_ObjSet_MatSE3_Struct * Rox_ObjSet_MatSE3;

//! Create a new Object set structure (a set of managed pointer to objects)
//! \param  [out]  obj            Address of the pointer for the newly created object
//! \param  [in ]  allocblocks    How many cells are allocated
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_matse3_new (
   Rox_ObjSet_MatSE3 * obj, 
   Rox_Uint allocblocks
);

//! Delete a objset structure (Will delete all attached objects)
//! \param  [out]     ptr         The pointer to the dynamic array to be deleted
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_matse3_del (
   Rox_ObjSet_MatSE3 * ptr
);

//! Retrieve the list of  managed pointers of an objset
//! \param  [out]  ptr            The objset array to get data from
//! \return the pointer
ROX_API Rox_ErrorCode rox_objset_matse3_get_data_pointer (
   Rox_MatSE3 ** data_pointer,
   Rox_ObjSet_MatSE3 ptr
);

//! Retrieve the number of blocks used in objset
//! \param  [out]  used           the size of the objset
//! \param  [in ]  ptr            the objset array to get data from
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_matse3_get_used (
   Rox_Uint* used, Rox_ObjSet_MatSE3 ptr
);

//! Reset a objset structure
//! \param  [out] ptr             The dynamic array to be reseted
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_matse3_reset (
   Rox_ObjSet_MatSE3 ptr
);

//! Stack two objset together
//! \param  [out]  ptr            The dynamic array to update
//! \param  [in ]  other          The dynamic array to add after
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_matse3_stack (
   Rox_ObjSet_MatSE3 ptr, 
   Rox_ObjSet_MatSE3 other
);

//! Clone one objset
//! \param  [out]  ptr            The dynamic array to update with other array
//! \param  [in ]  source         The dynamic array to clone
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_matse3_clone (
   Rox_ObjSet_MatSE3 ptr, 
   Rox_ObjSet_MatSE3 source
);

//! Add an element to the objset
//! \param  [out]  ptr            The object pointer to add
//! \param  [in ]  data           The value to add (MUST BE A REAL POINTER, NOT A POINTER TO A LOCAL VARIABLE)
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_matse3_append (
   Rox_ObjSet_MatSE3 ptr, 
   Rox_MatSE3 data
);

//! @}

#endif
