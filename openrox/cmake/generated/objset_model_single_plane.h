//==============================================================================
//
//    OPENROX   : File objset_model_single_plane.h
//
//    Contents  : API of objset_model_single_plane module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S.
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#ifndef __OPENROX_OBJSET_Model_Single_Plane__
#define __OPENROX_OBJSET_Model_Single_Plane__

#include <system/memory/datatypes.h>
#include <system/memory/memory.h>
#include <system/errors/errors.h>

#include <core/model/model_single_plane.h>
#include <core/features/descriptors/ehid/ehid_target_struct.h>

//! \ingroup ObjSet
//! @defgroup ObjSet_Model_Single_Plane objset_Model_Single_Plane
//! @{

//! A Objset pointer
typedef struct Rox_ObjSet_Model_Single_Plane_Struct * Rox_ObjSet_Model_Single_Plane;

//! Create a new Object set structure (a set of managed pointer to objects)
//! \param  [out]  obj            Address of the pointer for the newly created object
//! \param  [in ]  allocblocks    How many cells are allocated
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_model_single_plane_new (
   Rox_ObjSet_Model_Single_Plane * obj, 
   Rox_Uint allocblocks
);

//! Delete a objset structure (Will delete all attached objects)
//! \param  [out]     ptr         The pointer to the dynamic array to be deleted
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_model_single_plane_del (
   Rox_ObjSet_Model_Single_Plane * ptr
);

//! Retrieve the list of  managed pointers of an objset
//! \param  [out]  ptr            The objset array to get data from
//! \return the pointer
ROX_API Rox_ErrorCode rox_objset_model_single_plane_get_data_pointer (
   Rox_Model_Single_Plane ** data_pointer,
   Rox_ObjSet_Model_Single_Plane ptr
);

//! Retrieve the number of blocks used in objset
//! \param  [out]  used           the size of the objset
//! \param  [in ]  ptr            the objset array to get data from
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_model_single_plane_get_used (
   Rox_Uint* used, Rox_ObjSet_Model_Single_Plane ptr
);

//! Reset a objset structure
//! \param  [out] ptr             The dynamic array to be reseted
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_model_single_plane_reset (
   Rox_ObjSet_Model_Single_Plane ptr
);

//! Stack two objset together
//! \param  [out]  ptr            The dynamic array to update
//! \param  [in ]  other          The dynamic array to add after
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_model_single_plane_stack (
   Rox_ObjSet_Model_Single_Plane ptr, 
   Rox_ObjSet_Model_Single_Plane other
);

//! Clone one objset
//! \param  [out]  ptr            The dynamic array to update with other array
//! \param  [in ]  source         The dynamic array to clone
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_model_single_plane_clone (
   Rox_ObjSet_Model_Single_Plane ptr, 
   Rox_ObjSet_Model_Single_Plane source
);

//! Add an element to the objset
//! \param  [out]  ptr            The object pointer to add
//! \param  [in ]  data           The value to add (MUST BE A REAL POINTER, NOT A POINTER TO A LOCAL VARIABLE)
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_model_single_plane_append (
   Rox_ObjSet_Model_Single_Plane ptr, 
   Rox_Model_Single_Plane data
);

//! @}

#endif
