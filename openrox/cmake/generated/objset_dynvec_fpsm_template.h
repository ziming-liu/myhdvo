//==============================================================================
//
//    OPENROX   : File objset_dynvec_fpsm_template.h
//
//    Contents  : API of objset_dynvec_fpsm_template module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S.
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#ifndef __OPENROX_OBJSET_DynVec_Fpsm_Template__
#define __OPENROX_OBJSET_DynVec_Fpsm_Template__

#include <system/memory/datatypes.h>
#include <system/memory/memory.h>
#include <system/errors/errors.h>

#include <generated/dynvec_fpsm_template.h>
#include <generated/dynvec_fpsm_template.h>

//! \ingroup ObjSet
//! @defgroup ObjSet_DynVec_Fpsm_Template objset_DynVec_Fpsm_Template
//! @{

//! A Objset pointer
typedef struct Rox_ObjSet_DynVec_Fpsm_Template_Struct * Rox_ObjSet_DynVec_Fpsm_Template;

//! Create a new Object set structure (a set of managed pointer to objects)
//! \param  [out]  obj            Address of the pointer for the newly created object
//! \param  [in ]  allocblocks    How many cells are allocated
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_dynvec_fpsm_template_new (
   Rox_ObjSet_DynVec_Fpsm_Template * obj, 
   Rox_Uint allocblocks
);

//! Delete a objset structure (Will delete all attached objects)
//! \param  [out]     ptr         The pointer to the dynamic array to be deleted
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_dynvec_fpsm_template_del (
   Rox_ObjSet_DynVec_Fpsm_Template * ptr
);

//! Retrieve the list of  managed pointers of an objset
//! \param  [out]  ptr            The objset array to get data from
//! \return the pointer
ROX_API Rox_ErrorCode rox_objset_dynvec_fpsm_template_get_data_pointer (
   Rox_DynVec_Fpsm_Template ** data_pointer,
   Rox_ObjSet_DynVec_Fpsm_Template ptr
);

//! Retrieve the number of blocks used in objset
//! \param  [out]  used           the size of the objset
//! \param  [in ]  ptr            the objset array to get data from
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_dynvec_fpsm_template_get_used (
   Rox_Uint* used, Rox_ObjSet_DynVec_Fpsm_Template ptr
);

//! Reset a objset structure
//! \param  [out] ptr             The dynamic array to be reseted
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_dynvec_fpsm_template_reset (
   Rox_ObjSet_DynVec_Fpsm_Template ptr
);

//! Stack two objset together
//! \param  [out]  ptr            The dynamic array to update
//! \param  [in ]  other          The dynamic array to add after
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_dynvec_fpsm_template_stack (
   Rox_ObjSet_DynVec_Fpsm_Template ptr, 
   Rox_ObjSet_DynVec_Fpsm_Template other
);

//! Clone one objset
//! \param  [out]  ptr            The dynamic array to update with other array
//! \param  [in ]  source         The dynamic array to clone
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_dynvec_fpsm_template_clone (
   Rox_ObjSet_DynVec_Fpsm_Template ptr, 
   Rox_ObjSet_DynVec_Fpsm_Template source
);

//! Add an element to the objset
//! \param  [out]  ptr            The object pointer to add
//! \param  [in ]  data           The value to add (MUST BE A REAL POINTER, NOT A POINTER TO A LOCAL VARIABLE)
//! \return An error code
ROX_API Rox_ErrorCode rox_objset_dynvec_fpsm_template_append (
   Rox_ObjSet_DynVec_Fpsm_Template ptr, 
   Rox_DynVec_Fpsm_Template data
);

//! @}

#endif
