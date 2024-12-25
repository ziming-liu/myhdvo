//==============================================================================
//
//    OPENROX   : File dllist_template.h
//
//    Contents  : API of dllist_template module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S.
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#ifndef __OPENROX_DLLIST_TYPED_UINT__
#define __OPENROX_DLLIST_TYPED_UINT__

#include <system/memory/datatypes.h>
#include <system/memory/memory.h>
#include <system/errors/errors.h>
#include <system/memory/datatypes.h>
#include <system/memory/datatypes.h>

//! \ingroup DlList
//! @defgroup DlList_UINT DlList_UINT
//! \brief DlList with type UINT.
//! @{

//! A DlList Node pointer
typedef struct Rox_Dllist_Uint_Node_Struct * Rox_Dllist_Uint_Node;

//! A DllList pointer
typedef struct Rox_Dllist_Uint_Struct * Rox_Dllist_Uint;

//! Create a new double linked list
//! \param  [out]  obj            A pointer to the newly created list
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_uint_new(Rox_Dllist_Uint * obj);

//! Delete a double linked list
//! \param  [in]   obj            A pointer to the list to delete
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_uint_del(Rox_Dllist_Uint * obj);

//! Append a node to the end of the list
//! \param  [out]  obj            A pointer to the list to modify
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_uint_append(Rox_Dllist_Uint obj);

//! Delete the last node of the list
//! \param  [out]  obj            A pointer to the list to modify
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_uint_removelast(Rox_Dllist_Uint obj);

//! Delete the first node of the list
//! \param  [out]  obj            A pointer to the list to modify
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_uint_removefirst(Rox_Dllist_Uint obj);

//! Reset the list
//! \param  [out]  obj            A pointer to the list to modify
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_uint_reset(Rox_Dllist_Uint obj);

//! Add value to the list
//! \param  [out] obj A pointer to the list to modify
//! \param  [in]     val The value to add
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_uint_add(Rox_Dllist_Uint obj, Rox_Uint_Struct * val);

//! @}

#endif
