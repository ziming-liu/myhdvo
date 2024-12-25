//==============================================================================
//
//    OPENROX   : File dllist_struct_template.h
//
//    Contents  : Structure of dllist_template module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S.
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#ifndef __OPENROX_DLLIST_STRUCT_TYPED_UINT__
#define __OPENROX_DLLIST_STRUCT_TYPED_UINT__

#include <system/memory/datatypes.h>
#include <system/memory/memory.h>
#include <system/errors/errors.h>
#include <system/memory/datatypes.h>

//! \ingroup DlList
//! @defgroup DlList_UINT DlList_UINT
//! \brief DlList with type UINT.
//! @{

//! A DLList Node
struct Rox_Dllist_Uint_Node_Struct
{
   //!Pointer to previous node
   struct Rox_Dllist_Uint_Node_Struct * previous;

   //!Pointer to next node
   struct Rox_Dllist_Uint_Node_Struct * next;

   //!Data container for this node
   Rox_Uint_Struct data;
};

//! A DlList Node pointer
typedef struct Rox_Dllist_Uint_Node_Struct Rox_Dllist_Uint_Node_Struct ;

//! A structure for basic double linked list
struct Rox_Dllist_Uint_Struct
{
   //! How many nodes are used
   Rox_Uint used;

   //! How manu nodes are allocated
   Rox_Uint allocated;

   //! First node of the list
   Rox_Dllist_Uint_Node_Struct * first;

   //! Last USED node of the list.
   //! A node may be allocated previously but not needed for the moment
   Rox_Dllist_Uint_Node_Struct * last_used;

   //! Last node of the list
   Rox_Dllist_Uint_Node_Struct * last;
};

//! A DllList pointer
typedef struct Rox_Dllist_Uint_Struct Rox_Dllist_Uint_Struct;

//! @}

#endif
