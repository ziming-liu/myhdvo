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

#ifndef __OPENROX_DLLIST_TYPED_QUADTREE_ITEM__
#define __OPENROX_DLLIST_TYPED_QUADTREE_ITEM__

#include <system/memory/datatypes.h>
#include <system/memory/memory.h>
#include <system/errors/errors.h>
#include <core/occupancy/quadtree_item.h>
#include <core/occupancy/quadtree_item_struct.h>

//! \ingroup DlList
//! @defgroup DlList_QUADTREE_ITEM DlList_QUADTREE_ITEM
//! \brief DlList with type QUADTREE_ITEM.
//! @{

//! A DlList Node pointer
typedef struct Rox_Dllist_QuadTree_Item_Node_Struct * Rox_Dllist_QuadTree_Item_Node;

//! A DllList pointer
typedef struct Rox_Dllist_QuadTree_Item_Struct * Rox_Dllist_QuadTree_Item;

//! Create a new double linked list
//! \param  [out]  obj            A pointer to the newly created list
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_quadtree_item_new(Rox_Dllist_QuadTree_Item * obj);

//! Delete a double linked list
//! \param  [in]   obj            A pointer to the list to delete
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_quadtree_item_del(Rox_Dllist_QuadTree_Item * obj);

//! Append a node to the end of the list
//! \param  [out]  obj            A pointer to the list to modify
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_quadtree_item_append(Rox_Dllist_QuadTree_Item obj);

//! Delete the last node of the list
//! \param  [out]  obj            A pointer to the list to modify
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_quadtree_item_removelast(Rox_Dllist_QuadTree_Item obj);

//! Delete the first node of the list
//! \param  [out]  obj            A pointer to the list to modify
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_quadtree_item_removefirst(Rox_Dllist_QuadTree_Item obj);

//! Reset the list
//! \param  [out]  obj            A pointer to the list to modify
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_quadtree_item_reset(Rox_Dllist_QuadTree_Item obj);

//! Add value to the list
//! \param  [out] obj A pointer to the list to modify
//! \param  [in]     val The value to add
//! \return An error code
ROX_API Rox_ErrorCode rox_dllist_quadtree_item_add(Rox_Dllist_QuadTree_Item obj, Rox_QuadTree_Item_Struct * val);

//! @}

#endif
