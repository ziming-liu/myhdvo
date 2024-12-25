//==============================================================================
//
//    OPENROX   : File kdtree_node_struct.h
//
//    Contents  : API of kdtree node module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S. and Inria
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#ifndef __OPENROX_KDTREE_NODE_STRUCT__
#define __OPENROX_KDTREE_NODE_STRUCT__

//! Pointer to a branch of the kdtree 
struct Rox_Kdtree_Node_Struct
{
   //! Index of the node 
   int _index;

   //! Dimension cut value 
   double _cut_val;

   //! Dimension index of cut  
   int _cut_index;

   //! Pointer to left part  
   struct Rox_Kdtree_Node_Struct * _child_left;

   //! Pointer to right part  
   struct Rox_Kdtree_Node_Struct * _child_right;

   //! Pointer to optional data associated to the node 
   void *_data;

};

//! Kdtree node
typedef struct Rox_Kdtree_Node_Struct * Rox_Kdtree_Node;

#endif
