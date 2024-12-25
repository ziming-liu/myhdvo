//==============================================================================
//
//    OPENROX   : File kdtree.h
//
//    Contents  : API of kdtree module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S. and Inria
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#ifndef __OPENROX_KDTREE__
#define __OPENROX_KDTREE__

#include <generated/dynvec_uint.h> 
#include <generated/dynvec_uint_struct.h> 
#include <generated/dynvec_point3d_double.h> 
#include <generated/dynvec_point3d_double_struct.h> 

#include <baseproc/geometry/point/point3d.h> 

#include "kdtree_node_struct.h"
#include "kdtree_heap_branch.h"


//! define 
#define MEAN_SAMPLES 100
//! define 
#define TOP_RAND 3


//! Pointer to a kdtree 
struct Rox_Kdtree_Struct
{
   //! count of trees inside kdtre structure 
   Rox_Uint _count_trees;

   //! count of trees leaves inside kdtre structure 
   Rox_Uint _count_leaves;

   //! Buffer used for what ? to be discovered
   Rox_DynVec_Uint _checked;

   //! Buffer used for what ? to be discovered
   Rox_Kdtree_Heap_Branch _heap;

   //! Trees 
   Rox_Kdtree_Node * _roots;
};


//! Kdtree 
typedef struct Rox_Kdtree_Struct * Rox_Kdtree;

//! Create a new kdtree
//! \param [out] kdtree pointer to the object created
//! \param [in ] count_trees number of subtrees created
//! \return An error code
//! \todo to be tested
ROX_API Rox_ErrorCode rox_kdtree_new(Rox_Kdtree * kdtree, Rox_Sint k);

//! Delete a kdtree 
//! \param [out] kdtree pointer to the object  to delete
//! \return An error code
//! \todo to be tested
ROX_API Rox_ErrorCode rox_kdtree_del(Rox_Kdtree * kdtree);

//! Clean a kdtree
//! \param [in ] kdtree          the object to clean
//! \return en error code
//! \todo to be tested
ROX_API Rox_ErrorCode rox_kdtree_clean(Rox_Kdtree kdtree);

//! Build a kdtree given a dynamic vector of features (3D points)
//! \param [in ] kdtree the object  to build into
//! \param [in ] features the features to index
//! \return An error code
//! \todo to be tested
ROX_API Rox_ErrorCode rox_kdtree_build(Rox_Kdtree kdtree, Rox_DynVec_Point3D_Double features);

//! Search for features neighboors
//! \param [in] results the result set of closest features
//! \param [in] kdtree the object  to search into
//! \param [in] features the features to which are indexed in this tree
//! \param [in] feat the feature to search for
//! \return en error code
//! \todo to be tested
ROX_API Rox_ErrorCode rox_kdtree_search(Rox_DynVec_Point3D_Double results, Rox_Kdtree kdtree, Rox_DynVec_Point3D_Double features, Rox_Point3D_Double feat);

//! Save the index to a file
//! \param [in] kdtree the object  to save
//! \param [in] filename the file name to save to
//! \return en error code
//! \todo to be tested
ROX_API Rox_ErrorCode rox_kdtree_save(Rox_Kdtree kdtree, char *filename);

//! Load the index from a file
//! \param [in] kdtree the object  to load
//! \param [in] filename the file name to load from
//! \return en error code
//! \todo to be tested
ROX_API Rox_ErrorCode rox_kdtree_load(Rox_Kdtree kdtree, char *filename);

#endif
