//==============================================================================
//
//    OPENROX   : File dynvec_sparse_value.h
//
//    Contents  : API of dynvec_sparse_value module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S.
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#ifndef __OPENROX_DYNVEC_Sparse_Value__
#define __OPENROX_DYNVEC_Sparse_Value__

#include <system/memory/datatypes.h>
#include <system/memory/memory.h>
#include <system/errors/errors.h>

#include <system/memory/datatypes.h>
#include <system/memory/datatypes.h>


//! \ingroup DynVec
//! \defgroup DynVec_Sparse_Value DynVec_Sparse_Value
//! @{

//! Pointer to dynvec structure
typedef struct Rox_DynVec_Sparse_Value_Struct * Rox_DynVec_Sparse_Value;

//! Create a new dynamic array structure
//!
//! \param  [out]  obj            Address of the pointer for the newly created object
//! \param  [in ]  allocblocks    Number of additional cells allocated when maximum capacity is reached, also the number of cells pre-allocated
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_sparse_value_new (
   Rox_DynVec_Sparse_Value * obj, 
   Rox_Uint allocblocks
);

//! Delete a dynvec structure
//!
//! \param  [out]  ptr            The pointer to the dynamic array to be deleted
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_sparse_value_del (
   Rox_DynVec_Sparse_Value * ptr
);

//! Retrieve the list of pointers of a vector
//!
//! \param [in]  ptr          The dynamic array to get data from
//! \return the pointer
ROX_API Rox_ErrorCode rox_dynvec_sparse_value_get_data_pointer (
   Rox_Sparse_Value_Struct ** data_pointer, 
   Rox_DynVec_Sparse_Value ptr
);

//! Retrieve the number of blocks used in vector
//!
//! \param  [out]  nb_used        The size of the vector
//! \param  [in ]  ptr            The dynamic array to get data from
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_sparse_value_get_used (
   Rox_Uint * nb_used, Rox_DynVec_Sparse_Value ptr
);

//! Reset the counter of used blocks of a dynvec structure
//!
//! \param  [out]  ptr            The dynamic array to reset
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_sparse_value_reset (
   Rox_DynVec_Sparse_Value ptr
);

//! Increase the counter of used blocks of a dynvec structure, allocate memory if needed
//! Warning: if the counter of used blocks is a multiple of allocblocks, the allocated memory will actually be of size ( used + allocblocks )
//!
//! \param  [out]  ptr            The dynamic array to update
//! \param  [in ]  nbcells        The number of additional cells which have been used newly
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_sparse_value_usecells (
   Rox_DynVec_Sparse_Value ptr, 
   Rox_Uint nbcells
);

//!  Requests that the vector capacity be at least enough to contain n elements
//!
//! \param  [out]  ptr            The dynamic array to update
//! \param  [in ]  nbcells        Minimum capacity for the vector
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_sparse_value_reserve (
   Rox_DynVec_Sparse_Value ptr, 
   Rox_Uint nbcells
);

//! Stack two vectors together
//!
//! \param  [out]  ptr             The dynamic array to update
//! \param  [in ]  other           The dynamic array to add after
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_sparse_value_stack (
   Rox_DynVec_Sparse_Value ptr, 
   Rox_DynVec_Sparse_Value other
);

//! Clone one vector
//!
//! \param  [out]  ptr            The dynamic array to update with other array
//! \param  [in ]  source         The dynamic array to clone
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_sparse_value_clone (
   Rox_DynVec_Sparse_Value ptr, 
   Rox_DynVec_Sparse_Value source
);

//! Add an element to the vector, allocate memory if needed
//! Warning: if the counter of used blocks is a multiple of allocblocks, the allocated memory will actually be of size ( used + allocblocks )
//!
//! \param  [out]  ptr            The dynamic array to update with other array
//! \param  [in ]  data           The value to add
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_sparse_value_append (
   Rox_DynVec_Sparse_Value ptr, 
   Rox_Sparse_Value_Struct * elem
);

//! @}

#endif
