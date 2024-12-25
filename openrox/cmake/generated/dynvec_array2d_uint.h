//==============================================================================
//
//    OPENROX   : File dynvec_array2d_uint.h
//
//    Contents  : API of dynvec_array2d_uint module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S.
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#ifndef __OPENROX_DYNVEC_Array2D_Uint__
#define __OPENROX_DYNVEC_Array2D_Uint__

#include <system/memory/datatypes.h>
#include <system/memory/memory.h>
#include <system/errors/errors.h>

#include <generated/array2d_uint.h>
#include <generated/array2d_uint.h>


//! \ingroup DynVec
//! \defgroup DynVec_Array2D_Uint DynVec_Array2D_Uint
//! @{

//! Pointer to dynvec structure
typedef struct Rox_DynVec_Array2D_Uint_Struct * Rox_DynVec_Array2D_Uint;

//! Create a new dynamic array structure
//!
//! \param  [out]  obj            Address of the pointer for the newly created object
//! \param  [in ]  allocblocks    Number of additional cells allocated when maximum capacity is reached, also the number of cells pre-allocated
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_array2d_uint_new (
   Rox_DynVec_Array2D_Uint * obj, 
   Rox_Uint allocblocks
);

//! Delete a dynvec structure
//!
//! \param  [out]  ptr            The pointer to the dynamic array to be deleted
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_array2d_uint_del (
   Rox_DynVec_Array2D_Uint * ptr
);

//! Retrieve the list of pointers of a vector
//!
//! \param [in]  ptr          The dynamic array to get data from
//! \return the pointer
ROX_API Rox_ErrorCode rox_dynvec_array2d_uint_get_data_pointer (
   Rox_Array2D_Uint ** data_pointer, 
   Rox_DynVec_Array2D_Uint ptr
);

//! Retrieve the number of blocks used in vector
//!
//! \param  [out]  nb_used        The size of the vector
//! \param  [in ]  ptr            The dynamic array to get data from
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_array2d_uint_get_used (
   Rox_Uint * nb_used, Rox_DynVec_Array2D_Uint ptr
);

//! Reset the counter of used blocks of a dynvec structure
//!
//! \param  [out]  ptr            The dynamic array to reset
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_array2d_uint_reset (
   Rox_DynVec_Array2D_Uint ptr
);

//! Increase the counter of used blocks of a dynvec structure, allocate memory if needed
//! Warning: if the counter of used blocks is a multiple of allocblocks, the allocated memory will actually be of size ( used + allocblocks )
//!
//! \param  [out]  ptr            The dynamic array to update
//! \param  [in ]  nbcells        The number of additional cells which have been used newly
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_array2d_uint_usecells (
   Rox_DynVec_Array2D_Uint ptr, 
   Rox_Uint nbcells
);

//!  Requests that the vector capacity be at least enough to contain n elements
//!
//! \param  [out]  ptr            The dynamic array to update
//! \param  [in ]  nbcells        Minimum capacity for the vector
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_array2d_uint_reserve (
   Rox_DynVec_Array2D_Uint ptr, 
   Rox_Uint nbcells
);

//! Stack two vectors together
//!
//! \param  [out]  ptr             The dynamic array to update
//! \param  [in ]  other           The dynamic array to add after
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_array2d_uint_stack (
   Rox_DynVec_Array2D_Uint ptr, 
   Rox_DynVec_Array2D_Uint other
);

//! Clone one vector
//!
//! \param  [out]  ptr            The dynamic array to update with other array
//! \param  [in ]  source         The dynamic array to clone
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_array2d_uint_clone (
   Rox_DynVec_Array2D_Uint ptr, 
   Rox_DynVec_Array2D_Uint source
);

//! Add an element to the vector, allocate memory if needed
//! Warning: if the counter of used blocks is a multiple of allocblocks, the allocated memory will actually be of size ( used + allocblocks )
//!
//! \param  [out]  ptr            The dynamic array to update with other array
//! \param  [in ]  data           The value to add
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_array2d_uint_append (
   Rox_DynVec_Array2D_Uint ptr, 
   Rox_Array2D_Uint * elem
);

//! @}

#endif
