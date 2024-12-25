//==============================================================================
//
//    OPENROX   : File dynvec_rect_sint.h
//
//    Contents  : API of dynvec_rect_sint module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S.
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#ifndef __OPENROX_DYNVEC_Rect_Sint__
#define __OPENROX_DYNVEC_Rect_Sint__

#include <system/memory/datatypes.h>
#include <system/memory/memory.h>
#include <system/errors/errors.h>

#include <baseproc/geometry/rectangle/rectangle.h>
#include <baseproc/geometry/rectangle/rectangle_struct.h>


//! \ingroup DynVec
//! \defgroup DynVec_Rect_Sint DynVec_Rect_Sint
//! @{

//! Pointer to dynvec structure
typedef struct Rox_DynVec_Rect_Sint_Struct * Rox_DynVec_Rect_Sint;

//! Create a new dynamic array structure
//!
//! \param  [out]  obj            Address of the pointer for the newly created object
//! \param  [in ]  allocblocks    Number of additional cells allocated when maximum capacity is reached, also the number of cells pre-allocated
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_rect_sint_new (
   Rox_DynVec_Rect_Sint * obj, 
   Rox_Uint allocblocks
);

//! Delete a dynvec structure
//!
//! \param  [out]  ptr            The pointer to the dynamic array to be deleted
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_rect_sint_del (
   Rox_DynVec_Rect_Sint * ptr
);

//! Retrieve the list of pointers of a vector
//!
//! \param [in]  ptr          The dynamic array to get data from
//! \return the pointer
ROX_API Rox_ErrorCode rox_dynvec_rect_sint_get_data_pointer (
   Rox_Rect_Sint_Struct ** data_pointer, 
   Rox_DynVec_Rect_Sint ptr
);

//! Retrieve the number of blocks used in vector
//!
//! \param  [out]  nb_used        The size of the vector
//! \param  [in ]  ptr            The dynamic array to get data from
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_rect_sint_get_used (
   Rox_Uint * nb_used, Rox_DynVec_Rect_Sint ptr
);

//! Reset the counter of used blocks of a dynvec structure
//!
//! \param  [out]  ptr            The dynamic array to reset
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_rect_sint_reset (
   Rox_DynVec_Rect_Sint ptr
);

//! Increase the counter of used blocks of a dynvec structure, allocate memory if needed
//! Warning: if the counter of used blocks is a multiple of allocblocks, the allocated memory will actually be of size ( used + allocblocks )
//!
//! \param  [out]  ptr            The dynamic array to update
//! \param  [in ]  nbcells        The number of additional cells which have been used newly
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_rect_sint_usecells (
   Rox_DynVec_Rect_Sint ptr, 
   Rox_Uint nbcells
);

//!  Requests that the vector capacity be at least enough to contain n elements
//!
//! \param  [out]  ptr            The dynamic array to update
//! \param  [in ]  nbcells        Minimum capacity for the vector
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_rect_sint_reserve (
   Rox_DynVec_Rect_Sint ptr, 
   Rox_Uint nbcells
);

//! Stack two vectors together
//!
//! \param  [out]  ptr             The dynamic array to update
//! \param  [in ]  other           The dynamic array to add after
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_rect_sint_stack (
   Rox_DynVec_Rect_Sint ptr, 
   Rox_DynVec_Rect_Sint other
);

//! Clone one vector
//!
//! \param  [out]  ptr            The dynamic array to update with other array
//! \param  [in ]  source         The dynamic array to clone
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_rect_sint_clone (
   Rox_DynVec_Rect_Sint ptr, 
   Rox_DynVec_Rect_Sint source
);

//! Add an element to the vector, allocate memory if needed
//! Warning: if the counter of used blocks is a multiple of allocblocks, the allocated memory will actually be of size ( used + allocblocks )
//!
//! \param  [out]  ptr            The dynamic array to update with other array
//! \param  [in ]  data           The value to add
//! \return An error code
ROX_API Rox_ErrorCode rox_dynvec_rect_sint_append (
   Rox_DynVec_Rect_Sint ptr, 
   Rox_Rect_Sint_Struct * elem
);

//! @}

#endif
