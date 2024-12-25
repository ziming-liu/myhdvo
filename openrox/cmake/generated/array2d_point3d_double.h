//============================================================================
//
//    OPENROX   : File array2d_point3d_double.h
//
//    Contents  : API of array2d_point3d_double module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S.
//
//    License   : LGPL v3 or commercial license S.A.S.
//
//============================================================================

#ifndef __OPENROX_ARRAY2D_TYPED_Point3D_Double__
#define __OPENROX_ARRAY2D_TYPED_Point3D_Double__

#ifdef __cplusplus
extern "C" {
#endif

#include <system/memory/array2d.h>

#include <baseproc/geometry/point/point3d_struct.h>

//! \ingroup Array2D
//! @defgroup Array2D_Point3D_Double Array2D_Point3D_Double
//! @{

#ifndef DIRECTINCLUDE_ARRAY2D
   //! pointer to Rox_Array2D structure
   typedef struct _Rox_Array2D_Point3D_Double * Rox_Array2D_Point3D_Double;

   //! pointer to Rox_Array2D structure collection
   typedef struct _Rox_Array2D_Point3D_Double_Collection * Rox_Array2D_Point3D_Double_Collection;
#else
   //! pointer to Rox_Array2D structure
   typedef _Rox_Array2D_Point3D_Double * Rox_Array2D_Point3D_Double;

   //! pointer to Rox_Array2D structure collection
   typedef _Rox_Array2D_Point3D_Double_Collection * Rox_Array2D_Point3D_Double_Collection;
#endif

//! Create a new 2D array
//! \param  [out]  obj            Pointer to created array
//! \param  [in ]  rows           Height of array
//! \param  [in ]  cols           Width of array
//! \return An error code
ROX_API Rox_ErrorCode rox_array2d_point3d_double_new ( 
   Rox_Array2D_Point3D_Double * obj, 
   const Rox_Sint rows, 
   const Rox_Sint cols 
);

//! Delete a 2D array
//! \param  [out]  ptr            pointer to 2D array to delete
//! \return An error code
ROX_API Rox_ErrorCode rox_array2d_point3d_double_del ( 
   Rox_Array2D_Point3D_Double * ptr 
);

//! Create a view on a parent 2D array. Any change on this view will affect the elements in the parent array !
//! Delete the subarray using rox_array2d_point3d_double_del without affect the elements in the parent array !
//! \param  [out]  sub            Destination view
//! \param  [in ]  parent         2D array
//! \param  [in ]  initial_row    First row of the parent array to consider
//! \param  [in ]  initial_col    First col of the parent array to consider
//! \param  [in ]  rows           Number of rows of the parent array to consider
//! \param  [in ]  cols           Number of cols of the parent array to consider
//! \return An error code
ROX_API Rox_ErrorCode rox_array2d_point3d_double_new_subarray2d (
   Rox_Array2D_Point3D_Double * sub, 
   Rox_Array2D_Point3D_Double parent, 
   const Rox_Sint initial_row, 
   const Rox_Sint initial_col, 
   const Rox_Sint rows, 
   const Rox_Sint cols
);

//! Given a view on a parent 2D array and its parent, try to move the view on a different ROI without changing the size of the view. Any change on this view will affect the elements in the parent array !
//! \param  [out]  sub            Destination view
//! \param  [in ]  parent         2D array
//! \param  [in ]  initial_row    First row of the parent array to consider
//! \param  [in ]  initial_col    First col of the parent array to consider
//! \return An error code
ROX_API Rox_ErrorCode rox_array2d_point3d_double_subarray2d_shift (
   Rox_Array2D_Point3D_Double sub, 
   Rox_Array2D_Point3D_Double parent, 
   const Rox_Sint initial_row, 
   const Rox_Sint initial_col
);

//! Create a coalesced memory array of block pointers
//! \param  [in ]  ptr            The array to use
//! \param  [in ]  rows           Rows per blocks
//! \return the list of row pointers
ROX_API Rox_ErrorCode rox_array2d_point3d_double_create_blocksptr (
   Rox_Array2D_Point3D_Double ptr, 
   const Rox_Sint rows
);

//! Retrieve the list of per row pointers of a 2D array
//! \param  [out]  data_pointer   The pointer to the data of the array2d
//! \param  [in ]  array          The array to get data pointer from
//! \return the rows pointers
ROX_API Rox_ErrorCode rox_array2d_point3d_double_get_data_pointer_to_pointer (
   Rox_Point3D_Double_Struct *** data_pointer,
   Rox_Array2D_Point3D_Double array
);

//! Retrieve the list of per row pointers of a 2D array
//! \param  [out]  data_pointer   The pointer to the data of the array2d
//! \param  [in ]  array          The array to get data pointer from
//! \return the rows pointers
ROX_API Rox_ErrorCode rox_array2d_point3d_double_get_data_pointer (
   Rox_Point3D_Double_Struct ** data_pointer,
   Rox_Array2D_Point3D_Double array
);

//! Retrieve the list of per blocks pointers of a 2D array
//! \param  [in ]  array          The array to get data from
//! \return the blocks pointers
ROX_API Rox_ErrorCode rox_array2d_point3d_double_get_blocks (
   Rox_Point3D_Double_Struct **** blocks_pointer,
   Rox_Array2D_Point3D_Double array
);

//! Get the height of 2D array
//! \param  [out]  rows           The pointer to the rows
//! \param  [out]  cols           The pointer to the cols
//! \param  [in ]  array          Input array
//! \return An error code
ROX_API Rox_ErrorCode rox_array2d_point3d_double_get_size (
   Rox_Sint * rows, 
   Rox_Sint * cols, 
   const Rox_Array2D_Point3D_Double array
);

//! Get the height of 2D array
//! \param  [out]  rows           The pointer to the rows
//! \param  [in ]   array          Input array
//! \return An error code
ROX_API Rox_ErrorCode rox_array2d_point3d_double_get_rows (
   Rox_Sint * rows, 
   const Rox_Array2D_Point3D_Double array
);

//! Get the width of 2D array
//! \param  [out]  cols           The pointer to the cols
//! \param  [in ]  array          Input array
//! \return An error code
ROX_API Rox_ErrorCode rox_array2d_point3d_double_get_cols (
   Rox_Sint * cols, 
   const Rox_Array2D_Point3D_Double array
);

//! Get the stride (in bytes) of rows of 2D array
//! \param  [in ]  array          Input array
//! \return the width
ROX_API Rox_ErrorCode rox_array2d_point3d_double_get_stride (
   Rox_Sint * stride, 
   const Rox_Array2D_Point3D_Double array
);

//! Check if two 2D arrays have the same dimension
//! \param  [in ]  array1         Input array
//! \param  [in ]  array2         Input array
//! \return Rox_ErrorNone if match is ok or an errorcode otherwise
ROX_API Rox_ErrorCode rox_array2d_point3d_double_match_size (
   Rox_Array2D_Point3D_Double array1, 
   Rox_Array2D_Point3D_Double array2
);

//! Check if one 2D arrays have the good dimensions
//! \param  [in ]  input          Input array
//! \param  [in ]  height         The height to compare
//! \param  [in ]  width          The width to compare
//! \return Rox_ErrorNone if match is ok or an errorcode otherwise
ROX_API Rox_ErrorCode rox_array2d_point3d_double_check_size (
   Rox_Array2D_Point3D_Double input, 
   const Rox_Sint height, 
   const Rox_Sint width
);

//! Copy an 2D array in another
//! \param  [in ]  dest           Destination array
//! \param  [in ]  source         Source array
//! \return Rox_ErrorNone if match is ok or an errorcode otherwise
ROX_API Rox_ErrorCode rox_array2d_point3d_double_copy (
   Rox_Array2D_Point3D_Double dest, 
   const Rox_Array2D_Point3D_Double source
);

//! Set the value of a 2D array for one element
//! \param  [in ]  array          The array to change
//! \param  [in ]  r              Row to modify
//! \param  [in ]  c              Col to modify
//! \param  [in ]  val            Value to set
//! \return Rox_ErrorNone if dimension and type are ok or an errorcode otherwise
ROX_API Rox_ErrorCode rox_array2d_point3d_double_set_value (
   Rox_Array2D_Point3D_Double array, 
   const Rox_Sint r, 
   const Rox_Sint c, 
   const Rox_Point3D_Double_Struct val
);

//! Set the row of a 2D array 
//! \param  [in ]  array          The array to change
//! \param  [in ]  r              Row to modify
//! \param  [in ]  row            Values to set
//! \return Rox_ErrorNone if dimension and type are ok or an errorcode otherwise
ROX_API Rox_ErrorCode rox_array2d_point3d_double_set_row (
   Rox_Array2D_Point3D_Double array, 
   const Rox_Sint r, 
   const Rox_Array2D_Point3D_Double row
);

//! Set the col of a 2D array 
//! \param  [in ]  array          The array to change
//! \param  [in ]  c              Col to modify
//! \param  [in ]  col            Values to set
//! \return Rox_ErrorNone if dimension and type are ok or an errorcode otherwise
ROX_API Rox_ErrorCode rox_array2d_point3d_double_set_col (
   Rox_Array2D_Point3D_Double array, 
   const Rox_Sint c, 
   const Rox_Array2D_Point3D_Double col
);

//! Get the col of a 2D array 
//! \param  [in ]  array          The array to change
//! \param  [in ]  j              Col to modify
//! \param  [in ]  col            Values to set
//! \return Rox_ErrorNone if dimension and type are ok or an errorcode otherwise
ROX_API Rox_ErrorCode rox_array2d_point3d_double_get_col (
   Rox_Array2D_Point3D_Double col, 
   const Rox_Array2D_Point3D_Double array, 
   const Rox_Sint j
);

//! Set the values of a 2D array given a buffer without per line padding
//! \param  [in ]  array          The array to change
//! \param  [in ]  buf            Buffer to copy
//! \return Rox_ErrorNone if dimension and type are ok or an errorcode otherwise
ROX_API Rox_ErrorCode rox_array2d_point3d_double_set_buffer_no_stride (
   Rox_Array2D_Point3D_Double array, 
   const Rox_Point3D_Double_Struct * buf
);

//! Set the values of a 2D array given a buffer without per line padding
//! and perform a vetical flip
//! \param  [in ]  array          The array with a vertical flip 
//! \param  [in ]  buf            Buffer to copy 
//! \return Rox_ErrorNone if dimension and type are ok or an errorcode otherwise
ROX_API Rox_ErrorCode rox_array2d_point3d_double_set_buffer_no_stride_vertical_flip ( 
   Rox_Array2D_Point3D_Double array, 
   const Rox_Point3D_Double_Struct * buf 
);

//! Get the value of a 2D array without per line padding
//! \param  [out]  buf            Buffer to get values (must be allocated with the right size)
//! \param  [in ]  array          The array to read values from
//! \return An error code
ROX_API Rox_ErrorCode rox_array2d_point3d_double_get_buffer_no_stride (
   Rox_Point3D_Double_Struct * buf, 
   const Rox_Array2D_Point3D_Double array
);

//! Get the value of a 2D array for one double element
//! \param  [out]  val            Value to get
//! \param  [in ]  array          The array to read
//! \param  [in ]  i              Row to read
//! \param  [in ]  j              Col to read
//! \return Rox_ErrorNone if dimension and type are ok or an errorcode otherwise
ROX_API Rox_ErrorCode rox_array2d_point3d_double_get_value (
   Rox_Point3D_Double_Struct *val, 
   Rox_Array2D_Point3D_Double array, 
   const Rox_Sint row, 
   const Rox_Sint col
);

//! Create a new collection of 2D array
//! \param  [out]  obj            A pointer ot the created object
//! \param  [in ]  count          Number of arrays
//! \param  [in ]  rows           Height of array
//! \param  [in ]  cols           Width of array
//! \return An error code
ROX_API Rox_ErrorCode rox_array2d_point3d_double_collection_new (
   Rox_Array2D_Point3D_Double_Collection * obj, 
   const Rox_Sint count, 
   const Rox_Sint rows, 
   const Rox_Sint cols
);

//! Delete a 2D array collection
//! \param  [in ]  ptr            Pointer to 2D array to delete
//! \return An error code
ROX_API Rox_ErrorCode rox_array2d_point3d_double_collection_del (
   Rox_Array2D_Point3D_Double_Collection * ptr
);

//! Retrieve a 2D array from the 2D array collection
//! \param  [in ]  ptr            The collection to use
//! \param  [in ]  id             The id of the array in the collection
//! \return the needed array or 0 if an error occured
ROX_API Rox_Array2D_Point3D_Double rox_array2d_point3d_double_collection_get (
   Rox_Array2D_Point3D_Double_Collection ptr, 
   const Rox_Sint id
);

//! Retrieve the number of arrays from a 2D array collection
//! \param  [in ]  ptr            The collection to use
//! \return the size or 0 if an error occured
ROX_API Rox_Sint rox_array2d_point3d_double_collection_get_count (
   Rox_Array2D_Point3D_Double_Collection ptr
);

//! Cycle to the left the arrays (first array being the last array, etc)
//! \param  [in ]  ptr            The collection to use
//! \return An error code
ROX_API Rox_ErrorCode rox_array2d_point3d_double_collection_cycleleft (
   Rox_Array2D_Point3D_Double_Collection ptr
);

//! Create a 2D array and copy another
//! \param  [out]  output         Destination array
//! \param  [in ]  input          Source array
//! \return Rox_ErrorNone if match is ok or an errorcode otherwise
ROX_API Rox_ErrorCode rox_array2d_point3d_double_new_copy (
   Rox_Array2D_Point3D_Double * output, 
   const Rox_Array2D_Point3D_Double input
);

//! Create a new 2D array from an existing buffer
//! \remarks buffer must be aligned on ROX_DEFAULT_ALIGNMENT
//! \param  [out]  obj            Pointer to created array
//! \param  [in ]  rows           Height of array
//! \param  [in ]  cols           Width of array
//! \param  [in ]  buffer         Pre-allocated aligned buffer
//! \return An error code
ROX_API Rox_ErrorCode rox_array2d_point3d_double_new_frombuffer (
   Rox_Array2D_Point3D_Double * obj, 
   const Rox_Sint rows, 
   const Rox_Sint cols, 
   Rox_Void * buffer
);

//! @}

#ifdef __cplusplus
}
#endif

#endif
