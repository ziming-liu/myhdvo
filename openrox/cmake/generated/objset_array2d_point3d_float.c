//==============================================================================
//
//    OPENROX   : File objset_array2d_point3d_float.c
//
//    Contents  : Implementation of objset_array2d_point3d_float module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S.
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#include "generated/objset_array2d_point3d_float.h"
#include "generated/objset_array2d_point3d_float_struct.h"
#include <string.h>
#include <inout/system/errors_print.h>

Rox_ErrorCode rox_objset_array2d_point3d_float_new(Rox_ObjSet_Array2D_Point3D_Float * obj, Rox_Uint allocblocks)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   Rox_ObjSet_Array2D_Point3D_Float ret_objset;
   if (!obj) {error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error)}

   *obj = NULL;

   // Allocate structure
   ret_objset = (Rox_ObjSet_Array2D_Point3D_Float) rox_memory_allocate(sizeof(struct Rox_ObjSet_Array2D_Point3D_Float_Struct), 1);
   if (!ret_objset) {error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error)}

   // Allocate data
   ret_objset->data = (Rox_Array2D_Point3D_Float*) rox_memory_allocate(sizeof(Rox_Array2D_Point3D_Float), allocblocks);
   if (!ret_objset->data)
   {
      rox_memory_delete(ret_objset);
      {error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error)}
   }

   // Set information
   ret_objset->used = 0;
   ret_objset->allocated = allocblocks;
   ret_objset->allocblocks = allocblocks;

   *obj = ret_objset;

function_terminate:
   return error;
}

Rox_ErrorCode rox_objset_array2d_point3d_float_del(Rox_ObjSet_Array2D_Point3D_Float * ptr)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_ObjSet_Array2D_Point3D_Float todel = NULL;

   if ( !ptr ) 
   { error = ROX_ERROR_NULL_POINTER; goto function_terminate; }

   todel = *ptr;
   *ptr = NULL;

   if ( !todel ) 
   { error = ROX_ERROR_NULL_POINTER; goto function_terminate; }

   if (todel->data)
   {
      for ( Rox_Uint id = 0; id < todel->used; id++)
      {
         // rox_array2d_point3d_float_del ( &todel->data[id] );
         rox_array2d_point3d_float_del ( &todel->data[id] );
      }
      rox_memory_delete(todel->data);
   }

   rox_memory_delete(todel);

function_terminate:
   return error;
}

Rox_ErrorCode rox_objset_array2d_point3d_float_get_data_pointer ( Rox_Array2D_Point3D_Float ** ptr_data, Rox_ObjSet_Array2D_Point3D_Float ptr )
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   if ( !ptr || !ptr_data )
   { error = ROX_ERROR_NULL_POINTER; goto function_terminate; }

   *ptr_data = ptr->data;

function_terminate:
   return error;
}

Rox_ErrorCode rox_objset_array2d_point3d_float_get_used ( Rox_Uint * used, Rox_ObjSet_Array2D_Point3D_Float ptr)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   if (!ptr) { error = ROX_ERROR_NULL_POINTER; goto function_terminate; }

   *used = ptr->used;


function_terminate:
   return error;
}


Rox_ErrorCode rox_objset_array2d_point3d_float_reset(Rox_ObjSet_Array2D_Point3D_Float ptr)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   if (!ptr){ error = ROX_ERROR_NULL_POINTER; goto function_terminate; }

   for ( Rox_Uint id = 0; id < ptr->used; id++)
   {
      // error = rox_array2d_point3d_float_del ( &ptr->data[id] );
      error = rox_array2d_point3d_float_del(&ptr->data[id]);
      if(error)goto function_terminate;
   }

   ptr->used = 0;

function_terminate:
   return error;
}

Rox_ErrorCode rox_objset_array2d_point3d_float_stack(Rox_ObjSet_Array2D_Point3D_Float ptr, Rox_ObjSet_Array2D_Point3D_Float other)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   if (!ptr || !other) {error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error)}
   if (other->used == 0) {error = ROX_ERROR_NONE; goto function_terminate;}

   Rox_Uint lastcount = ptr->used;
   ptr->used = ptr->used + other->used;

   if (ptr->used >= ptr->allocated)
   {
      // Be sure we allocate a multiple of allocblocks
      Rox_Uint blockused = ptr->used / ptr->allocblocks;
      ptr->allocated = (blockused + 1) * ptr->allocblocks;

      // Update memory allocation
      ptr->data = (Rox_Array2D_Point3D_Float*)rox_memory_reallocate(ptr->data, sizeof(Rox_Array2D_Point3D_Float), ptr->allocated);
   }

   memcpy(&ptr->data[lastcount], other->data, sizeof(Rox_Array2D_Point3D_Float) * other->used);

function_terminate:
   return error;
}

Rox_ErrorCode rox_objset_array2d_point3d_float_clone(Rox_ObjSet_Array2D_Point3D_Float ptr, Rox_ObjSet_Array2D_Point3D_Float source)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   if (!ptr || !source) {error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error)}

   ptr->used = source->used;

   if (ptr->allocated != source->allocated)
   {
      // Be sure we allocate a multiple of allocblocks
      ptr->allocated = source->allocated;

      // Update memory allocation
      ptr->data = (Rox_Array2D_Point3D_Float*)rox_memory_reallocate(ptr->data, sizeof(Rox_Array2D_Point3D_Float), ptr->allocated);
   }

   memcpy(ptr->data, source->data, sizeof(Rox_Array2D_Point3D_Float) * source->used);

function_terminate:
   return error;
}

Rox_ErrorCode rox_objset_array2d_point3d_float_append(Rox_ObjSet_Array2D_Point3D_Float ptr, Rox_Array2D_Point3D_Float data)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Uint current_index = 0;

   if (!ptr) {error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error)}
   if (!data) {error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error)}

   //increment used slots
   current_index = ptr->used;
   ptr->used++;

   //resize if needed
   if (current_index >= ptr->allocated)
   {
      // Be sure we allocate a multiple of allocblocks
      Rox_Uint blockused = ptr->used / ptr->allocblocks;
      ptr->allocated = (blockused + 1) * ptr->allocblocks;

      // Update memory allocation
      ptr->data = (Rox_Array2D_Point3D_Float*)rox_memory_reallocate(ptr->data, sizeof(Rox_Array2D_Point3D_Float), ptr->allocated);
   }

   //assign value to store
   ptr->data[current_index] = data;

function_terminate:
   return error;
}
