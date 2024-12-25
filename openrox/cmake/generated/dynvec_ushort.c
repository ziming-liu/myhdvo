//==============================================================================
//
//    OPENROX   : File dynvec_ushort.c
//
//    Contents  : Implementation of dynvec_ushort module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S.
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#include "generated/dynvec_ushort.h"
#include "generated/dynvec_ushort_struct.h"
#include <string.h>

#include <system/memory/datatypes.h>

#include <inout/system/errors_print.h>

Rox_ErrorCode rox_dynvec_ushort_new(Rox_DynVec_Ushort * obj, Rox_Uint allocblocks)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_DynVec_Ushort ret_dynvec = NULL;

   if (!obj)
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error ); }

   *obj = NULL;

   // Allocate structure
   ret_dynvec = (Rox_DynVec_Ushort) rox_memory_allocate(sizeof(struct Rox_DynVec_Ushort_Struct), 1);
   if (!ret_dynvec)
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error ); }

   ret_dynvec->data = NULL;

   // Allocate data
   ret_dynvec->data = (Rox_Ushort *) rox_memory_allocate(sizeof(Rox_Ushort), allocblocks);
   if (!ret_dynvec->data)
   {
      rox_memory_delete(ret_dynvec);
      { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error ); }
   }

   // Set information
   ret_dynvec->used        = 0;
   ret_dynvec->allocated   = allocblocks;
   ret_dynvec->allocblocks = allocblocks;

   *obj = ret_dynvec;

function_terminate:
   return error;
}

Rox_ErrorCode rox_dynvec_ushort_del(Rox_DynVec_Ushort * ptr)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   Rox_DynVec_Ushort todel = NULL;

   if (!ptr)
   { error = ROX_ERROR_NULL_POINTER; goto function_terminate; }

   todel = *ptr;
   *ptr = NULL;

   if (!todel)
   { error = ROX_ERROR_NULL_POINTER; goto function_terminate; }

   if (todel->data)
   {
      rox_memory_delete(todel->data);
   }

   rox_memory_delete(todel);

function_terminate:
   return error;
}


Rox_ErrorCode rox_dynvec_ushort_get_data_pointer ( Rox_Ushort ** data_pointer, Rox_DynVec_Ushort ptr )
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
  
   if (!ptr)
   { error = ROX_ERROR_NULL_POINTER; goto function_terminate; }

   *data_pointer = ptr->data;

function_terminate:
   return error;
}


Rox_ErrorCode rox_dynvec_ushort_get_used(Rox_Uint * nb_used, Rox_DynVec_Ushort ptr)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   if (!nb_used)
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error ); }

   if (!ptr)
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error ); }

   *nb_used = ptr->used;

function_terminate:
   return error;
}


Rox_ErrorCode rox_dynvec_ushort_reset(Rox_DynVec_Ushort ptr)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   if (!ptr)
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error ); }

   ptr->used = 0;

function_terminate:
   return error;
}


Rox_ErrorCode rox_dynvec_ushort_usecells(Rox_DynVec_Ushort ptr, Rox_Uint nbcells)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   if (!ptr)
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error ); }

   ptr->used += nbcells;

   if (ptr->used >= ptr->allocated)
   {
      // Be sure we allocate a multiple of allocblocks
      Rox_Uint blockused = ptr->used / ptr->allocblocks;
      ptr->allocated = (blockused + 1) * ptr->allocblocks; //XXX wrong behavior if blockused is a multiple of allocblocks ?

      // Update memory allocation
      ptr->data = (Rox_Ushort *) rox_memory_reallocate(ptr->data, sizeof(Rox_Ushort), ptr->allocated);
   }

function_terminate:
   return error;
}


Rox_ErrorCode rox_dynvec_ushort_reserve(Rox_DynVec_Ushort ptr, Rox_Uint nbcells)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   if (!ptr)
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error ); }

   if (nbcells > ptr->allocated)
   {
      // Be sure we allocate a multiple of allocblocks
      Rox_Uint blocks_needed = nbcells / ptr->allocblocks;
      blocks_needed += ( nbcells % ptr->allocblocks ) ? 1 : 0;

      ptr->allocated = blocks_needed * ptr->allocblocks;

      // Update memory allocation
      ptr->data = (Rox_Ushort *) rox_memory_reallocate(ptr->data, sizeof(Rox_Ushort), ptr->allocated);
   }

function_terminate:
   return error;
}


Rox_ErrorCode rox_dynvec_ushort_stack(Rox_DynVec_Ushort ptr, Rox_DynVec_Ushort other)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   if (!ptr || !other)
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error ); }

   if (other->used == 0)
   { error = ROX_ERROR_NONE; goto function_terminate; }

   Rox_Uint lastcount = ptr->used;
   ptr->used = ptr->used + other->used;

   if (ptr->used >= ptr->allocated)
   {
      // Be sure we allocate a multiple of allocblocks
      Rox_Uint blockused = ptr->used / ptr->allocblocks;
      ptr->allocated = (blockused + 1) * ptr->allocblocks;

      // Update memory allocation
      ptr->data = (Rox_Ushort *) rox_memory_reallocate(ptr->data, sizeof(Rox_Ushort), ptr->allocated);
   }

   memcpy(&ptr->data[lastcount], other->data, sizeof(Rox_Ushort) *other->used);

function_terminate:
   return error;
}


Rox_ErrorCode rox_dynvec_ushort_clone(Rox_DynVec_Ushort ptr, Rox_DynVec_Ushort source)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   if (!ptr || !source)
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error ); }

   ptr->used = source->used;

   if (ptr->allocated != source->allocated)
   {
      // Be sure we allocate a multiple of allocblocks
      ptr->allocated = source->allocated;

      // Update memory allocation
      ptr->data = (Rox_Ushort *) rox_memory_reallocate(ptr->data, sizeof(Rox_Ushort), ptr->allocated);
   }

   memcpy(ptr->data, source->data, sizeof(Rox_Ushort) *source->used);

function_terminate:
   return error;
}


Rox_ErrorCode rox_dynvec_ushort_append(Rox_DynVec_Ushort ptr, Rox_Ushort * elem)
{
   Rox_ErrorCode error         = ROX_ERROR_NONE;
   Rox_Uint      current_index = 0;

   if (!ptr)
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error ); }

   if (!elem)
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error ); }

   //increment used slots
   current_index = ptr->used;
   ptr->used++;

   //resize if needed
   if (current_index >= ptr->allocated)
   {
      // Be sure we allocate a multiple of allocblocks
      Rox_Uint blockused = ptr->used / ptr->allocblocks;
      ptr->allocated = (blockused + 1) * ptr->allocblocks;//XXX wrong behavior if blockused is a multiple of allocblocks ?

      // Update memory allocation
      ptr->data = (Rox_Ushort *) rox_memory_reallocate(ptr->data, sizeof(Rox_Ushort), ptr->allocated);
   }

   //assign value to store
   ptr->data[current_index] = *elem;

function_terminate:
   return error;
}
