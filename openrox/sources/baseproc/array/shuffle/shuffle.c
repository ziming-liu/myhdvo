//==============================================================================
//
//    OPENROX   : File shuffle.c
//
//    Contents  : Implementation of shuffle module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S. 
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#include "shuffle.h"

void shuffle_array_uint(Rox_Uint * array, Rox_Uint size)
{
   Rox_Uint buf = 0;

   int pos = size - 1;
   while (pos > 0)
   {
      int id = rox_rand() % pos;
      buf = array[id];
      array[id] = array[pos];
      array[pos] = buf;

      pos--;
   }
}
