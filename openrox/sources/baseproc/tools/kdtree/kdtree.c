//==============================================================================
//
//    OPENROX   : File kdtree.c
//
//    Contents  : Implementation of kdtree module
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S. and Inria
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

#include "kdtree.h"
#include "kdtree_heap_branch.h"

#include <string.h>

#include <baseproc/maths/random/random.h>
#include <baseproc/array/shuffle/shuffle.h>
#include <baseproc/geometry/point/point3d_tools.h>
#include <inout/system/errors_print.h>

Rox_ErrorCode rox_kdtree_node_new(Rox_Kdtree_Node * kdtree_node);
Rox_ErrorCode rox_kdtree_node_del(Rox_Kdtree_Node * kdtree_node);

Rox_ErrorCode rox_kdtree_node_new(Rox_Kdtree_Node * kdtree_node)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Kdtree_Node ret = NULL;

   if (!kdtree_node) 
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error); }

   *kdtree_node = NULL;

   ret = (Rox_Kdtree_Node) rox_memory_allocate(sizeof(*ret), 1);
   if (!ret) 
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error); }

   ret->_child_left = NULL;
   ret->_child_right = NULL;

   *kdtree_node = ret;

function_terminate:
   if (error) rox_memory_delete(ret);
   return error;
}

Rox_ErrorCode rox_kdtree_node_del(Rox_Kdtree_Node * kdtree_node)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Kdtree_Node todel = NULL;

   if (!kdtree_node) 
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error); }

   todel = *kdtree_node;
   if (!todel) 
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error); }

   *kdtree_node = NULL;

   rox_kdtree_node_del(&todel->_child_left);
   rox_kdtree_node_del(&todel->_child_right);

   rox_memory_delete(todel);

function_terminate:
   return error;
}

Rox_ErrorCode rox_kdtree_new(Rox_Kdtree * kdtree, Rox_Sint dimension)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Kdtree ret = NULL;

   if (!kdtree) 
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error); }

   if (dimension < 1) 
   { error = ROX_ERROR_BAD_SIZE; ROX_ERROR_CHECK_TERMINATE(error); }

   *kdtree = NULL;

   ret = (Rox_Kdtree) rox_memory_allocate(sizeof(*ret), 1);
   if (!ret) { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error); }

   ret->_count_leaves = 0;
   ret->_roots = NULL;
   ret->_checked = NULL;
   
   ret->_heap = NULL; 

   ret->_count_trees = dimension;

   error = rox_dynvec_uint_new(&ret->_checked, 100);
   ROX_ERROR_CHECK_TERMINATE ( error );

   ret->_roots = (Rox_Kdtree_Node*) rox_memory_allocate(sizeof(Rox_Kdtree_Node), dimension);

   if (!ret->_roots) 
   {error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error );}

   // Set all pointers to NULL
   for (Rox_Sint idtree = 0; idtree < dimension; idtree++)
   {
      ret->_roots[idtree] = NULL;
   }

   *kdtree = ret;

function_terminate:
   if (error) rox_kdtree_del(&ret);
   return error;
}

Rox_ErrorCode rox_kdtree_clean(Rox_Kdtree kdtree)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   if (!kdtree) 
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error); }

   rox_dynvec_uint_reset(kdtree->_checked);
   rox_kdtree_heap_branch_del(&kdtree->_heap); 

   for (Rox_Uint idtree = 0; idtree < kdtree->_count_trees; idtree++)
   {
      rox_kdtree_node_del(&kdtree->_roots[idtree]);
   }

function_terminate:
   return error;
}

Rox_ErrorCode rox_kdtree_del(Rox_Kdtree * kdtree)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Kdtree todel = NULL;

   if (!kdtree) 
   {error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error)}

   todel = *kdtree;
   if (!todel) 
   {error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error)}

   *kdtree = NULL;

   rox_kdtree_heap_branch_del(&todel->_heap);  

   rox_dynvec_uint_del(&todel->_checked);

   if (todel->_roots)
   {
      rox_kdtree_clean(todel);
      rox_memory_delete(todel->_roots);
   }

   rox_memory_delete(todel);

function_terminate:
   return error;
}


Rox_ErrorCode rox_kdtree_node_split(Rox_Kdtree_Node kdtree_node, Rox_DynVec_Point3D_Double features, Rox_Uint * indices, Rox_Uint indices_count)
{
   Rox_Sint count_features;
   Rox_Sint id_idx, index, elem;
   Rox_Double feat_sum[3];
   Rox_Double feat_mean[3];
   Rox_Double feat_var[3];
   Rox_Float denom, sigma;
   Rox_Sint topidx[TOP_RAND];
   Rox_Sint pos, cut_dim, cut_val;
   Rox_Sint lim1, lim2, left, right, swap, halfcount;
   Rox_Sint j;

   memset(feat_sum, 0, sizeof(Rox_Double) * 3);

   count_features = indices_count;
   if (count_features > MEAN_SAMPLES) count_features = MEAN_SAMPLES;

   // Compute sum of set of features
   for (id_idx = 0; id_idx < count_features; id_idx++)
   {
      index = indices[id_idx];
   
      feat_sum[0] += features->data[index].X;
      feat_sum[1] += features->data[index].Y;
      feat_sum[2] += features->data[index].Z;
   }

   // Compute mean of set of features
   denom = 1.0f / count_features;
   for (elem = 0; elem < 3; elem++)
   {
      feat_mean[elem] = denom * ((Rox_Float) feat_sum[elem]);
      feat_var[elem] = 0;
   }

   // Compute variance*/
   for (id_idx = 0; id_idx < count_features; id_idx++)
   {
      index = indices[id_idx];

      sigma = features->data[index].X - feat_mean[0];
      feat_var[0] += sigma * sigma;
      sigma = features->data[index].Y - feat_mean[1];
      feat_var[1] += sigma * sigma;
      sigma = features->data[index].Z - feat_mean[2];
      feat_var[2] += sigma * sigma;
   }

   pos = 0;
   for (elem = 0; elem < 3; elem++)
   {
      if ((pos < TOP_RAND) || (feat_var[elem] > feat_var[topidx[pos - 1]]))
      {
         if (pos < TOP_RAND)
         {
            topidx[pos++] = elem;
         }
         else
         {
            topidx[pos-1] = elem;
         }
         j = pos - 1;
         while (j > 0 && feat_var[topidx[j]] > feat_var[topidx[j - 1]])
         {
            swap = topidx[j];
            topidx[j] = topidx[j - 1];
            topidx[j - 1] = swap;
            --j;
         }
      }
   }

   // Select one random cutting dimension among the top_rand dimensions
   cut_dim = topidx[rox_rand() % TOP_RAND];

   // rox_log("Select one random cutting dimension among the top_rand dimensions = %d\n", cut_dim);

   // Separation point on this dimension is the mean
   cut_val = feat_mean[cut_dim];

   // Sort indices so that they respect the subdivision cut
   left = 0;
   right = indices_count - 1;
   while (1)
   {
      //while (left <= right && features->data[indices[left]].descriptor[cut_dim] < cut_val)
      //   ++left;
      //while (left <= right && features->data[indices[right]].descriptor[cut_dim] >= cut_val)
      //   --right;

      if (cut_dim == 0)
      {
         while (left <= right && features->data[indices[left]].X < cut_val)
           ++left;
         while (left <= right && features->data[indices[right]].X >= cut_val)
           --right;
      }
      if (cut_dim == 1)
      {
         while (left <= right && features->data[indices[left]].Y < cut_val)
           ++left;
         while (left <= right && features->data[indices[right]].Y >= cut_val)
           --right;
      }
      if (cut_dim == 2)
      {
         while (left <= right && features->data[indices[left]].Z < cut_val)
           ++left;
         while (left <= right && features->data[indices[right]].Z >= cut_val)
           --right;
      }

      if (left > right) break;

      swap = indices[left];
      indices[left] = indices[right];
      indices[right] = swap;
      left++;
      right--;
   }
   lim1 = left;
   right = indices_count - 1;

   while (1)
   {
      //while (left <= right && features->data[indices[left]].descriptor[cut_dim] <= cut_val)
      //   ++left;
      //while (left <= right && features->data[indices[right]].descriptor[cut_dim] > cut_val)
      //   --right;

      if (cut_dim == 0)
      {
         while (left <= right && features->data[indices[left]].X <= cut_val)
           ++left;
         while (left <= right && features->data[indices[right]].X > cut_val)
           --right;
      }

      if (cut_dim == 1)
      {
         while (left <= right && features->data[indices[left]].Y <= cut_val)
           ++left;
         while (left <= right && features->data[indices[right]].Y > cut_val)
           --right;
      }

      if (cut_dim == 2)
      {
         while (left <= right && features->data[indices[left]].Z <= cut_val)
           ++left;
         while (left <= right && features->data[indices[right]].Z > cut_val)
           --right;
      }

      if (left > right) break;

      swap = indices[left];
      indices[left] = indices[right];
      indices[right] = swap;
      left++;
      right--;
   }
   lim2 = left;

   halfcount = indices_count / 2;
   if (lim1 > halfcount) index = lim1;
   else if (lim2 < halfcount) index = lim2;
   else index = halfcount;

   if (lim1 == indices_count) index = halfcount;
   else if (lim2 == 0) index = halfcount;

   kdtree_node->_index = index;
   kdtree_node->_cut_index = cut_dim;
   kdtree_node->_cut_val = cut_val;

   // rox_log("kdtree_node->_cut_index = %d \n", kdtree_node->_cut_index);

   return ROX_ERROR_NONE;
}

Rox_ErrorCode rox_kdtree_node_new_from_list(Rox_Kdtree_Node * kdtree_node, Rox_DynVec_Point3D_Double features, Rox_Uint * indices, Rox_Uint indices_count)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Kdtree_Node node;

   error = rox_kdtree_node_new(kdtree_node);
   ROX_ERROR_CHECK_TERMINATE ( error );

   node = *kdtree_node;

   // rox_log("indices_count = %d \n", indices_count);

   if (indices_count == 0)
   {
      {error = ROX_ERROR_BAD_SIZE; ROX_ERROR_CHECK_TERMINATE(error);}
   }
   else if (indices_count == 1)
   {
      node->_index = indices[0];
      node->_child_left = NULL;
      node->_child_right = NULL;
   }
   else
   {
      error = rox_kdtree_node_split(node, features, indices, indices_count);
      ROX_ERROR_CHECK_TERMINATE ( error );

      error = rox_kdtree_node_new_from_list(&node->_child_left, features, indices, node->_index);
      ROX_ERROR_CHECK_TERMINATE ( error );

      error = rox_kdtree_node_new_from_list(&node->_child_right, features, indices + node->_index, indices_count - node->_index);
      ROX_ERROR_CHECK_TERMINATE ( error );
   }

function_terminate:
   return error;
}

Rox_ErrorCode rox_kdtree_build(Rox_Kdtree kdtree, Rox_DynVec_Point3D_Double features)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_DynVec_Uint indices;

   // clean previous data in kdtree
   rox_kdtree_clean(kdtree);

   // The number of leaves will be the same number of features
   kdtree->_count_leaves = features->used;

   rox_dynvec_uint_reset(kdtree->_checked);
   error = rox_dynvec_uint_usecells(kdtree->_checked, kdtree->_count_leaves);
   ROX_ERROR_CHECK_TERMINATE ( error );

   error = rox_kdtree_heap_branch_new(&kdtree->_heap, kdtree->_count_leaves);
   ROX_ERROR_CHECK_TERMINATE ( error );

   error = rox_dynvec_uint_new(&indices, 100);
   ROX_ERROR_CHECK_TERMINATE ( error );

   for ( Rox_Uint id_feat = 0; id_feat < kdtree->_count_leaves; id_feat++)
   {
      rox_dynvec_uint_append(indices, &id_feat);
   }

   for (Rox_Sint id_tree = 0; id_tree < kdtree->_count_trees; id_tree++)
   {
      // Randomize indices of features
      shuffle_array_uint(indices->data, indices->used);

      // Build a new kd-tree
      error = rox_kdtree_node_new_from_list(&kdtree->_roots[id_tree], features, indices->data, indices->used);
      if (error) break;
   }

function_terminate:
   rox_dynvec_uint_del(&indices);
   return error;
}

Rox_ErrorCode rox_kdtree_search_node(Rox_DynVec_Point3D_Double results, Rox_Kdtree_Node kdtree_node, Rox_DynVec_Point3D_Double features, Rox_Point3D_Double feat, Rox_DynVec_Uint checked, Rox_Uint * checks, Rox_Uint maxchecks, Rox_Double mindist, Rox_Kdtree_Heap_Branch heap)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Double diff = 0.0;
   Rox_Double dist = 0.0;
   Rox_Point3D_Double featdb = NULL;
   Rox_Kdtree_Node best, other;

   if (kdtree_node == NULL) 
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE(error); }
   
   //rox_log("Cut index = %d \n", kdtree_node->_cut_index);

   //rox_log("Did we reach a leaf node ? \n");

   // Did we reach a leaf node ?
   if (kdtree_node->_child_left == NULL || kdtree_node->_child_right == NULL)
   {
      //rox_log("YES and *checks = %d features = %d \n", *checks, features->used);
      if ( (*checks >= maxchecks) || (*checks >= features->used) )// && rox_sraid_matchresultset_isfull(results)) 
      { error = ROX_ERROR_NONE; goto function_terminate; }

      if (checked->data[kdtree_node->_index]) 
      { error = ROX_ERROR_NONE; goto function_terminate; }

      checked->data[kdtree_node->_index] = 1;
      *checks = (*checks) + 1;

      // Compute full distance between searched feature and indexed feature
      featdb = &features->data[kdtree_node->_index];
      
      // dist = rox_sraid_match(featdb->descriptor, feat->descriptor);
      
      error = rox_point3d_double_distance ( &dist,  featdb, feat);
      ROX_ERROR_CHECK_TERMINATE(error);


      // rox_sraid_matchresultset_addresult(results, kdtree_node->_index, dist);
      error = rox_dynvec_point3d_double_append(results, featdb);
      ROX_ERROR_CHECK_TERMINATE(error);

      //rox_log("distance = %f \n", dist);
      error = ROX_ERROR_NONE; goto function_terminate;
   }

   // Space was divided in two : In which half should our feature lie considering only this dimension ?
   // By implementation choice, left child is the lower half.

   if (kdtree_node->_cut_index == 0)
   {
      diff = feat->X - kdtree_node->_cut_val;
   }
   if (kdtree_node->_cut_index == 1)
   {
      diff = feat->Y - kdtree_node->_cut_val;
   }
   if (kdtree_node->_cut_index == 2)
   {
      diff = feat->Z - kdtree_node->_cut_val;
   }


   // diff = ((Rox_Sint) feat->descriptor[kdtree_node->_index]) - kdtree_node->_cut_val;
   if (diff < 0)
   {
      best = kdtree_node->_child_left;
      other = kdtree_node->_child_right;
   }
   else
   {
      best = kdtree_node->_child_right;
      other = kdtree_node->_child_left;
   }

   // Maybe considering only this dimensions guide us to a wrong subspace, keep the branch if needed in memory for more results
   dist = mindist + diff * diff;

   error = rox_kdtree_heap_branch_push(heap, other, dist);
   ROX_ERROR_CHECK_TERMINATE ( error );

   // Recurse using the best children
   error = rox_kdtree_search_node(results, best, features, feat, checked, checks, maxchecks, mindist, heap);
   ROX_ERROR_CHECK_TERMINATE ( error );

function_terminate:
   return error;
}

Rox_ErrorCode rox_kdtree_search(Rox_DynVec_Point3D_Double results, Rox_Kdtree kdtree, Rox_DynVec_Point3D_Double features, Rox_Point3D_Double feat)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   const Rox_Uint max_checks = 32;
   Rox_Double score = 0.0;
   Rox_Uint checks = 0;
   Rox_Kdtree_Node node;

   if (!kdtree || !features || !feat) 
   { error = ROX_ERROR_NULL_POINTER; ROX_ERROR_CHECK_TERMINATE ( error ); }

   // rox_matchresultset_clear(results); 
   rox_dynvec_point3d_double_reset(results);
   rox_kdtree_heap_branch_reset(kdtree->_heap);

   // Reset checked flags
   memset(kdtree->_checked->data, 0, sizeof(Rox_Uint) * features->used);

   //rox_log("Try to find closest features for all %d trees \n", kdtree->_count_trees);

   // Try to find closest features for all trees
   for (Rox_Sint id_tree = 0; id_tree < kdtree->_count_trees; id_tree++)
   {
      error = rox_kdtree_search_node(results, kdtree->_roots[id_tree], features, feat, kdtree->_checked, &checks, max_checks, 0.0, kdtree->_heap);
      if (error) break;
   }

if(1)   
{
   rox_log("Make some new trials based on potentially erroneous branching \n");
   getchar();

   // Make some new trials based on potentially erroneous branching
   error = rox_kdtree_heap_branch_pop(kdtree->_heap, &node, &score);
   while (error == ROX_ERROR_NONE && (checks < max_checks || checks < features->used) ) // && (rox_sraid_matchresultset_isfull(results) == 0 )
   {
      error = rox_kdtree_search_node(results, node, features, feat, kdtree->_checked, &checks, max_checks, score, kdtree->_heap);
      if (error) break;

      error = rox_kdtree_heap_branch_pop(kdtree->_heap, &node, &score);
   }
}
   ROX_ERROR_CHECK_TERMINATE ( error );

function_terminate:
   return error;
}
