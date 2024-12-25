/*
 * @Author: Ziming Liu
 * @Date: 2022-06-21 15:41:53
 * @LastEditors: Ziming
 * @LastEditTime: 2022-06-26 02:58:58
 * @Description: reference the "examples/rox_example_odometry_dense_depthmap.c"
 * @Dependent packages: don't need any extral dependency
 */

//====== INCLUDED HEADERS   ====================================================
#include <stdio.h>
#include <baseproc/array/conversion/array2d_float_from_uchar.h>
#include <baseproc/array/fill/fillval.h>
#include <baseproc/array/inverse/inverse.h>
#include <baseproc/maths/linalg/matse3.h>
#include <generated/array2d_double.h>
#include <baseproc/array/conversion/array2d_double_from_uchar.h>
#include <baseproc/array/conversion/array2d_uint_from_uchar.h>
#include <core/odometry/depth/odometry_dense_depthmap.h>
#include <inout/numeric/scalar_save.h>
#include <inout/numeric/array_save.h>
#include <inout/numeric/array2d_save.h>
#include <inout/system/print.h>

Rox_MatSE3  ddo(Rox_Array2D_Uchar Ir_uchar, Rox_Array2D_Uchar Z_uchar, Rox_Array2D_Uchar Ic_uchar, Rox_Array2D_Uchar imask_uchar, Rox_Array2D_Uchar cTr_given, Rox_Array2D_Uchar Kuchar);


Rox_Sint get_size_row(Rox_Array2D_Double input);

Rox_Sint get_size_col(Rox_Array2D_Double input);