//==============================================================================
//
//    OPENROX   : File rox_example_odometry_dense_depthmap.c
//
//    Contents  : A simple example program for odometry with dense depthmap
//
//    Author(s) : Inria ACENTAURI team directed by Ezio MALIS
//
//    Copyright : LGPL or commercial license by Robocortex
//
//==============================================================================

//====== INCLUDED HEADERS   ====================================================

#include <stdio.h>
#include <baseproc/array/conversion/array2d_float_from_uchar.h>
#include <baseproc/array/fill/fillval.h>
#include <baseproc/array/inverse/inverse.h>
#include <baseproc/maths/linalg/matse3.h>
#include <core/odometry/depth/odometry_dense_depthmap.h>
#include <inout/numeric/scalar_save.h>
#include <inout/numeric/array_save.h>
#include <inout/numeric/array2d_save.h>
#include <inout/system/print.h>

//====== INTERNAL MACROS    ====================================================

//#define DEPTH
// #define IMASK

// #define DATA_PATH    "/media/emalis/T7/"
// #define DATA_PATH    "/data/acentauri/user/emalis/openrox/"
#define DATA_PATH "/data/acentauri/user/ziliu/data/openrox_test_data/kitti/"

#define SEQ 10

#define IMAGE_PATH   DATA_PATH"sequences/%02d/image_2/%06d.pgm"
#define IMASK_PATH   DATA_PATH"imasks/%02d/image_2/%06d.pgm"
#define K_PATH       DATA_PATH"sequences/%02d/image_2/intrinsics_parameters.txt"
#define Z_PATH       DATA_PATH"depths/%02d/image_2/txt/%06d.txt"
#define Zi_PATH      DATA_PATH"idepths/%02d/image_2/txt/%06d.txt"

// Seq 0, 1, 2, 13, 14, 15, 16, 17, 18, 19, 20, 21
// #define FU 718.8560
// #define FV 718.8560
// #define CU 607.1928
// #define CV 185.2157

// Seq 3
// #define FU 721.5377
// #define FV 721.5377
// #define CU 609.5593
// #define CV 172.8540

// Seq 4, 5, 6, 7, 8, 9, 10, 11, 12
// #define FU 707.0912 
// #define FV 707.0912
// #define CU 601.8873
// #define CV 183.1104


//====== INTERNAL VARIABLES ================================================

//====== INTERNAL FUNCTIONS ================================================

//====== MAIN PROGRAM ======================================================

Rox_Sint main(Rox_Sint argc, Rox_Char *argv[])
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Char filename[FILENAME_MAX];
   Rox_Double score = 1.0;
   Rox_Odometry_Dense_DepthMap odometry_dense_depthmap = NULL;

   // Current frame relative to the origin frame
   Rox_MatSE3 oTc_est = NULL;
   error = rox_matse3_new ( &oTc_est );
   ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_Double ** oTc_est_data = NULL; 
   error = rox_array2d_double_get_data_pointer_to_pointer ( &oTc_est_data, oTc_est );
   ROX_ERROR_CHECK_TERMINATE ( error );

   // Current frame relative to the origin frame
   Rox_MatSE3 rTc_est = NULL;
   error = rox_matse3_new ( &rTc_est );
   ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_MatSE3 cTr_est = NULL;
   error = rox_matse3_new ( &cTr_est );
   ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_MatSE3 cTr = NULL;
   error = rox_matse3_new ( &cTr );
   ROX_ERROR_CHECK_TERMINATE ( error );

   // Rox_Double tra[3] = {0.05, 0.0, 0.0};
   Rox_Double tra[3] = {0.00, 0.0, 0.0};

   error = rox_matse3_set_translation ( cTr, tra );
   ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_MatUT3 K = NULL;
   error = rox_matut3_new ( &K );
   ROX_ERROR_CHECK_TERMINATE ( error );

   // error = rox_transformtools_build_calibration_matrix ( K, FU, FV, CU, CV );
   // ROX_ERROR_CHECK_TERMINATE ( error );
   // rox_matse3_print(K);

   sprintf(filename, K_PATH, SEQ);
   rox_log("Reading file : %s \n",filename);
   
   error = rox_array2d_double_read ( K, filename );
   ROX_ERROR_CHECK_TERMINATE ( error );

   rox_matse3_print(K);

   sprintf(filename, IMAGE_PATH, SEQ, 0);
   rox_log("Reading image : %s \n",filename);

   Rox_Image Ir_uchar = NULL;
   error = rox_image_new_read_pgm ( &Ir_uchar, filename );
   ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_Sint width = 0, height = 0;
   error = rox_image_get_size ( &height, &width, Ir_uchar );
   ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_Image Ic_uchar = NULL;
   error = rox_image_new ( &Ic_uchar, width, height );
   ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_Array2D_Float Ir_float = NULL;
   error = rox_array2d_float_new ( &Ir_float, height, width );
   ROX_ERROR_CHECK_TERMINATE ( error );

   error = rox_array2d_float_from_uchar ( Ir_float, Ir_uchar );
   ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_Array2D_Float Ic_float = NULL;
   error = rox_array2d_float_new ( &Ic_float, height, width );
   ROX_ERROR_CHECK_TERMINATE ( error );

   error = rox_array2d_float_from_uchar ( Ic_float, Ic_uchar );
   ROX_ERROR_CHECK_TERMINATE ( error );

   rox_log("Using a float image \n");
   Rox_Array2D_Float Ic = Ic_float;
   Rox_Array2D_Float Ir = Ir_float;

   Rox_Array2D_Float depth = NULL;
   error = rox_array2d_float_new ( &depth, height, width );
   ROX_ERROR_CHECK_TERMINATE ( error );

   sprintf(filename, Z_PATH, SEQ, 0);
   rox_log("Reading image : %s \n",filename); 

   error = rox_array2d_float_read ( depth, filename );
   ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_Array2D_Float Zir = NULL;
   error = rox_array2d_float_new ( &Zir, height, width );
   ROX_ERROR_CHECK_TERMINATE ( error );

   sprintf(filename, Zi_PATH, SEQ, 0);
   rox_log("Reading image : %s \n",filename);

   //error = rox_array2d_float_read ( Zir, filename );
   //ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_Array2D_Float Ziur = NULL;
   error = rox_array2d_float_new ( &Ziur, height, width );
   ROX_ERROR_CHECK_TERMINATE ( error );

   error = rox_array2d_float_fillval ( Ziur, 0.0f );
   ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_Array2D_Float Zivr = NULL;
   error = rox_array2d_float_new ( &Zivr, height, width );
   ROX_ERROR_CHECK_TERMINATE ( error );

   error = rox_array2d_float_fillval ( Zivr, 0.0f );
   ROX_ERROR_CHECK_TERMINATE ( error );

   sprintf(filename, IMASK_PATH, SEQ, 0);
   rox_log("Reading imask : %s \n",filename);

   Rox_Imask imask = NULL;
   error = rox_imask_new_read_pgm ( &imask, filename );
   ROX_ERROR_CHECK_TERMINATE ( error );

   // If IMASK is not defined do not use an image mask 
#ifndef IMASK
   error = rox_imask_set_ones ( imask );
   ROX_ERROR_CHECK_TERMINATE ( error );
#endif




   error = rox_odometry_dense_depthmap_new ( &odometry_dense_depthmap, width, height );
   ROX_ERROR_CHECK_TERMINATE ( error );

   error = rox_odometry_dense_depthmap_set_calibration ( odometry_dense_depthmap, K );
   ROX_ERROR_CHECK_TERMINATE ( error );

   // If we have a prediction of the pose we can use it
   // error = rox_odometry_dense_depthmap_set_pose ( odometry_dense_depthmap, cTr );
   // ROX_ERROR_CHECK_TERMINATE ( error );

   // save the pose in kitti format
   sprintf ( filename, "%02d.txt", SEQ );
   error = rox_array_double_save ( filename, oTc_est_data[0], 12 );
   ROX_ERROR_CHECK_TERMINATE ( error );

   sprintf ( filename, "poses_%02d.txt", SEQ );
   rox_log(filename);
   error = rox_array2d_double_save ( filename, oTc_est ); 
   ROX_ERROR_CHECK_TERMINATE ( error );

   sprintf ( filename, "scores_%02d.txt", SEQ );
   rox_log(filename);
   error = rox_double_save (filename, score ); 
   ROX_ERROR_CHECK_TERMINATE ( error );

   rox_log("Starting loop \n");

   for ( Rox_Sint i = 1; i <=10; i++ )
   {
      // Set the previous displacement to init (use module filter_matse3 to get a better prediction)
      error = rox_odometry_dense_depthmap_set_pose ( odometry_dense_depthmap, cTr_est );
      ROX_ERROR_CHECK_TERMINATE ( error );

      // Read the reference image
      sprintf(filename, IMAGE_PATH, SEQ, i-1);
      rox_log("Reading reference image : %s \n",filename);

      error = rox_image_read_pgm ( Ir_uchar, filename );
      ROX_ERROR_CHECK_TERMINATE ( error );

      error = rox_array2d_float_from_uchar ( Ir_float, Ir_uchar );
      ROX_ERROR_CHECK_TERMINATE ( error );

#ifdef IMASK
      sprintf(filename, IMASK_PATH, SEQ, i-1);
      rox_log("Reading reference imask : %s \n",filename);

      error = rox_imask_read_pgm ( imask, filename );
      ROX_ERROR_CHECK_TERMINATE ( error );
#endif
      // Use the inverse depth
      sprintf(filename, Z_PATH, SEQ, i-1);
      rox_log("Reading reference depth : %s \n",filename);
   
      error = rox_array2d_float_read ( depth, filename );
      ROX_ERROR_CHECK_TERMINATE ( error );


#ifdef DEPTH
      // Use the depth
      error = rox_odometry_dense_depthmap_set_reference_depth ( odometry_dense_depthmap, Ir, depth, imask );
      ROX_ERROR_CHECK_TERMINATE ( error );
#else
      error = rox_array2d_float_inverse ( Zir, depth );
      ROX_ERROR_CHECK_TERMINATE ( error );

      error = rox_odometry_dense_depthmap_set_reference ( odometry_dense_depthmap, Ir, Zir, Ziur, Zivr, imask );
      ROX_ERROR_CHECK_TERMINATE ( error );
#endif

      // Read the current image
      sprintf(filename, IMAGE_PATH, SEQ, i);
      rox_log("Reading current image : %s \n", filename);

      error = rox_image_read_pgm ( Ic_uchar, filename );
      ROX_ERROR_CHECK_TERMINATE ( error );
      
      error = rox_array2d_float_from_uchar ( Ic_float, Ic_uchar );
      ROX_ERROR_CHECK_TERMINATE ( error );
      printf("Ic \n");
      Rox_Float ** M_data = NULL;
      error = rox_array2d_float_get_data_pointer_to_pointer ( &M_data, Ic_float );
      ROX_ERROR_CHECK_TERMINATE ( error );
      Rox_Sint row=0, col=0;
      for ( row=0; row < 5; row++){
            for(col = 0; col < 5; col++){
               printf("%f,  ", M_data[row][col]);
            }
            printf(";\n");
      }
      // Make dense direct odometry with ESM optimization
      error = rox_odometry_dense_depthmap_make ( odometry_dense_depthmap, Ic, 0);
      //ROX_ERROR_CHECK_TERMINATE ( error );

      // Get the estimated pose
      Rox_Sint success = 0;
      error = rox_odometry_dense_depthmap_get_results ( &success, &score, cTr_est, odometry_dense_depthmap );
      ROX_ERROR_CHECK_TERMINATE ( error );
      
      // Display and save results
      rox_log("success = %d \n", success);
      rox_log("score = %f \n", score);
      rox_matse3_print(cTr_est);

      rox_matse3_inv (rTc_est, cTr_est);

      // Update current pose in the worold frame
      error = rox_matse3_update_matse3_right ( oTc_est, rTc_est );
      ROX_ERROR_CHECK_TERMINATE ( error );

      // rox_matse3_print(oTc_est); 

      // save the pose in kitti format
      sprintf ( filename, "%02d.txt", SEQ );      
      error = rox_array_double_save_append ( filename, oTc_est_data[0], 12 );
      ROX_ERROR_CHECK_TERMINATE ( error );
      
      sprintf ( filename, "poses_%02d.txt", SEQ );
      error = rox_array2d_double_save_append (filename, oTc_est ); 
      ROX_ERROR_CHECK_TERMINATE ( error );

      sprintf ( filename, "scores_%02d.txt", SEQ );
      error = rox_double_save_append ( filename,  score );
      ROX_ERROR_CHECK_TERMINATE ( error );
   }

function_terminate:
   
   rox_error_print(error);

   //=======================================================================================

   error = rox_odometry_dense_depthmap_del ( &odometry_dense_depthmap );
   ROX_ERROR_CHECK_TERMINATE ( error );

   error = rox_image_del ( &Ir_uchar );
   ROX_ERROR_CHECK_TERMINATE ( error );

   error = rox_image_del ( &Ic_uchar );
   ROX_ERROR_CHECK_TERMINATE ( error );

   error = rox_array2d_float_del ( &Ir_float );
   ROX_ERROR_CHECK_TERMINATE ( error );

   error = rox_array2d_float_del ( &Ic_float );
   ROX_ERROR_CHECK_TERMINATE ( error );
}
