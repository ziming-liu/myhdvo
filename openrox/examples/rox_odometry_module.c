/*
 * @Author: Ziming Liu
 * @Date: 2022-06-21 15:41:53
 * @LastEditors: Ziming Liu
 * @LastEditTime: 2023-05-04 03:21:09
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
#include <inout/system/print.h>

//====== INTERNAL MACROS    ====================================================

// #define DEPTH
//#define IMASK

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
void floatreadfrombuffer(Rox_Array2D_Float output, Rox_Uchar * frame_data, Rox_Sint height, Rox_Sint width){
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Float ** out = NULL;
   error = rox_array2d_float_get_data_pointer_to_pointer( &out, output );

   int count = 0;
   Rox_Sint row=0, col=0;
   for ( row=0; row < height; row++){
         for(col = 0; col < width; col++){
            if(col<5 && row<1){
            printf("%f  ", (float) frame_data[count]);}
            out[row][col] = (float) frame_data[count];
            count++;
      }
      if(row==0){
      printf(" \n");}
   }
   return ;   
}
void zfloatreadfrombuffer(Rox_Array2D_Float output, Rox_Double * frame_data, Rox_Sint height, Rox_Sint width){
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Float ** out = NULL;
   error = rox_array2d_float_get_data_pointer_to_pointer( &out, output );

   int count = 0;
   Rox_Sint row=0, col=0;
   for ( row=0; row < height; row++){
         for(col = 0; col < width; col++){
            /*if(col<5 && row<1){
            printf("%f  ", (float) frame_data[count]);}*/
            out[row][col] = (Rox_Float) frame_data[count];
            count++;
      }
      /* if(row==0){
      printf(" \n");}*/
   }
   return ;   
}
void doublereadfrombuffer(Rox_Double ** out, Rox_Double * frame_data, Rox_Sint height, Rox_Sint width){
   /**
    * @description: load 4x4 matrix for camera pose and camera intrinsics
    * @return: {*}
    * @param {4 &&} width
    */   
   //assert (height==4 && width==4);
   //Rox_ErrorCode error = ROX_ERROR_NONE;
   //Rox_Double ** out = NULL;
   //error = rox_array2d_double_get_data_pointer_to_pointer( &out, output );

   int count = 0;
   Rox_Sint row=0, col=0;
   for ( row=0; row < height; row++){
         for(col = 0; col < width; col++){
            out[row][col] =  frame_data[count];
            count++;
      }
       
   }
   return ;
}

void uintreadfrombuffer(Rox_Array2D_Uint output, Rox_Double * frame_data, Rox_Sint height, Rox_Sint width){
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Uint ** out = NULL;
   error = rox_array2d_uint_get_data_pointer_to_pointer( &out, output );

   Rox_Sint cols = 0, rows = 0;
   error = rox_array2d_uint_get_size ( &rows, &cols, output ); 
   //ROX_ERROR_CHECK_TERMINATE( error );
   if(cols!=width || rows!=height){printf("mask size h %d w %d is not equal to the input h %d w %d \n", rows, cols, height, width); return;}
   int count = 0;
   Rox_Sint row=0, col=0;
   for ( row=0; row < height; row++){
         for(col = 0; col < width; col++){
            //if(col<5 && row<1){
            //printf("%d  ", frame_data[count]);
            //}
            const Rox_Uint onevalue  = ~0; 
            const Rox_Uint zerovalue =  0;
            Rox_Uint val = (Rox_Uint) frame_data[count];
            if( val ==1){ out[row][col] = onevalue; } //continue;
            else{out[row][col] = zerovalue;}
            //out[row][col] = val; //val << 24 | val << 16 | val << 8 | val  ;
            //printf("val: %d ", val);
            //printf("input data %f == out data %d  \n", (float) frame_data[count], out[row][col]);
            count++;
      }
      //if(row==0){
      //printf(" \n");}
   }
   //printf(" count %d  ", count);
   //printf("row x col %d  ", row*col);
   return ;   
}

void mattostring(Rox_Double * pose_out_ptr, Rox_Double ** pose, Rox_Sint height, Rox_Sint width){
   int count=0;
   Rox_Sint row=0, col=0;
   for ( row=0; row < height; row++){
         for(col = 0; col < width; col++){
               pose_out_ptr[count] = pose[row][col];
               count++;
      }
   }
}
 
Rox_Double *  ddo(Rox_Double * Ir_uchar, Rox_Double * Z_float, Rox_Double * Ic_uchar, Rox_Double * imask_uchar, Rox_Double * cTr_given, Rox_Double * K_uchar, Rox_Sint rows, Rox_Sint cols, Rox_Sint ifmask, Rox_Sint disp_log, Rox_Sint ifrobust){
   /**
    * @description: 
    * @parameter: cTr_est, init relatice pose for odometry.
    *             
    * @return: {*}
    */      
    if(ifmask == 1){
       //printf("define imask >>\n");
       #define IMASK
    }

   Rox_Char filename[FILENAME_MAX];

   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Double score = 1.0;
   Rox_Odometry_Dense_DepthMap odometry_dense_depthmap = NULL;

   // Current frame relative to the origin frame
   Rox_MatSE3 oTc_est = NULL;
   error = rox_matse3_new ( &oTc_est );
   //ROX_ERROR_CHECK_TERMINATE ( error );

   //Rox_Double ** oTc_est_data = NULL; 
   //error = rox_array2d_double_get_data_pointer_to_pointer ( &oTc_est_data, oTc_est );
   ////ROX_ERROR_CHECK_TERMINATE ( error );

   // Current frame relative to the origin frame
   Rox_MatSE3 rTc_est = NULL;
   error = rox_matse3_new ( &rTc_est );
   //ROX_ERROR_CHECK_TERMINATE ( error );



   Rox_MatSE3 cTr_est = NULL;
   error = rox_matse3_new ( &cTr_est );
   //ROX_ERROR_CHECK_TERMINATE ( error );
   
   Rox_Double ** cTr_est_data = NULL; 
   error = rox_array2d_double_get_data_pointer_to_pointer ( &cTr_est_data, cTr_est );
   //ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_MatSE3 cTr = NULL;
   error = rox_matse3_new ( &cTr );
   //ROX_ERROR_CHECK_TERMINATE ( error );
   // Rox_Double tra[3] = {0.05, 0.0, 0.0};
   Rox_Double tra[3] = {0.00, 0.0, 0.0};
   error = rox_matse3_set_translation ( cTr, tra );
   //ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_MatUT3 K = NULL;
   error = rox_matut3_new ( &K );
   //ROX_ERROR_CHECK_TERMINATE ( error );
   Rox_Double ** K_data = NULL;
   error = rox_array2d_double_get_data_pointer_to_pointer ( &K_data, K );
   //ROX_ERROR_CHECK_TERMINATE ( error );
   doublereadfrombuffer(K_data, K_uchar, 3, 3);
   if(disp_log==1){rox_log("camera instrinsics K >> \n"); rox_matut3_print(K);}

   //sprintf(filename, IMAGE_PATH, SEQ, 0);
   //rox_log("Reading image : %s \n",filename);

   //Rox_Image Ir_uchar = NULL;
   //error = rox_image_new_read_pgm ( &Ir_uchar, filename );
   ////ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_Sint width = cols, height = rows;
  
   //Rox_Image Ic_uchar = NULL;
   //error = rox_image_new ( &Ic_uchar, width, height );
   ////ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_Array2D_Float Ir_float = NULL;
   error = rox_array2d_float_new ( &Ir_float, height, width );
   //ROX_ERROR_CHECK_TERMINATE ( error );
   //printf("\n>>>>>load Ir_uchar data\n");

   zfloatreadfrombuffer(Ir_float, Ir_uchar, rows, cols);
   //printf("\n >> end load Ir \n");

   Rox_Array2D_Float Ic_float = NULL;
   error = rox_array2d_float_new ( &Ic_float, height, width );
   //ROX_ERROR_CHECK_TERMINATE ( error );

   zfloatreadfrombuffer(Ic_float, Ic_uchar, rows, cols);

   //rox_log("Using a float image \n");
   Rox_Array2D_Float Ic = Ic_float;
   Rox_Array2D_Float Ir = Ir_float;

   Rox_Array2D_Float depth = NULL;
   error = rox_array2d_float_new ( &depth, height, width );
   //ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_Array2D_Float Zir = NULL;
   error = rox_array2d_float_new ( &Zir, height, width );
   //ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_Array2D_Float Ziur = NULL;
   error = rox_array2d_float_new ( &Ziur, height, width );
   //ROX_ERROR_CHECK_TERMINATE ( error );

   error = rox_array2d_float_fillval ( Ziur, 0.0f );
   //ROX_ERROR_CHECK_TERMINATE ( error );

   Rox_Array2D_Float Zivr = NULL;
   error = rox_array2d_float_new ( &Zivr, height, width );
   //ROX_ERROR_CHECK_TERMINATE ( error );

   error = rox_array2d_float_fillval ( Zivr, 0.0f );
   //ROX_ERROR_CHECK_TERMINATE ( error );

   //sprintf(filename, IMASK_PATH, SEQ, 0);
   //rox_log("Reading imask : %s \n",filename);
   Rox_Imask imask = NULL;  
   error = rox_imask_new( &imask, width, height  ); // don't forget to allocate memory, if not, we can't give input data to imask successfully.
   ////ROX_ERROR_CHECK_TERMINATE ( error );
   uintreadfrombuffer(imask, imask_uchar, rows, cols);

   /*
   Rox_Uint ** M_data = NULL;
      error = rox_array2d_uint_get_data_pointer_to_pointer ( &M_data, imask );
      //ROX_ERROR_CHECK_TERMINATE ( error );
      Rox_Sint row=0, col=0;
      for ( row=0; row < 5; row++){
            for(col = 0; col < 5; col++){
               printf("%f,  ", M_data[row][col]);
            }
            printf(";\n");
      }*/
   // If IMASK is not defined do not use an image mask 

#ifndef IMASK
   printf(">> undef mask, set mask ones\n");
   error = rox_imask_set_ones ( imask );
   //ROX_ERROR_CHECK_TERMINATE ( error );

#endif

error = rox_odometry_dense_depthmap_new ( &odometry_dense_depthmap, width, height );
//ROX_ERROR_CHECK_TERMINATE ( error );

error = rox_odometry_dense_depthmap_set_calibration ( odometry_dense_depthmap, K );
//ROX_ERROR_CHECK_TERMINATE ( error );

// If we have a prediction of the pose we can use it, give initial cTr pose
//if(cTr_given !=NULL){ 
doublereadfrombuffer(cTr_est_data, cTr_given, 4, 4);
if(disp_log==1){
   rox_log("cTr pose init >> \n");
   rox_matse3_print(cTr_est);
}
error = rox_odometry_dense_depthmap_set_pose ( odometry_dense_depthmap, cTr_est );
//ROX_ERROR_CHECK_TERMINATE ( error );

//}
    

// Use the inverse depth
//printf("read depth >> \n");
zfloatreadfrombuffer(depth, Z_float, rows, cols);
if(disp_log==1){
   rox_log("imask >> \n");
   rox_array2d_uint_print(imask);}
#ifdef DEPTH
      // Use the depth
      error = rox_odometry_dense_depthmap_set_reference_depth ( odometry_dense_depthmap, Ir, depth, imask );
      //ROX_ERROR_CHECK_TERMINATE ( error );
#else
      //printf(">>Use the inverse depth \n");
      /*printf("z\n");
      Rox_Float ** M_data = NULL;
      error = rox_array2d_float_get_data_pointer_to_pointer ( &M_data, depth );
      //ROX_ERROR_CHECK_TERMINATE ( error );
      Rox_Sint row=0, col=0;
      for ( row=0; row < 5; row++){
            for(col = 0; col < 5; col++){
               printf("%f,  ", M_data[row][col]);
            }
            printf(";\n");
      }*/
      error = rox_array2d_float_inverse ( Zir, depth );
      //ROX_ERROR_CHECK_TERMINATE ( error );
      /*printf("inverse z\n");
      Rox_Float ** M2_data = NULL;
      error = rox_array2d_float_get_data_pointer_to_pointer ( &M2_data, Zir );
      //ROX_ERROR_CHECK_TERMINATE ( error );
      Rox_Sint row2=0, col2=0;
      for ( row2=0; row2 < 5; row2++){
            for(col2 = 0; col2 < 5; col2++){
               printf("%f,  ", M2_data[row2][col2]);
            }
            printf(";\n");
      }*/
      error = rox_odometry_dense_depthmap_set_reference ( odometry_dense_depthmap, Ir, Zir, Ziur, Zivr, imask );
      //ROX_ERROR_CHECK_TERMINATE ( error );
#endif


      // rox_log("Ziur >> \n");
      //rox_array2d_float_print(odometry_dense_depthmap->Ziur);

      //printf(">> perform the vo \n");
      
      // Make dense direct odometry with ESM optimization
      /*
      printf("Ir\n");
      Rox_Float ** M_data = NULL;
      error = rox_array2d_float_get_data_pointer_to_pointer ( &M_data, Ir_float );
      //ROX_ERROR_CHECK_TERMINATE ( error );
      Rox_Sint row=0, col=0;
      for ( row=100; row < 105; row++){
            for(col = 0; col < 5; col++){
               printf("%f,  ", M_data[row][col]);
            }
            printf(";\n");
      }
      printf("Ic\n");
      Rox_Float ** M2_data = NULL;
      error = rox_array2d_float_get_data_pointer_to_pointer ( &M2_data, Ic );
      //ROX_ERROR_CHECK_TERMINATE ( error );
      row=0, col=0;
      for ( row=100; row < 105; row++){
            for(col = 0; col < 5; col++){
               printf("%f,  ", M2_data[row][col]);
            }
            printf(";\n");
      }*/      
      error = rox_odometry_dense_depthmap_make( odometry_dense_depthmap, Ic, ifrobust);
      //rox_log("max iters %d \n", odometry_dense_depthmap->max_iters);
      //rox_array2d_float_print(odometry_dense_depthmap->Ziur);
      //rox_log("solution >> ");
      //rox_array2d_double_print(odometry_dense_depthmap->solution);
      //rox_log("Iu  >> \n");
      //rox_array2d_float_print(odometry_dense_depthmap->Iu);
      //rox_log("JtJ >> \n");
      //rox_array2d_double_print(odometry_dense_depthmap->JtJ);
      //char str[50];
      //sprintf(str, "%lf", odometry_dense_depthmap->alpha);
      //rox_log("alpha >> \n");
      //rox_log(str);

      //char str2[50];
      //sprintf(str2, "%lf", odometry_dense_depthmap->beta);
      //rox_log("beta >> \n");
      //rox_log(str2);
      //ROX_ERROR_CHECK_TERMINATE ( error );

      // Get the estimated pose
      Rox_Sint success = 0;
      error = rox_odometry_dense_depthmap_get_results ( &success, &score, cTr_est, odometry_dense_depthmap );
      //ROX_ERROR_CHECK_TERMINATE ( error );
      
      // Display and save results
      //rox_log("score = %f \n", score);
      if(disp_log == 1){
         rox_log("success = %d \n", success);
         rox_log("score = %f \n", score);
         rox_log("estimated cTr >> \n");
         rox_matse3_print(cTr_est);
      }

      //rox_matse3_inv (rTc_est, cTr_est);

      // Update current pose in the worold frame
      //error = rox_matse3_update_matse3_right ( oTc_est, rTc_est );
      ////ROX_ERROR_CHECK_TERMINATE ( error );
      Rox_Double * output_pose_ptr = malloc(16 *sizeof(Rox_Double));
      mattostring(output_pose_ptr, cTr_est_data, 4,4);
   return output_pose_ptr;
}


void main(){
   return ;
}
