//==============================================================================
//
//    OPENROX   : File test_kdtree.cpp
//
//    Contents  : Tests for kdtree.c
//
//    Author(s) : R&D department directed by Ezio MALIS
//
//    Copyright : 2022 Robocortex S.A.S. and Inria
//
//    License   : LGPL v3 or commercial license
//
//==============================================================================

//=== INCLUDED HEADERS   =======================================================

#include <openrox_tests.hpp>

extern "C"
{
   #include <baseproc/tools/kdtree/kdtree.h>
   #include <inout/geometry/point/dynvec_point3d_print.h>
   #include <inout/geometry/point/point3d_print.h>
   #include <inout/system/errors_print.h>
   #include <inout/system/print.h>
}

//=== INTERNAL MACROS    =======================================================

ROX_TEST_SUITE_BEGIN(kdtree)

//=== INTERNAL TYPESDEFS =======================================================

//=== INTERNAL DATATYPES =======================================================

//=== INTERNAL VARIABLES =======================================================

//=== INTERNAL FUNCTDEFS =======================================================

//=== INTERNAL FUNCTIONS =======================================================

// get a random double between -10 and 10 
double rd( void );

double rd( void ) 
{
  return (double)rand()/RAND_MAX * 20.0 - 10.0;
}

//=== EXPORTED FUNCTIONS =======================================================


ROX_TEST_CASE_DECLARE(rox::OpenROXTest, test_kdtree_new_del)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Kdtree kdtree = NULL;
   Rox_Sint dimension = 3;

   error = rox_kdtree_new(&kdtree, dimension);
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );

   error = rox_kdtree_del(&kdtree);
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );
}

ROX_TEST_CASE_DECLARE(rox::OpenROXTest, test_kdtree_clean)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Kdtree kdtree = NULL;
   Rox_Sint dimension = 3;

   error = rox_kdtree_new(&kdtree, dimension);
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );

   error = rox_kdtree_clean(kdtree);
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );

   error = rox_kdtree_del(&kdtree);
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );
}

ROX_TEST_CASE_DECLARE(rox::OpenROXTest, test_kdtree_build_search)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;
   Rox_Kdtree kdtree = NULL;
   Rox_Sint dimension = 3;
   Rox_DynVec_Point3D_Double m = NULL;
   Rox_Point3D_Double_Struct ms;
   Rox_DynVec_Point3D_Double results;

   rox_log("prepare data for kdtree\n");

   error = rox_dynvec_point3d_double_new ( &results, 1);
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );

   error = rox_dynvec_point3d_double_new ( &m, 1001);
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );

   ms.X = rd(); ms.Y = rd(); ms.Z = rd();
   rox_dynvec_point3d_double_append(m, &ms);

   rox_log("The search point\n");
   rox_point3d_double_print(&ms);

   for(int k = 0; k < 1000; k++)
   {
      Rox_Point3D_Double_Struct mk;

      mk.X = rd(); mk.Y = rd(); mk.Z = rd();

      rox_dynvec_point3d_double_append(m, &mk);
   }

   rox_log("create kdtree\n");

   error = rox_kdtree_new(&kdtree, dimension);
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );
   

   rox_log("build kdtree\n");

   error = rox_kdtree_build(kdtree, m);
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );
   
   rox_log("search kdtree\n");

   error = rox_kdtree_search(results, kdtree, m, &ms);
   rox_error_print(error);
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );
   
   rox_log("display result\n");

   rox_dynvec_point3d_double_print(results);

   rox_log("delete kdtree\n");

   error = rox_dynvec_point3d_double_del ( &m );
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );

   error = rox_kdtree_del(&kdtree);
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );
}

ROX_TEST_CASE_DECLARE(rox::OpenROXTest, test_kdtree_search)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   ROX_TEST_MESSAGE ( "This test has not been implemented yet !!! \n" );
   
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );
}

ROX_TEST_CASE_DECLARE(rox::OpenROXTest, test_kdtree_save)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   ROX_TEST_MESSAGE ( "This test has not been implemented yet !!! \n" );
   
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );
}

ROX_TEST_CASE_DECLARE(rox::OpenROXTest, test_kdtree_load)
{
   Rox_ErrorCode error = ROX_ERROR_NONE;

   ROX_TEST_MESSAGE ( "This test has not been implemented yet !!! \n" );
   
   ROX_TEST_CHECK_EQUAL ( error, ROX_ERROR_NONE );
}

ROX_TEST_SUITE_END()
