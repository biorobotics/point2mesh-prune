/****************************************************************************
* Based on trimesh_closest.cpp from:
* VCGLib                                                            o o     *
* Visual and Computer Graphics Library                            o     o   *
*                                                                _   O  _   *
* Copyright(C) 2004-2016                                           \/)\/    *
* Visual Computing Lab                                            /\/|      *
* ISTI - Italian National Research Council                           |      *
*                                                                    \      *
* All rights reserved.                                                      *
*                                                                           *
* This program is free software; you can redistribute it and/or modify      *
* it under the terms of the GNU General Public License as published by      *
* the Free Software Foundation; either version 2 of the License, or         *
* (at your option) any later version.                                       *
*                                                                           *
* This program is distributed in the hope that it will be useful,           *
* but WITHOUT ANY WARRANTY; without even the implied warranty of            *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
* GNU General Public License (http://www.gnu.org/licenses/gpl.txt)          *
* for more details.                                                         *
*                                                                           *
****************************************************************************/

// stuff to define the mesh
#include <vcg/complex/complex.h>
#include <vcg/simplex/face/component_ep.h>
#include <vcg/complex/algorithms/update/component_ep.h>
#include <vcg/complex/algorithms/point_sampling.h>

#include <wrap/io_trimesh/import.h>
#include <wrap/io_trimesh/export_ply.h>


class BaseVertex;
class BaseEdge;
class BaseFace;

struct BaseUsedTypes: public vcg::UsedTypes<vcg::Use<BaseVertex>::AsVertexType,vcg::Use<BaseEdge>::AsEdgeType,vcg::Use<BaseFace>::AsFaceType>{};

class BaseVertex  : public vcg::Vertex< BaseUsedTypes,
	vcg::vertex::Coord3f, vcg::vertex::Normal3f, vcg::vertex::BitFlags  > {};

class BaseEdge : public vcg::Edge< BaseUsedTypes> {};

class BaseFace    : public vcg::Face< BaseUsedTypes,
	vcg::face::Normal3f, vcg::face::VertexRef, vcg::face::BitFlags, vcg::face::Mark, vcg::face::EmptyEdgePlane > {};

class BaseMesh    : public vcg::tri::TriMesh<std::vector<BaseVertex>, std::vector<BaseFace> > {};


class RTVertex;
class RTEdge;
class RTFace;

struct RTUsedTypes: public vcg::UsedTypes<vcg::Use<RTVertex>::AsVertexType,vcg::Use<RTEdge>::AsEdgeType,vcg::Use<RTFace>::AsFaceType>{};

class RTVertex  : public vcg::Vertex< RTUsedTypes,
	vcg::vertex::Coord3f, vcg::vertex::Normal3f, vcg::vertex::BitFlags  > {};

class RTEdge : public vcg::Edge< RTUsedTypes> {};

class RTFace    : public vcg::Face< RTUsedTypes,
	vcg::face::Normal3f, vcg::face::VertexRef, vcg::face::EdgePlane, vcg::face::Mark, vcg::face::BitFlags > {};

class RTMesh    : public vcg::tri::TriMesh<std::vector<RTVertex>, std::vector<RTFace> > {};


using namespace vcg;

clock_t CLOCKS_PER_MS=CLOCKS_PER_SEC/1000;

void Usage()
{
	printf( "\nUsage:  trimesh_closest mesh.ply loops voxel_spacing sample_scale (as fraction of AABB of mesh) samplenum1 samplenum2 ...");
    printf("If voxel_spacing is non positive, set voxel size so that there are two voxels per face of the mesh\n");
	exit(-1);
}

// Testing of closest point on a mesh functionalities
// Two main options
// - using or not precomputed edges and planes
// - using the simple wrapper or the basic functions of the grid.
// - using the fn as size of the grid or the edge lenght as cell side

template <class MeshType, bool useEdge,bool useWrap>
bool testClosest(char *filename1, int loops, int numS, float dispPerc, std::vector<int> resultVec,float voxel_spacing)
{
  typedef typename MeshType::ScalarType ScalarType;
  typedef typename MeshType::CoordType CoordType;
  typedef typename MeshType::FaceType FaceType;
  typedef GridStaticPtr<FaceType, ScalarType> TriMeshGrid;

  MeshType mr;
  int err=vcg::tri::io::Importer<MeshType>::Open(mr,filename1);
  tri::UpdateBounding<MeshType>::Box(mr);
//  tri::UpdateNormals<MeshType>::PerFaceNormalized(mr);
  tri::UpdateNormal<MeshType>::PerFace(mr);
  if(err)
  {
      std::cerr << "Unable to open mesh " << filename1 << " : " << vcg::tri::io::Importer<MeshType>::ErrorMsg(err) << std::endl;
      exit(-1);
  }
  int endOpen = clock();

  bool use_auto_grid=voxel_spacing<=0.0;

  Point3f mesh_center = mr.bbox.Center();
  Point3f mesh_max = mr.bbox.max;
  Point3f mesh_min = mr.bbox.min;
  Point3f mesh_size = mesh_max-mesh_min;
  float dispAbs = mr.bbox.Diag()*dispPerc;
  std::vector<float> dur;
  for(int count=0;count<loops;count++){
    printf("Loop %i: ",count+1);
    int startSampling = clock();

    std::vector<Point3f> MontecarloSamples;
    // First step build the sampling
    typedef tri::TrivialSampler<MeshType> BaseSampler;
    BaseSampler mcSampler(MontecarloSamples);
    tri::SurfaceSampling<MeshType,BaseSampler>::SamplingRandomGenerator().initialize(123);
    tri::SurfaceSampling<MeshType,BaseSampler>::Montecarlo(mr, mcSampler, numS);
    math::MarsenneTwisterRNG rnd;
    rnd.initialize(123);

    for(size_t i=0;i<MontecarloSamples.size();++i)
    {
        Point3f pp(rnd.generate01(),rnd.generate01(),rnd.generate01());
        pp = (pp+Point3f(-0.5f,-0.5f,-0.5f)).Scale(mesh_size)*(dispPerc/2.0f)+mesh_center;
        MontecarloSamples[i]+=pp;
    }
    int endSampling = clock();

    printf("Sampling  %6.3f ms - ",float(endSampling-startSampling)/CLOCKS_PER_MS);

    int startGridInit = clock();
    TriMeshGrid TRGrid;
    if(use_auto_grid)
    {
        TRGrid.Set(mr.face.begin(),mr.face.end(),mr.FN()*2);
    }
    else
    {
        TRGrid.SetWithRadius(mr.face.begin(),mr.face.end(),voxel_spacing);
    }

    if(useEdge)
        tri::UpdateComponentEP<MeshType>::Set(mr);

    int endGridInit = clock();
    printf("Grid Init %6.3f ms - ",float(endGridInit-startGridInit)/CLOCKS_PER_MS);

    const ScalarType maxDist=std::max(dispAbs*10.0f,mr.bbox.Diag()/1000.f);
    CoordType closest;
    ScalarType dist;
    int startGridQuery = clock();
    double avgDist=0;
    resultVec.resize(MontecarloSamples.size());
    if(useEdge && useWrap)
        for(size_t i=0;i<MontecarloSamples.size();++i)
        {
        resultVec[i]=tri::Index(mr,tri::GetClosestFaceEP(mr,TRGrid,MontecarloSamples[i], maxDist,dist,closest));
        if(resultVec[i]) avgDist += double(dist);
        }
    if(!useEdge && useWrap)
        for(size_t i=0;i<MontecarloSamples.size();++i)
        {
        resultVec[i]=tri::Index(mr,tri::GetClosestFaceBase(mr,TRGrid,MontecarloSamples[i], maxDist,dist,closest));
        if(resultVec[i]) avgDist += double(dist);
        }
    if(useEdge && !useWrap)
    {
        typedef tri::FaceTmark<MeshType> MarkerFace;
        MarkerFace mf;
        mf.SetMesh(&mr);
        face::PointDistanceBaseFunctor<ScalarType> PDistFunct;
        for(size_t i=0;i<MontecarloSamples.size();++i)
        {
        resultVec[i]=tri::Index(mr,TRGrid.GetClosest(PDistFunct,mf,MontecarloSamples[i],maxDist,dist,closest));
        if(resultVec[i]) avgDist += double(dist);
        }
    }
    if(!useEdge && !useWrap)
    {
        typedef tri::FaceTmark<MeshType> MarkerFace;
        MarkerFace mf;
        mf.SetMesh(&mr);
        face::PointDistanceBaseFunctor<ScalarType> PDistFunct;
        for(size_t i=0;i<MontecarloSamples.size();++i)
        {
        resultVec[i]=tri::Index(mr,TRGrid.GetClosest(PDistFunct,mf,MontecarloSamples[i],maxDist,dist,closest));
        if(resultVec[i]) avgDist += double(dist);
        }
    }

    int endGridQuery = clock();
    dur.push_back(float(endGridQuery-startGridQuery)/CLOCKS_PER_MS);
    printf("Grid Size %3i %3i %3i - ",TRGrid.siz[0],TRGrid.siz[1],TRGrid.siz[2]);
    printf("Avg dist %6.9lf - ",avgDist / float(MontecarloSamples.size()));
    printf("Grid Query %6.3f ms\n", dur.back());
  }
  float mean=0.0f;
  for(auto it=dur.begin();it<dur.end();it++){mean+=*it;}
  mean/=loops;
  printf("Average query time %6.3f ms\n",mean);
  return true;
}

int main(int argc ,char**argv)
{
  int nargs=5;
  if(argc<nargs) Usage();
  int loops = atoi(argv[2]);
  float voxel_spacing = atof(argv[3]);
  float dispPerc = atof(argv[4]);
  std::vector<int> resultVecRT11;

  if(voxel_spacing>0){
    printf("Using voxel grid of edge length %f\n",voxel_spacing);
  }

  for(size_t idx=nargs;idx<argc;idx++){
    int numS=atoi(argv[idx]);
    printf("Trying %i points\n",numS);
    testClosest<RTMesh, true,  true>     (argv[1],loops,numS,dispPerc,resultVecRT11,voxel_spacing);
  }

  return 0;
}
