//
//  exact_solution_pf.h
//  provide exact solutions for Stokes problem with a point force(unbounded)
//
//  Created by Xujun Zhao on 10/30/14.
//
//

#ifndef _exact_solution_pf_h
#define _exact_solution_pf_h


// C++ include files that we need
#include <iostream>
#include <algorithm>
#include <math.h>


// Basic include file needed for the mesh functionality.
#include "libmesh/libmesh.h"
#include "libmesh/libmesh_common.h"
#include "libmesh/point.h"


#include "polymer_chain.h"  // need this for point force vector
#include "GeomTools.h"

// Bring in everything from the libMesh namespace
using namespace libMesh;


// =============================================================================================
Real Kronecker_delta(const unsigned int i,
                     const unsigned int j)
{
  return (i==j) ? 1.0:0.0;
}


// =============================================================================================
// return the value of green tensor for stokes flow at a point p with respect to p0
// Note i j k = 0, 1, 2
Number G_tensor(const Point& p,
                const Point& p0,
                const unsigned int i,
                const unsigned int j,
                const unsigned int dim)  // problem dimension, (2 or 3)
{
  // vector and distance between the p and p0
  const Point xx = p - p0;
  const Real r = GeomTools::point_distance(p,p0);
  
  // delta function
  const Real d_ij = Kronecker_delta(i,j);
  
  // Green functions for 2D and 3D problem
  if(dim==2)
    return -d_ij*log(r) + xx(i)*xx(j)/(r*r);
  else if(dim==3)
    return d_ij/r + xx(i)*xx(j)/(r*r*r);
  else
    libmesh_example_requires(false, "only 2D and 3D Green functions are supported!");
  // end if-else
  
  return 0.0;
}


// =============================================================================================
// return the derivatives of green tensor for stokes flow at a point p with respect to p0
Number G_tensor_deriv(const Point& p,
                      const Point& p0,
                      const unsigned int i,
                      const unsigned int j,
                      const unsigned int k,
                      const unsigned int dim)  // problem dimension, (2 or 3)
{
  // distance to the point0, and relative coordiantes x
  const Real r = GeomTools::point_distance(p,p0);
  const Point xx = p - p0;
  
  // delta function
  const Real d_ij = Kronecker_delta(i,j);
  const Real d_ik = Kronecker_delta(i,k);
  const Real d_jk = Kronecker_delta(j,k);
  
  // three parts:
  // Green functions for 2D and 3D problem
  Real G1, G2, G3;
  if(dim==2)
  {
    G1 = -d_ij*xx(k)/(r*r);
    G2 = ( d_ik*xx(j) + d_jk*xx(i) )/(r*r);
    G3 = -2.0*xx(i)*xx(j)*xx(k)/(r*r*r*r);
  }
  else if(dim==3)
  {
    G1 = -d_ij*xx(k)/(r*r*r);
    G2 = ( d_ik*xx(j) + d_jk*xx(i) )/(r*r*r);
    G3 = -3.0*xx(i)*xx(j)*xx(k)/(r*r*r*r*r);
  }
  else
    libmesh_example_requires(false, "only 2D and 3D Green functions are supported!");
  // end if-else
  
  return G1+G2+G3;
}


// =============================================================================================
Number P_tensor(const Point& p,
                const Point& p0,
                const unsigned int i,     // Note i = 0, 1, 2
                const unsigned int dim)   // problem dimension, (2 or 3)
{
  // vector and distance between the p and p0
  const Point xx = p - p0;
  const Real r = GeomTools::point_distance(p,p0);
  
  // Green functions for 2D and 3D problem
  if(dim==2)
    return 2.0*xx(i)/(r*r);
  else if(dim==3)
    return 2.0*xx(i)/(r*r*r);
  else
    libmesh_example_requires(false, "only 2D and 3D Green functions are supported!");
  // end if-else
  
  return 0.0;
}



// =============================================================================================
Number P_tensor_deriv(const Point& p,
                      const Point& p0,
                      const unsigned int i,     // Note i j = 0, 1, 2
                      const unsigned int j,
                      const unsigned int dim)   // problem dimension, (2 or 3)
{
  // distance to the point0, and relative coordiantes x
  const Real r = GeomTools::point_distance(p,p0);
  const Point xx = p - p0;
  
  // delta function
  const Real d_ij = Kronecker_delta(i,j);
  
  // three parts:
  // Green functions for 2D and 3D problem
  Real G1, G2;
  if(dim==2)
  {
    G1 = d_ij/(r*r);
    G2 = -2.0*xx(i)*xx(j)/(r*r*r*r);
  }
  else if(dim==3)
  {
    G1 = d_ij/(r*r*r);
    G2 = -3.0*xx(i)*xx(j)/(r*r*r*r*r);
  }
  else
    libmesh_example_requires(false, "only 2D and 3D Green functions are supported!");
  // end if-else
  
  return 2.0*(G1+G2);
}


// =============================================================================================
// exact solution for velocity u and pressure p
Number exact_solution(const Point& p,
                      const Parameters& parameters,
                      const std::string&  sys_name,    // sys_name
                      const std::string&  unk_name)    // unk_name
{
  // define polymer chain and retrieve the point force
  PolymerChain polymer_chain_one_bead;
  const Point p_force = polymer_chain_one_bead.bead_force(0);
  
  // bead coordinate
  const Point p0 = polymer_chain_one_bead.particle_coordinate(0);
  
  // viscosity of fluid
  const Real mu =  parameters.get<Real> ("viscosity");
  const unsigned int mesh_dim = parameters.get<unsigned int>("mesh dimension");
  
  unsigned int component = 10;
  if( unk_name=="u" )
    component = 0;
  else if( unk_name=="v" )
    component = 1;
  else if( unk_name=="w" )
    component = 2;
  else if( unk_name=="p" )
    component = 3;
  else
    libmesh_example_requires(false, "unknow name can be only u, v and w or p");
  // end if
  
  //std::cout<<"*************** component = "<<component<<", unk_name = "<<unk_name<<std::endl;
  
  Real coef = 1.0/pi;    // pi = LibMesh::pi;
  for(unsigned int j=0;j<mesh_dim; ++j)
    coef /= 2.0;
  
  // evaluate the exact solution for different components
  Real u = 0.0;
  if(component==3)    // compute pressure p
  {
    for (unsigned int j=0; j<mesh_dim; ++j)
      u += P_tensor(p, p0, j, mesh_dim)*p_force(j);
    
    const Real pres_inf = 0.0;
    u = u*coef + pres_inf;
  }
  else                // compute u
  {
    for (unsigned int j=0; j<mesh_dim; ++j)
      u += G_tensor(p, p0, component, j, mesh_dim)*p_force(j);
    
    u = u*coef/mu;
  }
  
  return u;
}



// =============================================================================================
// derivatives of exact solution for velocity u and pressure p: [d/dx d/dy d/dz]
Gradient exact_solution_deriv(const Point& p,
                              const Parameters& parameters,
                              const std::string&  sys_name,    // sys_name
                              const std::string&  unk_name)    // unk_name
{
  // define polymer chain and retrieve the point force
  PolymerChain polymer_chain_one_bead;
  const Point p_force = polymer_chain_one_bead.bead_force(0);
  
  // bead coordinate
  const Point p0 = polymer_chain_one_bead.particle_coordinate(0);
  
  // viscosity of fluid
  const Real mu =  parameters.get<Real> ("viscosity");
  const unsigned int mesh_dim = parameters.get<unsigned int>("mesh dimension");
  
  unsigned int component = 10;
  if( unk_name=="u" )
    component = 0;
  else if( unk_name=="v" )
    component = 1;
  else if( unk_name=="w" )
    component = 2;
  else if( unk_name=="p" )
    component = 3;
  else
    libmesh_example_requires(false, "unknow name can be only u, v and w or p");
  // end if
  
  //std::cout<<"*************** component = "<<component<<", unk_name = "<<unk_name<<std::endl;
  
  Real coef = 1.0/pi;    // pi = LibMesh::pi;
  for(unsigned int j=0;j<mesh_dim; ++j)
    coef /= 2.0;
  
  // evaluate the exact solution for different components
  Gradient grad_u = 0.0;
  if(component==3)    // compute pressure p
  {
    for (unsigned int k=0; k<mesh_dim; ++k)     // Gradient component
      for (unsigned int j=0; j<mesh_dim; ++j)   // summation indices
        grad_u(k) += P_tensor_deriv(p, p0, j, k, mesh_dim)*p_force(j);

    grad_u = grad_u*coef;
  }
  else                // compute u
  {
    for (unsigned int k=0; k<mesh_dim; ++k)     // Gradient component
      for (unsigned int j=0; j<mesh_dim; ++j)   // summation indices
        grad_u(k) += G_tensor_deriv(p, p0, component, j, k, mesh_dim)*p_force(j);
    
    grad_u = grad_u*coef/mu;
  }
  
  return grad_u;
}



// =============================================================================================


#endif
