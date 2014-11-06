#!/bin/bash

#set -x

#source $LIBMESH_DIR/examples/run_common.sh

#example_name=systems_of_equations_ex1

#run_example "$example_name" -ksp_type cg

#run_example "$example_name" -ksp_type preonly -pc_type lu -pc_factor_mat_solver_package superlu_dist

#run_example "$example_name" -ksp_type preonly -pc_type lu -pc_factor_mat_solver_package mumps -mat_mumps_icntl_7 2

#run_example -np 2 "$example_name" -ksp_type preonly -pc_type lu -pc_factor_mat_solver_package superlu_dist


# # # mpirun
#mpirun -np 3 ./example-opt -ksp_type cg -log_summary

mpirun -np 3 ./example-opt -ksp_type preonly -pc_type lu -pc_factor_mat_solver_package superlu_dist -log_summary