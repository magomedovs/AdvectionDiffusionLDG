#include "AdvectionDiffusionLDG.h"

namespace LDG
{
  using namespace dealii;
  
  template <int dim>
  AdvectionDiffusionProblem<dim>::AdvectionDiffusionProblem(
    const MappingQ<dim>&  imapping,
    Triangulation<dim>&   itriangulation,
    std::unique_ptr<const TensorFunction<1, dim>> iadvection,
    std::unique_ptr<const Function<dim>> idiffusion,
    std::unique_ptr<const Function<dim>> irhs_function,
    std::unique_ptr<const Function<dim>> iDirichlet_boundary_function,
    std::unique_ptr<const TensorFunction<1, dim>> iNeumann_boundary_flux_function
  )
    : mapping(imapping)
    , triangulation(itriangulation)
    , u_poly_degree(1)                        // p
    , sigma_poly_degree(u_poly_degree + 1)    // p+1
    , quadrature(sigma_poly_degree + 1)
    , face_quadrature(sigma_poly_degree + 1)
    , fe(	FE_DGQ<dim>(sigma_poly_degree), dim, 
					FE_DGQ<dim>(u_poly_degree), 1 )		/* 2 fe basis sets, corresponding to \vec{\sigma} and u */
    , dof_handler(triangulation)
    , ExactSolution(std::make_unique<Functions::ZeroFunction<dim>>())
    , exact_solution_given(false)
  {
    advection = std::move(iadvection);
    diffusion = std::move(idiffusion);

    rhs_function                   = std::move(irhs_function);
    Dirichlet_boundary_function    = std::move(iDirichlet_boundary_function);
    Neumann_boundary_flux_function = std::move(iNeumann_boundary_flux_function);
  }


  template <int dim>
  AdvectionDiffusionProblem<dim>::AdvectionDiffusionProblem(
    Triangulation<dim>&   itriangulation,
    std::unique_ptr<const TensorFunction<1, dim>> iadvection,
    std::unique_ptr<const Function<dim>> idiffusion,
    std::unique_ptr<const Function<dim>> irhs_function,
    std::unique_ptr<const Function<dim>> iDirichlet_boundary_function,
    std::unique_ptr<const TensorFunction<1, dim>> iNeumann_boundary_flux_function
  )
    : AdvectionDiffusionProblem(MappingQ1<dim>(), 
                                itriangulation,
                                std::move(iadvection),
                                std::move(idiffusion),
                                std::move(irhs_function),
                                std::move(iDirichlet_boundary_function),
                                std::move(iNeumann_boundary_flux_function))
  {}


  template <int dim>
  void AdvectionDiffusionProblem<dim>::set_exact_solution(std::unique_ptr<const Function<dim>> iExactSolution)
  {
    ExactSolution = std::move(iExactSolution);
    exact_solution_given = true;
  }


  template <int dim>
  void AdvectionDiffusionProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }


  template <int dim>
  void AdvectionDiffusionProblem<dim>::assemble_system()
  {
    const auto cell_worker = [&](const auto &cell, auto &scratch_data, auto &copy_data) {
      const FEValues<dim> &fe_v          = scratch_data.reinit(cell);
      const unsigned int   dofs_per_cell = fe_v.dofs_per_cell;
      copy_data.reinit(cell, dofs_per_cell);

      const auto        &q_points    = scratch_data.get_quadrature_points();
      const unsigned int n_q_points  = q_points.size();
      const std::vector<double> &JxW = scratch_data.get_JxW_values();

      std::vector<double> rhs(n_q_points);
      rhs_function->value_list(q_points, rhs);

      const FEValuesExtractors::Vector sigma(0);
      const FEValuesExtractors::Scalar u(dim);
      
      std::vector<Tensor<1, dim>> phi_sigma(dofs_per_cell);
      std::vector<double> div_phi_sigma(dofs_per_cell);
      std::vector<double> phi_u(dofs_per_cell);
      std::vector<Tensor<1, dim>> grad_phi_u(dofs_per_cell);

      for (unsigned int point = 0; point < n_q_points; ++point)
      {
        const auto advection_q = advection->value(q_points[point]);
        const auto diffusion_q = diffusion->value(q_points[point]);

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_sigma[k]        = fe_v[sigma].value(k, point);
          div_phi_sigma[k]    = fe_v[sigma].divergence(k, point);
          phi_u[k]            = fe_v[u].value(k, point);
          grad_phi_u[k]       = fe_v[u].gradient(k, point);
        }

        for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < fe_v.dofs_per_cell; ++j)
          {
            copy_data.cell_matrix(i, j) += (
              1./diffusion_q * phi_sigma[i] * phi_sigma[j] +
              -div_phi_sigma[i] * phi_u[j] +
              -grad_phi_u[i] * advection_q * phi_u[j] +
              -grad_phi_u[i] * phi_sigma[j]
            ) * JxW[point];                // dx
          }

          copy_data.cell_rhs(i) += phi_u[i] *           // v_h
                                    rhs[point] *        // F
                                    JxW[point];         // dx
        }
      }
    };

    const auto boundary_worker = [&](const auto &        cell,
                                     const unsigned int &face_no,
                                     auto &              scratch_data,
                                     auto &              copy_data) {
      const FEFaceValuesBase<dim> &fe_fv = scratch_data.reinit(cell, face_no);

      const auto        &q_points      = scratch_data.get_quadrature_points();
      const unsigned int n_q_points    = q_points.size();
      const unsigned int dofs_per_cell = fe_fv.dofs_per_cell;

      const std::vector<double>             &JxW = scratch_data.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals = scratch_data.get_normal_vectors();

      std::vector<double> Dirichlet_boundary_values(n_q_points);
      Dirichlet_boundary_function->value_list(q_points, Dirichlet_boundary_values);

      const FEValuesExtractors::Vector sigma(0);
			const FEValuesExtractors::Scalar u(dim);
      
      std::vector<Tensor<1, dim>> phi_sigma(dofs_per_cell);
      std::vector<double>         phi_u(dofs_per_cell);

      for (unsigned int point = 0; point < n_q_points; ++point)
      {
        const auto advection_q = advection->value(q_points[point]);
        const auto diffusion_q = diffusion->value(q_points[point]);
        const auto u_N         = Neumann_boundary_flux_function->value(q_points[point]);

        const double p = static_cast<double>(u_poly_degree);
        const double h = cell->diameter(); //cell->measure();
        const double eta = 2 * p*p / h;

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_sigma[k] = fe_fv[sigma].value(k, point);
          phi_u[k]     = fe_fv[u].value(k, point);
        }
        
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {     
            if (cell->face(face_no)->boundary_id() == Dirichlet)
            {
              copy_data.cell_matrix(i, j) += (
                phi_u[i] * (phi_sigma[j] * normals[point] + diffusion_q * eta * (phi_u[j]))
              ) * JxW[point];                // dx
            }       
            else if (cell->face(face_no)->boundary_id() == Neumann)
            {
              copy_data.cell_matrix(i, j) += (
                phi_sigma[i] * normals[point] * phi_u[j]
              ) * JxW[point];                // dx
            }
            else
            {
              /* Throw exception ! */
            }
            
          }
        
          if (cell->face(face_no)->boundary_id() == Dirichlet)
          {
            copy_data.cell_rhs(i) += (
              -phi_sigma[i] * normals[point] * Dirichlet_boundary_values[point] +
              -phi_u[i] * Dirichlet_boundary_values[point] * advection_q * normals[point]
              + phi_u[i] * diffusion_q * eta * Dirichlet_boundary_values[point]
            ) * JxW[point];                // dx
          }
          else if (cell->face(face_no)->boundary_id() == Neumann)
          {
            copy_data.cell_rhs(i) += (              
              -phi_u[i] * u_N * normals[point]        
            ) * JxW[point];                // dx
          }
          else
          {
            /* Throw exception ! */
          }
         
        }
      }
    };

    const auto face_worker = [&](const auto &        cell,
                                 const unsigned int &f,
                                 const unsigned int &sf,
                                 const auto &        ncell,
                                 const unsigned int &nf,
                                 const unsigned int &nsf,
                                 auto &              scratch_data,
                                 auto &              copy_data) {
      
      const FEInterfaceValues<dim> &fe_iv = scratch_data.reinit(cell, f, sf, ncell, nf, nsf);

      copy_data.face_data.emplace_back();
      CopyDataFace      &copy_data_face = copy_data.face_data.back();
      const unsigned int n_dofs_face    = fe_iv.n_current_interface_dofs();
      copy_data_face.joint_dof_indices  = fe_iv.get_interface_dof_indices();
      copy_data_face.cell_matrix.reinit(n_dofs_face, n_dofs_face);

      const std::vector<double>         &JxW     = fe_iv.get_JxW_values();
      const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();

      const auto &q_points = fe_iv.get_quadrature_points();

      const FEValuesExtractors::Vector sigma(0);
			const FEValuesExtractors::Scalar u(dim);
      
      std::vector<Tensor<1, dim>> phi_sigma_jump(n_dofs_face);
      std::vector<Tensor<1, dim>> phi_sigma_avg(n_dofs_face);
      std::vector<double> phi_u_jump(n_dofs_face);
      std::vector<double> phi_u_avg(n_dofs_face);
      std::vector<double> phi_u_upwind(n_dofs_face);

      for (const unsigned int point : fe_iv.quadrature_point_indices())
      {
        const auto advection_q = advection->value(q_points[point]);
        const auto diffusion_q = diffusion->value(q_points[point]);
        
        const Tensor<1, dim> beta = normals[point] / 2.; 
        const double p = static_cast<double>(u_poly_degree);
        const double h = std::min(cell->diameter(), ncell->diameter()); // std::max(cell->measure(), ncell->measure());
        const double eta = 2 * p*p / h;

        const double advection_dot_n = advection_q * normals[point];

        for (unsigned int k = 0; k < n_dofs_face; ++k)
        {
          phi_sigma_jump[k]        = fe_iv[sigma].jump_in_values(k, point);
          phi_sigma_avg[k]         = fe_iv[sigma].average_of_values(k, point);
          phi_u_jump[k]            = fe_iv[u].jump_in_values(k, point);
          phi_u_avg[k]             = fe_iv[u].average_of_values(k, point);
          phi_u_upwind[k]          = fe_iv[u].value((advection_dot_n > 0), k, point);
        }

        for (const unsigned int i : fe_iv.dof_indices())
        {
          for (const unsigned int j : fe_iv.dof_indices())
          {
            copy_data_face.cell_matrix(i, j) += (
              phi_sigma_jump[i] * normals[point] * (phi_u_avg[j] - beta * normals[point] * phi_u_jump[j]) +
              phi_u_jump[i] * phi_u_upwind[j] * advection_dot_n +
              phi_u_jump[i] * (phi_sigma_avg[j] * normals[point] + beta * normals[point] * phi_sigma_jump[j] * normals[point] + diffusion_q * eta * phi_u_jump[j])
            ) * JxW[point];                                 // dx
          }
        }
      }
    };

    AffineConstraints<double> constraints;
    constraints.close();
    const auto copier = [&](const auto &c) {
      constraints.distribute_local_to_global(c.cell_matrix,
                                             c.cell_rhs,
                                             c.local_dof_indices,
                                             system_matrix,
                                             system_rhs);

      for (auto &cdf : c.face_data)
      {
        constraints.distribute_local_to_global(cdf.cell_matrix,
                                                cdf.joint_dof_indices,
                                                system_matrix);
      }
    };

    const UpdateFlags cell_flags = update_values | update_gradients |
                                   update_quadrature_points | update_JxW_values;
    const UpdateFlags face_flags = update_values | update_gradients |
                                   update_quadrature_points |
                                   update_normal_vectors | update_JxW_values;

    ScratchData scratch_data(mapping, fe, quadrature, cell_flags, face_quadrature, face_flags);
    CopyData copy_data;

    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker,
                          copier,
                          scratch_data,
                          copy_data,
                          MeshWorker::assemble_own_cells |
                            MeshWorker::assemble_boundary_faces |
                            MeshWorker::assemble_own_interior_faces_once,
                          boundary_worker,
                          face_worker);
  }


  template <int dim>
  void AdvectionDiffusionProblem<dim>::solve()
  {
    SparseDirectUMFPACK A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult(solution, system_rhs);
  }


  template <int dim>
  void AdvectionDiffusionProblem<dim>::process_solution(const unsigned int cycle)
  {  
    if (exact_solution_given)
    {
      Vector<float> difference_per_cell(triangulation.n_active_cells());

      ComponentSelectFunction<dim> u_select(dim, dim + 1);
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        *ExactSolution,
                                        difference_per_cell,
                                        QGauss<dim>(sigma_poly_degree + 3),
                                        VectorTools::L2_norm,
                                        &u_select);
      
      const double u_L2_error = VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);


      ComponentSelectFunction<dim> sigma_select(std::pair<unsigned int, unsigned int>(0, dim), dim + 1);
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        *ExactSolution,
                                        difference_per_cell,
                                        QGauss<dim>(sigma_poly_degree + 3),
                                        VectorTools::L2_norm,
                                        &sigma_select);
      
      const double sigma_L2_error = VectorTools::compute_global_error(triangulation,
                                          difference_per_cell,
                                          VectorTools::L2_norm);


      const unsigned int n_active_cells = triangulation.n_active_cells();
      const unsigned int n_dofs         = dof_handler.n_dofs();

      convergence_table.add_value("cycle", cycle);
      convergence_table.add_value("cells", n_active_cells);
      convergence_table.add_value("dofs", n_dofs);

      convergence_table.add_value("u_L2", u_L2_error);
      convergence_table.add_value("sigma_L2", sigma_L2_error);

      convergence_table.set_scientific("u_L2", true);
      convergence_table.set_scientific("sigma_L2", true);

      convergence_table.set_precision("u_L2", 8);
      convergence_table.set_precision("sigma_L2", 8);
  
      convergence_table.set_tex_caption("cells", "$\\#$ cells");
      convergence_table.set_tex_caption("dofs", "$\\#$ dofs");
      convergence_table.set_tex_caption("u_L2", "$\\| u - u^h \\|_{L^2}$");
      convergence_table.set_tex_caption("sigma_L2", "$\\| \\sigma - \\sigma^h \\|_{L^2}$");
  
      convergence_table.set_tex_format("cells", "r");
      convergence_table.set_tex_format("dofs", "r");

      convergence_table.evaluate_convergence_rates(
            "u_L2", "cells", ConvergenceTable::reduction_rate_log2, dim);

      convergence_table.evaluate_convergence_rates(
            "sigma_L2", "cells", ConvergenceTable::reduction_rate_log2, dim);
  
    }
  }


  template <int dim>
  void AdvectionDiffusionProblem<dim>::output_results(const unsigned int cycle)
  {
    std::vector<std::string> solution_names(dim, "sigma");
    solution_names.emplace_back("u");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution,
                              solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
    data_out.build_patches(u_poly_degree + sigma_poly_degree);

    std::ofstream output(
      "solutionLDG_p" + Utilities::int_to_string(u_poly_degree) + "-" + Utilities::int_to_string(cycle, 2) + ".vtk");
    data_out.write_vtk(output);


    if (exact_solution_given)
    {
      {    
        std::cout << std::endl;
        convergence_table.write_text(std::cout);

        std::string error_filename_text = "errorsLDG_p" + Utilities::int_to_string(u_poly_degree) + ".text";
        std::ofstream error_table_file_text(error_filename_text);
        convergence_table.write_text(error_table_file_text);
  
        std::string error_filename_tex = "errorsLDG_p" + Utilities::int_to_string(u_poly_degree) + ".tex";
        std::ofstream error_table_file_tex(error_filename_tex);
        convergence_table.write_tex(error_table_file_tex);
      }
    
    }

    // std::ofstream out_sp("sparsity-pattern-"  + Utilities::int_to_string(cycle, 2) + ".svg");
    // sparsity_pattern.print_svg(out_sp);

  }


  template <int dim>
  void AdvectionDiffusionProblem<dim>::run(const unsigned int n_refinements)
  {
    unsigned int cycle = 0;
    while (cycle < n_refinements + 1)
    {
      std::cout << "Cycle " << cycle << std::endl;
      
      if (cycle != 0)
      {
        triangulation.refine_global(1);
      }

      std::cout << "  Number of active cells       : " << triangulation.n_active_cells() << std::endl;
      
      std::cout << "  Setting up system ...";
      setup_system();
      std::cout << "  Done. " << std::endl;

      std::cout << "  Number of degrees of freedom : " << dof_handler.n_dofs() << std::endl;

      std::cout << "  Assembling system ...";
      assemble_system();
      std::cout << "  Done. " << std::endl;
      
      std::cout << "  Solving system ...";
      solve();
      std::cout << "  Done. " << std::endl;

      if (exact_solution_given)
      {
        process_solution(cycle);
      }
      
      output_results(cycle);
    
      std::cout << std::endl;

      ++cycle;
    }   
  }


  template class AdvectionDiffusionProblem<1>;
  template class AdvectionDiffusionProblem<2>;
  template class AdvectionDiffusionProblem<3>;

} // namespace LDG
