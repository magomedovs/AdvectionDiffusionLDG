#ifndef ADVECTION_DIFFUSION_LDG
#define ADVECTION_DIFFUSION_LDG

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/derivative_approximation.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/scratch_data.h>

namespace LDG
{
  using namespace dealii;


  template <typename T>
  bool are_equal(const T &a, const T &b, const double tol = 1e-10)
  {
    if (std::abs( a - b ) < tol)
    {
      return true;
    }
    else
    {
      return false;
    }
  }


  struct CopyDataFace
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
    std::array<double, 2>                values;
    std::array<unsigned int, 2>          cell_indices;
  };


  struct CopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace>            face_data;
    double                               value;
    unsigned int                         cell_index;


    template <class Iterator>
    void reinit(const Iterator &cell, const unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);
      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
    }
  };


  template <int dim>
  class AdvectionDiffusionProblem
  {
  public:
    enum
    {
      Dirichlet,
      Neumann
    };

    AdvectionDiffusionProblem(
      const MappingQ<dim>                           &imapping,
      Triangulation<dim>                            &itriangulation,
      std::unique_ptr<const TensorFunction<1, dim>> iadvection,
      std::unique_ptr<const Function<dim>>          idiffusion,
      std::unique_ptr<const Function<dim>>          irhs_function,
      std::unique_ptr<const Function<dim>>          iDirichlet_boundary_function,
      std::unique_ptr<const TensorFunction<1, dim>> iNeumann_boundary_flux_function = std::make_unique<ZeroTensorFunction<1, dim>>()
    );

    AdvectionDiffusionProblem(
      Triangulation<dim>                            &itriangulation,
      std::unique_ptr<const TensorFunction<1, dim>> iadvection,
      std::unique_ptr<const Function<dim>>          idiffusion,
      std::unique_ptr<const Function<dim>>          irhs_function,
      std::unique_ptr<const Function<dim>>          iDirichlet_boundary_function,
      std::unique_ptr<const TensorFunction<1, dim>> iNeumann_boundary_flux_function = std::make_unique<ZeroTensorFunction<1, dim>>()
    );

    void set_exact_solution(std::unique_ptr<const Function<dim>> iExactSolution);
    void run(const unsigned int n_refinements = 0);

  private:
    void setup_system();
    void assemble_system();
    void solve();

    void process_solution(const unsigned int cycle);
    void output_results(const unsigned int cycle);

    const MappingQ<dim> mapping;
    Triangulation<dim>  &triangulation;
    
    using ScratchData = MeshWorker::ScratchData<dim>;

    const unsigned int u_poly_degree;
    const unsigned int sigma_poly_degree;
    const QGauss<dim>     quadrature;
    const QGauss<dim - 1> face_quadrature;
    
    const FESystem<dim> fe;
    DoFHandler<dim>     dof_handler;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double>       solution;
    Vector<double>       system_rhs;

    std::unique_ptr<const TensorFunction<1, dim>> advection;
    std::unique_ptr<const Function<dim>>          diffusion;

    std::unique_ptr<const Function<dim>>          rhs_function;
    std::unique_ptr<const Function<dim>>          Dirichlet_boundary_function;
    std::unique_ptr<const TensorFunction<1, dim>> Neumann_boundary_flux_function;

    std::unique_ptr<const Function<dim>> ExactSolution;
    bool exact_solution_given;

    ConvergenceTable convergence_table;
    TimerOutput computing_timer;
  };


} // namespace LDG

#endif //ADVECTION_DIFFUSION_LDG
