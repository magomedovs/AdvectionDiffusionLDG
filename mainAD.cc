#include "AdvectionDiffusionLDG.h"

namespace LDG
{
  using namespace dealii;

  template <int dim>
  class BoundaryValuesPlate : public Function<dim>
  {
  public:
    BoundaryValuesPlate()
      : Function<dim>()
    {}
    virtual double value(const Point<dim> & p,
                            const unsigned int component = 0) const override
    {
      const Point<2> center(0., 0.);
      const double inner_radius = 0.1, outer_radius = 0.25;
      const double padding = 0.25;

      const double distance_from_center = center.distance(p);
      if (distance_from_center <= outer_radius)
      {
        return 0.;
        // return p(0) + p(1);
      }
      else
      {
        return 0.;
        // return p(0) + p(1);
      }
    }                          

  };


  template <int dim>
  class RightHandSidePlate : public Function<dim>
  {
  public:
    RightHandSidePlate()
      : Function<dim>()
    {}

    virtual double value(const Point<dim> &,
                            const unsigned int component = 0) const override
    {
      // return 0.;
      return 1.;
    }
  };


  template <int dim>
  class DiffusionPlate : public Function<dim>
  {
  public:
    DiffusionPlate()
      : Function<dim>()
    {}

    virtual double value(const Point<dim> &,
                            const unsigned int component = 0) const override
    {
      return 0.02;
    }
  };


  template <int dim>
  class AdvectionPlate : public TensorFunction<1, dim>
  {
  public:
    AdvectionPlate()
      : TensorFunction<1, dim>()
    {}

    virtual Tensor<1, dim> value(const Point<dim> &) const override
    {
      // Assert(dim >= 2, ExcNotImplemented());

      Tensor<1, dim> adv;
      switch (dim)
      {
        case 1:
          adv[0] = 1.;
          break;
        case 2:
          adv[0] = 1. / std::sqrt(2.);
          adv[1] = 1. / std::sqrt(2.);
          break;
        case 3:
          adv[0] = 1. / std::sqrt(3.);
          adv[1] = 1. / std::sqrt(3.);
          adv[2] = 1. / std::sqrt(3.);
          break;
        default:
          Assert(false, ExcNotImplemented());
      }

      double magnitude = 1.;
      return magnitude * adv;
    }
  };

}

void plate_with_a_hole_case(const unsigned int n_refinements = 0)
{
  using namespace dealii;
  using namespace LDG;

  constexpr unsigned int dim = 2;

  Triangulation<dim>    triangulation;

  const Point<2> center(0., 0.);
  const double inner_radius = 0.1, outer_radius = 0.25;
  const double padding = 0.25;

  GridGenerator::plate_with_a_hole(triangulation, inner_radius, outer_radius, padding, padding, padding, padding, center);


  for (const auto &cell : triangulation.active_cell_iterators())
  {
    if (cell->at_boundary())
    {
      for (const auto &face : cell->face_iterators())
      {
        if (face->at_boundary())
        {
          const double distance_from_center = center.distance(face->center());
          if (distance_from_center <= inner_radius)
          {
            face->set_boundary_id(AdvectionDiffusionProblem<dim>::Dirichlet);
          }

          if (are_equal(face->center()[0], center[0] - outer_radius - padding) || 
              are_equal(face->center()[0], center[0] + outer_radius + padding) || 
              are_equal(face->center()[1], center[1] - outer_radius - padding) || 
              are_equal(face->center()[1], center[1] + outer_radius + padding))
          {
            face->set_boundary_id(AdvectionDiffusionProblem<dim>::Dirichlet);
          }
        }

      }
    }
  }

  AdvectionDiffusionProblem<dim> problem(
    MappingQ<dim>(2),
    triangulation,
    std::make_unique<AdvectionPlate<dim>>(),
    std::make_unique<DiffusionPlate<dim>>(),
    std::make_unique<RightHandSidePlate<dim>>(),
    std::make_unique<BoundaryValuesPlate<dim>>()
    // , std::make_unique<ZeroTensorFunction<1, dim>>()
  );

  problem.run(n_refinements);
}

/*========================================================================================================*/


namespace LDG
{
  using namespace dealii;

  template <int dim>
  class SolutionBase
  {
  protected:
    static const unsigned int n_source_centers = 3;
    static const Point<dim>   source_centers[n_source_centers];
    static const double       width;
  };


  template <>
  const Point<1>
    SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers] =
      {Point<1>(-1.0 / 3.0), Point<1>(0.0), Point<1>(+1.0 / 3.0)};


  template <>
  const Point<2>
    SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers] =
      {Point<2>(-1., +0.5), Point<2>(-0.5, -0.5), Point<2>(+0.8, -0.15)};

  template <>
  const Point<3>
    SolutionBase<3>::source_centers[SolutionBase<3>::n_source_centers] = {
      Point<3>(-0.5, +0.5, 0.25),
      Point<3>(-0.6, -0.5, -0.125),
      Point<3>(+0.5, -0.5, 0.5)};

  template <int dim>
  const double SolutionBase<dim>::width = 1. / 3.;


  template <int dim>
  class DiffusionExp : public Function<dim>
  {
  public:
    DiffusionExp()
      : Function<dim>()
    {}

    virtual double value(const Point<dim> &,
                            const unsigned int component = 0) const override
    {
      return 0.02;
    }
  };


  template <int dim>
  class AdvectionExp : public TensorFunction<1, dim>
  {
  public:
    AdvectionExp()
      : TensorFunction<1, dim>()
    {}

    virtual Tensor<1, dim> value(const Point<dim> &p) const override
    {
      Tensor<1, dim> advection;
      switch (dim)
        {
          case 1:
            advection[0] = 1.;
            break;
          case 2:
            advection[0] = p[1];
            advection[1] = -p[0];
            break;
          case 3:
            advection[0] = p[1];
            advection[1] = -p[0];
            advection[2] = 1.;
            break;
          default:
            Assert(false, ExcNotImplemented());
        }
      return advection;
    }
  };

  template <int dim>
  class ExactSolutionExp : public Function<dim>, protected SolutionBase<dim>
  {
  public:
    virtual double value(const Point<dim> &p,
                          const unsigned int /*component*/ = 0) const override
    {
      double sum = 0;
      for (unsigned int i = 0; i < this->n_source_centers; ++i)
        {
          const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];
          sum += std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
        }

      return sum;
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
              const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> sum;
      double gamma = this->width;
      for (unsigned int i = 0; i < this->n_source_centers; ++i)
      {
        const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];

        sum += (
            -2. / (gamma * gamma) *
            std::exp(-x_minus_xi.norm_square() / (gamma * gamma)) *
            x_minus_xi
        );
      }

      return sum;
    }
  };

  template <int dim>
  class BoundaryValuesExp : public Function<dim>
  {
  public:
    BoundaryValuesExp()
      : Function<dim>()
    {}
    virtual double value(const Point<dim> & p,
                            const unsigned int component = 0) const override
    {
      ExactSolutionExp<dim> sol;
      return sol.value(p);
    }                          

  };

  template <int dim>
  class NeumannFluxExp : public TensorFunction<1, dim>
  {
  public:
    NeumannFluxExp()
      : TensorFunction<1, dim>()
    {}

    virtual Tensor<1, dim> value(const Point<dim> &p) const override
    {
      AdvectionExp<dim>     advection;
      DiffusionExp<dim>     diffusion;
      ExactSolutionExp<dim> solution;

      Tensor<1, dim> a      = advection.value(p);
      double kappa          = diffusion.value(p);
      double u              = solution.value(p);
      Tensor<1, dim> grad_u = solution.gradient(p);

      Tensor<1, dim> flux;
      flux = a * u - kappa * grad_u;
      return flux;
    }

  };


  template <int dim>
  class RightHandSideExp : public Function<dim>, protected SolutionBase<dim>
  {
  public:
    virtual double value(const Point<dim> &p,
                          const unsigned int /*component*/ = 0) const override
    {
      AdvectionExp<dim> advection;
      DiffusionExp<dim> diffusion;
      Tensor<1, dim>    a = advection.value(p);
      double kappa        = diffusion.value(p);
      double   sum = 0;
      double gamma = this->width;
      for (unsigned int i = 0; i < this->n_source_centers; ++i)
      {
        const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];

        sum += (
          std::exp(-x_minus_xi.norm_square() / (gamma * gamma)) * (-2. / (gamma * gamma)) *
          (a * x_minus_xi + kappa * ((2. / (gamma * gamma)) * x_minus_xi.norm_square() - dim))
        );
      }

      return sum;
    }
  };

  template <int dim>
  class ExactSolutionAndGradientExp : public Function<dim>, protected SolutionBase<dim>
  {
  public:
    ExactSolutionAndGradientExp()
      : Function<dim>(dim + 1)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  v) const override
    {
      AssertDimension(v.size(), dim + 1);
      
      DiffusionExp<dim> diffusion;
      double kappa          = diffusion.value(p);
      
      ExactSolutionExp<dim>  solution;
      double u              = solution.value(p);
      Tensor<1, dim> sigma  = -kappa * solution.gradient(p);

      for (unsigned int d = 0; d < dim; ++d)
        v[d] = sigma[d];
      v[dim] = u;
    }
  };



}

void exp_test_case(const unsigned int n_refinements = 0)
{
  using namespace dealii;
  using namespace LDG;

  constexpr unsigned int dim = 2;

  Triangulation<dim>    triangulation;

  GridGenerator::hyper_cube(triangulation, -1., 1.);

  triangulation.refine_global(1);

  for (const auto &cell : triangulation.active_cell_iterators())
  {
    if (cell->at_boundary())
    {
      for (const auto &face : cell->face_iterators())
      {
        if (face->at_boundary())
        {
          if (are_equal(face->center()[0], -1.) || are_equal(face->center()[1], 1.))
          {
            // face->set_boundary_id(AdvectionDiffusionProblem<dim>::Dirichlet);
            face->set_boundary_id(AdvectionDiffusionProblem<dim>::Neumann);
          }
          else
          {
            face->set_boundary_id(AdvectionDiffusionProblem<dim>::Dirichlet);
          }
        }
      }
    }
  }

  AdvectionDiffusionProblem<dim> problem(
    // MappingQ<dim>(2),
    triangulation,
    std::make_unique<AdvectionExp<dim>>(),
    std::make_unique<DiffusionExp<dim>>(),
    std::make_unique<RightHandSideExp<dim>>(),
    std::make_unique<BoundaryValuesExp<dim>>(),
    std::make_unique<NeumannFluxExp<dim>>()
  );

  problem.set_exact_solution(std::make_unique<ExactSolutionAndGradientExp<dim>>());
  problem.run(n_refinements);
}


int main()
{
  try
    {
      // plate_with_a_hole_case(5);
      exp_test_case(5);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    };

  return 0;
}
