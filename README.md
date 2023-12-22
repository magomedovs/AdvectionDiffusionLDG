# Advection–diffusion equation and deal.II library

Solver for the stationary advection–diffusion equation in the domain $\Omega$ with Dirichlet and Neumann boundary conditions on $\partial\Omega_{D} \cup \partial\Omega_{N} = \partial\Omega$

$$\begin{align}
 \nabla \cdot (\boldsymbol{a} u) - \nabla \cdot (\kappa \nabla u) &= F(\boldsymbol{x}), \quad &\mathrm{in}\ \Omega, \\
 u &= u_D, \quad &\mathrm{on}\ \partial\Omega_{D}, \\
 (\boldsymbol{a} u - \kappa \nabla u) \cdot \boldsymbol{\mathrm{n}} &= u_N, \quad &\mathrm{on}\ \partial\Omega_{N}.
 \end{align}$$

Here $u$ is the variable of interest, $\kappa(\boldsymbol{x}) > 0$ is the diffusion coefficient, $\boldsymbol{a}(\boldsymbol{x})$ is the vector field, such that $\nabla \cdot  \boldsymbol{a} \equiv 0$, and $\boldsymbol{\mathrm{n}}$ denotes the unit outward normal to the boundary. The equation is solved by the Local Discontinuous Galerkin (LDG) method using an open source finite element library [deal.II](https://www.dealii.org).