# Lagrangian Solver

Sympy implementation of solver class with methods for solving Lagrangian and Hamiltonian mechanics problems using SymPy. The solver handles n-dimensional systems and provides clean LaTeX output.

## Features

- Automatic derivation of Euler-Lagrange equations
- Computation of second derivatives (accelerations)
- Conversion from Lagrangian to Hamiltonian formalism
- Generation of Hamilton's equations of motion
- Pretty printing with LaTeX formatting
- Support for custom coordinate variables
- Handles n-dimensional mechanical systems

## Usage Example

```python
from lagrangian import solver, pprint
import sympy

"""
Example: Simple Harmonic Oscillator
The Lagrangian function's argument MUST be of the format 
(q1, q2, ..., qn, dq1, dq2, ..., dqn) 
even if the coordinates are not used
"""

def harmonic_oscillator(q, dq):
    m = sympy.Symbol('m')
    k = sympy.Symbol('k')
    return sympy.Rational(1,2) * m * dq**2 - sympy.Rational(1,2) * k * q**2

# Create solver instance for 1-dimensional system
system = solver(n=1, lagrangian=harmonic_oscillator, custom_coords=['x'])

# Get equations of motion
eom = system.euler_lagrange_equations()[0]
pprint(eom)

# Get accelaration
accelaration = system.get_second_derivatives()[0]
pprint(accelaration)

# Print Lagrangian
system.print_lagrangian()

# Get Hamiltonian equations
p_eqs, q_eqs = system.hamitonian_equations()
pprint(p_eqs[0])
pprint(q_eqs[0])

# Print Hamiltonian
system.print_hamiltonian()

```

## Class Methods

- `euler_lagrange_equations(latex=True, pretty_diffs=True)`: Derives the equations of motion
- `get_second_derivatives(latex=True, pretty_diffs=True, simplify=False)`: Solves for accelerations
- `print_lagrangian(pretty_diffs=True, display_name=True)`: Displays the Lagrangian
- `print_hamiltonian(pretty_diffs=True, display_name=True)`: Displays the Hamiltonian
- `hamitonian_equations(latex=True, pretty_diffs=True)`: Returns Hamilton's equations

## Check out the example notebooks

- [simple_examples.ipynb](simple_examples.ipynb) - Basic mechanical systems
- [more_examples.ipynb](more_examples.ipynb) - Advanced applications
