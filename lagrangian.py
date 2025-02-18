'''
Provides a solver class for obtaining the Euler Lagrange equations of motion and Hamiltonian equations of motion, given a Lagrangian.
Also provides a function pprint for pretty printing of latex strings returned by sympy.
'''

import sympy
from inspect import signature
from IPython.display import display, Math

class solver:
    def __init__(self, n, lagrangian, custom_coords=None):
        '''
        Creates a solver class for obtaining the Euler Lagrange equations of motion and Hamiltonian equations of motion, given a Lagrangian. 
        
        Parameters -
        n: number of generalised position coordinates
        lagrangian: lagrangian function, MUST have 2*n arguments, first n being the n generalised positions and the next n being the n generalised velocities
        custom_coords: list of custom position coordinates, if not provided, defaults to q_1, q_2, ..., q_n
        '''
        sig = signature(lagrangian)
        self.system_name = lagrangian.__name__.replace('_',' ')
        lagrangian_args = len(sig.parameters)
        if lagrangian_args!=2*n:
            raise Exception(f"Lagrangian function should have {2*n} arguments, got {lagrangian_args}")
        
        if custom_coords and len(custom_coords)!=n:
            raise Exception(f"Custom coordinates should have {n} elements, got {len(custom_coords)}")
        
        self.n = n
        self.lagrangian = lagrangian

        if custom_coords:
            self.coords = custom_coords
        else:
            self.coords = [f'q_{i}' for i in range(n)]

        self.q = [sympy.Symbol(self.coords[i]) for i in range(n)]
        self.dq = [sympy.Symbol('d' + self.coords[i]) for i in range(n)]
        self.ddq = [sympy.Symbol('dd' + self.coords[i]) for i in range(n)]
        self.p = [sympy.Symbol('p_{' + self.coords[i] + '}') for i in range(n)]
        self.dp = [sympy.Symbol('dp_{' + self.coords[i] + '}') for i in range(n)]

        self.t = sympy.Symbol('t')

    def __get_lagrangian(self):
        lagrangian = self.lagrangian(*(self.q + self.dq))
        return lagrangian
    
    def __total_derivative(self, expr):
        dexpr_dt = sympy.diff(expr, self.t)
        for j in range(self.n):
            dexpr_dt += sympy.diff(expr, self.q[j]) * self.dq[j]
            dexpr_dt += sympy.diff(expr, self.dq[j]) * self.ddq[j]
        return dexpr_dt

    def euler_lagrange_equations(self, latex=True, pretty_diffs = True):
        '''
        Returns the Euler Lagrange equations of motion.

        Parameters - 
        latex (defaults to True): if True, returns equations as latex strings, else returns sympy equations objects
        pretty_diffs (defaults to True): if True, returns equations with latex format derivates, i.e. \dot{<coord>} and \ddot{<coord>} as opposed to d<coord> and dd<coord>.
        '''
        equations = []
        lagrangian = self.__get_lagrangian()
        for i in range(self.n):
            dL_dq = sympy.diff(lagrangian, self.q[i])
            dL_ddq = sympy.diff(lagrangian, self.dq[i])
            eq = sympy.Eq(self.__total_derivative(dL_ddq) - dL_dq,0)
            equations.append(sympy.simplify(eq))
        if pretty_diffs:
            subs_dict = {}
            for i in range(self.n):
                subs_dict[self.dq[i]] = sympy.Symbol(r'\dot{ ' + self.coords[i] +'}')
                subs_dict[self.ddq[i]] = sympy.Symbol(r'\ddot{ ' +  self.coords[i] +'}')
            equations = [eq.subs(subs_dict) for eq in equations]
        if latex:
            equations = [sympy.latex(equation) for equation in equations]
        return equations
    
    def get_second_derivatives(self, latex=True, pretty_diffs = True, simplify = False):
        '''
        Solves the system of Euler Lagrange equations for the second derivatives of the generalised coordinates (i.e. accelarations).

        Parameters -
        latex (defaults to True): if True, returns equations as latex strings, else returns sympy equations objects
        pretty_diffs (defaults to True): if True, returns equations with latex format derivates, i.e. \dot{<coord>} and \ddot{<coord>} as opposed to d<coord> and dd<coord>.
        simplify (defaults to False): if True, simplifies the final equations for the accelarations, else lets them be as it is.  
        '''
        equations = self.euler_lagrange_equations(latex=False, pretty_diffs=False)
        sol = sympy.solve(equations, self.ddq, dict=True)[0]
        second_derivatives = [sympy.Eq(self.ddq[i], sol[self.ddq[i]]) for i in range(self.n)]
        if pretty_diffs:
            subs_dict = {}
            for i in range(self.n):
                subs_dict[self.dq[i]] = sympy.Symbol(r'\dot{ ' + self.coords[i] +'}')
                subs_dict[self.ddq[i]] = sympy.Symbol(r'\ddot{ ' +  self.coords[i] +'}')
            second_derivatives = [eq.subs(subs_dict) for eq in second_derivatives]
        if simplify:
            second_derivatives = [sympy.simplify(eq) for eq in second_derivatives]
        if latex:
            second_derivatives = [sympy.latex(equation) for equation in second_derivatives]
        return second_derivatives
    
    def __get__hamiltonian(self):
        H = 0
        lagrangian = self.__get_lagrangian()
        eqs = [sympy.Eq(self.p[i], sympy.diff(lagrangian, self.dq[i])) for i in range(self.n)]
        sol = sympy.solve(eqs, self.dq, dict=True)
        if not sol:
            raise Exception("Could not invert the velocity-momentum relations; the Lagrangian may be degenerate.")
        sol_dict = sol[0] 
        dq_sol = [sol_dict[self.dq[i]] for i in range(self.n)]
        H = sum(self.p[i] * dq_sol[i] for i in range(self.n)) - self.lagrangian(*(self.q + dq_sol))
        H = sympy.simplify(H)
        return H
    
    def hamitonian_equations(self, latex=True, pretty_diffs = True):
        '''
        Calculates Hamiltonian from the Lagrangian and rettains the Hamiltonian equations of motion.

        Parameters -
        latex (defaults to True): if True, returns equations as latex strings, else returns sympy equations objects
        pretty_diffs (defaults to True): if True, returns equations with latex format derivates, i.e. \dot{<coord>} and \dot{p_{<coord>}} as opposed to d<coord> and dp_<coord>.
        '''
        p_eqs = []
        q_eqs = [] 
        H = self.__get__hamiltonian()
        for i in range(self.n):
            p_eqs.append(sympy.Eq(self.dp[i], -sympy.diff(H, self.q[i])))
            q_eqs.append(sympy.Eq(self.dq[i], sympy.diff(H, self.p[i])))
        if pretty_diffs:
            subs_dict = {}
            for i in range(self.n):
                subs_dict[self.dq[i]] = sympy.Symbol(r'\dot{'+self.coords[i]+'}')
                subs_dict[self.dp[i]] = sympy.Symbol(r'\dot{p_{'+self.coords[i]+'}}')
            p_eqs = [eq.subs(subs_dict) for eq in p_eqs]
            q_eqs = [eq.subs(subs_dict) for eq in q_eqs]
        if latex:
            p_eqs = [sympy.latex(equation) for equation in p_eqs]
            q_eqs = [sympy.latex(equation) for equation in q_eqs]
        return p_eqs, q_eqs
    
    def print_lagrangian(self, pretty_diffs=True, display_name = True):
        '''
        Prints the Lagrangian of the system.

        Parameters -
        pretty_diffs (defaults to True): if True, returns equations with latex format derivates, i.e. \dot{<coord>} and \ddot{<coord>} as opposed to d<coord> and dd<coord>.
        display_name (defaults to True): if True, prints the name of the lagrangian function passed to the object (i.e. if the object is created with solver(n, lagrangian=Lagrangian_of_the_system), then the name will be 'Lagrangian of the system'.
        '''
        lagrangian = sympy.Eq(sympy.Symbol('\mathcal{L}'), self.__get_lagrangian())
        if pretty_diffs:
            subs_dict = {}
            for i in range(self.n):
                subs_dict[self.dq[i]] = sympy.Symbol(r'\dot{ ' + self.coords[i] +'}')
                subs_dict[self.ddq[i]] = sympy.Symbol(r'\ddot{ ' +  self.coords[i] +'}')
            lagrangian = lagrangian.subs(subs_dict)
        if display_name: 
            print(f'Lagrangian for {self.system_name}:')
        pprint(sympy.latex(lagrangian))

    def print_hamiltonian(self, pretty_diffs=True, display_name = True):
        '''
        Prints the Hamiltonian of the system.
        
        Parameters -
        pretty_diff (defaults to True): if True, returns equations with latex format derivates, i.e. \dot{<coord>} and \dot{p_{<coord>}} as opposed to d<coord> and dp_<coord>.
        display_name (defaults to True): if True, prints the name of the lagrangian function passed to the object (i.e. if the object is created with solver(n, lagrangian=Lagrangian_of_the_system), then the name will be 'Lagrangian of the system'.
        '''
        hamiltonian = sympy.Eq(sympy.Symbol('\mathcal{H}'), self.__get__hamiltonian())
        if pretty_diffs:
            subs_dict = {}
            for i in range(self.n):
                subs_dict[self.dq[i]] = sympy.Symbol(r'\dot{'+self.coords[i]+'}')
                subs_dict[self.dp[i]] = sympy.Symbol(r'\dot{p_{'+self.coords[i]+'}}')
            hamiltonian = hamiltonian.subs(subs_dict)
        if display_name:
            print(f'Hamiltonian for {self.system_name}:')
        pprint(sympy.latex(hamiltonian))

    
def pprint(equation):
    display(Math(equation))        


