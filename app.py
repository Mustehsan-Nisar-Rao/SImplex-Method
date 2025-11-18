import streamlit as st
import numpy as np
import pandas as pd
import re

# Page configuration
st.set_page_config(
    page_title="Simplex Method Solver",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .tableau-table {
        width: 100%;
        border-collapse: collapse;
    }
    .tableau-table th {
        background-color: #1f77b4;
        color: white;
        padding: 0.5rem;
        text-align: center;
        border: 1px solid #ddd;
    }
    .tableau-table td {
        padding: 0.5rem;
        text-align: center;
        border: 1px solid #ddd;
    }
    .tableau-table tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    .tableau-table tr:hover {
        background-color: #e9ecef;
    }
    .solution-box {
        background-color: #e7f3ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .iteration-box {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffd54f;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SimplexSolver:
    def __init__(self):
        self.tableau = None
        self.basic_vars = []
        self.var_names = []
        self.num_decision_vars = 0
        self.num_constraints = 0
        self.phase = 1
        self.iterations = []
    
    def parse_coefficients(self, input_str, expected_count):
        """Parse coefficients from user input - handles both formats"""
        # Clean the input string
        input_str = input_str.strip()
        
        # If it contains 'x', parse as mathematical expression
        if 'x' in input_str.lower():
            return self._parse_expression(input_str, expected_count)
        else:
            # Parse as space-separated numbers
            return self._parse_space_separated(input_str, expected_count)
    
    def _parse_expression(self, input_str, expected_count):
        """Parse mathematical expressions like '2x1 + 3x2 - x3'"""
        coeffs = [0.0] * expected_count
        
        # Remove all spaces for easier parsing
        clean_str = input_str.replace(' ', '')
        
        # Handle cases where expression starts without sign (assume positive)
        if clean_str and clean_str[0].isdigit() or clean_str[0] == 'x':
            clean_str = '+' + clean_str
        
        # Regular expression to match terms like +2x1, -3x2, +x3, -x4, etc.
        pattern = r'([+-]?)(\d*\.?\d*)(x\d+)'
        
        # Find all matches
        matches = re.finditer(pattern, clean_str)
        
        for match in matches:
            sign = match.group(1) or '+'
            number_str = match.group(2)
            var_str = match.group(3)
            
            # Extract variable number
            var_num = int(var_str[1:])  # Remove 'x' and convert to int
            
            # Skip if variable number is out of range
            if var_num < 1 or var_num > expected_count:
                continue
            
            # Determine coefficient value
            if number_str == '':
                # No number specified (e.g., +x1, -x2)
                coeff = 1.0
            else:
                coeff = float(number_str)
            
            # Apply sign
            if sign == '-':
                coeff = -coeff
            
            # Assign to the correct position (variables are 1-indexed)
            coeffs[var_num - 1] = coeff
        
        return coeffs
    
    def _parse_space_separated(self, input_str, expected_count):
        """Parse space-separated numbers"""
        if not input_str.strip():
            return [0.0] * expected_count
        
        parts = input_str.split()
        coeffs = []
        
        for part in parts:
            try:
                coeffs.append(float(part))
            except ValueError:
                st.error(f"Invalid number: '{part}'. Please enter valid numbers.")
                return [0.0] * expected_count
        
        # Pad with zeros if not enough coefficients provided
        if len(coeffs) < expected_count:
            coeffs.extend([0.0] * (expected_count - len(coeffs)))
        # Truncate if too many coefficients provided
        elif len(coeffs) > expected_count:
            coeffs = coeffs[:expected_count]
        
        return coeffs
    
    def build_tableau(self, constraints, constraint_types, rhs_values):
        """Build the initial simplex tableau"""
        slack_count = 0
        surplus_count = 0
        artificial_count = 0
        
        # Count additional variables needed
        for c_type in constraint_types:
            if c_type == '<=':
                slack_count += 1
            elif c_type == '>=':
                surplus_count += 1
                artificial_count += 1
            elif c_type == '=':
                artificial_count += 1
        
        self.needs_phase1 = (artificial_count > 0)
        
        # Build variable names
        self.var_names = [f'x{i+1}' for i in range(self.num_decision_vars)]
        
        # Add slack variables
        for i in range(slack_count):
            self.var_names.append(f's{i+1}')
        
        # Add surplus variables
        for i in range(surplus_count):
            self.var_names.append(f'surp{i+1}')
        
        # Add artificial variables
        for i in range(artificial_count):
            self.var_names.append(f'A{i+1}')
        
        self.var_names.append('RHS')
        
        # Initialize tableau
        total_vars = self.num_decision_vars + slack_count + surplus_count + artificial_count
        self.tableau = np.zeros((self.num_constraints + 1, total_vars + 1))
        
        # Fill constraint rows
        slack_idx = self.num_decision_vars
        surplus_idx = self.num_decision_vars + slack_count
        artificial_idx = self.num_decision_vars + slack_count + surplus_count
        
        s_counter = 0
        surplus_counter = 0
        a_counter = 0
        
        for i, (coeffs, c_type, rhs) in enumerate(zip(constraints, constraint_types, rhs_values)):
            # Add decision variable coefficients
            self.tableau[i, :self.num_decision_vars] = coeffs
            
            # Add slack/surplus/artificial variables
            if c_type == '<=':
                self.tableau[i, slack_idx + s_counter] = 1
                self.basic_vars.append(self.var_names[slack_idx + s_counter])
                s_counter += 1
            elif c_type == '>=':
                self.tableau[i, surplus_idx + surplus_counter] = -1
                self.tableau[i, artificial_idx + a_counter] = 1
                self.basic_vars.append(self.var_names[artificial_idx + a_counter])
                surplus_counter += 1
                a_counter += 1
            elif c_type == '=':
                self.tableau[i, artificial_idx + a_counter] = 1
                self.basic_vars.append(self.var_names[artificial_idx + a_counter])
                a_counter += 1
            
            # Add RHS
            self.tableau[i, -1] = rhs
        
        if self.needs_phase1:
            # Phase 1: Minimize sum of artificial variables
            for i in range(artificial_count):
                self.tableau[-1, artificial_idx + i] = -1
            
            # Eliminate artificial variables from objective row
            for i, basic_var in enumerate(self.basic_vars):
                if basic_var.startswith('A'):
                    self.tableau[-1] += self.tableau[i]
        else:
            # No artificial variables, use original objective directly
            self.tableau[-1, :self.num_decision_vars] = [-c for c in self.obj_coeffs]
    
    def find_pivot_column(self):
        """Find the entering variable (most negative in objective row)"""
        obj_row = self.tableau[-1, :-1]
        min_val = np.min(obj_row)
        
        if min_val >= -1e-10:  # All non-negative (with tolerance)
            return -1  # Optimal solution reached
        
        return np.argmin(obj_row)
    
    def find_pivot_row(self, pivot_col):
        """Find the leaving variable using minimum ratio test"""
        ratios = []
        for i in range(self.num_constraints):
            if self.tableau[i, pivot_col] > 1e-10:  # Positive coefficient
                ratio = self.tableau[i, -1] / self.tableau[i, pivot_col]
                ratios.append(ratio)
            else:
                ratios.append(float('inf'))
        
        if all(r == float('inf') for r in ratios):
            return -1  # Unbounded solution
        
        return np.argmin(ratios)
    
    def pivot(self, pivot_row, pivot_col):
        """Perform pivoting operation"""
        # Store iteration data for display
        iteration_data = {
            'tableau': self.tableau.copy(),
            'basic_vars': self.basic_vars.copy(),
            'pivot_row': pivot_row,
            'pivot_col': pivot_col,
            'pivot_element': self.tableau[pivot_row, pivot_col]
        }
        self.iterations.append(iteration_data)
        
        # Make pivot element = 1
        pivot_element = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row] /= pivot_element
        
        # Make other elements in pivot column = 0
        for i in range(len(self.tableau)):
            if i != pivot_row:
                multiplier = self.tableau[i, pivot_col]
                self.tableau[i] -= multiplier * self.tableau[pivot_row]
        
        # Update basic variable
        self.basic_vars[pivot_row] = self.var_names[pivot_col]
    
    def solve_problem(self):
        """Solve the linear programming problem"""
        self.iterations = []  # Reset iterations
        
        if self.needs_phase1:
            # PHASE 1: Find initial feasible solution
            self.phase = 1
            iteration = 0
            
            # Store initial tableau
            self.iterations.append({
                'tableau': self.tableau.copy(),
                'basic_vars': self.basic_vars.copy(),
                'pivot_row': None,
                'pivot_col': None,
                'pivot_element': None
            })
            
            while True:
                pivot_col = self.find_pivot_column()
                
                if pivot_col == -1:
                    # Phase 1 complete
                    if abs(self.tableau[-1, -1]) > 1e-6:
                        return False, "NO_FEASIBLE_SOLUTION"
                    break
                
                pivot_row = self.find_pivot_row(pivot_col)
                
                if pivot_row == -1:
                    return False, "UNBOUNDED_PHASE1"
                
                self.pivot(pivot_row, pivot_col)
            
            # Remove artificial variables and setup Phase 2
            artificial_indices = [i for i, name in enumerate(self.var_names[:-1]) if name.startswith('A')]
            
            # Remove artificial variable columns
            keep_cols = [i for i in range(len(self.var_names)) if i not in artificial_indices]
            self.tableau = self.tableau[:, keep_cols]
            self.var_names = [self.var_names[i] for i in keep_cols]
            
            # Update basic variables (remove artificial ones)
            new_basic_vars = []
            for var in self.basic_vars:
                if not var.startswith('A'):
                    new_basic_vars.append(var)
            self.basic_vars = new_basic_vars
            
            # Set up Phase 2 objective function
            self.tableau[-1, :] = 0
            for i in range(self.num_decision_vars):
                self.tableau[-1, i] = -self.obj_coeffs[i]
            
            # Eliminate basic variables from objective row
            for i, basic_var in enumerate(self.basic_vars):
                var_idx = self.var_names.index(basic_var)
                if abs(self.tableau[-1, var_idx]) > 1e-10:
                    multiplier = self.tableau[-1, var_idx]
                    self.tableau[-1] -= multiplier * self.tableau[i]
        
        # PHASE 2: Optimize
        self.phase = 2
        
        # Store Phase 2 initial tableau
        self.iterations.append({
            'tableau': self.tableau.copy(),
            'basic_vars': self.basic_vars.copy(),
            'pivot_row': None,
            'pivot_col': None,
            'pivot_element': None,
            'phase': 2
        })
        
        while True:
            pivot_col = self.find_pivot_column()
            
            if pivot_col == -1:
                break
            
            pivot_row = self.find_pivot_row(pivot_col)
            
            if pivot_row == -1:
                return False, "UNBOUNDED"
            
            self.pivot(pivot_row, pivot_col)
        
        return True, "OPTIMAL"

def display_problem_formulation(solver, obj_type, constraints, constraint_types, rhs_values):
    """Display the problem formulation in a nice format"""
    st.markdown("---")
    st.markdown('<div class="sub-header">📝 Problem Formulation</div>', unsafe_allow_html=True)
    
    # Objective function
    obj_terms = []
    for i, coeff in enumerate(solver.original_obj_coeffs):
        if abs(coeff) > 1e-10:
            if i == 0:
                if coeff < 0:
                    obj_terms.append(f"- {abs(coeff):g}x{i+1}")
                else:
                    obj_terms.append(f"{coeff:g}x{i+1}")
            else:
                sign = "+" if coeff >= 0 else "-"
                obj_terms.append(f"{sign} {abs(coeff):g}x{i+1}")
    
    obj_str = " ".join(obj_terms)
    st.write(f"**Objective:** {obj_type.upper()} Z = {obj_str}")
    
    # Constraints
    st.write("**Subject to:**")
    for i in range(solver.num_constraints):
        constraint_terms = []
        for j, coeff in enumerate(constraints[i]):
            if abs(coeff) > 1e-10:
                if j == 0:
                    if coeff < 0:
                        constraint_terms.append(f"- {abs(coeff):g}x{j+1}")
                    else:
                        constraint_terms.append(f"{coeff:g}x{j+1}")
                else:
                    sign = "+" if coeff >= 0 else "-"
                    constraint_terms.append(f"{sign} {abs(coeff):g}x{j+1}")
        
        constraint_str = " ".join(constraint_terms)
        st.write(f"{constraint_str} {constraint_types[i]} {rhs_values[i]:g}")

def display_tableau(tableau, var_names, basic_vars, phase, iteration_num):
    """Display a simplex tableau"""
    # Create header
    headers = ["Basic Var"] + var_names
    
    # Create data rows
    data = []
    for i in range(len(tableau) - 1):
        row_data = [basic_vars[i]]
        for val in tableau[i]:
            row_data.append(f"{val:.4f}")
        data.append(row_data)
    
    # Add objective row
    obj_label = "w" if phase == 1 else "Z"
    obj_row = [obj_label]
    for val in tableau[-1]:
        obj_row.append(f"{val:.4f}")
    data.append(obj_row)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)
    
    # Display with styling
    st.dataframe(df, use_container_width=True, hide_index=True)

def display_solution(solver, obj_type):
    """Display the final solution"""
    st.markdown('<div class="solution-box">', unsafe_allow_html=True)
    st.markdown("### ✅ Optimal Solution Found!")
    
    # Display variable values
    st.write("**Decision Variables:**")
    cols = st.columns(4)
    var_values = []
    
    for i in range(solver.num_decision_vars):
        var_name = f'x{i+1}'
        if var_name in solver.basic_vars:
            idx = solver.basic_vars.index(var_name)
            value = solver.tableau[idx, -1]
        else:
            value = 0
        var_values.append(value)
        
        with cols[i % 4]:
            st.metric(f"x{i+1}", f"{value:.4f}")
    
    # Display objective value
    optimal_value = solver.tableau[-1, -1]
    if not solver.is_maximization:
        optimal_value = -optimal_value
    
    st.metric("**Optimal Objective Value (Z)**", f"{optimal_value:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">📊 Simplex Method Solver</div>', unsafe_allow_html=True)
    
    # Sidebar with instructions
    with st.sidebar:
        st.markdown("### 📖 Instructions")
        st.markdown("""
        **Objective Function:**
        - Enter coefficients for decision variables
        - Choose MAX or MIN
        
        **Constraints:**
        - Use format: `coefficients inequality RHS`
        - Example: `1 2 <= 10` means x₁ + 2x₂ ≤ 10
        - Supported inequalities: `<=`, `>=`, `=`
        
        **Input Formats:**
        - **Space-separated**: `2 5 1` 
        - **Mathematical expression**: `2x1 + 5x2 + x3`
        - **Mixed**: `2x1 + 3x2 - 1x3` or `2x1+3x2-1x3`
        """)
        
        st.markdown("### 💡 Valid Expression Examples")
        st.markdown("""
        - `2x1 + 3x2 - x3`
        - `x1 - 2x2 + 5x3`
        - `3x1+2x2-4x3` (no spaces)
        - `-x1 + x2 - x3`
        - `2x1 - 3x2`
        """)
        
        st.markdown("### ⚠️ Common Issues")
        st.markdown("""
        - Don't use `*` between coefficient and variable
        - Include `+` or `-` between terms
        - Variable names must be `x1`, `x2`, etc.
        - Ensure RHS values are non-negative
        """)

    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">🔧 Problem Setup</div>', unsafe_allow_html=True)
        
        # Problem parameters
        num_vars = st.number_input("Number of Decision Variables", min_value=1, max_value=10, value=2, step=1, key="num_vars")
        num_constraints = st.number_input("Number of Constraints", min_value=1, max_value=10, value=2, step=1, key="num_constraints")
        
        # Objective function
        st.markdown("#### Objective Function")
        obj_type = st.radio("Optimization Type:", ["max", "min"], horizontal=True, key="obj_type")
        
        obj_format = st.radio("Input Format:", ["Space-separated numbers", "Mathematical expression"], 
                            horizontal=True, key="obj_format")
        
        if obj_format == "Space-separated numbers":
            obj_input = st.text_input(
                f"Enter coefficients for x₁ to x{num_vars}:", 
                value="3 2",
                help=f"Enter {num_vars} numbers separated by spaces. Example: '3 2' for 3x₁ + 2x₂",
                key="obj_input"
            )
            if obj_input:
                st.caption(f"Interpreted as: {obj_input.replace(' ', 'x₁ + ')}x{num_vars}")
        else:
            obj_input = st.text_input(
                "Enter objective function:", 
                value="3x1 + 2x2",
                help="Example: 2x1 + 3x2 - x3 or 2x1+3x2-1x3",
                key="obj_expr"
            )
            if obj_input:
                st.caption("Make sure to use format like: 2x1 + 3x2 - x3")

        # Show parsing preview
        if obj_input:
            try:
                preview_solver = SimplexSolver()
                preview_coeffs = preview_solver.parse_coefficients(obj_input, num_vars)
                preview_terms = []
                for i, coeff in enumerate(preview_coeffs):
                    if abs(coeff) > 1e-10:
                        if i == 0:
                            preview_terms.append(f"{coeff:g}x{i+1}")
                        else:
                            sign = "+" if coeff >= 0 else "-"
                            preview_terms.append(f"{sign} {abs(coeff):g}x{i+1}")
                if preview_terms:
                    st.info(f"**Parsed as:** {' '.join(preview_terms)}")
            except:
                pass
    
    with col2:
        st.markdown('<div class="sub-header">📋 Constraints</div>', unsafe_allow_html=True)
        
        constraints = []
        constraint_types = []
        rhs_values = []
        
        for i in range(num_constraints):
            st.markdown(f"**Constraint {i+1}**")
            col_a, col_b, col_c = st.columns([3, 1, 2])
            
            with col_a:
                if obj_format == "Space-separated numbers":
                    coeff_input = st.text_input(
                        f"Coefficients {i+1}", 
                        value="2 1" if i == 0 else "1 1",
                        key=f"coeff_{i}",
                        label_visibility="collapsed",
                        help=f"Enter {num_vars} numbers separated by spaces"
                    )
                else:
                    coeff_input = st.text_input(
                        f"Expression {i+1}", 
                        value="2x1 + 1x2" if i == 0 else "1x1 + 1x2",
                        key=f"expr_{i}",
                        label_visibility="collapsed",
                        help="Example: 2x1 + 3x2 - x3"
                    )
            
            with col_b:
                inequality = st.selectbox(
                    "Inequality", 
                    ["<=", ">=", "="], 
                    key=f"ineq_{i}", 
                    label_visibility="collapsed"
                )
            
            with col_c:
                rhs = st.number_input(
                    "RHS", 
                    value=18.0 if i == 0 else 10.0, 
                    key=f"rhs_{i}", 
                    label_visibility="collapsed"
                )
            
            constraints.append(coeff_input)
            constraint_types.append(inequality)
            rhs_values.append(rhs)

    # Solve button
    if st.button("🚀 Solve using Simplex Method", use_container_width=True, type="primary"):
        if not obj_input.strip():
            st.error("❌ Please enter the objective function coefficients.")
            return
            
        try:
            # Initialize solver
            solver = SimplexSolver()
            solver.num_decision_vars = num_vars
            solver.num_constraints = num_constraints
            solver.is_maximization = (obj_type == "max")
            
            # Parse objective function
            solver.original_obj_coeffs = solver.parse_coefficients(obj_input, num_vars)
            
            if not solver.is_maximization:
                solver.obj_coeffs = [-c for c in solver.original_obj_coeffs]
            else:
                solver.obj_coeffs = solver.original_obj_coeffs.copy()
            
            # Parse constraints
            parsed_constraints = []
            for i in range(num_constraints):
                coeffs = solver.parse_coefficients(constraints[i], num_vars)
                parsed_constraints.append(coeffs)
            
            # Build tableau
            solver.build_tableau(parsed_constraints, constraint_types, rhs_values)
            
            # Display problem formulation
            display_problem_formulation(solver, obj_type, parsed_constraints, constraint_types, rhs_values)
            
            # Solve the problem
            st.markdown("---")
            st.markdown('<div class="sub-header">🔄 Solution Process</div>', unsafe_allow_html=True)
            
            success, message = solver.solve_problem()
            
            if not success:
                if message == "NO_FEASIBLE_SOLUTION":
                    st.markdown('<div class="error-box">❌ No Feasible Solution Exists!</div>', unsafe_allow_html=True)
                    st.write("The problem constraints are inconsistent and no feasible solution can be found.")
                elif message == "UNBOUNDED_PHASE1":
                    st.markdown('<div class="error-box">❌ Unbounded in Phase 1!</div>', unsafe_allow_html=True)
                elif message == "UNBOUNDED":
                    st.markdown('<div class="warning-box">⚠️ Unbounded Solution!</div>', unsafe_allow_html=True)
                    st.write("The objective function can be improved indefinitely without violating constraints.")
                return
            
            # Display iterations
            st.markdown("### 📊 Solution Steps")
            
            for i, iteration in enumerate(solver.iterations):
                with st.expander(f"{'Phase 1' if i == 0 or (i > 0 and 'phase' not in iteration) else 'Phase 2'} - Iteration {i}", expanded=i == len(solver.iterations)-1):
                    if iteration['pivot_row'] is not None:
                        st.write(f"**Pivot:** Row {iteration['pivot_row'] + 1} ({solver.basic_vars[iteration['pivot_row']] if i < len(solver.iterations)-1 else iteration['basic_vars'][iteration['pivot_row']]}), "
                                f"Column {iteration['pivot_col'] + 1} ({solver.var_names[iteration['pivot_col']] if i < len(solver.iterations)-1 else ['x' + name for name in solver.var_names if 'x' in name][iteration['pivot_col']]})")
                        st.write(f"**Pivot Element:** {iteration['pivot_element']:.4f}")
                    
                    phase = 1 if i == 0 or (i > 0 and 'phase' not in iteration) else 2
                    display_tableau(iteration['tableau'], 
                                  solver.var_names if i < len(solver.iterations)-1 else [name for name in solver.var_names if not name.startswith('A')],
                                  iteration['basic_vars'], 
                                  phase, i)
            
            # Display final solution
            display_solution(solver, obj_type)
            
        except Exception as e:
            st.error(f"❌ Error solving the problem: {str(e)}")
            st.info("💡 Please check your input format. For mathematical expressions, use format like: '2x1 + 3x2 - x3'")

    # Examples section
    with st.expander("📚 Example Problems", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Example 1: Maximization**")
            st.code("""
Maximize: Z = 3x1 + 2x2
Subject to:
  2x1 + x2 ≤ 18
  2x1 + 3x2 ≤ 42
  3x1 + x2 ≤ 24
  x1, x2 ≥ 0

Input format (expression):
Objective: 3x1 + 2x2
Constraints: 
  2x1 + 1x2 <= 18
  2x1 + 3x2 <= 42  
  3x1 + 1x2 <= 24
            """)
        
        with col2:
            st.markdown("**Example 2: Minimization**")
            st.code("""
Minimize: Z = 4x1 + x2
Subject to:
  3x1 + x2 = 3
  4x1 + 3x2 ≥ 6
  x1 + 2x2 ≤ 4
  x1, x2 ≥ 0

Input format (expression):
Objective: 4x1 + 1x2
Constraints:
  3x1 + 1x2 = 3
  4x1 + 3x2 >= 6
  1x1 + 2x2 <= 4
            """)

if __name__ == "__main__":
    main()
