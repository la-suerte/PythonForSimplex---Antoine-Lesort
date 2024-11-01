import csv 
import pandas as pd
from ortools.linear_solver import pywraplp
import argparse
import numpy as np


def get_lp_input():
    print("Enter the number of variables:")
    n_vars = int(input())

    print("Enter the objective function coefficients (e.g., '3 4' for 2 variables or '1 2 3' for 3 variables):")
    obj_coeffs = list(map(float, input().split()))

    if len(obj_coeffs) != n_vars:
        print("Error: Number of coefficients must match the number of variables.")
        return None, None, None

    print("Enter the number of constraints:")
    n_constraints = int(input())
    constraints = []
    bounds = []

    for i in range(n_constraints):
        print(f"Enter coefficients for constraint {i+1} (e.g., '1 2' for 2 variables or '1 2 3' for 3 variables):")
        coeffs = list(map(float, input().split()))
        if len(coeffs) != n_vars:
            print(f"Error: Number of coefficients must match the number of variables ({n_vars}).")
            return None, None, None

        print("Enter the inequality (<=, >=, =):")
        inequality = input()
        print("Enter the bound for this constraint:")
        bound = float(input())
        constraints.append((coeffs, inequality, bound))

    return n_vars, obj_coeffs, constraints



def get_lp_input_2(filename):
    """
    Read linear programming problem from an Excel file for n-dimensional problems.
    The first row contains the coefficients of the objective function.
    Subsequent rows contain the coefficients of constraints in the form:
    coeff1, coeff2, ..., coeffN, inequality, bound
    """
    constraints = []
    
    df = pd.read_excel(filename, header=None)
    
    rows = df.values.tolist()

    obj_coeffs = [float(x) for x in rows[0] if pd.notna(x)]

    n_vars = len(obj_coeffs)

    print("The constraints : \n")
    for row in rows[1:]:
        coeffs = [float(x) for x in row[:n_vars]]
        inequality = row[n_vars]
        bound = float(row[n_vars + 1])
        print(f'{coeffs} {inequality} {bound}')


        if len(coeffs) != n_vars:
            raise ValueError(f"Number of coefficients in constraint does not match number of objective function variables. Expected {n_vars}, got {len(coeffs)}.")
        
        
        if inequality == '<=':
            constraints.append((coeffs, '<=', bound))
        elif inequality == '>=':
            constraints.append((coeffs, '>=', bound))
        elif inequality == '=':
            constraints.append((coeffs, '=', bound))
        else:
            raise ValueError(f"Unknown inequality: {inequality}")
    return n_vars, obj_coeffs, constraints


# Method 1: Solving with Google OR-Tools
def solve_with_ortools(n_vars, obj_coeffs, constraints):
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        print("Solver not available.")
        return

    vars = [solver.NumVar(0, solver.infinity(), f"x{i}") for i in range(n_vars)]
    solver.Maximize(sum(obj_coeffs[i] * vars[i] for i in range(n_vars)))

    for coeffs, inequality, bound in constraints:
        expr = sum(coeffs[i] * vars[i] for i in range(n_vars))
        if inequality == "<=":
            solver.Add(expr <= bound)
        elif inequality == ">=":
            solver.Add(expr >= bound)
        else:
            solver.Add(expr == bound)

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print("Solution found:")
        print(f"Objective value = {solver.Objective().Value():0.1f}")
        for i, var in enumerate(vars):
            print(f"x{i} = {var.solution_value():0.1f}")
    else:
        print("The problem does not have an optimal solution.")

# Method 2: Simplex Method by Tableau 


def initialize_tableau(n_vars, obj_coeffs, constraints):
    """
    Initializes the simplex tableau with slack/surplus variables as needed.
    """
    num_constraints = len(constraints)
    tableau = np.zeros((num_constraints + 1, n_vars + num_constraints + 1))

    tableau[-1, :n_vars] = -np.array(obj_coeffs)  # Negative for maximization

    for i, (coeffs, inequality, bound) in enumerate(constraints):
        tableau[i, :n_vars] = coeffs  # Coefficients for the variables
        
        if inequality == "<=":
            tableau[i, n_vars + i] = 1  # Slack variable
        elif inequality == ">=":
            tableau[i, n_vars + i] = -1  # Surplus variable
        else:
            tableau[i, n_vars + i] = 1
        
        tableau[i, -1] = bound

    return tableau


def find_pivot_column(tableau):
    """
    Find the pivot column (most negative indicator in the objective row).
    """
    last_row = tableau[-1, :-1]
    pivot_col = np.argmin(last_row)
    if last_row[pivot_col] >= 0:
        return -1  # No negative indicators, optimal solution found
    return pivot_col

def find_pivot_row(tableau, pivot_col):
    """
    Find the pivot row using the minimum positive ratio test.
    """
    rhs = tableau[:-1, -1]
    pivot_col_values = tableau[:-1, pivot_col]
    ratios = []

    for i, val in enumerate(pivot_col_values):
        if val > 0:  # Positive entries only
            ratios.append(rhs[i] / val)
        else:
            ratios.append(np.inf)

    pivot_row = np.argmin(ratios)
    if ratios[pivot_row] == np.inf:
        return -1  # Unbounded
    return pivot_row

def pivot(tableau, pivot_row, pivot_col):
    """
    Perform the pivot operation to update the tableau.
    """
    pivot_value = tableau[pivot_row, pivot_col]
    tableau[pivot_row] /= pivot_value  # Make pivot 1

    for i in range(tableau.shape[0]):
        if i != pivot_row:
            row_factor = tableau[i, pivot_col]
            tableau[i] -= row_factor * tableau[pivot_row]  # Zero out pivot column entries
            
def print_tableau(tableau, pivot_col=None):
    """
    Print the tableau, highlighting the pivot column in red.
    """
    red_color = '\033[91m'  
    reset_color = '\033[0m' 

    for i, row in enumerate(tableau):
        for j, value in enumerate(row):
            if j == pivot_col:
                print(f"{red_color}{value:>10.2f}{reset_color}", end=" ")
            else:
                print(f"{value:>10.2f}", end=" ")
        print()  # Newline after each row

def solve_with_simplex(n_vars, obj_coeffs, constraints):
    """
    Solve the LP using the Simplex method with tableau representation.
    """
    tableau = initialize_tableau(n_vars, obj_coeffs, constraints)
    print("Initial Tableau:")
    print_tableau(tableau)
    print()

    while True:
        pivot_col = find_pivot_column(tableau)
        if pivot_col == -1:
            print("Optimal solution found.")
            break  # Optimal solution found

        pivot_row = find_pivot_row(tableau, pivot_col)
        if pivot_row == -1:
            print("The problem is unbounded.")
            return  # Unbounded solution

        pivot(tableau, pivot_row, pivot_col)

        print("Updated Tableau:")
        print_tableau(tableau, pivot_col)
        print()

    # Extracting solution
    solution = np.zeros(n_vars)
    for i in range(n_vars):
        column = tableau[:, i]
        if np.sum(column == 1) == 1 and np.sum(column) == 1:
            one_index = np.where(column == 1)[0][0]
            solution[i] = tableau[one_index, -1]

    objective_value = tableau[-1, -1]
    print(f"Optimal Objective Value: {objective_value}")
    print("Solution:", solution)



# Main menu to run the program
def main():
    print("Welcome to the Linear Programming Solver.")
    choice = input("Do you want to manually input the data (D) or use a pre-made file? (P) ")
    if(choice == 'P'):
        choice_2 = input("Do you want to use the data from Google example (G) or Exercise 5 (5) or Exercise 6 (6) Exercise 7 (7) ")
        if(choice_2 == "G"):
            n_vars, obj_coeffs, constraints = get_lp_input_2(r"C:\Users\lesor\OneDrive\Dokumenter\WORK ESILV\A3 - VIC\PythonForSimplex - Antoine Lesort\OR data.xlsx")
        elif(choice_2 == "5"):
            n_vars, obj_coeffs, constraints = get_lp_input_2(r"C:\Users\lesor\OneDrive\Dokumenter\WORK ESILV\A3 - VIC\PythonForSimplex - Antoine Lesort\Exercise5.xlsx")
        elif(choice_2 == "6"):
            n_vars, obj_coeffs, constraints = get_lp_input_2(r"C:\Users\lesor\OneDrive\Dokumenter\WORK ESILV\A3 - VIC\PythonForSimplex - Antoine Lesort\Exercise6.xlsx")
        elif(choice_2 == "7"):
            n_vars, obj_coeffs, constraints = get_lp_input_2(r"C:\Users\lesor\OneDrive\Dokumenter\WORK ESILV\A3 - VIC\PythonForSimplex - Antoine Lesort\Exercise7.xlsx")
        else:
            print("Incorrect input")
    elif(choice=='D'):
      n_vars, obj_coeffs, constraints = get_lp_input()  



    print("\n----------         1 - Google OR-Tools          ----------")

    solve_with_ortools(n_vars, obj_coeffs, constraints)
    
    print("\n----------    2 - Simplex Method by Tableau    ----------")
    solve_with_simplex(n_vars, obj_coeffs, constraints)

if __name__ == "__main__":
    main()
