import os
import sys
import argparse
from magic_square_genetic import run_experiment, compare_algorithms
import numpy as np
import random
import time


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the program header."""
    print("\n" + "=" * 80)
    print(f"{'MAGIC SQUARE GENETIC ALGORITHM':^80}")
    print("=" * 80)
    print("Computational Biology Assignment - Magic Square Generator using Genetic Algorithms")
    print("-" * 80)


def print_menu():
    """Print the main menu options."""
    print("\nMAIN MENU:")
    print("1. Run a single algorithm")
    print("2. Compare all algorithms")
    print("3. Quick demo (3x3 magic square)")
    print("4. Advanced settings")
    print("5. What is a Magic Square?")
    print("6. Exit")
    print("\nEnter your choice (1-6): ")


def get_valid_int_input(prompt, min_val=1, max_val=None):
    """Get a valid integer input from the user."""
    while True:
        try:
            value = int(input(prompt))
            if min_val is not None and value < min_val:
                print(f"Please enter a value >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Please enter a value <= {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid integer.")


def get_valid_float_input(prompt, min_val=0.0, max_val=1.0):
    """Get a valid float input from the user."""
    while True:
        try:
            value = float(input(prompt))
            if value < min_val or value > max_val:
                print(f"Please enter a value between {min_val} and {max_val}")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


def get_yes_no_input(prompt):
    """Get a yes/no input from the user."""
    while True:
        response = input(prompt).strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'.")


def run_single_algorithm():
    """Run a single genetic algorithm with user-specified parameters."""
    clear_screen()
    print_header()
    print("\nRUN SINGLE ALGORITHM")
    print("-" * 80)

    # Get algorithm type
    print("\nChoose algorithm type:")
    print("1. Classical genetic algorithm")
    print("2. Darwinian genetic algorithm")
    print("3. Lamarckian genetic algorithm")

    choice = get_valid_int_input("Enter your choice (1-3): ", 1, 3)
    algorithm_types = ["classical", "darwinian", "lamarckian"]
    algorithm_type = algorithm_types[choice - 1]

    # Get magic square size
    n = get_valid_int_input("\nEnter the size of the magic square (n x n): ", 3, 15)

    # Check if user wants most perfect magic square
    most_perfect = False
    if n % 4 == 0 and n > 0:
        most_perfect = get_yes_no_input("\nDo you want to generate a Most Perfect Magic Square? (y/n): ")

    # Get optional parameters or use defaults
    if get_yes_no_input("\nDo you want to adjust advanced settings? (y/n): "):
        population_size = get_valid_int_input("Population size (recommended: 100): ", 10)
        max_generations = get_valid_int_input("Maximum generations (recommended: 1000): ", 10)
        mutation_rate = get_valid_float_input("Mutation rate (0.0-1.0, recommended: 0.1): ", 0.0, 1.0)
        elite_size = get_valid_int_input(f"Elite size (recommended: {min(10, population_size // 10)}): ", 1,
                                         population_size // 2)
        tournament_size = get_valid_int_input(f"Tournament size (recommended: {min(5, population_size // 20)}): ", 2,
                                              population_size // 2)
        crossover_rate = get_valid_float_input("Crossover rate (0.0-1.0, recommended: 0.8): ", 0.0, 1.0)
        optimization_steps = get_valid_int_input(f"Optimization steps (recommended: {n}): ", 1)
    else:
        # Use default values
        population_size = 100
        max_generations = 1000
        mutation_rate = 0.1
        elite_size = 10
        tournament_size = 5
        crossover_rate = 0.8
        optimization_steps = n

    # Confirm and run
    print("\nRunning algorithm with the following parameters:")
    print(f"Algorithm type: {algorithm_type.capitalize()}")
    print(f"Magic square size: {n}x{n}")
    print(f"Most perfect: {most_perfect}")
    print(f"Population size: {population_size}")
    print(f"Maximum generations: {max_generations}")
    print(f"Mutation rate: {mutation_rate}")
    print(f"Elite size: {elite_size}")
    print(f"Tournament size: {tournament_size}")
    print(f"Crossover rate: {crossover_rate}")
    print(f"Optimization steps: {optimization_steps}")

    if get_yes_no_input("\nStart algorithm with these parameters? (y/n): "):
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        run_experiment(
            n=n,
            algorithm_type=algorithm_type,
            max_generations=max_generations,
            population_size=population_size,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            optimization_steps=optimization_steps,
            most_perfect=most_perfect
        )

        input("\nPress Enter to return to the main menu...")
    else:
        print("Operation cancelled.")


def run_algorithm_comparison():
    """Run and compare all three algorithm types."""
    clear_screen()
    print_header()
    print("\nCOMPARE ALL ALGORITHMS")
    print("-" * 80)

    # Get magic square size
    n = get_valid_int_input("\nEnter the size of the magic square (n x n): ", 3, 15)

    # Check if user wants most perfect magic square
    most_perfect = False
    if n % 4 == 0 and n > 0:
        most_perfect = get_yes_no_input("\nDo you want to generate a Most Perfect Magic Square? (y/n): ")

    # Get optional parameters or use defaults
    if get_yes_no_input("\nDo you want to adjust advanced settings? (y/n): "):
        population_size = get_valid_int_input("Population size (recommended: 100): ", 10)
        max_generations = get_valid_int_input("Maximum generations (recommended: 1000): ", 10)
        mutation_rate = get_valid_float_input("Mutation rate (0.0-1.0, recommended: 0.1): ", 0.0, 1.0)
        elite_size = get_valid_int_input(f"Elite size (recommended: {min(10, population_size // 10)}): ", 1,
                                         population_size // 2)
        tournament_size = get_valid_int_input(f"Tournament size (recommended: {min(5, population_size // 20)}): ", 2,
                                              population_size // 2)
        crossover_rate = get_valid_float_input("Crossover rate (0.0-1.0, recommended: 0.8): ", 0.0, 1.0)
        optimization_steps = get_valid_int_input(f"Optimization steps (recommended: {n}): ", 1)
    else:
        # Use default values
        population_size = 100
        max_generations = 1000
        mutation_rate = 0.1
        elite_size = 10
        tournament_size = 5
        crossover_rate = 0.8
        optimization_steps = n

    # Confirm and run
    print("\nRunning comparison with the following parameters:")
    print(f"Magic square size: {n}x{n}")
    print(f"Most perfect: {most_perfect}")
    print(f"Population size: {population_size}")
    print(f"Maximum generations: {max_generations}")
    print(f"Mutation rate: {mutation_rate}")
    print(f"Elite size: {elite_size}")
    print(f"Tournament size: {tournament_size}")
    print(f"Crossover rate: {crossover_rate}")
    print(f"Optimization steps: {optimization_steps}")

    if get_yes_no_input("\nStart comparison with these parameters? (y/n): "):
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        compare_algorithms(
            n=n,
            max_generations=max_generations,
            population_size=population_size,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            optimization_steps=optimization_steps,
            most_perfect=most_perfect
        )

        input("\nPress Enter to return to the main menu...")
    else:
        print("Operation cancelled.")


def run_quick_demo():
    """Run a quick demo with a 3x3 magic square."""
    clear_screen()
    print_header()
    print("\nQUICK DEMO - 3x3 MAGIC SQUARE")
    print("-" * 80)

    print("\nRunning a quick demonstration with a 3x3 magic square...")
    print("Using Lamarckian genetic algorithm with default parameters.")

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Run the experiment with default parameters
    run_experiment(
        n=3,
        algorithm_type="lamarckian",
        max_generations=1000,
        population_size=100,
        mutation_rate=0.1,
        elite_size=10,
        tournament_size=5,
        crossover_rate=0.8,
        optimization_steps=3
    )

    input("\nPress Enter to return to the main menu...")


def advanced_settings():
    """Show advanced settings and explanations."""
    clear_screen()
    print_header()
    print("\nADVANCED SETTINGS EXPLANATION")
    print("-" * 80)

    print("\nMagic Square Size (n):")
    print("  - Determines the dimensions of the magic square (n x n)")
    print("  - Larger sizes are more challenging to solve")
    print("  - Recommended: 3-5 for regular magic squares")
    print("  - For Most Perfect Magic Squares, use values divisible by 4 (4, 8, 12)")

    print("\nAlgorithm Types:")
    print("  1. Classical Genetic Algorithm")
    print("     - Traditional GA with selection, crossover, and mutation")
    print("  2. Darwinian Genetic Algorithm")
    print("     - Individuals undergo local optimization before fitness evaluation")
    print("     - But original (unoptimized) individuals are used for reproduction")
    print("  3. Lamarckian Genetic Algorithm")
    print("     - Individuals undergo local optimization before fitness evaluation")
    print("     - Optimized individuals are used for reproduction")

    print("\nPopulation Size:")
    print("  - Number of individual solutions in each generation")
    print("  - Larger populations provide more diversity but increase computation time")
    print("  - Recommended: 100-200")

    print("\nMaximum Generations:")
    print("  - Maximum number of generations to evolve")
    print("  - The algorithm will stop earlier if a solution is found")
    print("  - Recommended: 1000-2000")

    print("\nMutation Rate:")
    print("  - Probability of mutation for each gene (0.0-1.0)")
    print("  - Higher values increase exploration but may disrupt good solutions")
    print("  - Recommended: 0.05-0.2")

    print("\nElite Size:")
    print("  - Number of top individuals preserved unchanged in each generation")
    print("  - Ensures the best solutions are not lost")
    print("  - Recommended: 5-10% of population size")

    print("\nTournament Size:")
    print("  - Number of individuals in each selection tournament")
    print("  - Larger values increase selection pressure")
    print("  - Recommended: 3-7")

    print("\nCrossover Rate:")
    print("  - Probability of crossover between parents (0.0-1.0)")
    print("  - Higher values promote more genetic exchange")
    print("  - Recommended: 0.7-0.9")

    print("\nOptimization Steps:")
    print("  - Number of local optimization steps per generation")
    print("  - Only applicable for Darwinian and Lamarckian algorithms")
    print("  - Recommended: Same as the magic square size (n)")

    input("\nPress Enter to return to the main menu...")


def show_magic_square_info():
    """Display information about magic squares."""
    clear_screen()
    print_header()
    print("\nWHAT IS A MAGIC SQUARE?")
    print("-" * 80)

    print("\nA Magic Square is an n×n grid filled with numbers from 1 to n² such that:")
    print("  - Every row sum equals the same value (magic sum)")
    print("  - Every column sum equals the same value (magic sum)")
    print("  - Both main diagonal sums equal the same value (magic sum)")

    print("\nThe magic sum for an n×n magic square is: n(n²+1)/2")
    print("\nExample of a 3×3 magic square:")
    print("  8  1  6")
    print("  3  5  7")
    print("  4  9  2")
    print("\nHere, all rows, columns, and diagonals sum to 15.")

    print("\nA Most Perfect Magic Square is a special type that satisfies additional conditions:")
    print("  - It can only exist when n is a multiple of 4 (4, 8, 12, etc.)")
    print("  - All pairs of numbers that are equidistant from the center sum to n²+1")
    print("  - The sum of all 2×2 subsquares equals 2(n²+1)")

    input("\nPress Enter to return to the main menu...")


def main():
    """Main program loop."""
    # Check if the required module exists
    try:
        import magic_square_genetic
    except ImportError:
        print("Error: magic_square_genetic.py not found in the current directory.")
        print("Please ensure the file is in the same directory as this script.")
        sys.exit(1)

    # Check if matplotlib is installed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not found. Graphs will not be displayed.")
        print("To install matplotlib, run: pip install matplotlib")

    while True:
        clear_screen()
        print_header()
        print_menu()

        try:
            choice = int(input())

            if choice == 1:
                run_single_algorithm()
            elif choice == 2:
                run_algorithm_comparison()
            elif choice == 3:
                run_quick_demo()
            elif choice == 4:
                advanced_settings()
            elif choice == 5:
                show_magic_square_info()
            elif choice == 6:
                clear_screen()
                print("Thank you for using the Magic Square Genetic Algorithm. Goodbye!")
                sys.exit(0)
            else:
                print("Invalid choice. Please enter a number between 1 and 6.")
                time.sleep(1)
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 6.")
            time.sleep(1)
        except KeyboardInterrupt:
            clear_screen()
            print("\nProgram interrupted. Exiting...")
            sys.exit(0)


if __name__ == "__main__":
    # Parse command line arguments if provided
    parser = argparse.ArgumentParser(description='Magic Square Genetic Algorithm Menu')
    parser.add_argument('--direct', action='store_true', help='Skip menu and run with command-line arguments')
    parser.add_argument('--n', type=int, default=3, help='Size of the magic square (default: 3)')
    parser.add_argument('--alg', type=str, default='all', choices=['all', 'classical', 'darwinian', 'lamarckian'],
                        help='Algorithm type (default: all)')
    parser.add_argument('--pop', type=int, default=100, help='Population size (default: 100)')
    parser.add_argument('--gen', type=int, default=1000, help='Maximum generations (default: 1000)')
    parser.add_argument('--mut', type=float, default=0.1, help='Mutation rate (default: 0.1)')
    parser.add_argument('--elite', type=int, default=10, help='Elite size (default: 10)')
    parser.add_argument('--tournament', type=int, default=5, help='Tournament size (default: 5)')
    parser.add_argument('--crossover', type=float, default=0.8, help='Crossover rate (default: 0.8)')
    parser.add_argument('--opt_steps', type=int, default=None, help='Optimization steps (default: n)')
    parser.add_argument('--most_perfect', action='store_true', help='Generate a Most Perfect Magic Square')

    args = parser.parse_args()

    # If direct mode is specified, bypass the menu
    if args.direct:
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        # Use provided arguments to run the algorithm
        if args.alg == 'all':
            compare_algorithms(
                n=args.n,
                max_generations=args.gen,
                population_size=args.pop,
                mutation_rate=args.mut,
                elite_size=args.elite,
                tournament_size=args.tournament,
                crossover_rate=args.crossover,
                optimization_steps=args.opt_steps if args.opt_steps else args.n,
                most_perfect=args.most_perfect
            )
        else:
            run_experiment(
                n=args.n,
                algorithm_type=args.alg,
                max_generations=args.gen,
                population_size=args.pop,
                mutation_rate=args.mut,
                elite_size=args.elite,
                tournament_size=args.tournament,
                crossover_rate=args.crossover,
                optimization_steps=args.opt_steps if args.opt_steps else args.n,
                most_perfect=args.most_perfect
            )
    else:
        # Start the interactive menu
        main()
