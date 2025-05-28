import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import List, Tuple, Optional
from enum import Enum


class AlgorithmType(Enum):
    """Enumeration for the different genetic algorithm variants."""
    CLASSICAL = "classical"
    DARWINIAN = "darwinian"
    LAMARCKIAN = "lamarckian"


class MagicSquareGA:
    """
    Genetic algorithm for solving magic squares.

    This algorithm can solve both standard magic squares and most-perfect magic squares
    using classical, Darwinian, or Lamarckian genetic algorithms.
    """

    def __init__(self,
                 n: int,
                 population_size: int = 100,
                 mutation_rate: float = 0.1,
                 elite_size: int = 10,
                 tournament_size: int = 5,
                 crossover_rate: float = 0.8,
                 algorithm_type: str = "classical",
                 optimization_steps: Optional[int] = None,
                 most_perfect: bool = False):
        """
        Initialize the genetic algorithm for magic square generation.

        Args:
            n: Size of the magic square (n x n)
            population_size: Number of individuals in each generation
            mutation_rate: Probability of mutation
            elite_size: Number of the best individuals to keep unchanged in next generation
            tournament_size: Number of individuals in each tournament selection
            crossover_rate: Probability of crossover between parents
            algorithm_type: "classical", "darwinian", or "lamarckian"
            optimization_steps: Number of local optimization steps (default is n)
            most_perfect: Whether to search for a most-perfect magic square
        """
        self.n = n
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.initial_mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.algorithm_type = algorithm_type
        self.optimization_steps = n if optimization_steps is None else optimization_steps
        self.most_perfect = most_perfect

        # Calculate the magic sum (target sum for rows, columns, and diagonals)
        self.magic_sum = n * (n ** 2 + 1) // 2

        # Tracking variables
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.evaluation_calls = 0

        # Initialize the population
        self.population = self._initialize_population()

    def _initialize_population(self) -> List[np.ndarray]:
        """
        Initialize a population of random magic square candidates.

        Returns:
            List of n x n numpy arrays representing magic square candidates
        """
        population = []
        for _ in range(self.population_size):
            # Generate a random permutation of numbers 1 to n^2
            numbers = list(range(1, self.n ** 2 + 1))
            random.shuffle(numbers)
            square = np.array(numbers).reshape(self.n, self.n)
            population.append(square)
        return population

    def fitness(self, square: np.ndarray) -> float:
        """
        Calculate the fitness of a magic square candidate.

        Lower values indicate better fitness, with 0 being a perfect solution.
        Fitness is measured as the total deviation from the magic sum across all
        rows, columns, and diagonals.

        Args:
            square: n x n numpy array representing a magic square candidate

        Returns:
            Total deviation from the magic sum (lower is better, 0 is perfect)
        """
        self.evaluation_calls += 1
        total_deviation = 0

        # Check row sums
        row_sums = np.sum(square, axis=1)
        total_deviation += np.sum(np.abs(row_sums - self.magic_sum))

        # Check column sums
        col_sums = np.sum(square, axis=0)
        total_deviation += np.sum(np.abs(col_sums - self.magic_sum))

        if self.most_perfect:
            # All broken diagonals must also have the magic sum for most-perfect squares
            for k in range(self.n):
                diag1_sum = 0
                diag2_sum = 0
                for i in range(self.n):
                    diag1_sum += square[i, (i + k) % self.n]
                    diag2_sum += square[i, (k - i + self.n) % self.n]
                total_deviation += abs(diag1_sum - self.magic_sum)
                total_deviation += abs(diag2_sum - self.magic_sum)
        else:
            # For standard magic squares, only main diagonals matter
            main_diag_sum = np.sum(np.diag(square))
            anti_diag_sum = np.sum(np.diag(np.fliplr(square)))
            total_deviation += abs(main_diag_sum - self.magic_sum)
            total_deviation += abs(anti_diag_sum - self.magic_sum)

        if self.most_perfect and self.n % 4 == 0:
            # Additional constraints for most-perfect magic squares

            # 1. 2x2 sub-square sum constraint
            sub_square_sum_target = 2 * (self.n ** 2 + 1)
            for r_idx in range(self.n):
                for c_idx in range(self.n):
                    s = (square[r_idx, c_idx] +
                         square[r_idx, (c_idx + 1) % self.n] +
                         square[(r_idx + 1) % self.n, c_idx] +
                         square[(r_idx + 1) % self.n, (c_idx + 1) % self.n])
                    total_deviation += abs(s - sub_square_sum_target)

            # 2. Complementary pair constraint
            pair_sum_target = self.n ** 2 + 1
            n_half = self.n // 2
            for r_p2 in range(n_half):
                for c_p2 in range(self.n):
                    val1 = square[r_p2, c_p2]
                    partner_r = (r_p2 + n_half) % self.n
                    partner_c = (c_p2 + n_half) % self.n
                    val2 = square[partner_r, partner_c]
                    total_deviation += abs(val1 + val2 - pair_sum_target)

        return total_deviation

    def is_solution(self, square: np.ndarray) -> bool:
        """
        Check if a square is a valid magic square.

        Args:
            square: n x n numpy array representing a magic square candidate

        Returns:
            True if the square is a valid magic square, False otherwise
        """
        return self.fitness(square) == 0

    def tournament_selection(self, population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
        """
        Select an individual using tournament selection.

        Args:
            population: List of individuals to select from
            fitnesses: List of fitness values corresponding to the population

        Returns:
            Selected individual (copied to avoid modifying the original)
        """
        # Select tournament_size individuals randomly
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

        # Find the best individual in the tournament
        best_idx_in_tournament = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
        return population[best_idx_in_tournament].copy()

    def order_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Perform order crossover between two parents.

        This preserves the relative order of elements from one parent while
        filling in the remaining positions with elements from the other parent.

        Args:
            parent1: First parent magic square
            parent2: Second parent magic square

        Returns:
            Child magic square
        """
        # Flatten the parents for easier manipulation
        flat_parent1 = parent1.flatten()
        flat_parent2 = parent2.flatten()
        length = len(flat_parent1)

        # Select a random segment from parent1
        start, end = sorted(random.sample(range(length), 2))

        # Initialize child with -1s
        child = np.full(length, -1)

        # Copy the segment from parent1
        child[start:end + 1] = flat_parent1[start:end + 1]

        # Fill the remaining positions with elements from parent2
        pointer = 0
        # Process positions before and after the segment
        for i in list(range(start)) + list(range(end + 1, length)):
            # Find next element in parent2 that's not already in child
            while flat_parent2[pointer] in child:
                pointer += 1
            child[i] = flat_parent2[pointer]
            pointer += 1

        # Reshape back to a square
        return child.reshape(self.n, self.n)

    def mutation(self, square: np.ndarray) -> np.ndarray:
        """
        Perform mutation on a magic square by swapping random pairs of elements.

        Args:
            square: Magic square to mutate

        Returns:
            Mutated magic square
        """
        mutated = square.copy()

        # Calculate number of swaps based on mutation rate and square size
        num_swaps = max(1, int(self.n ** 2 * self.mutation_rate))

        for _ in range(num_swaps):
            # Select two random positions
            i1, j1 = random.randint(0, self.n - 1), random.randint(0, self.n - 1)
            i2, j2 = random.randint(0, self.n - 1), random.randint(0, self.n - 1)

            # Make sure they're different positions
            while i1 == i2 and j1 == j2:
                i2, j2 = random.randint(0, self.n - 1), random.randint(0, self.n - 1)

            # Swap the elements
            mutated[i1, j1], mutated[i2, j2] = mutated[i2, j2], mutated[i1, j1]

        return mutated

    def local_optimization(self, square: np.ndarray) -> np.ndarray:
        """
        Perform local optimization on a magic square by making incremental improvements.

        Args:
            square: Magic square to optimize

        Returns:
            Optimized magic square
        """
        optimized_square = square.copy()

        # For most-perfect squares, we can't use incremental evaluation efficiently
        # due to the complex fitness function
        use_incremental = not self.most_perfect

        for _ in range(self.optimization_steps * self.n):
            current_fitness = self.fitness(optimized_square)

            # If we've found a perfect solution, stop optimizing
            if current_fitness == 0:
                break

            best_swap_coords = None
            best_found_new_fitness = current_fitness
            if current_fitness >= 3:
                num_trials = self.n ** 2
            else:
                num_trials = self.n ** 3

            # Try random swaps and keep the best one
            for _ in range(num_trials):
                # Select two random positions to swap
                r1, c1 = random.randint(0, self.n - 1), random.randint(0, self.n - 1)
                r2, c2 = random.randint(0, self.n - 1), random.randint(0, self.n - 1)

                # Make sure they're different positions
                while r1 == r2 and c1 == c2:
                    r2, c2 = random.randint(0, self.n - 1), random.randint(0, self.n - 1)

                val_at_r1c1 = optimized_square[r1, c1]
                val_at_r2c2 = optimized_square[r2, c2]

                if use_incremental:
                    # Calculate the change in fitness incrementally (optimization)
                    delta_deviation = 0

                    # Calculate row sum changes
                    if r1 != r2:
                        old_sum_r1 = np.sum(optimized_square[r1, :])
                        new_sum_r1 = old_sum_r1 - val_at_r1c1 + val_at_r2c2
                        delta_deviation += (abs(new_sum_r1 - self.magic_sum) - abs(old_sum_r1 - self.magic_sum))

                        old_sum_r2 = np.sum(optimized_square[r2, :])
                        new_sum_r2 = old_sum_r2 - val_at_r2c2 + val_at_r1c1
                        delta_deviation += (abs(new_sum_r2 - self.magic_sum) - abs(old_sum_r2 - self.magic_sum))

                    # Calculate column sum changes
                    if c1 != c2:
                        old_sum_c1 = np.sum(optimized_square[:, c1])
                        new_sum_c1 = old_sum_c1 - val_at_r1c1 + val_at_r2c2
                        delta_deviation += (abs(new_sum_c1 - self.magic_sum) - abs(old_sum_c1 - self.magic_sum))

                        old_sum_c2 = np.sum(optimized_square[:, c2])
                        new_sum_c2 = old_sum_c2 - val_at_r2c2 + val_at_r1c1
                        delta_deviation += (abs(new_sum_c2 - self.magic_sum) - abs(old_sum_c2 - self.magic_sum))

                    # Calculate main diagonal changes
                    old_md_sum = np.sum(np.diag(optimized_square))
                    md_sum_change = 0
                    if r1 == c1:
                        md_sum_change += (val_at_r2c2 - val_at_r1c1)
                    if r2 == c2:
                        md_sum_change += (val_at_r1c1 - val_at_r2c2)
                    new_md_sum_after_swap = old_md_sum + md_sum_change
                    delta_deviation += (abs(new_md_sum_after_swap - self.magic_sum) - abs(old_md_sum - self.magic_sum))

                    # Calculate anti-diagonal changes
                    old_od_sum = np.sum(np.diag(np.fliplr(optimized_square)))
                    od_sum_change = 0
                    if r1 + c1 == self.n - 1:
                        od_sum_change += (val_at_r2c2 - val_at_r1c1)
                    if r2 + c2 == self.n - 1:
                        od_sum_change += (val_at_r1c1 - val_at_r2c2)
                    new_od_sum_after_swap = old_od_sum + od_sum_change
                    delta_deviation += (abs(new_od_sum_after_swap - self.magic_sum) - abs(old_od_sum - self.magic_sum))

                    # Calculate new fitness
                    trial_new_fitness = current_fitness + delta_deviation
                else:
                    # Perform the swap temporarily and calculate the full fitness
                    optimized_square[r1, c1], optimized_square[r2, c2] = val_at_r2c2, val_at_r1c1
                    trial_new_fitness = self.fitness(optimized_square)
                    # Undo the swap
                    optimized_square[r1, c1], optimized_square[r2, c2] = val_at_r1c1, val_at_r2c2

                # Keep track of the best swap
                if trial_new_fitness < best_found_new_fitness:
                    best_found_new_fitness = trial_new_fitness
                    best_swap_coords = (r1, c1, r2, c2)

            # Apply the best swap if we found one
            if best_swap_coords:
                r1, c1, r2, c2 = best_swap_coords
                optimized_square[r1, c1], optimized_square[r2, c2] = optimized_square[r2, c2], optimized_square[r1, c1]

        return optimized_square

    def evolve(self, max_generations: int = 1000, target_fitness: float = 0,
               stagnation_limit: int = 100) -> Tuple[np.ndarray, int, float]:
        """
        Evolve the population to find a magic square solution.

        Args:
            max_generations: Maximum number of generations to evolve
            target_fitness: Stop when this fitness is reached (0 for perfect solution)
            stagnation_limit: Stop if no improvement for this many generations

        Returns:
            Tuple of (best_solution, generations, best_fitness)
        """
        best_solution = self.population[0].copy()
        gen_best_fitness = self.fitness(best_solution)

        generations_without_improvement = 0
        if not hasattr(self, 'initial_mutation_rate'):
            self.initial_mutation_rate = self.mutation_rate

        for generation in range(1, max_generations + 1):
            current_pop_fitnesses = []

            # Apply local optimization for Darwinian and Lamarckian variants
            if self.algorithm_type in ["darwinian", "lamarckian"]:
                temp_optimized_pop = []
                for ind in self.population:
                    temp_optimized_pop.append(self.local_optimization(ind))
                eval_population = temp_optimized_pop
            else:
                eval_population = self.population

            # Evaluate the fitness of each individual
            for individual in eval_population:
                current_pop_fitnesses.append(self.fitness(individual))

            # Find the best individual in this generation
            gen_best_idx = np.argmin(current_pop_fitnesses)
            gen_best_fitness = current_pop_fitnesses[gen_best_idx]
            gen_best_individual = eval_population[gen_best_idx]

            # Track history
            self.best_fitness_history.append(gen_best_fitness)
            self.avg_fitness_history.append(np.mean(current_pop_fitnesses))

            best_solution = gen_best_individual.copy()

            # Periodic reporting
            print(
                f"Generation {generation}: Best Fitness = {gen_best_fitness:.2f} "
                f"Avg Fitness = {self.avg_fitness_history[-1]:.2f}, "
                f"OptSteps: {self.optimization_steps}"
            )

            # Check if we've found a solution
            if gen_best_fitness <= target_fitness:
                print(f"Solution found at generation {generation}!")
                return best_solution, generation, gen_best_fitness

            # Check for stagnation
            if (generations_without_improvement >= stagnation_limit and
                    generation > stagnation_limit):
                print(f"Stopping due to stagnation after {stagnation_limit} generations without improvement.")
                return best_solution, generation, gen_best_fitness

            # Select population for breeding
            if self.algorithm_type == "lamarckian":
                # In Lamarckian evolution, we use the optimized individuals for breeding
                breeding_population = eval_population
                breeding_fitnesses = current_pop_fitnesses
            else:
                # In classical and Darwinian evolution, we use the original individuals
                breeding_population = self.population
                breeding_fitnesses = current_pop_fitnesses

            # Create the next generation
            next_generation = []

            # Sort by fitness (ascending, since lower is better)
            sorted_indices = np.argsort(current_pop_fitnesses)

            # Choose source population for elites
            source_for_elites = breeding_population
            if self.algorithm_type == "classical" or self.algorithm_type == "darwinian":
                source_for_elites = self.population

            # Elitism: Keep the best individuals
            for i in range(self.elite_size):
                next_generation.append(source_for_elites[sorted_indices[i]].copy())

            # Generate the rest of the next generation
            while len(next_generation) < self.population_size:
                # Select parents using tournament selection
                parent1 = self.tournament_selection(source_for_elites, breeding_fitnesses)
                parent2 = self.tournament_selection(source_for_elites, breeding_fitnesses)

                # Apply crossover with probability crossover_rate
                if random.random() < self.crossover_rate:
                    child = self.order_crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Apply mutation
                child = self.mutation(child)
                next_generation.append(child)

            # Update the population
            self.population = next_generation

        return best_solution, max_generations, gen_best_fitness

    def plot_progress(self):
        """Plot the progress of the genetic algorithm."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.best_fitness_history, label='Best Fitness')
        plt.plot(self.avg_fitness_history, label='Average Fitness')
        plt.title(f'Magic Square GA Progress ({self.algorithm_type}) for N={self.n}')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (lower is better)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Set y-axis scale to log if appropriate
        if min(self.best_fitness_history) > 0:
            non_zero_min = min(f for f in self.best_fitness_history if f > 0) if any(
                f > 0 for f in self.best_fitness_history) else 1
            if non_zero_min > 0:
                plt.ylim(bottom=max(0.1, non_zero_min * 0.5))
                plt.yscale('log')

        plt.tight_layout()
        plt.show()

    def print_square(self, square: np.ndarray):
        """
        Print and validate a magic square.

        Args:
            square: Magic square to print and validate
        """
        # Print header
        if self.most_perfect:
            print(f"\nMost Perfect Magic Square (N={self.n}, Magic Sum = {self.magic_sum}):")
        else:
            print(f"\nMagic Square (N={self.n}, Magic Sum = {self.magic_sum}):")

        if square is None:
            print("No solution found or square is None.")
            return

        # Print the square
        cell_width = len(str(self.n ** 2))
        separator = "-" * (self.n * (cell_width + 3) + 1)
        print(separator)
        for i in range(self.n):
            row_str = " | ".join(f"{square[i, j]:{cell_width}d}" for j in range(self.n))
            print(f"| {row_str} |")
            print(separator)

        # Validate the square
        original_eval_calls = self.evaluation_calls
        is_sol = self.is_solution(square)
        self.evaluation_calls = original_eval_calls

        if is_sol:
            if self.most_perfect:
                print("This is a valid Most Perfect Magic Square!")
            else:
                print("This is a valid Magic Square!")
        else:
            print(f"This is NOT a valid magic square. (Fitness: {self.fitness(square):.2f})")
            self.evaluation_calls = original_eval_calls

        # Verification
        print("\nVerification:")

        # Check rows
        for i in range(self.n):
            row_sum = np.sum(square[i, :])
            print(f"Row {i + 1} sum: {row_sum} {'✓' if row_sum == self.magic_sum else '✗'}")

        # Check columns
        for i in range(self.n):
            col_sum = np.sum(square[:, i])
            print(f"Column {i + 1} sum: {col_sum} {'✓' if col_sum == self.magic_sum else '✗'}")

        # Check main diagonals
        main_diag_sum = np.sum(np.diag(square))
        print(f"Main diagonal sum: {main_diag_sum} {'✓' if main_diag_sum == self.magic_sum else '✗'}")

        anti_diag_sum = np.sum(np.diag(np.fliplr(square)))
        print(f"Other diagonal sum: {anti_diag_sum} {'✓' if anti_diag_sum == self.magic_sum else '✗'}")

        # Extra checks for most perfect magic square
        if self.most_perfect and self.n % 4 == 0:
            sub_square_sum_target = 2 * (self.n ** 2 + 1)
            pair_sum_target = self.n ** 2 + 1
            n_half = self.n // 2

            # Check 2x2 sub-squares
            print(f"\n2x2 Sub-square sums (should be {sub_square_sum_target}):")
            valid_subsquares = 0
            total_checked = 0
            for r in range(0, self.n, 2):
                for c in range(0, self.n, 2):
                    s = (square[r, c] + square[r, (c + 1) % self.n] +
                         square[(r + 1) % self.n, c] + square[(r + 1) % self.n, (c + 1) % self.n])
                    print(f"Sub-square at ({r},{c}): {s} {'✓' if s == sub_square_sum_target else '✗'}")
                    valid_subsquares += (s == sub_square_sum_target)
                    total_checked += 1
            print(f"Valid sub-squares: {valid_subsquares}/{total_checked}")

            # Check symmetric pairs
            print(f"\nSymmetric pair sums (should be {pair_sum_target}):")
            valid_pairs = 0
            total_pairs = 0
            for r in range(n_half):
                for c in range(self.n):
                    val1 = square[r, c]
                    partner_r = (r + n_half) % self.n
                    partner_c = (c + n_half) % self.n
                    val2 = square[partner_r, partner_c]
                    pair_sum = val1 + val2
                    print(
                        f"Pair ({r},{c}) + ({partner_r},{partner_c}): {pair_sum} {'✓' if pair_sum == pair_sum_target else '✗'}")
                    valid_pairs += (pair_sum == pair_sum_target)
                    total_pairs += 1
            print(f"Valid symmetric pairs: {valid_pairs}/{total_pairs}")

def run_experiment(n: int, algorithm_type: str, max_generations: int = 1000,
                   population_size: int = 100, mutation_rate: float = 0.1,
                   elite_size: int = 10, tournament_size: int = 5,
                   crossover_rate: float = 0.8, optimization_steps: int = None,
                   most_perfect: bool = False):
    """
    Run a single experiment with the specified parameters.

    Args:
        n: Size of the magic square
        algorithm_type: "classical", "darwinian", or "lamarckian"
        max_generations: Maximum number of generations
        population_size: Size of the population
        mutation_rate: Mutation rate
        elite_size: Number of elites to keep
        tournament_size: Tournament size for selection
        crossover_rate: Crossover rate
        optimization_steps: Number of local optimization steps (default is n)
        most_perfect: decide whether search for perfect square or not

    Returns:
        Tuple of (best_solution, generations, best_fitness, evaluation_calls, runtime)
    """
    if optimization_steps is None:
        optimization_steps = n

    print(f"\n{'=' * 60}")
    print(f"Running {algorithm_type.capitalize()} Algorithm for {n}x{n} Magic Square")
    print(f"{'=' * 60}")

    start_time = time.time()

    # Initialize and run the GA
    ga = MagicSquareGA(
        n=n,
        population_size=population_size,
        mutation_rate=mutation_rate,
        elite_size=elite_size,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        algorithm_type=algorithm_type,
        optimization_steps=optimization_steps,
        most_perfect=most_perfect
    )

    best_solution, generations, best_fitness = ga.evolve(
        max_generations=max_generations,
        target_fitness=0,
        stagnation_limit=max(int(max_generations / 2), n * 10)
    )

    runtime = time.time() - start_time

    # Display results
    print(f"\nAlgorithm: {algorithm_type.capitalize()}")
    print(f"Generations: {generations}")
    print(f"Best Fitness: {best_fitness}")
    print(f"Evaluation Calls: {ga.evaluation_calls}")
    print(f"Runtime: {runtime:.2f} seconds")

    ga.print_square(best_solution)
    ga.plot_progress()

    return best_solution, generations, best_fitness, ga.evaluation_calls, runtime


def compare_algorithms(n: int, max_generations: int = 1000, population_size: int = 100,
                       mutation_rate: float = 0.1, elite_size: int = 10,
                       tournament_size: int = 5, crossover_rate: float = 0.8,
                       optimization_steps: int = None, most_perfect: bool = False):
    """
    Compare the three algorithm types for a given square size.

    Args:
        n: Size of the magic square
        max_generations: Maximum number of generations
        population_size: Size of the population
        mutation_rate: Mutation rate
        elite_size: Number of elites to keep
        tournament_size: Tournament size for selection
        crossover_rate: Crossover rate
        optimization_steps: Number of local optimization steps (default is n)
        most_perfect: decide whether search for perfect square or not
    """
    if optimization_steps is None:
        optimization_steps = n

    results = {}

    for algorithm_type in ["classical", "darwinian", "lamarckian"]:
        print(f"\n\n{'#' * 80}")
        print(f"# Running {algorithm_type.capitalize()} Algorithm for {n}x{n} Magic Square")
        print(f"{'#' * 80}")

        best_solution, generations, best_fitness, evaluation_calls, runtime = run_experiment(
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

        results[algorithm_type] = {
            "best_solution": best_solution,
            "generations": generations,
            "best_fitness": best_fitness,
            "evaluation_calls": evaluation_calls,
            "runtime": runtime,
            "solution_found": best_fitness == 0
        }

    # Print comparison
    print("\n\n" + "=" * 80)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 80)
    print(
        f"{'Algorithm':<15} {'Solution Found':<15} {'Generations':<15} {'Evaluation Calls':<20} {'Runtime (s)':<15} {'Best Fitness':<15}")
    print("-" * 80)

    for algorithm_type, result in results.items():
        print(
            f"{algorithm_type.capitalize():<15} {str(result['solution_found']):<15} {result['generations']:<15} {result['evaluation_calls']:<20} {result['runtime']:<15.2f} {result['best_fitness']:<15}")

    return results
