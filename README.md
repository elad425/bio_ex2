# Genetic Algorithm for Solving Magic Squares

This project implements a Genetic Algorithm (GA) to find solutions for N x N Magic Squares. It can solve both standard magic squares and the more constrained "Most Perfect Magic Squares." The implementation allows for comparison between Classical, Darwinian, and Lamarckian evolutionary strategies.

## Project Structure

The project consists of the following key files:

- `magic_square_genetic.py`: Contains the core logic for the Genetic Algorithm, including:
  - Population initialization
  - Fitness calculation (for standard and Most Perfect Magic Squares)
  - Selection mechanisms (e.g., Tournament Selection)
  - Crossover operations (e.g., Order Crossover)
  - Mutation operations (e.g., Swap Mutation)
  - Implementation of Classical, Darwinian, and Lamarckian evolutionary models.
- `magic_square_menu.py`: Provides a user-friendly command-line interface (CLI) to:
  - Run experiments with various configurations.
  - Adjust GA parameters.
  - Compare the performance of different algorithms.
  - It also supports direct command-line arguments for non-interactive (scripted) execution.
- `Genetic Algorithm for Solving Magic Squares report.docx`: the report for the assignment.
- `magic_square_menu.exe`: the execution file for running the code without downloading libraries.

## ⚙️ Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.x** (Recommended: Python 3.7 or newer)
- **NumPy**: A fundamental package for numerical computation in Python, used here for efficient array manipulations.
  ```bash
  pip install numpy
  ```
- **Matplotlib**: A comprehensive library for creating static, animated, and interactive visualizations in Python. Used here for plotting graphs of fitness over generations.
  ```bash
  pip install matplotlib
  ```

## 🚀 How to Run

The primary way to interact with and run the experiments is through the `magic_square_menu.py` script.

### 1. Interactive Menu Mode 🖱️

This mode is recommended for exploring the different options, configuring parameters easily, and getting a feel for the algorithm's behavior.

**Steps:**

1.  **File Placement**: Ensure `magic_square_genetic.py` and `magic_square_menu.py` are located in the same directory.
2.  **Open Terminal**: Launch your terminal or command prompt.
3.  **Navigate**: Change to the directory containing the Python scripts:
    ```bash
    cd path/to/your/scripts
    ```
4.  **Execute Script**: Run the menu script using Python:
    ```bash
    python magic_square_menu.py
    ```
5.  **Menu Navigation**: You will be greeted with the main menu:

    ```
    ================================================================================
                             MAGIC SQUARE GENETIC ALGORITHM
    ================================================================================
    Computational Biology Assignment - Magic Square Generator using Genetic Algorithms
    --------------------------------------------------------------------------------

    MAIN MENU:
    1. Run a single algorithm
    2. Compare all algorithms
    3. Quick demo (3x3 magic square)
    4. Advanced settings
    5. What is a Magic Square?
    6. Exit

    Enter your choice (1-6):
    ```

    Follow the on-screen prompts to:

    - Run a single algorithm (Classical, Darwinian, or Lamarckian) with specific or default parameters.
    - Compare all three algorithms for a given square size and type.
    - Execute a quick demonstration using a 3x3 magic square.
    - View detailed explanations of advanced GA settings.
    - Learn more about the definition and properties of Magic Squares.
