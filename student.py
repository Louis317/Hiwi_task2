"""This module provides classes and methods to represent TU Darmstadt students,
including specialized capabilities for robust data science students.

Classes:
    - TUDarmstadtStudent: Represents a generic TU Darmstadt student with basic student attributes.
        Methods:
            - __init__: Initializes a TU student with specific attributes.
            - name: Returns the student's name.
            - age: Returns the student's age.
            - registration_date: Returns the registration date.
            - study_program: Returns the study program.
            - reg_number: Returns the registration number.
            - courses: Returns or sets the list of completed courses.
            - favorite_course: Returns or sets the favorite course.

    - RobustDataScienceStudent: Extends TUDarmstadtStudent to add numerical programming capabilities.
        Methods:
            - solve_integral_problem: Solves and analyzes integral-based problems.
            - solve_linear_system: Solves a linear algebraic system of equations.
            - solve_least_squares: Performs multivariate least-squares regression.

Usage:
    To use the module, import it and instantiate the desired class. Example:

    >>> from students import RobustDataScienceStudent
    >>> student = RobustDataScienceStudent("Alice", 23, "15-09-2022", "Data Science", "123456",
                                           ["ML", "AI", "Stats", "Python", "Deep Learning"], "AI")
    >>> # Solve a regression problem
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> y = X @ np.array([1.2, -0.5, 0.3]) + np.random.randn(100) * 0.1
    >>> student.solve_least_squares(X, y, print_results=True)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate
import datetime


def main():
    student = RobustDataScienceStudent(
        "Bob",
        24,
        "01-10-2023",
        "AI",
        "654321",
        ["Convex", "ML", "DSP", "DSP Lab", "Matrix"],
        "AI",
    )

    # student.solve_integral_problem(
    #     x_range=(0, 12), x_stats=[4, 7], plot_derivative=False
    # )

    # student.solve_linear_system(
    #     A=[[3, 2, 3, 10], [2, -2, 5, 8], [3, 3, 4, 9], [3, 4, -3, -7]], b=[4, 1, 3, 2]
    # )

    # Generate synthetic data
    np.random.seed(42)  # For reproducibility
    num_samples = 100
    num_features = 3
    X = np.random.randn(num_samples, num_features)
    true_beta = np.random.randn(num_features)
    print(true_beta)
    # noise = np.random.randn(num_samples) * 0.1  # Add small noise
    y = X @ true_beta
    student.solve_least_squares(X, y, print_results=True)


class TUDarmstadtStudent:
    """Class representing a TU Darmstadt student.

    Attributes:
        - name (str): The student's name (private).
        - age (int): The student's age (private).
        - registration_date (datetime): Date of registration (private).
        - study_program (str): The student's field of study (private).
        - registration_number (int): Unique student registration number (private).
        - courses (list[str]): List of completed courses.
        - favorite_course (str): The student's favorite course.

    Methods:
        - __init__: Initializes a TU student with the given attributes.
        - name: Returns the student's name.
        - age: Returns the student's age.
        - registration_date: Returns the registration date.
        - study_program: Returns the study program.
        - registration_number: Returns the registration number.
        - courses: Returns or set the list of completed courses.
        - favorite_course: Returns or sets the favorite course.
    """

    def __init__(
        self,
        name,
        age,
        registration_date,
        study_program,
        reg_number,
        courses,
        favorite_course,
    ):
        """Initializes a TU Darmstadt student.

        Args:
            name (str): The name of the student.
            age (int): The age of the student.
            registration_date (str): Date of registration in 'DD-MM-YYYY' format.
            study_program (str): The field of study of the student.
            reg_number (int): Unique registration number of the student.
            courses (list[str]): List of completed courses (must contain 5 elements).
            favorite_course (str): The student's favorite course.

        Raises:
            ValueError: If any input is invalid.
        """
        if not isinstance(name, str):
            raise ValueError("Name must be a string.")
        if not isinstance(age, int):
            raise ValueError("Age must be an integer.")
        if (
            not isinstance(courses, list)
            or not all(isinstance(course, str) for course in courses)
            or len(courses) != 5
        ):
            raise ValueError("Courses must be a list of 5 strings.")
        if not isinstance(favorite_course, str):
            raise ValueError("Favorite course must be a single string.")
        try:
            datetime.datetime.strptime(registration_date, "%d-%m-%Y")
        except ValueError:
            raise ValueError("Registration date must be in the format DD-MM-YYYY.")
        self.__name = name
        self.__age = age
        self.__registration_date = registration_date
        self.__study_program = study_program
        self.__reg_number = reg_number
        self.courses = courses
        self.favorite_course = favorite_course

    # Getters
    @property
    def name(self):
        """Returns the student's name.

        Returns:
            str: The name of the student.
        """
        return self.__name

    @property
    def age(self):
        """Returns the student's age.

        Returns:
            int: The age of the student.
        """
        return self.__age

    @property
    def registration_date(self):
        """Returns the student's registration date.

        Returns:
            str: The registration date in 'DD-MM-YYYY' format.
        """
        return self.__registration_date

    @property
    def study_program(self):
        """Returns the student's field of study.

        Returns:
            str: The field of study.
        """
        return self.__study_program

    @property
    def reg_number(self):
        """Returns the student's registration number.

        Returns:
            int: The registration number.
        """
        return self.__reg_number

    # Getters and setters for courses and favorite course
    @property
    def courses(self):
        """Returns the list of completed courses.

        Returns:
            list[str]: The list of completed courses.
        """
        return self._courses

    @courses.setter
    def courses(self, courses):
        """Sets the list of completed courses.

        Args:
            courses (list[str]): A list of 5 course names.

        Raises:
            ValueError: If the courses list is invalid.
        """
        self._courses = courses

    @property
    def favorite_course(self):
        """Returns the student's favorite course.

        Returns:
            str: The favorite course.
        """
        return self._favorite_course

    @favorite_course.setter
    def favorite_course(self, favorite_course):
        """Sets the student's favorite course.

        Args:
            favorite_course (str): The new favorite course.

        Raises:
            ValueError: If the favorite course is not a string.
        """
        self._favorite_course = favorite_course


class RobustDataScienceStudent(TUDarmstadtStudent):
    """Class representing a robust data science student with additional functionalities.

    Inherits from:
    TUStudent

    Additional Methods:
        - solve_integral_problem: Solves and analyzes an integral-based problem.
        - solve_linear_system: Solves a linear algebraic system of equations.
        - solve_least_squares: Performs multivariate least-squares linear regression.
    """

    def solve_integral_problem(self, x_range, x_stats, plot_derivative=False):
        """Solves an integral problem with given numerical and statistical analysis.

        Args:
            x_range (tuple): Range for x, e.g., (0, 12).
            x_stats_range (tuple): Subrange of x for statistical analysis, e.g., (4, 7).
            plot_derivative (bool): If True, plots the derivative dy/dx, default is False.

        Returns:
            dict: A dictionary containing:
                - 'mean': Mean of y values in the statistical range.
                - 'variance': Variance of y values in the statistical range.
                - 'std_dev': Standard deviation of y values in the statistical range.
                - 'threshold': Threshold value of y where 70% of values fall below.
                - 'zero_index': Index of x where dy/dx equals zero.

        Raises:
            ValueError: If input arguments are invalid.
        """
        try:
            if not isinstance(x_range, (list, tuple)) or not all(
                isinstance(i, (int, float)) for i in x_range
            ):
                raise ValueError("x_range must be a list or tuple of numbers.")
            if not isinstance(x_stats, (list, tuple)) or not all(
                isinstance(i, (int, float)) for i in x_stats
            ):
                raise ValueError("x_stats must be a list or tuple of numbers.")

            x = np.linspace(x_range[0], x_range[1], 1200)
            y = np.exp(-x) * np.cos(x)

            x_stats_indices = (x >= x_stats[0]) & (x <= x_stats[1])
            y_stats = y[x_stats_indices]
            mean_y = np.mean(y_stats)
            var_y = np.var(y_stats)
            std_y = np.std(y_stats)
            threshold = np.percentile(y_stats, 70)

            dy_dx = -np.exp(-x) * np.cos(x) - np.exp(-x) * np.sin(x)
            zero_derivative_idx = np.where(np.isclose(dy_dx, 0, atol=1e-3))[0]

            if plot_derivative:
                plt.figure(figsize=(10, 6))
                plt.plot(x, y, label=r"$y = e^{-x} \cos(x)$")
                plt.plot(x, dy_dx, label=r"$\frac{dy}{dx}$", linestyle="--")
                plt.xlabel(r"$x$")
                plt.ylabel(r"$y$")
                plt.legend()
                plt.title("Function and its Derivative")
                plt.show()

            # Nicely formatted results
            results = [
                ["Mean of y", mean_y],
                ["Variance of y", var_y],
                ["Standard deviation of y", std_y],
                ["Threshold (70 percent)", threshold],
                ["First Zero Crossing Index of x", x[zero_derivative_idx[0]]],
            ]

            print(tabulate(results, headers=["Metric", "Value"], tablefmt="orgtbl"))
            # print(tabulate(results, headers=["Metric", "Value"], tablefmt="pretty", numalign="right"))
            return mean_y, var_y, std_y, threshold, x[zero_derivative_idx]

        except Exception as e:
            raise e

    def solve_linear_system(self, A, b):
        """Solves a linear system of equations Ax = b.

        Args:
            A (list[list[float]]): The system matrix.
            b (list[float]): The solution vector.

        Returns:
            numpy.ndarray: The unkown variable vector.

        Raises:
            ValueError: If the dimensions or data types of A and b are invalid.
        """
        try:
            A = np.array(A)
            b = np.array(b)
            if A.shape[0] != b.shape[0] or A.shape[0] != A.shape[1]:
                raise ValueError("Dimensions of A and b must match or A must be a square matrix.")
            if not (
                np.issubdtype(A.dtype, np.number) and np.issubdtype(b.dtype, np.number)
            ):
                raise ValueError("All elements in A and b must be numeric.")

            solution = np.linalg.solve(A, b)

            # Nicely formatted results
            results = [[f"V{i+1}", val] for i, val in enumerate(solution)]

            print(tabulate(results, headers=["Variable", "Value"], tablefmt="pretty"))

            return solution

        except Exception as e:
            raise e

    def solve_least_squares(self, X, y, print_results=False):
        """Performs multivariate least-squares linear regression.

        Args:
            X (list[list[float]]): The regressor matrix.
            y (list[float]): The response vector.
            print_results (bool): If True, plots the derivative dy/dx, default is False.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The beta vector (regression coefficients).
                - numpy.ndarray: The t-statistics for beta.
                - numpy.ndarray: The p-values for beta.

        Raises:
            ValueError: If the dimensions or data types of X and y are invalid.
        """
        try:
            X = np.array(X)
            # X = np.c_[np.ones(X.shape[0]), X]
            y = np.array(y)
            if X.shape[0] <= X.shape[1]:
                raise ValueError(
                    "Number of observations must exceed the number of variables."
                )
            if len(y) != X.shape[0]:
                raise ValueError(
                    "Response vector y must be compatible with regressor matrix X."
                )
            if not (
                np.issubdtype(X.dtype, np.number) and np.issubdtype(y.dtype, np.number)
            ):
                raise ValueError("All elements in X and y must be numeric.")

            # Perform least squares
            beta = np.linalg.lstsq(X, y, rcond=None)[0]

            # Compute statistics
            residuals = y - X @ beta
            variance = np.var(residuals)
            mse = np.sqrt(variance * np.linalg.inv(X.T @ X).diagonal())
            t_stats = beta / mse
            p_values = 2 * (
                1 - stats.t.cdf(np.abs(t_stats), df=X.shape[0] - X.shape[1])
            )

            results = [
                [f"β{i+1}", b, t, p]
                for i, (b, t, p) in enumerate(zip(beta, t_stats, p_values))
            ]

            if print_results:
                print(
                    tabulate(
                        results,
                        headers=["Coefficient", "Value", "t-stat", "p-value"],
                        tablefmt="pretty",
                    )
                )

            return beta, t_stats, p_values

        except Exception as e:
            raise e


if __name__ == "__main__":
    main()
