import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate
import datetime

def main():
    student = RobustDataScienceStudent("Bob", 24, "01-10-2023", "AI", "654321", ["Convex", "ML", "DSP", "DSP Lab", "Matrix"], "AI")

    student.solve_integral_problem(
        x_range=(0, 12), x_stats=[4, 7], plot_derivative=False
    )

    student.solve_linear_system(A=[[3, 2, 3, 10],
                                              [2, -2, 5, 8],
                                              [3, 3, 4, 9],
                                              [3, 4, -3, -7]],
                                              b=[4, 1, 3, 2]
    )
    # np.random.seed(42)
    # X = np.random.rand(10, 3)
    # y = np.random.rand(10)
    X = [[1, 2], [3, 4], [5, 6]]
    y = [3, 7, 11]

    student.solve_least_squares(X, y, print_results=True)



class TUDarmstadtStudent:
    """Class representing a TU Darmstadt student."""
    def __init__(self, name, age, registration_date, study_program, reg_number, courses, favorite_course):
        if not isinstance(name, str):
            raise ValueError("Name must be a string.")
        if not isinstance(age, int):
            raise ValueError("Age must be an integer.")
        if not isinstance(courses, list) or not all(isinstance(course, str) for course in courses) or len(courses) != 5:
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
        return self.__name

    @property
    def age(self):
        return self.__age

    @property
    def registration_date(self):
        return self.__registration_date

    @property
    def study_program(self):
        return self.__study_program

    @property
    def reg_number(self):
        return self.__reg_number

    # Getters and setters for courses and favorite course
    @property
    def courses(self):
        return self._courses

    @courses.setter
    def courses(self, courses):
        self._courses = courses

    @property
    def favorite_course(self):
        return self._favorite_course

    @favorite_course.setter
    def favorite_course(self, favorite_course):
        self._favorite_course = favorite_course


class RobustDataScienceStudent(TUDarmstadtStudent):
    """Class representing a robust data science student with additional functionalities."""

    def solve_integral_problem(self, x_range, x_stats, plot_derivative=False):
        try:
            if not isinstance(x_range, (list, tuple)) or not all(isinstance(i, (int, float)) for i in x_range):
                raise ValueError("x_range must be a list or tuple of numbers.")
            if not isinstance(x_stats, (list, tuple)) or not all(isinstance(i, (int, float)) for i in x_stats):
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
                plt.plot(x, dy_dx, label=r"$\frac{dy}{dx}$", linestyle='--')
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
        try:
            A = np.array(A)
            b = np.array(b)
            if A.shape[0] != b.shape[0]:
                raise ValueError("Dimensions of A and b must match.")
            if not (np.issubdtype(A.dtype, np.number) and np.issubdtype(b.dtype, np.number)):
                raise ValueError("All elements in A and b must be numeric.")

            solution = np.linalg.solve(A, b)

            # Nicely formatted results
            results = [[f"V{i+1}", val] for i, val in enumerate(solution)]

            print(tabulate(results, headers=["Variable", "Value"], tablefmt="pretty"))

            return solution

        except Exception as e:
            raise e

    def solve_least_squares(self, X, y, print_results=False):
        try:
            X = np.array(X)
            # X = np.c_[np.ones(X.shape[0]), X]
            y = np.array(y)
            if X.shape[0] <= X.shape[1]:
                raise ValueError("Number of observations must exceed the number of variables.")
            if len(y) != X.shape[0]:
                raise ValueError("Response vector y must be compatible with regressor matrix X.")
            if not (np.issubdtype(X.dtype, np.number) and np.issubdtype(y.dtype, np.number)):
                raise ValueError("All elements in X and y must be numeric.")



            # Perform least squares
            beta = np.linalg.lstsq(X, y, rcond=None)[0]

            # Compute statistics
            residuals = y - X @ beta
            variance = np.var(residuals)
            mse = np.sqrt(variance * np.linalg.inv(X.T @ X).diagonal())
            t_stats = beta / mse
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=X.shape[0] - X.shape[1]))

            results = [
            [f"β{i+1}", b, t, p]
            for i, (b, t, p) in enumerate(zip(beta, t_stats, p_values))
            ]

            if print_results:
                print(tabulate(results, headers=["Coefficient", "Value", "t-stat", "p-value"], tablefmt="pretty"))

            return beta, t_stats, p_values

        except Exception as e:
            raise e

if __name__ == "__main__":
    main()
