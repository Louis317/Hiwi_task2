import pytest
import numpy as np
from student import TUDarmstadtStudent, RobustDataScienceStudent

# Test TUDarmstadtStudent
def test_tudarmstadt_student():
    student = TUDarmstadtStudent(
        "Alice", 25, "01-10-2020", "iCE", "345678",
        ["Convex", "ML", "DSP", "DSP Lab", "Matrix"], "DSP"
    )
    assert student.name == "Alice"
    assert student.age == 25
    assert student.registration_date == "01-10-2020"
    assert student.study_program == "iCE"
    assert student.reg_number == "345678"
    assert student.courses == ["Convex", "ML", "DSP", "DSP Lab", "Matrix"]
    assert student.favorite_course == "DSP"
    student.courses = ["Convex", "ML", "DSP", "DSP Lab", "Matrix", "Math"]
    assert student.courses == ["Convex", "ML", "DSP", "DSP Lab", "Matrix", "Math"]
    student.favorite_course = "Math"
    assert student.favorite_course == "Math"
    with pytest.raises(AttributeError):
        student.name = "Bob"
    with pytest.raises(AttributeError):
        student.age = 24
    with pytest.raises(AttributeError):
        student.registration_date = "01-10-2021"
    with pytest.raises(AttributeError):
        student.study_program = "AI"
    with pytest.raises(AttributeError):
        student.reg_number = "654321"
    with pytest.raises(ValueError, match="Name must be a string."):
        student = TUDarmstadtStudent(
        111, 25, "2020-01-10", "iCE", "345678",
        ["Convex", "ML", "DSP", "DSP Lab", "Matrix"], "DSP"
    )
    with pytest.raises(ValueError, match="Age must be an integer."):
        student = TUDarmstadtStudent(
        "Alice", "24", "2020-01-10", "iCE", "345678",
        ["Convex", "ML", "DSP", "DSP Lab", "Matrix"], "DSP"
    )
    with pytest.raises(ValueError, match="Courses must be a list of 5 strings."):
        student = TUDarmstadtStudent(
        "Alice", 25, "2020-01-10", "iCE", "345678",
        ["Convex", "ML", "DSP", "DSP Lab", "Matrix", "Math"], "DSP"
    )
    with pytest.raises(ValueError, match="Favorite course must be a single string."):
        student = TUDarmstadtStudent(
        "Alice", 25, "2020-01-10", "iCE", "345678",
        ["Convex", "ML", "DSP", "DSP Lab", "Matrix"], ["DSP"]
    )
    with pytest.raises(ValueError, match="Registration date must be in the format DD-MM-YYYY."):
        student = TUDarmstadtStudent(
        "Alice", 25, "2020/01/10", "iCE", "345678",
        ["Convex", "ML", "DSP", "DSP Lab", "Matrix"], "DSP"
    )


# Test RobustDataScienceStudent
def test_solve_integral_problem():
    student = RobustDataScienceStudent("Bob", 24, "01-10-2021", "ETIT", "654321", ["Convex", "ML", "DSP", "DSP Lab", "Matrix"], "Data Science")
    mean_y, var_y, std_y, threshold, zero_derivative_idx = student.solve_integral_problem(
        x_range=[0, 12], x_stats=[4, 7]
    )
    assert isinstance(mean_y, float)
    assert isinstance(var_y, float)
    assert isinstance(std_y, float)
    assert len(zero_derivative_idx) > 0
    assert mean_y == pytest.approx(0.0002934895732504914, abs=0.00001)
    assert var_y == pytest.approx(1.2035904898598692e-05, abs=0.00001)
    assert std_y == pytest.approx(0.003469280170092737, abs=0.00001)
    assert threshold == pytest.approx(0.0023520121029938865, abs=0.00001)
    assert zero_derivative_idx[0] == pytest.approx(2.35, abs=0.01)
    with pytest.raises(ValueError, match="x_range must be a list or tuple of numbers."):
        student.solve_integral_problem(x_range=['0', 12], x_stats=['4', 7], plot_derivative=False)
    with pytest.raises(ValueError, match="x_range must be a list or tuple of numbers."):
        student.solve_integral_problem(x_range=2, x_stats=4, plot_derivative=False)



def test_solve_linear_system():
    student = RobustDataScienceStudent("Bob", 24, "01-10-2021", "AI", "654321", ["Convex", "ML", "DSP", "DSP Lab", "Matrix"], "AI")
    A = [[3, 2, 3, 10], [2, -2, 5, 8], [3, 3, 4, 9], [3, 4, -3, -7]]
    b = [4, 1, 3, 2]
    solution = student.solve_linear_system(A, b)
    assert solution == pytest.approx([0.78378378, 0.03603604, -0.67567568, 0.36036036], abs=0.00001)
    with pytest.raises(ValueError, match="Dimensions of A and b must match."):
        student.solve_linear_system(A, b=[4, 1, 3])
    with pytest.raises(ValueError, match="All elements in A and b must be numeric."):
        student.solve_linear_system(A=[[3, 2, '3'], [2, -2, 5], [3, 3, 4]], b=[4, 1, 3])


def test_solve_least_squares():
    student = RobustDataScienceStudent("Bob", 24, "01-10-2021", "AI", "654321", ["Convex", "ML", "DSP", "DSP Lab", "Matrix"], "AI")
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])
    beta, t_stats, p_values = student.solve_least_squares(X, y)
    assert len(beta) == 2
    assert len(t_stats) == 2
    assert len(p_values) == 2
    assert beta == pytest.approx([1, 1], abs=0.00001)
    assert t_stats == pytest.approx([1563571146130420, 1977778442196834], abs=10)
    assert p_values == pytest.approx([4.440892098500626e-16, 4.440892098500626e-16], abs=0.00001)
    with pytest.raises(ValueError, match="Number of observations must exceed the number of variables."):
        student.solve_least_squares(X=[[1, 2, 3], [3, 4, 5]], y=[3, 7])
    with pytest.raises(ValueError, match="Response vector y must be compatible with regressor matrix X."):
        student.solve_least_squares(X=[[1, 2], [3, 4], [5, 6]], y=[3, 7])
    with pytest.raises(ValueError, match="All elements in X and y must be numeric."):
        student.solve_least_squares(X=[[1, 2], [3, 4], [5, '6']], y=[3, 7, 11])
