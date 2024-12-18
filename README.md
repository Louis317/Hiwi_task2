# **TU Darmstadt Student and Robust Data Science Student Classes**

## Description

This repository provides a Python implementation of two classes:

​	•	TUDarmstadtStudent: Represents a general TU Darmstadt student.

​	•	RobustDataScienceStudent: Extends TUDarmstadtStudent with numerical and statistical problem-solving capabilities.

### **Files**

​	•	students.py: Contains the implementation of both classes.

​	•	unittest.py: Includes unit tests to verify functionality.



## **Installation**

Clone or download the repository, then ensure you have the following dependencies installed:

​	•	numpy

​	•	scipy

​	•	tabulate

​	•	datetime

​	•	pytest (for testing)

Install the dependencies using:

```bash
pip install numpy scipy tabulate matplotlib datetime
```

<br />

## **1. TUDarmstadtStudent Class**

### **Overview**

The TUDarmstadtStudent class provides core attributes and methods for handling student data. The TUDarmstadtStudent class represents a generic student with private attributes for sensitive data and public getters for accessing them. It provides:

### Attributes

| Attribute         | Type | Description                                                  |
| ----------------- | ---- | ------------------------------------------------------------ |
| name              | str  | The name of the student (**private**).                       |
| age               | int  | The age of the student (**private**).                        |
| registration_date | str  | Registration date at TU Darmstadt (**private**, in DD-MM-YYYY format). |
| study_program     | str  | The study program of the student (**private**).              |
| reg_number        | str  | The registration number of the student (**private**).        |
| courses           | list | A list of 5 completed courses.                               |
| favorite_course   | str  | The favorite course of the student.                          |

#### <u>**Public Methods**:</u>

​	•	Getter methods for private attributes.

​	•	Getter and setter methods for non-private attributes.

### Usage Example

```python
from students import TUDarmstadtStudent

# Create an instance of TUDarmstadtStudent
student = TUDarmstadtStudent(
    name="Alice",
    age=25,
    registration_date="01-10-2021",
    study_program="Data Science",
    reg_number="123456",
    courses=["Math", "Python", "Statistics", "Algorithms", "AI"],
    favorite_course="AI"
)

# Access private attributes using getter methods
print("Name:", student.name)
print("Age:", student.age)
print("Registration Date:", student.registration_date)
print("Study Program:", student.study_program)
print("Registration Number:", student.reg_number)

# Modify courses and favorite_course
student.courses = ["Math", "Python", "Optimization", "Statistics", "ML"]
student.favorite_course = "Optimization"

print("Updated Courses:", student.courses)
print("Favorite Course:", student.favorite_course)
```

<br />

## **2. RobustDataScienceStudent Class**


### **Overview**

The RobustDataScienceStudent class inherits from TUDarmstadtStudent and provides additional numerical problem-solving methods.

<br />

### **Method 1: Solve Integral Problem**

```python
solve_integral_problem(x_range, x_stats, plot_derivative=False)
```

​	•	Compute $y = e^{-x} \cos(x)$ for a range $x$ .

​	•	Calculate mean, variance, standard deviation, and 70% threshold of y within a subrange $x_{\text{stats}}$ .

​	•	Plot $\frac{dy}{dx}$ (if an additional Boolean input argument allows it).

​	•	Find the zero-crossing index of the derivative $\frac{dy}{dx}$ .

​	•	Output results in a well-formatted table.

#### Parameters

| Parameter name  | Type | Description                                           |
| --------------- | ---- | ----------------------------------------------------- |
| x_range         | list | Range of $x$ values (e.g., [0, 12]).                  |
| x_stats         | list | Subrange for statistical calculations (e.g., [4, 7]). |
| plot_derivative | bool | Whether to plot $\frac{dy}{dx}$ (default: ``False``). |

#### Example

```python
from students import RobustDataScienceStudent

robust_student = RobustDataScienceStudent(
    name="Bob",
    age=23,
    registration_date="01-10-2021",
    study_program="Robust Data Science",
    reg_number="654321",
    courses=["Math", "Python", "Optimization", "Statistics", "ML"],
    favorite_course="Optimization"
)

x_range = [0, 12]
x_stats = [4, 7]
robust_student.solve_integral_problem(x_range, x_stats, plot_derivative=True)
```

![Figure_1](https://github.com/user-attachments/assets/e86ef0e0-2404-422e-aeb8-8a1b0862549f)


```bash
| Metric                         |       Value |
|--------------------------------+-------------|
| Mean of y                      | 0.00029349  |
| Variance of y                  | 1.20359e-05 |
| Standard deviation of y        | 0.00346928  |
| Threshold (70 percent)         | 0.00235201  |
| First Zero Crossing Index of x | 2.35196     |
```

<br />

### **Method 2: Solve Linear Algebra Problem**

```python
solve_linear_system(A, b)
```

​	•	Solve a system of linear equations $A \mathbf{V} = b$ , where $A$ is the coefficient matrix, and $b$ is the solution vector.

​	•	Output the solution with variable names $V_1, V_2, \dots$ .

#### Parameters

| Parameter | Type | Description                                                  |
| --------- | ---- | ------------------------------------------------------------ |
| A         | list | System matrix (e.g., [[3, 2], [2, -2]]). Which will be later converted to numpy.array. |
| b         | list | Solution vector (e.g., [4, 1]). Which will be later converted to numpy.array. |

#### Example

```python
A = np.array([[3, 2, 3, 10], [2, -2, 5, 8], [3, 3, 4, 9], [3, 4, -3, -7]])
b = np.array([4, 1, 3, 2])

robust_student.solve_linear_system(A, b)
```

```bash
+----------+---------------------+
| Variable |        Value        |
+----------+---------------------+
|    V1    | 0.7837837837837835  |
|    V2    | 0.03603603603603629 |
|    V3    | -0.6756756756756754 |
|    V4    | 0.3603603603603603  |
+----------+---------------------+
```

<br />

### **Method 3: Multivariate Least-Squares Regression**

```python
solve_least_squares(X, y, print_results=False)
```

Perform multivariate least-squares regression with inputs $X$ (regressors) and $y$ (response) to calculate:

​	•	Coefficients $\beta$.

​	•	t-statistics and p-values for statistical significance.

​	•	Display results optionally based on a Boolean flag.

#### Parameters

| Parameter       | Type | Description                                                  |
| --------------- | ---- | ------------------------------------------------------------ |
| X               | list | Regressor matrix (e.g., [[1, 2], [3, 4]]). Which will be later converted to numpy.array. |
| y               | list | Response vector (e.g., [5, 6]). Which will be later converted to numpy.array. |
| display_results | bool | Whether to print results (default: `False`).                 |

#### Example

```python
np.random.seed(42)  # For reproducibility
num_samples = 100
num_features = 3
X = np.random.randn(num_samples, num_features)
true_beta = np.random.randn(num_features)
y = X @ true_beta
student.solve_least_squares(X, y, print_results=True)
```

```bash
+-------------+---------------------+-------------------------+---------+
| Coefficient |        Value        |         t-stat          | p-value |
+-------------+---------------------+-------------------------+---------+
|     β1      | -0.8289950109220728 | -2.183194870821761e+16  |   0.0   |
|     β2      | -0.5601810401969698 | -1.7661434472438342e+16 |   0.0   |
|     β3      | 0.7472936051232617  |  2.627354331366357e+16  |   0.0   |
+-------------+---------------------+-------------------------+---------+
```
<br />

## **Testing**

Run unit tests to validate the functionality of the module:

```bash
pytest unittest.py
```

The tests cover:

​	•	Getter and Setter methods for TUDarmstadtStudent.

​	•	Statistical and numerical results for RobustDataScienceStudent methods.

​	•	Input validation for all methods with errors messages that are clear and specific.
