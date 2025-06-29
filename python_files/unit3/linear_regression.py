import matplotlib.pyplot as plt
from scipy import stats

# Create the arrays that represent the values of the x and y axis
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]

# Execute a method that returns key values for Linear Regression
slope, intercept, r, p, std_err = stats.linregress(x, y)

# Measure the Pearson correlation
corr, _ = stats.pearsonr(x, y)
print('Pearson correlation: %.3f' % corr)

# Function to calculate y-values using regression line
def myfunc(x):
    return slope * x + intercept

# Calculate the fitted values (predicted y)
mymodel = list(map(myfunc, x))

# Plot the original data and the regression line
plt.scatter(x, y, label='Data points')
plt.plot(x, mymodel, color='red', label='Regression line')
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Linear Regression & Correlation")
plt.legend()
plt.show()
