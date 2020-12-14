from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
import matplotlib.pyplot as plt

# Making a y_true for true values and some prediction values
# Each prediction value will have different deviations to show the difference between
# the measures of fit
y_true = [-10, -5, 0, 5, 10, 15]
y_always_five = [-15, -10, -5, 0, 5, 10]
y_small_large = [-7, -2, 3, 12, 17, 22]
y_outlier = [-10, -5, 0, 5, 10, 45]

print("MAE y_always_five: ", mean_absolute_error(y_true, y_always_five))
print("MAE y_small_large: ", mean_absolute_error(y_true, y_small_large))
print("MAE y_outlier: ", mean_absolute_error(y_true, y_outlier), "\n")

print("MSE y_always_five: ", mean_squared_error(y_true, y_always_five))
print("MSE y_small_large: ", mean_squared_error(y_true, y_small_large))
print("MSE y_outlier: ", mean_squared_error(y_true, y_outlier), "\n")

print("RMSE y_always_five: ", sqrt(mean_squared_error(y_true, y_always_five)))
print("RMSE y_small_large: ", sqrt(mean_squared_error(y_true, y_small_large)))
print("RMSE y_outlier: ", sqrt(mean_squared_error(y_true, y_outlier)), "\n")

print("R2 y_always_five: ", r2_score(y_true, y_always_five))
print("R2 y_small_large: ", r2_score(y_true, y_small_large))
print("R2 y_outlier: ", r2_score(y_true, y_outlier))

# Time to make some beautiful plots!
x = [0, 1, 2, 3, 4, 5]

(y_true,) = plt.plot(y_true, color="b")
y_true = plt.legend([y_true], ["y_true"], loc="lower right")

y_always_five = plt.scatter(x, y_always_five, marker="D", color="r")
y_small_large = plt.scatter(x, y_small_large, marker="D", color="g")
y_outlier = plt.scatter(x, y_outlier, marker="D", color="y")

# Making a legend for the scatter plots
plt.legend(
    [y_always_five, y_small_large, y_outlier],
    ["y_always_five", "y_small_large", "y_outlier"],
    loc="upper left",
    scatterpoints=1,
)

# Adding a second legend, will remove the first
# By making y_true a separate artist, it will be added to the legend again
plt.gca().add_artist(y_true)
plt.show()
