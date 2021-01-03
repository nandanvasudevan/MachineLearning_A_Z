# Regression

## Simple Linear Regression

**One dependent & independent variable**

Simple linear regression is a simple straight line equation given by
```
an equation for a straight line
    y = mx + c

or in this case
    y = b0 + b1*c1

where 
 b0     -> is a constant
 y      -> is the dependent variable
 x      -> is the independent variable
```

Regression simply provides a straight line that best fits the data.

### Ordinary least square method

In this method the line that gives the lowest squared error is chosen.

1. Error is calculated as difference between the actual value and the predicted value
2. The errors from each point are squared and integrated 
3. The line with the lowest sum is chosen

### Linear Regression example
The dataset used here is the salary dataset.

![](simpleRegression.svg)