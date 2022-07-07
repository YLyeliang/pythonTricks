def quadratic_pred(p, coeff):
    return coeff[0] + p * coeff[1] + p ** 2 * coeff[2]


def cubic_pred(p, coeff):
    """
    calculate the y = c0 + c1*x + c2*x^2 + c3*x^3
    """
    return coeff[0] + p * coeff[1] + p ** 2 * coeff[2] + p ** 3 * coeff[3]
