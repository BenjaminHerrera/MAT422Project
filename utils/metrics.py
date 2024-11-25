def calculate_adjusted_r2(r_squared: float, n: int, p: int) -> float:
    """Calculates the adjusted R2 of a given R2

    Args:
        r_squared (float): Original R2
        n (int): Number of Observations
        p (int): Number of Predictors

    Returns:
        float: Adjusted R2
    """
    # Handle the case when the denominator is 0
    if not (n - p - 1):
        return 0

    # Return the actual adjusted R2 if no issures are present
    return 1 - (((1 - r_squared) * (n - 1)) / (n - p - 1))
