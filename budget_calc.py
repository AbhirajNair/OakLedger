def calculate_advanced_budget_recommendations(monthly_income, age_range, family_size, location):
    """
    Advanced budget calculation for Indian context considering:
    - Cost of living by Indian cities
    - Age-specific financial priorities in Indian context
    - Family size adjustments for Indian households
    - Income tier considerations for Indian economy
    """

    # Cost of living multipliers by Indian cities
    cost_of_living_multipliers = {
        'high': ['mumbai', 'delhi', 'gurgaon', 'noida', 'bangalore', 'hyderabad'],
        'medium-high': ['pune', 'chennai', 'kolkata', 'ahmedabad', 'jaipur', 'lucknow'],
        'medium': ['kochi', 'coimbatore', 'indore', 'bhopal', 'nagpur', 'vadodara'],
        'low': ['patna', 'ranchi', 'bhubaneswar', 'raipur', 'dehradun', 'guwahati']
    }

    # Determine cost of living tier based on location
    location_lower = location.lower()
    col_tier = 'medium'  # default

    for tier, cities in cost_of_living_multipliers.items():
        if any(city in location_lower for city in cities):
            col_tier = tier
            break

    # Cost of living adjustments for Indian context
    col_multipliers = {
        'high': {'housing': 1.5, 'food': 1.3, 'transportation': 1.2, 'utilities': 1.1},
        'medium-high': {'housing': 1.2, 'food': 1.15, 'transportation': 1.1, 'utilities': 1.05},
        'medium': {'housing': 1.0, 'food': 1.0, 'transportation': 1.0, 'utilities': 1.0},
        'low': {'housing': 0.7, 'food': 0.85, 'transportation': 0.9, 'utilities': 0.9}
    }

    # Base budget allocation percentages for Indian context
    base_percentages = {
        'Housing (Rent/EMI)': 0.25,
        'Food & Groceries': 0.15,
        'Transportation': 0.08,
        'Utilities & Bills': 0.05,
        'Healthcare': 0.06,
        'Savings & Investments': 0.18,
        'Insurance (Life/Health)': 0.04,
        'Entertainment & Leisure': 0.05,
        'Personal Care': 0.03,
        'Education & Skills': 0.04,
        'Shopping & Lifestyle': 0.04,
        'Emergency Fund': 0.03
    }

    # Age-based adjustments for Indian life stages
    age_adjustments = {
        '18-24': {
            'Savings & Investments': -0.04,
            'Education & Skills': +0.05,
            'Entertainment & Leisure': +0.02,
            'Shopping & Lifestyle': +0.02
        },
        '25-34': {
            'Savings & Investments': +0.03,
            'Housing (Rent/EMI)': +0.04,
            'Insurance (Life/Health)': +0.02,
            'Entertainment & Leisure': -0.01
        },
        '35-44': {
            'Savings & Investments': +0.05,
            'Education & Skills': +0.03,  # children's education
            'Healthcare': +0.02,
            'Insurance (Life/Health)': +0.01
        },
        '45-54': {
            'Savings & Investments': +0.07,
            'Healthcare': +0.03,
            'Emergency Fund': +0.02,
            'Entertainment & Leisure': +0.01
        },
        '55+': {
            'Healthcare': +0.05,
            'Savings & Investments': +0.04,
            'Entertainment & Leisure': +0.03,
            'Housing (Rent/EMI)': -0.04
        }
    }

    # Family size adjustments for Indian households
    family_adjustments = {
        '1': {
            'Housing (Rent/EMI)': -0.03,
            'Food & Groceries': -0.03,
            'Entertainment & Leisure': +0.02
        },
        '2': {
            'Food & Groceries': +0.03,
            'Housing (Rent/EMI)': +0.02,
            'Healthcare': +0.01
        },
        '3-4': {
            'Food & Groceries': +0.06,
            'Housing (Rent/EMI)': +0.05,
            'Education & Skills': +0.04,
            'Healthcare': +0.02,
            'Entertainment & Leisure': -0.02
        },
        '5+': {
            'Food & Groceries': +0.10,
            'Housing (Rent/EMI)': +0.08,
            'Education & Skills': +0.05,
            'Healthcare': +0.03,
            'Savings & Investments': -0.06,
            'Entertainment & Leisure': -0.04
        }
    }

    # Income tier adjustments for Indian economy
    income_tier_adjustments = {
        'low': {
            'Savings & Investments': -0.05,
            'Housing (Rent/EMI)': +0.04,
            'Food & Groceries': +0.03
        },
        'medium': {},  # No adjustments for medium income
        'high': {
            'Savings & Investments': +0.07,
            'Entertainment & Leisure': +0.02,
            'Shopping & Lifestyle': +0.02
        }
    }

    # Determine income tier based on Indian standards (monthly in rupees)
    if monthly_income < 40000:
        income_tier = 'low'
    elif monthly_income < 150000:
        income_tier = 'medium'
    else:
        income_tier = 'high'

    # Start with base percentages
    final_percentages = base_percentages.copy()

    # Apply age adjustments
    if age_range in age_adjustments:
        for category, adjustment in age_adjustments[age_range].items():
            if category in final_percentages:
                final_percentages[category] += adjustment

    # Apply family size adjustments
    if family_size in family_adjustments:
        for category, adjustment in family_adjustments[family_size].items():
            if category in final_percentages:
                final_percentages[category] += adjustment

    # Apply income tier adjustments
    if income_tier in income_tier_adjustments:
        for category, adjustment in income_tier_adjustments[income_tier].items():
            if category in final_percentages:
                final_percentages[category] += adjustment

    # Apply cost of living adjustments to specific categories
    col_multiplier = col_multipliers[col_tier]
    for category, multiplier in col_multiplier.items():
        adjusted_category = None
        if category == 'housing':
            adjusted_category = 'Housing (Rent/EMI)'
        elif category == 'food':
            adjusted_category = 'Food & Groceries'
        elif category == 'transportation':
            adjusted_category = 'Transportation'
        elif category == 'utilities':
            adjusted_category = 'Utilities & Bills'

        if adjusted_category and adjusted_category in final_percentages:
            final_percentages[adjusted_category] *= multiplier

    # Ensure percentages sum to 100% (normalize)
    total_percentage = sum(final_percentages.values())
    if abs(total_percentage - 1.0) > 0.01:  # Allow small floating point differences
        normalization_factor = 1.0 / total_percentage
        for category in final_percentages:
            final_percentages[category] *= normalization_factor

    # Calculate rupee amounts and prepare recommendations
    recommendations = {}
    for category, percentage in final_percentages.items():
        amount = monthly_income * percentage
        recommendations[category] = {
            'percentage': percentage * 100,
            'amount': amount,
            'display_percentage': f"{percentage * 100:.1f}%",
            'display_amount': f"₹{amount:,.0f}"
        }

    return recommendations