SAMPLE_PROFILES = {
    'young_professional': {
        'name': 'Young Professional',
        'description': 'Single working professional in an urban Indian city',
        # Monthly income in INR
        'monthly_income': 50000,
        'age_range': '25-34',
        'family_size': '1',
        'location': 'Bengaluru',
        'expenses': {
            # Typical monthly breakup in INR
            'housing_expense': 18000,      # rent in cities like Bengaluru/Mumbai (shared/studio)
            'transportation_expense': 4000,
            'food_expense': 8000,
            'utilities_expense': 1500,
            'healthcare_expense': 1000,
            'debt_expense': 3000,
            'discretionary_expense': 4500
        }
    },
    'family_of_four': {
        'name': 'Family of Four',
        'description': 'Dual-income family with two children living in a metro/suburban area',
        'monthly_income': 150000,
        'age_range': '35-44',
        'family_size': '3-4',
        'location': 'Mumbai',
        'expenses': {
            'housing_expense': 40000,
            'transportation_expense': 12000,
            'food_expense': 30000,
            'utilities_expense': 6000,
            'healthcare_expense': 8000,
            'debt_expense': 8000,
            'discretionary_expense': 10000
        }
    },
    'retired_couple': {
        'name': 'Retired Couple',
        'description': 'Retired couple living on pension/savings in a smaller city or town',
        'monthly_income': 70000,
        'age_range': '55+',
        'family_size': '2',
        'location': 'Pune',
        'expenses': {
            'housing_expense': 16000,
            'transportation_expense': 3000,
            'food_expense': 9000,
            'utilities_expense': 3000,
            'healthcare_expense': 12000,
            'debt_expense': 2000,
            'discretionary_expense': 8000
        }
    }
}