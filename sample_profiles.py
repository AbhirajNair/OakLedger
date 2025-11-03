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
        },
        'credit': {
            'total_debt': 120000,
            'monthly_debt_payment': 6000,
            'avg_interest_rate': 18,
            'credit_utilization_pct': 45,
            'on_time_payment_pct': 95,
            'open_credit_lines': 2,
            'hard_inquiries_12m': 1
        },
        'goal_preset': {
            'goal_name': 'Build Emergency Fund',
            'target_amount': 150000,
            'current_amount': 20000,
            'target_date': '2026-12-31'
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
        },
        'credit': {
            'total_debt': 450000,
            'monthly_debt_payment': 15000,
            'avg_interest_rate': 14,
            'credit_utilization_pct': 55,
            'on_time_payment_pct': 92,
            'open_credit_lines': 4,
            'hard_inquiries_12m': 2
        },
        'goal_preset': {
            'goal_name': 'Down Payment Fund',
            'target_amount': 1200000,
            'current_amount': 150000,
            'target_date': '2028-06-30'
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
        },
        'credit': {
            'total_debt': 80000,
            'monthly_debt_payment': 3000,
            'avg_interest_rate': 12,
            'credit_utilization_pct': 20,
            'on_time_payment_pct': 99,
            'open_credit_lines': 3,
            'hard_inquiries_12m': 0
        },
        'goal_preset': {
            'goal_name': 'Travel Fund',
            'target_amount': 200000,
            'current_amount': 50000,
            'target_date': '2027-03-31'
        }
    }
}