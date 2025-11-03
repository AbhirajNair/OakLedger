def generate_investment_guidance(monthly_income, age_range, financial_health=None, goals=None):
    """
    Provide simple investment guidance split into Low-Risk and Growth options.

    Inputs used:
    - monthly_income: float (₹)
    - age_range: str (e.g., '18-24', '25-34', '35-44', '45-54', '55+')
    - financial_health: dict with metrics like savings_rate (optional)
    - goals: list of goal dicts from session (optional)

    Returns a dict with:
    - risk_profile: 'Conservative' | 'Moderate' | 'Aggressive'
    - split: {'low_risk_pct': int, 'growth_pct': int}
    - suggested_allocation: {'low_risk': [{'name', 'percentage', 'amount', 'notes'}], 'growth': [...]}  
    - notes: list[str]
    """
    age_to_profile = {
        '18-24': 'Aggressive',
        '25-34': 'Aggressive',
        '35-44': 'Moderate',
        '45-54': 'Moderate',
        '55+': 'Conservative',
    }

    base_profile = age_to_profile.get(age_range, 'Moderate')

    # Adjust based on savings rate if available
    savings_rate = None
    if financial_health and isinstance(financial_health, dict):
        metrics = financial_health.get('metrics') or {}
        savings_rate = metrics.get('savings_rate')

    if isinstance(savings_rate, (int, float)):
        if savings_rate < 10 and base_profile == 'Aggressive':
            risk_profile = 'Moderate'
        elif savings_rate > 25 and base_profile == 'Moderate':
            risk_profile = 'Aggressive'
        else:
            risk_profile = base_profile
    else:
        risk_profile = base_profile

    if risk_profile == 'Aggressive':
        split = {'low_risk_pct': 30, 'growth_pct': 70}
    elif risk_profile == 'Moderate':
        split = {'low_risk_pct': 50, 'growth_pct': 50}
    else:
        split = {'low_risk_pct': 70, 'growth_pct': 30}

    # Ensure monthly_income is valid
    income = float(monthly_income or 0)

    # Suggested buckets and within-bucket allocations
    low_risk_options = [
        {'name': 'Emergency Fund (Liquid/Overnight Fund)', 'weight': 0.40, 'notes': 'Target 6 months expenses. Highly liquid, low volatility.'},
        {'name': 'Bank FD / RD', 'weight': 0.25, 'notes': 'Stable returns, consider laddering for liquidity.'},
        {'name': 'PPF / EPF / VPF', 'weight': 0.20, 'notes': 'Long-term, tax-efficient. Lock-in applies (PPF 15y).'},
        {'name': 'Short-Duration Debt Fund', 'weight': 0.15, 'notes': 'Low duration risk; suitable for 1–3 years horizon.'},
    ]

    growth_options = [
        {'name': 'Nifty 50 Index Fund / ETF', 'weight': 0.40, 'notes': 'Core equity exposure with low cost.'},
        {'name': 'Large & Flexi-cap Fund', 'weight': 0.30, 'notes': 'Diversified equity with risk-managed approach.'},
        {'name': 'ELSS (Tax-saving Equity)', 'weight': 0.15, 'notes': '3-year lock-in; only if you need 80C benefit.'},
        {'name': 'NPS (Equity Allocation)', 'weight': 0.15, 'notes': 'Retirement-focused; tiered liquidity and tax benefits.'},
    ]

    # Normalize weights to 1.0 just in case
    def normalize(options):
        total = sum(x['weight'] for x in options)
        return [dict(x, weight=(x['weight'] / total if total else 0)) for x in options]

    low_risk_options = normalize(low_risk_options)
    growth_options = normalize(growth_options)

    low_risk_amt = income * (split['low_risk_pct'] / 100.0)
    growth_amt = income * (split['growth_pct'] / 100.0)

    def allocate(options, bucket_amount):
        out = []
        for x in options:
            pct_of_income = (split['low_risk_pct'] if options is low_risk_options else split['growth_pct']) * x['weight']
            amt = bucket_amount * x['weight']
            out.append({
                'name': x['name'],
                'percentage': round(pct_of_income, 1),
                'amount': amt,
                'display_amount': f"₹{amt:,.0f}",
                'notes': x['notes'],
            })
        return out

    suggested = {
        'low_risk': allocate(low_risk_options, low_risk_amt),
        'growth': allocate(growth_options, growth_amt),
    }

    notes = []
    if goals:
        # If any short-term goal (< 24 months), nudge more low-risk focus
        try:
            import datetime
            soon = False
            for g in goals:
                target_date = g.get('target_date')
                # Templates may serialize dates to string; handle both
                if isinstance(target_date, str):
                    try:
                        target_date = datetime.datetime.fromisoformat(target_date)
                    except Exception:
                        target_date = None
                if target_date:
                    months = (target_date.year - datetime.datetime.now().year) * 12 + (target_date.month - datetime.datetime.now().month)
                    if months <= 24:
                        soon = True
                        break
            if soon and risk_profile != 'Conservative':
                notes.append('Short-term goals detected (≤ 24 months). Consider shifting 10% more towards low-risk instruments until goals are funded.')
        except Exception:
            pass

    # General best-practice notes for India context
    notes.extend([
        'Prioritize building a 3–6 month emergency fund before increasing equity exposure.',
        'Rebalance portfolio annually to maintain your chosen risk split.',
        'Use low-cost index funds as core holdings; add actives only if they fit your plan.',
    ])

    return {
        'risk_profile': risk_profile,
        'split': split,
        'suggested_allocation': suggested,
        'notes': notes,
    }
