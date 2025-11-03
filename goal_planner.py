from datetime import datetime, timedelta

def calculate_goal_feasibility(target_amount, current_savings, monthly_income, monthly_expenses, target_date=None):
    """
    Calculate if a financial goal is feasible and provide recommendations
    
    Parameters:
    target_amount (float): Goal target amount
    current_savings (float): Current amount saved
    monthly_income (float): Monthly income
    monthly_expenses (float): Total monthly expenses
    target_date (datetime): Target date for goal completion
    
    Returns:
    dict: Goal analysis including feasibility, required savings, and recommendations
    """
    monthly_savings_potential = monthly_income - monthly_expenses
    remaining_amount = target_amount - current_savings
    
    if target_date:
        months_until_target = (target_date - datetime.now()).days / 30
        required_monthly_saving = remaining_amount / months_until_target
        is_feasible = required_monthly_saving <= monthly_savings_potential
    else:
        required_monthly_saving = monthly_savings_potential
        months_to_goal = remaining_amount / monthly_savings_potential if monthly_savings_potential > 0 else float('inf')
        is_feasible = monthly_savings_potential > 0
        
    recommendations = []
    
    if not is_feasible:
        potential_expense_reduction = monthly_expenses * 0.1  # Suggest 10% expense reduction
        new_monthly_savings = monthly_savings_potential + potential_expense_reduction
        
        recommendations.extend([
            "Consider reducing monthly expenses by 10% to increase savings potential",
            f"Look for ways to save an additional ₹{potential_expense_reduction:,.2f} per month",
            "Review discretionary spending for potential savings"
        ])
        
        if target_date:
            extended_date = datetime.now() + timedelta(days=30 * (remaining_amount / new_monthly_savings))
            recommendations.append(f"Consider extending your target date to {extended_date.strftime('%B %Y')}")
    
    return {
        "is_feasible": is_feasible,
        "required_monthly_saving": required_monthly_saving,
        "current_monthly_savings": monthly_savings_potential,
        "remaining_amount": remaining_amount,
        "recommendations": recommendations
    }

def track_goal_progress(target_amount, current_amount, start_date, target_date=None):
    """
    Track progress towards a financial goal
    
    Parameters:
    target_amount (float): Goal target amount
    current_amount (float): Current amount saved
    start_date (datetime): Date the goal was started
    target_date (datetime): Target date for goal completion
    
    Returns:
    dict: Progress tracking information
    """
    progress_percentage = (current_amount / target_amount) * 100
    amount_remaining = target_amount - current_amount
    
    days_elapsed = (datetime.now() - start_date).days
    
    if target_date:
        total_days = (target_date - start_date).days
        days_remaining = (target_date - datetime.now()).days
        time_percentage = (days_elapsed / total_days) * 100
        
        # Check if progress is on track
        expected_amount = (days_elapsed / total_days) * target_amount
        is_on_track = current_amount >= expected_amount
    else:
        days_remaining = None
        time_percentage = None
        is_on_track = progress_percentage > 0
    
    milestones = [
        {"percentage": 25, "achieved": progress_percentage >= 25},
        {"percentage": 50, "achieved": progress_percentage >= 50},
        {"percentage": 75, "achieved": progress_percentage >= 75},
        {"percentage": 100, "achieved": progress_percentage >= 100}
    ]
    
    return {
        "progress_percentage": progress_percentage,
        "amount_remaining": amount_remaining,
        "days_elapsed": days_elapsed,
        "days_remaining": days_remaining,
        "time_percentage": time_percentage,
        "is_on_track": is_on_track,
        "milestones": milestones
    }