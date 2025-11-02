from flask import Flask, render_template, request, session, redirect, url_for
import datetime
from budget_calc import calculate_advanced_budget_recommendations

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Needed for sessions


def get_current_year():
    return datetime.datetime.now().year


@app.route("/", methods=['GET', 'POST'])
@app.route("/index.html", methods=['GET', 'POST'])
def index():
    year = get_current_year()

    if request.method == 'POST':
        # Get form data
        monthly_income = float(request.form.get('monthly_income', 0))
        age_range = request.form.get('age_range', '')
        family_size = request.form.get('family_size', '')
        location = request.form.get('location', '')

        # Store form data in session
        session['budget_data'] = {
            'monthly_income': monthly_income,
            'age_range': age_range,
            'family_size': family_size,
            'location': location
        }

        # Calculate budget recommendations and store in session
        recommendations = calculate_advanced_budget_recommendations(
            monthly_income, age_range, family_size, location
        )
        session['recommendations'] = recommendations

        # Redirect to generic.html after form submission
        return redirect(url_for('generic'))

    # For GET requests, just render the index page
    return render_template("index.html", year=year)


@app.route("/generic", methods=['GET'])
def generic():
    year = get_current_year()

    # Retrieve data from session
    budget_data = session.get('budget_data', {})
    recommendations = session.get('recommendations', {})

    # Prepare chart data for the pie chart
    chart_data = None
    if recommendations:
        chart_data = {
            'labels': list(recommendations.keys()),
            'percentages': [rec['percentage'] for rec in recommendations.values()],
            'amounts': [rec['amount'] for rec in recommendations.values()],
            'colors': [
                '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40',
                '#FF6384', '#C9CBCF', '#4BC0C0', '#FFCD56', '#36A2EB', '#FF6384'
            ]
        }

    return render_template(
        "generic.html",
        year=year,
        recommendations=recommendations,
        budget_data=budget_data,
        chart_data=chart_data
    )


@app.route("/elements")
def elements():
    year = get_current_year()
    return render_template("elements.html", year=year)


if __name__ == "__main__":
    app.run(debug=True)