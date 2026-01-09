# Sales Intelligence Copilot

AI-powered automation for sales forecasting & customer churn insights.

## Features
- 📊 Automated MRR forecasting (Prophet, 90-day horizon)
- 🎯 Customer churn prediction (Logistic Regression, 0.85 AUC)
- 🤖 AI-generated executive summaries (OpenAI GPT-4o-mini)
- 📈 Interactive dashboards (Streamlit + Plotly)
- 📄 Automated PowerPoint reports

## Quick Start
```bash
# Clone and setup
git clone https://github.com/yourusername/sales-intelligence-copilot.git
cd sales-intelligence-copilot
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Generate sample data
python data/generate_data.py

# Run app
streamlit run app.py
```

## Demo
[Add screenshot or video here]

## Tech Stack
- **ML**: Prophet, Scikit-learn
- **LLM**: OpenAI GPT-4o-mini
- **UI**: Streamlit, Plotly
- **Reports**: python-pptx

## Project Structure
```
sales-intelligence-copilot/
├── app.py                    # Streamlit interface
├── src/
│   ├── data_processing.py    # Data pipeline
│   ├── metrics.py            # Business metrics
│   ├── forecasting.py        # Prophet model
│   ├── churn_model.py        # Churn prediction
│   ├── insights_generator.py # LLM insights
│   └── report_builder.py     # PPT generation
├── config/settings.py        # Configuration
└── data/generate_data.py     # Synthetic data
```

## License
MIT
