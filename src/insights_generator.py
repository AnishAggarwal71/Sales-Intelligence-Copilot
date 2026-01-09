"""
AI Insights Generator Module
Uses OpenAI GPT to generate executive summaries and recommendations
"""

import os
from openai import OpenAI
from typing import Dict, List
import logging
from pathlib import Path
import sys

# Import settings
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    OPENAI_MODEL,
    OPENAI_MAX_TOKENS,
    OPENAI_TEMPERATURE,
    EXECUTIVE_SUMMARY_PROMPT,
    RECOMMENDATIONS_PROMPT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsightsGenerator:
    """Generate AI-powered business insights using LLM"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize insights generator
        
        Args:
            api_key: OpenAI API key (if None, reads from env)
        """
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in .env file."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = OPENAI_MODEL
        self.max_tokens = OPENAI_MAX_TOKENS
        self.temperature = OPENAI_TEMPERATURE
    
    def generate_executive_summary(self, metrics: Dict) -> str:
        """
        Generate executive summary from business metrics
        
        Args:
            metrics: Dictionary with business metrics
            
        Returns:
            Executive summary text
        """
        logger.info("Generating executive summary with LLM...")
        
        # Prepare prompt with metrics
        prompt = EXECUTIVE_SUMMARY_PROMPT.format(
            current_mrr=metrics.get('current_mrr', 0),
            mrr_growth=metrics.get('mrr_growth_mom', 0),
            forecast_mrr=metrics.get('forecast_mrr', metrics.get('current_mrr', 0) * 1.05),
            churn_rate=metrics.get('churn_rate', 0),
            active_customers=metrics.get('active_customers', 0),
            at_risk_count=metrics.get('at_risk_count', 0)
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a business analyst creating executive summaries for SaaS companies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info("✓ Executive summary generated")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._fallback_summary(metrics)
    
    def generate_recommendations(self, metrics: Dict, at_risk_customers: List[Dict]) -> str:
        """
        Generate actionable recommendations
        
        Args:
            metrics: Dictionary with business metrics
            at_risk_customers: List of at-risk customer data
            
        Returns:
            Recommendations text
        """
        logger.info("Generating recommendations with LLM...")
        
        # Prepare at-risk summary
        at_risk_summary = ""
        if at_risk_customers:
            top_5 = at_risk_customers[:5]
            at_risk_summary = ", ".join([
                f"Customer {c['customer_id']} (${c['current_revenue']:.0f}/mo, {c['churn_probability']*100:.0f}% risk)"
                for c in top_5
            ])
        
        # Determine MRR trend
        mrr_growth = metrics.get('mrr_growth_mom', 0)
        if mrr_growth > 5:
            mrr_trend = "strong growth"
        elif mrr_growth > 0:
            mrr_trend = "moderate growth"
        elif mrr_growth > -5:
            mrr_trend = "slight decline"
        else:
            mrr_trend = "concerning decline"
        
        prompt = RECOMMENDATIONS_PROMPT.format(
            at_risk_summary=at_risk_summary or "No high-risk customers identified",
            churn_rate=metrics.get('churn_rate', 0),
            mrr_trend=mrr_trend
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strategic business advisor for SaaS companies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            recommendations = response.choices[0].message.content.strip()
            logger.info("✓ Recommendations generated")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._fallback_recommendations(metrics)
    
    def _fallback_summary(self, metrics: Dict) -> str:
        """Fallback summary if API fails"""
        return f"""
EXECUTIVE SUMMARY

Revenue Performance:
• Current MRR stands at ${metrics.get('current_mrr', 0):,.0f} with {metrics.get('mrr_growth_mom', 0):+.1f}% month-over-month growth
• Annualized ARR of ${metrics.get('arr', 0):,.0f} serving {metrics.get('active_customers', 0):,} active customers

Customer Health:
• Current churn rate at {metrics.get('churn_rate', 0):.1f}% with {metrics.get('at_risk_count', 0)} high-value customers identified as at-risk
• Average revenue per user (ARPU) of ${metrics.get('arpu', 0):,.2f}

Key Implications:
• Proactive retention efforts needed for at-risk segment to protect revenue base
• Revenue trajectory indicates {"positive" if metrics.get('mrr_growth_mom', 0) > 0 else "concerning"} momentum requiring {"continued investment" if metrics.get('mrr_growth_mom', 0) > 0 else "immediate attention"}
        """.strip()
    
    def _fallback_recommendations(self, metrics: Dict) -> str:
        """Fallback recommendations if API fails"""
        churn = metrics.get('churn_rate', 0)
        
        return f"""
STRATEGIC RECOMMENDATIONS

1. Customer Retention Program:
Deploy targeted retention campaigns for the {metrics.get('at_risk_count', 0)} at-risk customers. Consider personalized outreach, loyalty discounts, or enhanced support to reduce churn from current {churn:.1f}%.

2. Revenue Protection:
Prioritize high-value at-risk customers for immediate intervention. Assign dedicated customer success managers to accounts with monthly revenue exceeding $200.

3. Product & Engagement:
Analyze usage patterns among churned customers to identify product gaps. Implement proactive engagement triggers when usage metrics drop below historical averages.
        """.strip()
    
    def generate_full_report_insights(self, metrics: Dict, at_risk_customers: List[Dict]) -> Dict[str, str]:
        """
        Generate all insights for a complete report
        
        Args:
            metrics: Business metrics dictionary
            at_risk_customers: List of at-risk customer data
            
        Returns:
            Dictionary with summary and recommendations
        """
        # Prepare at-risk customer list
        at_risk_list = []
        if at_risk_customers is not None and len(at_risk_customers) > 0:
            at_risk_list = at_risk_customers.to_dict('records') if hasattr(at_risk_customers, 'to_dict') else at_risk_customers
        
        return {
            'executive_summary': self.generate_executive_summary(metrics),
            'recommendations': self.generate_recommendations(metrics, at_risk_list)
        }


def generate_insights(metrics: Dict, at_risk_customers=None) -> Dict[str, str]:
    """
    Convenience function to generate all insights
    
    Args:
        metrics: Business metrics dictionary
        at_risk_customers: DataFrame or list of at-risk customers
        
    Returns:
        Dictionary with generated insights
    """
    try:
        generator = InsightsGenerator()
        
        # Convert DataFrame to list if needed
        at_risk_list = []
        if at_risk_customers is not None:
            if hasattr(at_risk_customers, 'to_dict'):
                at_risk_list = at_risk_customers.to_dict('records')
            else:
                at_risk_list = at_risk_customers
        
        return generator.generate_full_report_insights(metrics, at_risk_list)
        
    except Exception as e:
        logger.error(f"Error in insights generation: {e}")
        # Return fallback insights
        generator = InsightsGenerator.__new__(InsightsGenerator)
        return {
            'executive_summary': generator._fallback_summary(metrics),
            'recommendations': generator._fallback_recommendations(metrics)
        }


# Test the module
if __name__ == "__main__":
    # Test with sample metrics
    test_metrics = {
        'current_mrr': 245892.50,
        'arr': 2950710.00,
        'mrr_growth_mom': 2.34,
        'active_customers': 8234,
        'arpu': 29.87,
        'churn_rate': 5.21,
        'at_risk_count': 15,
        'forecast_mrr': 258187.13
    }
    
    test_at_risk = [
        {'customer_id': 'CUST_001', 'current_revenue': 499.0, 'churn_probability': 0.89},
        {'customer_id': 'CUST_002', 'current_revenue': 499.0, 'churn_probability': 0.87},
        {'customer_id': 'CUST_003', 'current_revenue': 299.0, 'churn_probability': 0.82}
    ]
    
    print("=" * 60)
    print("TESTING INSIGHTS GENERATOR")
    print("=" * 60)
    
    try:
        insights = generate_insights(test_metrics, test_at_risk)
        
        print("\n📊 EXECUTIVE SUMMARY:")
        print(insights['executive_summary'])
        
        print("\n💡 RECOMMENDATIONS:")
        print(insights['recommendations'])
        
        print("\n✓ Insights generation test successful!")
        
    except Exception as e:
        print(f"\n⚠️ Note: {e}")
        print("This is expected if OPENAI_API_KEY is not set in .env")
        print("Fallback templates will be used in production.")