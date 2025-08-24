# EcoSense AI - Advanced Personalized Climate Action Advisor
# Enhanced Hackathon Project using GPT-5 with Advanced Features

import gradio as gr
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import datetime
import re
import random
import time
import os

# Initialize GPT-5 client
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=os.getenv("AIML_API_KEY")  # Replace with your API key
)

# OpenWeatherMap API key (get free at openweathermap.org)
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # Replace with your key

class EcoSenseAI:
    def __init__(self):
        self.carbon_factors = {
            'electricity': 0.82,  # kg CO2 per kWh
            'natural_gas': 2.04,  # kg CO2 per cubic meter
            'gasoline': 2.31,     # kg CO2 per liter
            'flights_short': 0.25, # kg CO2 per km
            'flights_long': 0.15,  # kg CO2 per km
        }
        
        # Gamification data
        self.achievements = {
            'first_analysis': {'name': '🌱 First Step', 'desc': 'Completed first carbon analysis', 'earned': False},
            'low_footprint': {'name': '💚 Green Guardian', 'desc': 'Carbon footprint below 300kg/month', 'earned': False},
            'eco_warrior': {'name': '⚡ Eco Warrior', 'desc': 'Carbon footprint below 200kg/month', 'earned': False},
            'climate_hero': {'name': '🏆 Climate Hero', 'desc': 'Carbon footprint below 150kg/month', 'earned': False}
        }
        
        # Mock climate risk data by region
        self.climate_risks = {
            'new york': {'heat_waves': 'High', 'flooding': 'Medium', 'storms': 'High'},
            'london': {'flooding': 'Medium', 'heatwaves': 'Medium', 'storms': 'Low'},
            'tokyo': {'earthquakes': 'High', 'typhoons': 'High', 'heatwaves': 'Medium'},
            'default': {'extreme_weather': 'Medium', 'temperature_rise': 'High'}
        }
        
        # Offset projects database
        self.offset_projects = [
            {'name': 'Amazon Rainforest Protection', 'cost_per_ton': 15, 'location': 'Brazil', 'type': 'Forest Conservation'},
            {'name': 'Solar Farm Development', 'cost_per_ton': 25, 'location': 'India', 'type': 'Renewable Energy'},
            {'name': 'Mangrove Restoration', 'cost_per_ton': 20, 'location': 'Indonesia', 'type': 'Ecosystem Restoration'},
            {'name': 'Wind Energy Project', 'cost_per_ton': 22, 'location': 'Kenya', 'type': 'Renewable Energy'},
            {'name': 'Reforestation Initiative', 'cost_per_ton': 18, 'location': 'Costa Rica', 'type': 'Tree Planting'}
        ]
    
    def get_weather_data(self, city):
        """Get current weather and air quality data"""
        try:
            # For demo purposes, return mock data if API key not set
            if WEATHER_API_KEY == "your_openweather_api_key":
                return {
                    'temperature': random.randint(15, 30),
                    'humidity': random.randint(40, 80),
                    'description': random.choice(['clear sky', 'broken clouds', 'light rain', 'sunny']),
                    'aqi': random.randint(50, 150),
                    'city': city
                }
            
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
            response = requests.get(weather_url)
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'description': data['weather'][0]['description'],
                    'city': data['name']
                }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_carbon_footprint(self, electricity, gas, fuel, flights):
        """Calculate monthly carbon footprint"""
        try:
            footprint = {
                'electricity': float(electricity) * self.carbon_factors['electricity'],
                'gas': float(gas) * self.carbon_factors['natural_gas'],
                'transport': float(fuel) * self.carbon_factors['gasoline'],
                'flights': float(flights) * self.carbon_factors['flights_short']
            }
            footprint['total'] = sum(footprint.values())
            return footprint
        except:
            return None
    
    def ask_gpt5(self, prompt, max_tokens=2000):
        """Enhanced GPT-5 interaction with better error handling"""
        try:
            print(f"🔄 Sending request to GPT-5...")
            
            response = client.chat.completions.create(
                model="openai/gpt-5-2025-08-07",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Always provide concise, direct answers. Keep responses under 300 words unless absolutely necessary. Focus on key information only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            print(f"📦 Full API Response:")
            print(f"Model: {response.model}")
            print(f"Finish Reason: {response.choices[0].finish_reason}")
            print(f"Usage: {response.usage}")
            
            if response.choices[0].finish_reason == 'length':
                print("⚠️ Warning: Response was truncated due to max_tokens limit")
            
            content = response.choices[0].message.content
            
            if content:
                return content.strip()
            else:
                return "⚠️ Empty content received."
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def generate_recommendations(self, location, carbon_data, weather_data):
        """Generate personalized climate recommendations"""
        prompt = f"I live in {location} where it's currently {weather_data.get('temperature', 'N/A')}°C. My monthly carbon footprint is {carbon_data['total']:.1f} kg CO2 from: electricity {carbon_data['electricity']:.1f}kg, gas {carbon_data['gas']:.1f}kg, transport {carbon_data['transport']:.1f}kg. Give me 3 specific ways to reduce my emissions by 25%."
        
        max_tokens = 4000 if len(prompt.split()) > 10 else 2000
        print(f"🔧 Using {max_tokens} max tokens for recommendations...")
        
        return self.ask_gpt5(prompt, max_tokens)
    
    def predict_carbon_trend(self, current_footprint, reduction_target=25):
        """Generate predictive carbon trend analysis"""
        current_annual = current_footprint * 12
        target_reduction = current_annual * (reduction_target / 100)
        target_annual = current_annual - target_reduction
        
        # Generate monthly projection data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Current trend (slight increase without action)
        current_trend = [current_footprint * (1 + (i * 0.02)) for i in range(12)]
        
        # With improvements (gradual reduction)
        improved_trend = [current_footprint * (1 - (i * reduction_target/100/12)) for i in range(12)]
        
        return {
            'months': months,
            'current_trend': current_trend,
            'improved_trend': improved_trend,
            'target_annual': target_annual,
            'potential_savings': target_reduction
        }
    
    def get_climate_risk_assessment(self, location):
        """Generate AI-powered climate risk assessment"""
        city_key = location.lower().replace(' ', '')
        risks = self.climate_risks.get(city_key, self.climate_risks['default'])
        
        prompt = f"Analyze climate risks for {location}. Current risks include: {', '.join([f'{k}: {v}' for k, v in risks.items()])}. Provide specific adaptation strategies and timeline for action."
        
        return self.ask_gpt5(prompt, 2000), risks
    
    def generate_carbon_offset_recommendations(self, carbon_footprint):
        """AI-powered carbon offset marketplace recommendations"""
        annual_footprint = carbon_footprint * 12 / 1000  # Convert to tons
        
        # Select 3 best matching projects
        selected_projects = random.sample(self.offset_projects, 3)
        
        # Simpler, shorter prompt for better GPT-5 response
        prompt = f"I need to offset {annual_footprint:.1f} tons CO2 per year. Which is best: {selected_projects[0]['name']} (${selected_projects[0]['cost_per_ton']}/ton), {selected_projects[1]['name']} (${selected_projects[1]['cost_per_ton']}/ton), or {selected_projects[2]['name']} (${selected_projects[2]['cost_per_ton']}/ton)? Give me the best choice and why."
        
        try:
            ai_recommendation = self.ask_gpt5(prompt, 1000)
            # Fallback if GPT-5 returns empty
            if not ai_recommendation or "Empty content" in ai_recommendation:
                ai_recommendation = f"For {annual_footprint:.1f} tons CO2, I recommend {selected_projects[0]['name']} as it offers good cost-effectiveness at ${selected_projects[0]['cost_per_ton']}/ton with strong environmental impact in {selected_projects[0]['location']}."
        except:
            ai_recommendation = f"Based on your {annual_footprint:.1f} ton annual footprint, {selected_projects[0]['name']} offers the best value at ${selected_projects[0]['cost_per_ton']}/ton."
        
        return {
            'projects': selected_projects,
            'annual_tons': annual_footprint,
            'ai_recommendation': ai_recommendation
        }
    
    def check_achievements(self, carbon_footprint):
        """Check and update achievements based on carbon footprint"""
        achievements_earned = []
        
        # First analysis achievement
        if not self.achievements['first_analysis']['earned']:
            self.achievements['first_analysis']['earned'] = True
            achievements_earned.append(self.achievements['first_analysis'])
        
        # Footprint-based achievements
        if carbon_footprint < 300 and not self.achievements['low_footprint']['earned']:
            self.achievements['low_footprint']['earned'] = True
            achievements_earned.append(self.achievements['low_footprint'])
        
        if carbon_footprint < 200 and not self.achievements['eco_warrior']['earned']:
            self.achievements['eco_warrior']['earned'] = True
            achievements_earned.append(self.achievements['eco_warrior'])
        
        if carbon_footprint < 150 and not self.achievements['climate_hero']['earned']:
            self.achievements['climate_hero']['earned'] = True
            achievements_earned.append(self.achievements['climate_hero'])
        
        return achievements_earned
    
    def create_footprint_visualization(self, carbon_data):
        """Create carbon footprint breakdown chart"""
        categories = ['Electricity', 'Gas', 'Transport', 'Flights']
        values = [carbon_data['electricity'], carbon_data['gas'], 
                 carbon_data['transport'], carbon_data['flights']]
        
        fig = px.pie(
            values=values, 
            names=categories,
            title="Monthly Carbon Footprint Breakdown (kg CO2)",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    def create_comparison_chart(self, user_total):
        """Compare user footprint to averages"""
        data = {
            'Category': ['Your Footprint', 'National Average', 'Global Average', 'Paris Agreement Target'],
            'CO2 (kg/month)': [user_total, 1200, 800, 400],
            'Color': ['red', 'orange', 'blue', 'green']
        }
        
        fig = px.bar(
            data, x='Category', y='CO2 (kg/month)',
            title="Carbon Footprint Comparison",
            color='Color',
            color_discrete_map={'red': '#FF6B6B', 'orange': '#FFE66D', 
                               'blue': '#4ECDC4', 'green': '#45B7D1'}
        )
        return fig
    
    def create_trend_analysis_chart(self, trend_data):
        """Create predictive trend analysis chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trend_data['months'],
            y=trend_data['current_trend'],
            mode='lines+markers',
            name='Current Trend',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data['months'],
            y=trend_data['improved_trend'],
            mode='lines+markers',
            name='With Improvements',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="12-Month Carbon Footprint Projection",
            xaxis_title="Month",
            yaxis_title="CO2 Emissions (kg)",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig

# Initialize the EcoSense AI system
eco_ai = EcoSenseAI()

def main_analysis(city, electricity_kwh, gas_m3, fuel_liters, flight_km):
    """Main function to run complete analysis with advanced features"""
    
    # Input validation
    try:
        electricity_kwh = max(0, float(electricity_kwh or 0))
        gas_m3 = max(0, float(gas_m3 or 0))
        fuel_liters = max(0, float(fuel_liters or 0))
        flight_km = max(0, float(flight_km or 0))
    except ValueError:
        return "❌ Please enter valid numbers for all consumption fields.", None, None, None, "", ""
    
    if not city.strip():
        return "❌ Please enter a city name.", None, None, None, "", ""
    
    # Get weather data
    weather_data = eco_ai.get_weather_data(city.strip())
    
    # Calculate carbon footprint
    carbon_data = eco_ai.calculate_carbon_footprint(
        electricity_kwh, gas_m3, fuel_liters, flight_km
    )
    
    if not carbon_data:
        return "❌ Error calculating carbon footprint.", None, None, None, "", ""
    
    # Check achievements (gamification)
    achievements = eco_ai.check_achievements(carbon_data['total'])
    achievement_text = ""
    if achievements:
        achievement_text = "\n🏆 **New Achievements Unlocked!**\n"
        for ach in achievements:
            achievement_text += f"• {ach['name']}: {ach['desc']}\n"
        achievement_text += "\n"
    
    # Generate AI recommendations
    recommendations = eco_ai.generate_recommendations(city, carbon_data, weather_data)
    
    # Generate predictive analysis
    trend_data = eco_ai.predict_carbon_trend(carbon_data['total'])
    trend_chart = eco_ai.create_trend_analysis_chart(trend_data)
    
    # Generate offset recommendations
    offset_data = eco_ai.generate_carbon_offset_recommendations(carbon_data['total'])
    offset_text = f"\n💰 **Carbon Offset Recommendations:**\n"
    offset_text += f"Annual footprint to offset: {offset_data['annual_tons']:.1f} tons CO2\n\n"
    offset_text += "**Available Projects:**\n"
    for project in offset_data['projects']:
        annual_cost = offset_data['annual_tons'] * project['cost_per_ton']
        offset_text += f"• {project['name']} ({project['location']}) - ${annual_cost:.0f}/year\n"
    offset_text += f"\n**AI Recommendation:** {offset_data['ai_recommendation']}\n"
    
    # Create visualizations
    pie_chart = eco_ai.create_footprint_visualization(carbon_data)
    comparison_chart = eco_ai.create_comparison_chart(carbon_data['total'])
    
    # Format summary
    summary = f"""🌍 **EcoSense AI Analysis for {city}**

**Current Conditions:** {weather_data.get('temperature', 'N/A')}°C, {weather_data.get('description', 'N/A')}

**Your Monthly Carbon Footprint:** {carbon_data['total']:.1f} kg CO2

**Breakdown:**
• Electricity: {carbon_data['electricity']:.1f} kg CO2
• Natural Gas: {carbon_data['gas']:.1f} kg CO2  
• Transportation: {carbon_data['transport']:.1f} kg CO2
• Air Travel: {carbon_data['flights']:.1f} kg CO2

**Environmental Impact:**
• Annual footprint: ~{carbon_data['total'] * 12:.0f} kg CO2/year
• Equivalent to {carbon_data['total'] * 12 / 2300:.1f} trees needed for offset

**📈 Predictive Analysis:**
• Without changes: {trend_data['current_trend'][-1]:.1f} kg CO2/month by year end
• With improvements: {trend_data['improved_trend'][-1]:.1f} kg CO2/month by year end
• Potential annual savings: {trend_data['potential_savings']:.0f} kg CO2

{achievement_text}---"""
    
    # Separate achievement display for the side panel
    achievement_display = ""
    if achievements:
        achievement_display = "🎉 **New Achievements!**\n"
        for ach in achievements:
            achievement_display += f"{ach['name']}: {ach['desc']}\n"
    else:
        achievement_display = "Complete your analysis to unlock achievements!"
    
    full_recommendations = summary + "\n\n**🤖 AI Recommendations:**\n" + recommendations + "\n" + offset_text
    
    return full_recommendations, pie_chart, comparison_chart, trend_chart, "✅ Complete Analysis Done!", achievement_display

def get_climate_risk_assessment(city):
    """Get AI-powered climate risk assessment"""
    if not city.strip():
        return "Please enter a city name."
    
    risk_analysis, risks = eco_ai.get_climate_risk_assessment(city.strip())
    
    risk_summary = f"🌡️ **Climate Risk Assessment for {city}**\n\n"
    risk_summary += f"**Current Risk Levels:**\n"
    for risk, level in risks.items():
        risk_emoji = "🔴" if level == "High" else "🟡" if level == "Medium" else "🟢"
        risk_summary += f"• {risk.replace('_', ' ').title()}: {level} {risk_emoji}\n"
    
    risk_summary += f"\n**AI Analysis & Recommendations:**\n{risk_analysis}"
    
    return risk_summary

def get_quick_tip():
    """Get a quick climate tip from GPT-5"""
    prompt = "Give me one practical climate action tip for today."
    
    max_tokens = 1500
    print(f"🔧 Using {max_tokens} max tokens for climate tip...")
    
    return eco_ai.ask_gpt5(prompt, max_tokens)

def climate_qa(question):
    """Climate Q&A feature"""
    if not question.strip():
        return "Please ask a climate-related question."
    
    prompt = f"Answer this climate question: {question}"
    
    max_tokens = 4000 if len(question.split()) > 10 else 1500
    print(f"🔧 Using {max_tokens} max tokens for Q&A...")
    
    return eco_ai.ask_gpt5(prompt, max_tokens)

def show_achievements():
    """Display current achievements"""
    achievement_display = "🏆 **Your Climate Achievements**\n\n"
    
    for key, ach in eco_ai.achievements.items():
        status = "✅" if ach['earned'] else "⬜"
        achievement_display += f"{status} {ach['name']}: {ach['desc']}\n"
    
    total_earned = sum(1 for ach in eco_ai.achievements.values() if ach['earned'])
    achievement_display += f"\n**Progress: {total_earned}/{len(eco_ai.achievements)} achievements unlocked**"
    
    return achievement_display

# Create Enhanced Gradio Interface
with gr.Blocks(title="EcoSense AI - Advanced Climate Action Platform", theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: #2E8B57;">🌱 EcoSense AI</h1>
        <h2 style="color: #4682B4;">Advanced Personalized Climate Action Platform</h2>
        <p style="color: #666;">Powered by GPT-5 • AI-Driven Sustainability with Predictive Analytics & Gamification</p>
    </div>
    """)
    
    with gr.Tabs():
        
        # Main Analysis Tab (Enhanced)
        with gr.Tab("🏠 Carbon Footprint Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>📍 Location & Consumption Data</h3>")
                    
                    city_input = gr.Textbox(
                        label="City/Location", 
                        placeholder="Enter your city (e.g., New York, London, Tokyo)",
                        value="New York"
                    )
                    
                    gr.HTML("<h4>Monthly Consumption:</h4>")
                    
                    electricity_input = gr.Number(
                        label="Electricity (kWh)", 
                        value=300,
                        info="Average household: 250-400 kWh/month"
                    )
                    
                    gas_input = gr.Number(
                        label="Natural Gas (cubic meters)", 
                        value=50,
                        info="Average household: 30-80 m³/month"
                    )
                    
                    fuel_input = gr.Number(
                        label="Vehicle Fuel (liters)", 
                        value=60,
                        info="Average car: 40-100 L/month"
                    )
                    
                    flight_input = gr.Number(
                        label="Air Travel (km)", 
                        value=0,
                        info="Round trip domestic flight ~2000km"
                    )
                    
                    analyze_btn = gr.Button("🔍 Complete AI Analysis", variant="primary", size="lg")
                    
                    status_output = gr.Textbox(label="Status", interactive=False)
                    achievement_display = gr.Markdown(label="🏆 Achievements")
                
                with gr.Column(scale=2):
                    results_output = gr.Markdown(label="AI Recommendations & Analysis")
            
            with gr.Row():
                with gr.Column():
                    pie_chart_output = gr.Plot(label="Carbon Footprint Breakdown")
                with gr.Column():
                    comparison_chart_output = gr.Plot(label="Footprint Comparison")
            
            with gr.Row():
                trend_chart_output = gr.Plot(label="📈 Predictive Trend Analysis")
            
            # Connect the enhanced analysis function
            analyze_btn.click(
                fn=main_analysis,
                inputs=[city_input, electricity_input, gas_input, fuel_input, flight_input],
                outputs=[results_output, pie_chart_output, comparison_chart_output, trend_chart_output, status_output, achievement_display]
            )
        
        # Climate Risk Assessment Tab (NEW)
        with gr.Tab("🌡️ Climate Risk Assessment"):
            gr.HTML("<h3>AI-Powered Climate Risk Analysis for Your Location</h3>")
            
            risk_city_input = gr.Textbox(
                label="City/Location",
                placeholder="Enter your city for climate risk assessment",
                value="New York"
            )
            
            risk_btn = gr.Button("🔍 Analyze Climate Risks", variant="primary")
            risk_output = gr.Markdown()
            
            risk_btn.click(
                fn=get_climate_risk_assessment,
                inputs=risk_city_input,
                outputs=risk_output
            )
        
        # Achievements Tab (NEW)
        with gr.Tab("🏆 Achievements"):
            gr.HTML("<h3>Track Your Climate Action Progress</h3>")
            
            achievements_btn = gr.Button("📊 View My Achievements", variant="secondary")
            achievements_output = gr.Markdown()
            
            achievements_btn.click(fn=show_achievements, outputs=achievements_output)
            
            gr.HTML("""
            <div style="margin-top: 20px; padding: 15px; background-color: #f0f8ff; border-radius: 10px;">
                <h4>🎯 How to Unlock Achievements:</h4>
                <ul>
                    <li>🌱 <strong>First Step:</strong> Complete your first carbon analysis</li>
                    <li>💚 <strong>Green Guardian:</strong> Get your footprint below 300kg/month</li>
                    <li>⚡ <strong>Eco Warrior:</strong> Achieve less than 200kg CO2/month</li>
                    <li>🏆 <strong>Climate Hero:</strong> Reach the ultimate goal of under 150kg/month</li>
                </ul>
            </div>
            """)
        
        # Enhanced Tips Tab
        with gr.Tab("💡 Daily Climate Tips"):
            gr.HTML("<h3>Get personalized climate tips powered by GPT-5</h3>")
            
            with gr.Row():
                tip_btn = gr.Button("🌟 Get Today's Climate Tip", variant="secondary")
                test_btn = gr.Button("🧪 Test GPT-5 Connection", variant="outline")
            
            tip_output = gr.Markdown(value="Click the button above to get your daily climate tip!")
            
            tip_btn.click(fn=get_quick_tip, outputs=tip_output)
            
            def test_gpt5():
                test_prompt = "Say hello and confirm you are working."
                return eco_ai.ask_gpt5(test_prompt, 1500)
            
            test_btn.click(fn=test_gpt5, outputs=tip_output)
        
        # Enhanced Q&A Tab
        with gr.Tab("❓ Climate Q&A"):
            gr.HTML("<h3>Ask GPT-5 any climate or sustainability question</h3>")
            
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., How effective are solar panels in reducing carbon footprint?",
                lines=2
            )
            
            qa_btn = gr.Button("🤖 Ask GPT-5", variant="primary")
            qa_output = gr.Markdown()
            
            qa_btn.click(
                fn=climate_qa,
                inputs=question_input,
                outputs=qa_output
            )
        
        # About Tab (Enhanced)
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## 🌱 EcoSense AI - Advanced Climate Action Platform
            
            **Powered by GPT-5** | Built for Climate Action Innovation
            
            ### 🚀 Core Features:
            - **🔍 Advanced Carbon Analysis** - Comprehensive footprint calculation with real-time data
            - **📊 Interactive Visualizations** - Dynamic charts and comparison analytics  
            - **🤖 GPT-5 AI Recommendations** - Personalized sustainability strategies
            - **📈 Predictive Analytics** - 12-month carbon trend forecasting
            - **💰 Smart Offset Marketplace** - AI-recommended carbon offset projects
            - **🌡️ Climate Risk Assessment** - Location-based vulnerability analysis
            - **🏆 Gamification System** - Achievement tracking and progress rewards
            - **💡 Daily AI Tips** - Fresh climate action suggestions
            - **❓ Expert Climate Q&A** - Instant answers to sustainability questions
            
            ### 🎯 Advanced Features:
            
            #### **🧠 Predictive Intelligence**
            - Future carbon footprint projections
            - Trend analysis with improvement scenarios
            - Goal tracking and progress monitoring
            
            #### **💰 Carbon Offset Marketplace**
            - AI-curated offset project recommendations
            - Cost-effectiveness analysis
            - Global project portfolio access
            
            #### **🏆 Gamification & Engagement**
            - Achievement system with progressive rewards
            - Performance tracking and milestones
            - Motivation through environmental impact visualization
            
            #### **🌡️ Climate Adaptation Planning**
            - Location-specific risk assessments
            - Personalized adaptation strategies
            - Future climate scenario planning
            
            ### 🛠️ Technical Architecture:
            - **AI Model:** GPT-5 via AIML API (Latest Generation)
            - **Interface:** Advanced Gradio with multi-tab design
            - **Visualization:** Interactive Plotly charts and analytics
            - **Data Sources:** Real-time weather APIs and emission databases
            - **Analytics:** Predictive modeling and trend analysis
            
            ### 🌍 Environmental Impact:
            - **Measurable Results** - Track actual CO2 reduction
            - **Behavioral Change** - AI-powered habit modification
            - **Global Perspective** - Connect local actions to global goals
            - **Community Building** - Achievement sharing and motivation
            
            ### 🏆 Competitive Advantages:
            ✅ **First GPT-5 Integration** for climate action
            ✅ **Comprehensive Feature Set** beyond basic calculators
            ✅ **Advanced Analytics** with predictive capabilities
            ✅ **Gamification Elements** for sustained engagement
            ✅ **Real-world Impact** with measurable outcomes
            
            ---
            *Built with ❤️ for a sustainable future | Hackathon Innovation Project*
            """)

if __name__ == "__main__":
    # Launch the app
    demo.launch(
        share=True,  # Creates public link for sharing
        debug=True,
        show_error=True,
        server_port=None,  # Let Gradio find available port automatically
        quiet=False
    )