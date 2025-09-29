import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
import pytz
import math
import time
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Smart Energy Source Predictor",
    page_icon="âš¡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme and improved UI
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: white;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        color: white;
    }
    
    .control-panel {
        background-color: #2d2d2d;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .metric-card {
        background-color: #2d2d2d;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.7;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
    }
    
    .ai-prediction {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .prediction-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: white;
    }
    
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: white;
    }
    
    .prediction-desc {
        font-size: 1.1rem;
        opacity: 0.9;
        color: white;
    }
    
    .time-display {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-family: 'Courier New', monospace;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .location-status {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #4CAF50;
    }
    
    .location-error {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #ff6b6b;
    }
    
    .stRadio > div {
        flex-direction: row;
        gap: 2rem;
    }
    
    .stRadio label {
        background-color: #2d2d2d;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        cursor: pointer;
    }
    
    .location-btn {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 16px;
        margin: 10px 0;
    }
    
    .location-btn:hover {
        background: linear-gradient(135deg, #45a049, #3d8b40);
    }
</style>
""", unsafe_allow_html=True)

# Constants
OPENWEATHER_API_KEY = "a0eb17f6a7fcba0c0792ecd57ac8a17c"

def get_browser_location_html():
    """Generate HTML/JavaScript for browser geolocation"""
    return """
    <div>
        <button class="location-btn" onclick="getLocation()">📍 Use My Current Location</button>
        <div id="location-status"></div>
        <div id="location-result" style="display:none;"></div>
    </div>
    
    <script>
    function getLocation() {
        const statusDiv = document.getElementById('location-status');
        const resultDiv = document.getElementById('location-result');
        
        statusDiv.innerHTML = '<p>🔍 Detecting your location...</p>';
        
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                function(position) {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    const accuracy = position.coords.accuracy;
                    
                    // Store in result div for Streamlit to read
                    resultDiv.innerHTML = JSON.stringify({
                        lat: lat,
                        lon: lon,
                        accuracy: accuracy,
                        timestamp: Date.now()
                    });
                    resultDiv.style.display = 'block';
                    
                    statusDiv.innerHTML = `
                        <div class="location-status">
                            <p>✅ Location detected successfully!</p>
                            <p>📍 Latitude: ${lat.toFixed(6)}, Longitude: ${lon.toFixed(6)}</p>
                            <p>🎯 Accuracy: ${Math.round(accuracy)}m</p>
                        </div>
                    `;
                    
                    // Trigger Streamlit rerun
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: {lat: lat, lon: lon, accuracy: accuracy, timestamp: Date.now()}
                    }, '*');
                },
                function(error) {
                    let errorMsg = '';
                    switch(error.code) {
                        case error.PERMISSION_DENIED:
                            errorMsg = "Location access denied by user. Please enable location permissions.";
                            break;
                        case error.POSITION_UNAVAILABLE:
                            errorMsg = "Location information unavailable.";
                            break;
                        case error.TIMEOUT:
                            errorMsg = "Location request timed out.";
                            break;
                        default:
                            errorMsg = "An unknown error occurred.";
                            break;
                    }
                    statusDiv.innerHTML = `
                        <div class="location-error">
                            <p>❌ Error: ${errorMsg}</p>
                            <p>💡 Tip: Make sure location services are enabled in your browser</p>
                        </div>
                    `;
                },
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 60000
                }
            );
        } else {
            statusDiv.innerHTML = `
                <div class="location-error">
                    <p>❌ Geolocation is not supported by this browser.</p>
                </div>
            `;
        }
    }
    
    // Auto-trigger location on page load if needed
    if (window.location.search.includes('auto_locate=true')) {
        getLocation();
    }
    </script>
    """

def get_location_from_coordinates(lat, lon):
    """Get city/country info from coordinates using reverse geocoding"""
    try:
        # Use OpenWeatherMap's reverse geocoding API
        url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={OPENWEATHER_API_KEY}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                location_info = data[0]
                return {
                    'lat': lat,
                    'lon': lon,
                    'city': location_info.get('name', 'Unknown'),
                    'country': location_info.get('country', 'Unknown'),
                    'state': location_info.get('state', ''),
                    'timezone': 'UTC'  # Will be determined later
                }
    except Exception as e:
        st.warning(f"Reverse geocoding failed: {e}")
    
    # Fallback - just return coordinates
    return {
        'lat': lat,
        'lon': lon,
        'city': f'Location ({lat:.2f}, {lon:.2f})',
        'country': 'Unknown',
        'state': '',
        'timezone': 'UTC'
    }

def get_location():
    """Get location - prioritize browser geolocation, fallback to IP geolocation"""
    
    # Check if we have browser location data in session state
    if 'user_location' in st.session_state and st.session_state.user_location:
        loc = st.session_state.user_location
        if 'lat' in loc and 'lon' in loc:
            # Get detailed location info from coordinates
            return get_location_from_coordinates(loc['lat'], loc['lon'])
    
    # Try IP geolocation as fallback
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                return {
                    'lat': data['lat'],
                    'lon': data['lon'],
                    'city': data['city'],
                    'country': data['country'],
                    'timezone': data['timezone']
                }
    except Exception as e:
        st.warning(f"IP geolocation failed: {e}")
    
    # Final fallback to default location (Delhi, India)
    return {
        'lat': 28.6139,
        'lon': 77.2090,
        'city': 'Delhi',
        'country': 'India',
        'timezone': 'Asia/Kolkata'
    }

def get_weather_data(lat, lon):
    """Fetch live weather data from OpenWeatherMap API"""
    try:
        if OPENWEATHER_API_KEY and OPENWEATHER_API_KEY != "your_api_key_here":
            url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'wind_speed': data['wind']['speed'],
                    'cloud_cover': data['clouds']['all'],
                    'description': data['weather'][0]['description'],
                    'pressure': data['main']['pressure']
                }
            else:
                st.error(f"Weather API Error: {response.status_code} - {response.text}")
        
        # Mock data fallback
        return {
            'temperature': 22.5 + np.random.normal(0, 2),
            'humidity': 65 + np.random.normal(0, 10),
            'wind_speed': 3.2 + np.random.normal(0, 1),
            'cloud_cover': 25 + np.random.normal(0, 15),
            'description': 'Partly cloudy',
            'pressure': 1013.25
        }
        
    except Exception as e:
        st.error(f"Weather data fetch failed: {e}")
        # Fallback to mock data
        return {
            'temperature': 20.0,
            'humidity': 60,
            'wind_speed': 5.0,
            'cloud_cover': 30,
            'description': 'Clear sky',
            'pressure': 1013.25
        }

def calculate_sun_position(lat, lon, dt):
    """Calculate sun elevation angle for solar irradiance estimation"""
    # Convert to radians
    lat_rad = math.radians(lat)
    
    # Day of year
    day_of_year = dt.timetuple().tm_yday
    
    # Solar declination angle
    declination = math.radians(23.45) * math.sin(math.radians(360 * (284 + day_of_year) / 365))
    
    # Hour angle
    hour = dt.hour + dt.minute / 60.0
    hour_angle = math.radians(15 * (hour - 12))
    
    # Solar elevation angle
    elevation = math.asin(
        math.sin(declination) * math.sin(lat_rad) +
        math.cos(declination) * math.cos(lat_rad) * math.cos(hour_angle)
    )
    
    return math.degrees(elevation)

def estimate_solar(lat, lon, cloud_cover, dt):
    """Estimate solar irradiance based on location, time, and weather"""
    # Calculate sun elevation
    sun_elevation = calculate_sun_position(lat, lon, dt)
    
    # Base solar irradiance (clear sky)
    if sun_elevation <= 0:
        base_irradiance = 0  # No sun
    else:
        # Approximate clear-sky irradiance
        base_irradiance = 1000 * math.sin(math.radians(sun_elevation))
    
    # Apply cloud cover reduction
    cloud_factor = 1 - (cloud_cover / 100) * 0.75  # 75% reduction at 100% cloud cover
    
    estimated_irradiance = base_irradiance * cloud_factor
    return max(0, estimated_irradiance)

def estimate_wind_power(wind_speed):
    """Estimate wind power generation from wind speed"""
    # Simple wind power curve approximation
    if wind_speed < 3:  # Cut-in speed
        return 0
    elif wind_speed > 25:  # Cut-out speed
        return 0
    elif wind_speed > 12:  # Rated speed
        return 15.0  # Rated power in kW
    else:
        # Cubic relationship below rated speed
        return 15.0 * ((wind_speed - 3) / (12 - 3)) ** 3

def ai_decision(solar_irradiance, wind_speed, load_demand, battery_soc, current_time=None):
    """AI decision model for microgrid operation mode - considers time of day and actual power availability"""
    # Calculate available power
    solar_power = solar_irradiance * 0.015  # Assuming 15kW solar array
    wind_power = estimate_wind_power(wind_speed)
    
    # Check if it's daytime and solar is actually available
    is_daytime = solar_irradiance > 50  # Minimum solar irradiance threshold
    
    # Get current hour if time is provided
    current_hour = current_time.hour if current_time else 12
    is_night = current_hour < 6 or current_hour > 18  # Rough night time check
    
    # Logic: Only recommend solar if there's actual solar power available
    if is_night or solar_irradiance < 10:
        # During night or very low solar irradiance, always recommend wind
        return "Wind Power Recommended", "💨", "Solar not available - Night time or low sunlight"
    
    elif solar_irradiance < 100:  # Dawn/dusk with minimal solar
        if wind_power > 1.0:  # If there's decent wind
            return "Wind Power Recommended", "💨", "Low solar conditions - Wind more reliable"
        else:
            return "Solar Power Recommended", "☀️", "Minimal solar available but better than wind"
    
    elif solar_power >= wind_power and solar_irradiance >= 200:
        # Good solar conditions during day
        return "Solar Power Recommended", "☀️", "Strong solar conditions detected"
    
    else:
        # Wind is better or solar is too weak
        return "Wind Power Recommended", "💨", "Wind conditions favorable over solar"

def display_location_selector():
    """Display location selection interface"""
    st.markdown("""
    <div class="section-title">📍 Location Settings</div>
    """, unsafe_allow_html=True)
    
    # Location method selection
    location_method = st.radio(
        "Choose location method:",
        ["🌐 Use Browser Location (Recommended)", "🔧 Manual Coordinates", "🌍 IP-based Location"],
        horizontal=False
    )
    
    if location_method == "🌐 Use Browser Location (Recommended)":
        st.markdown("Click the button below to allow the app to access your current location:")
        
        # Browser geolocation component
        location_data = components.html(
            get_browser_location_html(),
            height=200,
            scrolling=True
        )
        
        # Handle location data from browser
        if location_data and isinstance(location_data, dict):
            if 'lat' in location_data and 'lon' in location_data:
                st.session_state.user_location = location_data
                st.success(f"✅ Location updated: {location_data['lat']:.6f}, {location_data['lon']:.6f}")
                st.rerun()
        
        # Show current location status
        if 'user_location' in st.session_state and st.session_state.user_location:
            loc = st.session_state.user_location
            st.success(f"📍 Current location: {loc.get('lat', 0):.6f}, {loc.get('lon', 0):.6f}")
        
    elif location_method == "🔧 Manual Coordinates":
        st.markdown("Enter your coordinates manually:")
        
        col1, col2 = st.columns(2)
        with col1:
            manual_lat = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=st.session_state.get('manual_lat', 28.6139),
                step=0.000001,
                format="%.6f"
            )
        
        with col2:
            manual_lon = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=st.session_state.get('manual_lon', 77.2090),
                step=0.000001,
                format="%.6f"
            )
        
        if st.button("📍 Use These Coordinates"):
            st.session_state.user_location = {
                'lat': manual_lat,
                'lon': manual_lon,
                'manual': True
            }
            st.session_state.manual_lat = manual_lat
            st.session_state.manual_lon = manual_lon
            st.success(f"📍 Location set to: {manual_lat:.6f}, {manual_lon:.6f}")
            st.rerun()
    
    else:  # IP-based location
        st.markdown("Using IP-based location detection (less accurate)")
        if st.button("🔍 Detect IP Location"):
            if 'user_location' in st.session_state:
                del st.session_state.user_location
            st.rerun()

def display_real_time_dashboard():
    """Display the real-time dashboard with improved UI"""
    
    # Main Header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">⚡ Smart Energy Source Predictor</div>
        <div class="main-subtitle">Real-time AI system to predict optimal renewable energy source</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Location Selector in sidebar or expandable section
    with st.expander("📍 Location Settings", expanded=False):
        display_location_selector()
    
    # Control Panel
    st.markdown("""
    <div class="control-panel">
        <div class="section-title">🎛️ Control Panel</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        mode = st.radio("Select Mode:", ["🔄 Real-Time Mode", "🧪 Test Mode"], horizontal=True)
    
    with col2:
        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()
    
    with col3:
        auto_refresh = st.toggle("🔄 Auto Refresh (5s)")
        if auto_refresh:
            time.sleep(5)
            st.rerun()
    
    if mode == "🧪 Test Mode":
        display_test_model()
        return
    
    # Get location and weather data
    with st.spinner("Fetching location and weather data..."):
        location = get_location()
        weather = get_weather_data(location['lat'], location['lon'])
        
        # Get current time in location timezone
        try:
            if location.get('timezone') and location['timezone'] != 'UTC':
                tz = pytz.timezone(location['timezone'])
            else:
                tz = pytz.UTC
            current_time = datetime.now(tz)
        except:
            current_time = datetime.now()
        
        # Estimate solar irradiance
        solar_irradiance = estimate_solar(
            location['lat'], 
            location['lon'], 
            weather['cloud_cover'], 
            current_time
        )
    
    # Real-time clock display
    location_display = f"{location['city']}, {location['country']}"
    if location.get('state'):
        location_display = f"{location['city']}, {location['state']}, {location['country']}"
    
    st.markdown(f"""
    <div class="time-display">
        📍 {location_display} | 🕐 {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}
    </div>
    """, unsafe_allow_html=True)
    
    # Show location source
    location_source = "Browser Location" if 'user_location' in st.session_state and st.session_state.user_location else "IP Location"
    if 'user_location' in st.session_state and st.session_state.user_location and st.session_state.user_location.get('manual'):
        location_source = "Manual Coordinates"
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1rem; opacity: 0.7; font-size: 0.9rem;">
        Location source: {location_source} | Coordinates: {location['lat']:.6f}, {location['lon']:.6f}
    </div>
    """, unsafe_allow_html=True)
    
    # Current Conditions Section
    st.markdown("""
    <div class="section-title">🌤️ Current Conditions</div>
    """, unsafe_allow_html=True)
    
    # Calculate powers
    solar_power = solar_irradiance * 0.015
    wind_power = estimate_wind_power(weather['wind_speed'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">☀️ Solar Irradiance</div>
            <div class="metric-value">{solar_irradiance:.1f} W/m²</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💨 Wind Speed</div>
            <div class="metric-value">{weather['wind_speed']:.1f} m/s</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚡ Solar Power</div>
            <div class="metric-value">{solar_power:.1f} kW</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌪️ Wind Power</div>
            <div class="metric-value">{wind_power:.1f} kW</div>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Prediction Section
    load_demand = 8.5 + np.random.normal(0, 1.5)  # kW
    battery_soc = max(10, min(100, 65 + np.random.normal(0, 10)))    # %
    
    decision, icon, description = ai_decision(
        solar_irradiance, 
        weather['wind_speed'], 
        load_demand, 
        battery_soc,
        current_time
    )
    
    st.markdown("""
    <div class="section-title">🤖 AI Prediction</div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #87CEEB, #4682B4);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    ">
        <div style="
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: white;
        ">{icon} {decision.upper()}</div>
        <div style="
            font-size: 1.2rem;
            opacity: 0.95;
            color: white;
        ">{description}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional metrics
    st.markdown("""
    <div class="section-title">📊 System Status</div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🌡️ Temperature",
            value=f"{weather['temperature']:.1f}°C"
        )
    
    with col2:
        st.metric(
            label="💧 Humidity",
            value=f"{weather['humidity']:.0f}%"
        )
    
    with col3:
        st.metric(
            label="⚡ Load Demand",
            value=f"{load_demand:.1f} kW"
        )
    
    with col4:
        st.metric(
            label="🔋 Battery SoC",
            value=f"{battery_soc:.1f}%"
        )

def display_test_model():
    """Display the test model interface"""
    st.markdown("""
    <div class="section-title">🧪 Test AI Decision Model</div>
    """, unsafe_allow_html=True)
    
    st.write("Input custom parameters to test the AI decision-making algorithm:")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        solar_irradiance = st.slider(
            "☀️ Solar Irradiance (W/m²)",
            min_value=0.0,
            max_value=1200.0,
            value=600.0,
            step=50.0
        )
        
        wind_speed = st.slider(
            "💨 Wind Speed (m/s)",
            min_value=0.0,
            max_value=25.0,
            value=8.0,
            step=0.5
        )
    
    with col2:
        load_demand = st.slider(
            "⚡ Load Demand (kW)",
            min_value=1.0,
            max_value=20.0,
            value=10.0,
            step=0.5
        )
        
        battery_soc = st.slider(
            "🔋 Battery SoC (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        )
    
    # Calculate and display results
    decision, icon, description = ai_decision(
        solar_irradiance,
        wind_speed, 
        load_demand, 
        battery_soc
    )
    
    # Display decision with new UI - matching the image style
    st.markdown("""
    <div class="section-title">🤖 AI Prediction</div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #87CEEB, #4682B4);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    ">
        <div style="
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
            color: white;
        ">{icon} {decision.upper()}</div>
        <div style="
            font-size: 1.2rem;
            opacity: 0.95;
            color: white;
        ">{description}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show power calculations
    solar_power = solar_irradiance * 0.015  # 15kW array
    wind_power = estimate_wind_power(wind_speed)
    total_renewable = solar_power + wind_power
    
    st.markdown("""
    <div class="section-title">⚡ Power Analysis</div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Solar Power", f"{solar_power:.1f} kW")
    
    with col2:
        st.metric("Wind Power", f"{wind_power:.1f} kW")
    
    with col3:
        st.metric("Total Renewable", f"{total_renewable:.1f} kW")
    
    with col4:
        net_power = total_renewable - load_demand
        st.metric("Net Power", f"{net_power:.1f} kW")
    
    # Create power flow visualization
    if st.checkbox("Show Power Flow Chart"):
        labels = ['Solar', 'Wind', 'Load Demand']
        values = [solar_power, wind_power, load_demand]
        colors = ['#FF9800', '#2196F3', '#f44336']
        
        fig = px.bar(
            x=labels,
            y=values,
            color=labels,
            color_discrete_sequence=colors,
            title="Power Generation vs Load Demand"
        )
        fig.update_layout(
            showlegend=False, 
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    # Initialize session state
    if 'user_location' not in st.session_state:
        st.session_state.user_location = None
    
    # API Key setup notification
    if OPENWEATHER_API_KEY == "your_api_key_here":
        st.info("📝 **Note:** Using mock weather data. Add your OpenWeatherMap API key for real data.")
    else:
        st.success("🌐 **Connected to live weather data**")
    
    display_real_time_dashboard()

if __name__ == "__main__":
    main()
