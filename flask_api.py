from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import numpy as np
import math

app = Flask(__name__)
CORS(app)

WAQI_TOKEN = "cff8fda4247e8f55fdfb854b2c236f216c4c07fd"

T_TIME_SECONDS = 31536000

def normalize(text):
    return text.lower().strip().replace(' ', '')

def get_wikidata_data(city_name):
    endpoint_url = "https://query.wikidata.org/sparql"
    
    query = f"""
    SELECT ?population ?area ?elevation WHERE {{
      ?city wdt:P31/wdt:P279* wd:Q515; rdfs:label "{city_name.title()}"@en.
      OPTIONAL {{ ?city wdt:P1082 ?population. }}
      OPTIONAL {{ ?city wdt:P2046 ?area. }}
      OPTIONAL {{ ?city wdt:P2044 ?elevation. }}
    }}
    LIMIT 1
    """

    try:
        response = requests.get(endpoint_url, params={'query': query, 'format': 'json'}, timeout=10)
        data = response.json()
        bindings = data['results']['bindings']
        
        if bindings:
            result = bindings[0]
            return {
                'population': int(result['population']['value']) if 'population' in result else None,
                'area_km2': float(result['area']['value']) if 'area' in result else None,
                'elevation': float(result['elevation']['value']) if 'elevation' in result else 10
            }
    except Exception as e:
        print(f"Wikidata error: {e}")
        
    return None

class TreePredictionAI:
    def __init__(self):
        self.species_database = [
            {
                'name': 'Alstonia scholaris (Pulai)',
                'rates': {'co': 622.1, 'o3': 7007.4, 'no2': 2782.3, 'so2': 1812.6, 'pm10': 24012.5, 'pm25': 2097.0},
                'maintenance': 'Medium'
            },
            {
                'name': 'Bombax ceiba (Randu Alas)',
                'rates': {'co': 712.0, 'o3': 5491.5, 'no2': 1535.7, 'so2': 1428.2, 'pm10': 19920.2, 'pm25': 1208.3},
                'maintenance': 'Medium'
            },
            {
                'name': 'Artocarpus heterophyllus (Nangka)',
                'rates': {'co': 123.8, 'o3': 1393.0, 'no2': 553.2, 'so2': 360.2, 'pm10': 4773.6, 'pm25': 417.0},
                'maintenance': 'Low'
            },
            {
                'name': 'Caryota urens (Palem)',
                'rates': {'co': 116.0, 'o3': 1300.5, 'no2': 516.5, 'so2': 336.2, 'pm10': 4457.3, 'pm25': 389.3},
                'maintenance': 'High'
            },
            {
                'name': 'Albizia lebbeck (Tekik)',
                'rates': {'co': 133.2, 'o3': 1027.8, 'no2': 287.4, 'so2': 267.4, 'pm10': 3728.4, 'pm25': 226.2},
                'maintenance': 'Low'
            },
            {
                'name': 'Azadirachta indica (Mimba)',
                'rates': {'co': 75.4, 'o3': 848.9, 'no2': 337.1, 'so2': 219.6, 'pm10': 2909.1, 'pm25': 254.0},
                'maintenance': 'Low'
            },
            {
                'name': 'Ailanthus excelsa',
                'rates': {'co': 83.0, 'o3': 640.3, 'no2': 179.0, 'so2': 166.5, 'pm10': 2322.5, 'pm25': 140.9},
                'maintenance': 'Medium'
            },
            {
                'name': 'Bauhinia racemosa',
                'rates': {'co': 12.4, 'o3': 95.9, 'no2': 26.8, 'so2': 25.0, 'pm10': 348.0, 'pm25': 21.1},
                'maintenance': 'Low'
            },
            {
                'name': 'Callistemon viminalis',
                'rates': {'co': 1.1, 'o3': 12.9, 'no2': 5.1, 'so2': 3.3, 'pm10': 44.1, 'pm25': 3.9},
                'maintenance': 'Medium'
            }
        ]
    
    def calculate_pollution_load(self, concentration_ug_m3, area_km2, wind_speed):
        C_ug_m3 = concentration_ug_m3
        
        area_m2 = area_km2 * 1_000_000
        
        v_ms = wind_speed
        
        Q = v_ms * area_m2
        
        E_ug_year = C_ug_m3 * Q * T_TIME_SECONDS
        
        E_g_year = E_ug_year / 1_000_000
        
        return E_g_year, wind_speed, Q

    def get_recommendations_sorted(self, pollutants_data, area_km2, wind_speed):
        results = []
        
        dominant_type = max(pollutants_data, key=pollutants_data.get) if pollutants_data else 'pm25'
        dominant_value = pollutants_data.get(dominant_type, 0)
        
        total_load_grams, U_used, Q_calculated = self.calculate_pollution_load(dominant_value, area_km2, wind_speed)
        
        STREET_CANYON_FACTOR = 0.4
        
        for species in self.species_database:
            absorption_rate = species['rates'].get(dominant_type, 0.1) 
            
            if absorption_rate <= 0:
                absorption_rate = 0.1
                
            raw_trees_needed = total_load_grams / absorption_rate
            adjusted_trees_needed = math.ceil(raw_trees_needed * STREET_CANYON_FACTOR)
            
            results.append({
                'species': species['name'],
                'trees_needed': int(adjusted_trees_needed),
                'absorption_rate': absorption_rate,
                'pollutant_target': dominant_type.upper(),
                'maintenance': species['maintenance'],
                'all_rates': species['rates']
            })
        
        sorted_results = sorted(results, key=lambda x: x['absorption_rate'], reverse=True)
        
        return sorted_results, total_load_grams, dominant_type, U_used

tree_ai = TreePredictionAI()

@app.route('/api/city-info/<city>', methods=['GET'])
def get_city_info(city):
    print(f"Mencari data {city} di Wikidata...")
    wiki_data = get_wikidata_data(city)
    
    if wiki_data and wiki_data['area_km2']:
        city_data = {
            'area_km2': wiki_data['area_km2']
        }
        return jsonify({'success': True, 'source': 'wikidata', 'data': city_data})

    return jsonify({
        'success': True,
        'source': 'default_fallback',
        'data': {
            'area_km2': 500
        }
    })

@app.route('/api/predict-trees', methods=['POST'])
def predict_trees():
    try:
        data = request.json
        
        pollutants = {
            'pm25': float(data.get('pm25', 0)),
            'pm10': float(data.get('pm10', 0)),
            'no2': float(data.get('no2', 0)),
            'so2': float(data.get('so2', 0)),
            'co': float(data.get('co', 0)),
            'o3': float(data.get('o3', 0))
        }
        
        area_km2 = float(data.get('area_km2', 500))
        wind_speed = float(data.get('wind_speed', 3.0))
        
        recommendations, total_load_grams, dominant_pollutant, U_wind = tree_ai.get_recommendations_sorted(
            pollutants, area_km2, wind_speed
        )
        
        best_option = recommendations[0]
        
        return jsonify({
            'success': True,
            'data': {
                'analysis_summary': {
                    'dominant_pollutant': dominant_pollutant.upper(),
                    'total_pollution_load_per_year': f"{total_load_grams:,.2f} grams",
                    'area_assumed': f"{area_km2} km2",
                    'wind_speed_used': f"{U_wind:.2f} m/s"
                },
                'best_recommendation': {
                    'species': best_option['species'],
                    'trees_needed': best_option['trees_needed'],
                    'note': "Most effective species"
                },
                'all_species_ranked': recommendations 
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/air-quality/<city>', methods=['GET'])
def get_air_quality(city):
    try:
        url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"
        response = requests.get(url, timeout=10)
        data = response.json()
        if data['status'] == 'ok':
            return jsonify({'success': True, 'data': data['data']})
        return jsonify({'success': False, 'error': 'City not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-city/<city>', methods=['GET'])
def analyze_city(city):
    try:
        aqi_res = get_air_quality(city).get_json()
        if not aqi_res['success']: 
            return jsonify(aqi_res)
        waqi_data = aqi_res['data']
        
        iaqi = waqi_data.get('iaqi', {})
        pollutants = {
            'pm25': iaqi.get('pm25', {}).get('v', 0),
            'pm10': iaqi.get('pm10', {}).get('v', 0),
            'no2': iaqi.get('no2', {}).get('v', 0),
            'so2': iaqi.get('so2', {}).get('v', 0),
            'co': iaqi.get('co', {}).get('v', 0),
            'o3': iaqi.get('o3', {}).get('v', 0)
        }
        
        wind_speed_data = iaqi.get('w', {}).get('v', None)
        if wind_speed_data is not None:
            wind_speed = float(wind_speed_data)
        else:
            wind_speed = 3.0
        
        city_info_res = get_city_info(city).get_json()
        if not city_info_res['success']:
            return jsonify(city_info_res)
        
        city_data = city_info_res['data']
        area_km2 = city_data['area_km2']
        
        recommendations, total_load, dom_pol, U_w = tree_ai.get_recommendations_sorted(
            pollutants, area_km2, wind_speed
        )
        
        return jsonify({
            'success': True,
            'city': waqi_data.get('city', {}).get('name', city),
            'city_info': {
                'area_km2': area_km2,
                'source': city_info_res['source']
            },
            'air_quality': {
                'aqi': waqi_data.get('aqi'),
                'dominant': dom_pol.upper(),
                'pollutants_raw': pollutants,
                'wind_speed_ms': wind_speed
            },
            'physics_calculation': {
                'formula': 'E_total = C * Area * u * T',
                'wind_speed_ms': U_w,
                'wind_speed_source': 'WAQI API' if wind_speed_data is not None else 'Default (3.0 m/s)',
                'total_load_grams_year': total_load
            },
            'recommendations_ranked': recommendations
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Physics-Based Tree Calculation API with Wikidata Integration...")
    app.run(debug=True, host='0.0.0.0', port=5000)
