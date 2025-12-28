import requests
import pandas as pd
from datetime import datetime, timedelta
import time

class CryptoDataFetcher:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
        
    def get_current_price(self, coin_id='bitcoin'):
        """Get current price for a coin"""
        url = f"{self.base_url}/simple/price"
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_24hr_vol': 'true',
            'include_market_cap': 'true'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'coin': coin_id,
                'price': data[coin_id]['usd'],
                'change_24h': data[coin_id].get('usd_24h_change', 0),
                'volume_24h': data[coin_id].get('usd_24h_vol', 0),
                'market_cap': data[coin_id].get('usd_market_cap', 0),
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None
    
    def get_historical_data(self, coin_id='bitcoin', days=90):
        """Get historical price data"""
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['date'] = df['timestamp'].dt.date
            
            # Add volumes if available
            if 'total_volumes' in data:
                volumes = data['total_volumes']
                df['volume'] = [v[1] for v in volumes]
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def get_multiple_coins(self, coin_ids=['bitcoin', 'ethereum', 'cardano']):
        """Get current prices for multiple coins"""
        results = []
        for coin_id in coin_ids:
            price_data = self.get_current_price(coin_id)
            if price_data:
                results.append(price_data)
            time.sleep(1)  # Be polite to API
        
        return pd.DataFrame(results)

# Test
if __name__ == "__main__":
    fetcher = CryptoDataFetcher()
    
    # Test current price
    print("Fetching Bitcoin price...")
    btc = fetcher.get_current_price('bitcoin')
    print(btc)
    
    # Test historical data
    print("\nFetching historical data...")
    hist = fetcher.get_historical_data('bitcoin', days=30)
    if hist is not None:
        print(hist.head())
        print(f"\nTotal records: {len(hist)}")
    else:
        print("Failed to fetch historical data")