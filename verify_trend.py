import numpy as np
import pandas as pd
import sys
import os

# Add current directory to path to import take3
sys.path.append(os.getcwd())

from take3 import AdvancedStatisticalAnalyzer, Config

# Mock Config if needed (though we import it)
Config.MIN_SAMPLES_FOR_STATS = 50

def generate_trending_data(length=1000):
    """Generate stochastic trending data (Biased Random Walk)"""
    # Returns with positive mean = Trend
    returns = np.random.normal(0.05, 0.1, length) 
    price = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({'close': price})

def generate_mean_reverting_data(length=1000):
    """Generate mean reverting (ranging) data"""
    t = np.linspace(0, 50, length)
    # Sine wave is perfectly mean reverting
    oscillations = np.sin(t) * 5
    noise = np.random.normal(0, 1, length)
    price = 100 + oscillations + noise
    return pd.DataFrame({'close': price})

def generate_random_walk_data(length=1000):
    """Generate random walk data"""
    # Cumsum of random normal returns = Random Walk
    returns = np.random.normal(0, 0.01, length)
    price = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({'close': price})

def test_regime_detection():
    analyzer = AdvancedStatisticalAnalyzer()
    
    print("=== TESTING TREND DETECTION LOGIC ===")
    print(f"Hurst Thresholds: Trending > {Config.HURST_TRENDING_THRESHOLD}, MeanRev < {Config.HURST_MEANREVERTING_THRESHOLD}")
    print("-" * 50)

    # 1. Test Trending
    trend_data = generate_trending_data()
    trend_result = analyzer.calculate_market_regime(trend_data)
    print(f"\n[TRENDING DATA TEST]")
    print(f"Detected Regime: {trend_result['regime'].upper()}")
    print(f"Hurst Exponent:  {trend_result['hurst']:.4f}")
    print(f"Confidence:      {trend_result['confidence']:.2%}")
    
    # 2. Test Mean Reverting
    range_data = generate_mean_reverting_data()
    range_result = analyzer.calculate_market_regime(range_data)
    print(f"\n[MEAN REVERTING DATA TEST]")
    print(f"Detected Regime: {range_result['regime'].upper()}")
    print(f"Hurst Exponent:  {range_result['hurst']:.4f}")
    print(f"Confidence:      {range_result['confidence']:.2%}")

    # 3. Test Random Walk
    rw_data = generate_random_walk_data()
    rw_result = analyzer.calculate_market_regime(rw_data)
    print(f"\n[RANDOM WALK DATA TEST]")
    print(f"Detected Regime: {rw_result['regime'].upper()}")
    print(f"Hurst Exponent:  {rw_result['hurst']:.4f}")
    print(f"Confidence:      {rw_result['confidence']:.2%}")

if __name__ == "__main__":
    test_regime_detection()
