# Written by @TTurbold

"""
Crypto Volatility Predictor Module
Integrates with existing volatility analysis and adds 4/2RS predictive modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Import your existing volatility analysis
from volatility import df_getter  # Replace with your actual module name

# Import the crypto-adapted 4/2RS model
from cryptomodel import CryptoFourTwoRSVolatilityModel


class CryptoVolatilityPredictor:
    """
    Main class that integrates your existing volatility analysis with 4/2RS predictive modeling
    """

    def __init__(self):
        self.metrics_df = None
        self.new_data_dict = None
        self.calibrated_models = {}
        self.volatility_forecasts = {}

    def load_existing_data(self):
        """
        Load your existing volatility data and calculations
        """
        print("Loading existing volatility data...")
        self.metrics_df, self.new_data_dict = df_getter()
        print(f"Loaded data for {len(self.new_data_dict)} cryptocurrencies")
        return self

    def calibrate_models_from_data(self):
        """
        Calibrate 4/2RS models using your existing 7-day and 14-day volatility calculations
        """
        if self.new_data_dict is None:
            self.load_existing_data()

        print("\nCalibrating 4/2RS models from empirical volatility data...")

        for coin_symbol, df in self.new_data_dict.items():
            if df is None or df.empty:
                continue

            print(f"Calibrating {coin_symbol}...")

            try:
                # Extract your calculated volatilities
                vol_7day = df['vol_7d'].dropna() if 'vol_7d' in df.columns else None
                vol_14day = df['vol_14d'].dropna() if 'vol_14d' in df.columns else None

                if vol_7day is None or vol_14day is None or len(vol_7day) < 10:
                    print(f"  Skipping {coin_symbol}: insufficient volatility data")
                    continue

                # Create and calibrate model
                model = CryptoFourTwoRSVolatilityModel()
                calibrated_params = self._calibrate_single_coin(
                    coin_symbol, vol_7day, vol_14day, df['close']
                )

                # Update model with calibrated parameters
                for param, value in calibrated_params.items():
                    setattr(model, param, value)

                self.calibrated_models[coin_symbol] = model
                print(f"  ✓ Calibrated: H={model.H:.3f}, θ={model.theta1:.4f}")

            except Exception as e:
                print(f"  ✗ Failed to calibrate {coin_symbol}: {e}")
                continue

        print(f"\nSuccessfully calibrated {len(self.calibrated_models)} models")
        return self

    def _calibrate_single_coin(self, coin_symbol: str, vol_7day: pd.Series,
                               vol_14day: pd.Series, prices: pd.Series) -> dict:
        """
        Calibrate parameters for a single cryptocurrency using your volatility data
        """
        # Convert volatilities to variances
        var_7day = vol_7day ** 2
        var_14day = vol_14day ** 2

        # 1. Estimate long-term variance from your data
        theta1 = self._estimate_long_term_variance(var_7day, var_14day)

        # 2. Estimate roughness from multi-scale volatility relationship
        H = self._estimate_hurst_from_volatility(vol_7day, vol_14day)

        # 3. Estimate vol-of-vol from your volatility changes
        sigma1 = self._estimate_vol_of_vol(vol_7day, vol_14day)

        # 4. Estimate mean reversion from volatility persistence
        k1 = self._estimate_mean_reversion(var_7day)

        # 5. Estimate jump parameters from price data
        jump_intensity, jump_size_mean = self._estimate_jump_parameters(prices)

        # 6. Coin-specific adjustments
        mu, sigma2 = self._coin_specific_adjustments(coin_symbol, vol_7day, vol_14day)

        calibrated_params = {
            'H': H,
            'k1': k1,
            'theta1': theta1,
            'sigma1': sigma1,
            'k2': k1 * 4.0,  # Smooth component has slower mean reversion
            'theta2': theta1 * 0.8,
            'sigma2': sigma2,
            'mu': mu,
            'jump_intensity': jump_intensity,
            'jump_size_mean': jump_size_mean,
            'rho1': -0.3,
            'rho2': -0.2,
            'V1': theta1,  # Initialize state variables
            'V2': theta1 * 0.8
        }

        return calibrated_params

    def _estimate_long_term_variance(self, var_7day: pd.Series, var_14day: pd.Series) -> float:
        """Estimate long-term variance level from your multi-scale data"""
        combined_vars = pd.concat([var_7day, var_14day])

        # Robust estimation (trim outliers common in crypto)
        q_low = combined_vars.quantile(0.10)
        q_high = combined_vars.quantile(0.90)
        trimmed_vars = combined_vars[(combined_vars >= q_low) & (combined_vars <= q_high)]

        long_term_var = trimmed_vars.median()
        return max(min(long_term_var, 0.5), 0.01)  # Reasonable bounds

    def _estimate_hurst_from_volatility(self, vol_7day: pd.Series, vol_14day: pd.Series) -> float:
        """Estimate roughness parameter H from your 7-day vs 14-day volatility"""
        # Align time series
        common_index = vol_7day.index.intersection(vol_14day.index)
        if len(common_index) < 10:
            return 0.15  # Default

        vol_7day_aligned = vol_7day.loc[common_index]
        vol_14day_aligned = vol_14day.loc[common_index]

        # Calculate volatility ratio variability
        vol_ratio = (vol_7day_aligned / vol_14day_aligned).dropna()
        ratio_variability = vol_ratio.std() / vol_ratio.mean()

        # Map to H parameter (higher variability → rougher → lower H)
        if ratio_variability > 1.0:
            H = 0.10  # Very rough
        elif ratio_variability > 0.7:
            H = 0.15  # Rough
        elif ratio_variability > 0.4:
            H = 0.20  # Moderately rough
        else:
            H = 0.25  # Less rough

        return H

    def _estimate_vol_of_vol(self, vol_7day: pd.Series, vol_14day: pd.Series) -> float:
        """Estimate volatility-of-volatility from your rolling volatility series"""
        vol_changes_7day = vol_7day.pct_change().dropna().abs()
        vol_changes_14day = vol_14day.pct_change().dropna().abs()

        all_vol_changes = pd.concat([vol_changes_7day, vol_changes_14day])

        # Remove extreme outliers
        q99 = all_vol_changes.quantile(0.99)
        filtered_changes = all_vol_changes[all_vol_changes <= q99]

        vol_of_vol = filtered_changes.std()
        sigma1 = vol_of_vol * 2.5  # Empirical scaling

        return max(min(sigma1, 4.0), 0.5)

    def _estimate_mean_reversion(self, var_7day: pd.Series) -> float:
        """Estimate mean reversion speed from volatility autocorrelation"""
        if len(var_7day) < 20:
            return 2.0

        autocorr = var_7day.autocorr(lag=1)
        if pd.isna(autocorr) or autocorr <= 0:
            return 3.0

        k1 = 1.0 / (autocorr + 0.1)  # Empirical relationship
        return max(min(k1, 5.0), 0.5)

    def _estimate_jump_parameters(self, prices: pd.Series):
        """Estimate jump parameters from price series"""
        if prices is None or len(prices) < 30:
            return 0.15, 0.08

        returns = np.log(prices / prices.shift(1)).dropna()
        returns_std = returns.std()
        jump_threshold = 3 * returns_std

        large_moves = returns.abs() > jump_threshold
        jump_intensity = large_moves.mean() * 365

        jump_sizes = returns[large_moves].abs()
        jump_size_mean = jump_sizes.mean() if len(jump_sizes) > 0 else 0.08

        return min(jump_intensity, 1.0), min(jump_size_mean, 0.25)

    def _coin_specific_adjustments(self, coin_symbol: str, vol_7day: pd.Series,
                                   vol_14day: pd.Series):
        """Make coin-specific adjustments based on volatility characteristics"""
        vol_stability = vol_7day.std() / vol_7day.mean()

        # Classify by volatility profile
        if vol_stability > 0.8:
            mu, sigma2 = 0.8, 4.0  # Very unstable coins
        elif vol_stability > 0.5:
            mu, sigma2 = 0.7, 3.0  # Moderately unstable
        else:
            mu, sigma2 = 0.6, 2.0  # Relatively stable

        # Additional adjustments for known coin categories
        coin_upper = coin_symbol.upper()
        if any(x in coin_upper for x in ['BTC', 'ETH', 'ADA', 'DOT', 'SOL']):
            mu = 0.65  # Major coins less rough
        elif any(x in coin_upper for x in ['DOGE', 'SHIB', 'PEPE', 'FLOKI']):
            mu, sigma2 = 0.85, 5.0  # Memecoins very rough

        return mu, sigma2

    def generate_volatility_forecasts(self, horizon_days: int = 30,
                                      n_simulations: int = 1000) -> dict:
        """
        Generate volatility forecasts for all calibrated models
        """
        return_dict = {}
        print(f"\nGenerating {horizon_days}-day volatility forecasts...")

        self.volatility_forecasts = {}

        for coin_symbol, model in self.calibrated_models.items():
            try:
                forecast = model.forecast_crypto_volatility(
                    horizon=horizon_days,
                    n_simulations=n_simulations
                )
                self.volatility_forecasts[coin_symbol] = forecast
                return_dict[coin_symbol] = forecast['mean']
                print(f"  ✓ {coin_symbol}: {forecast['mean']:.1%} mean forecast")

            except Exception as e:
                print(f"  ✗ Failed to forecast {coin_symbol}: {e}")
                continue

        return return_dict

    def get_forecast_summary(self) -> pd.DataFrame:
        """
        Create a summary table of all volatility forecasts
        """
        summary_data = []

        for coin_symbol, forecast in self.volatility_forecasts.items():
            clean_symbol = coin_symbol.replace('/USDT', '')

            summary_data.append({
                'Coin': clean_symbol,
                'Current_Vol': f"{np.sqrt(self.calibrated_models[coin_symbol].theta1):.1%}",
                'Forecast_Mean': f"{forecast['mean']:.1%}",
                'Forecast_5th': f"{forecast['q5']:.1%}",
                'Forecast_95th': f"{forecast['q95']:.1%}",
                'High_Vol_Prob': f"{forecast['probability_high_vol']:.1%}",
                'Crisis_Prob': f"{forecast['probability_crisis']:.1%}",
                'Roughness_H': f"{self.calibrated_models[coin_symbol].H:.3f}",
                'Dominant_Regime': max(forecast['regime_distribution'].items(),
                                       key=lambda x: x[1])[0] if forecast['regime_distribution'] else 'N/A'
            })

        summary_df = pd.DataFrame(summary_data)
        return summary_df.sort_values('Forecast_Mean', ascending=False)


def main():
    """
    Main execution function that integrates everything
    """
    print("=== Crypto Volatility Prediction System ===")
    print("Integrating existing volatility analysis with 4/2RS predictive modeling...")

    # Initialize predictor
    predictor = CryptoVolatilityPredictor()

    # Step 1: Load your existing data
    predictor.load_existing_data()

    # Step 2: Calibrate models using your 7-day and 14-day volatilities
    predictor.calibrate_models_from_data()

    # Step 3: Generate 30-day volatility forecasts
    forecasts = predictor.generate_volatility_forecasts(horizon_days=1, n_simulations=1000)

    # Step 4: Create summary and diagnostics
    summary_df = predictor.get_forecast_summary()

    print("\n" + "=" * 80)
    print("VOLATILITY FORECAST SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Step 5: Plot diagnostics
    print("\nGenerating diagnostic plots...")
    predictor.plot_calibration_diagnostics(top_n=15)

    # Step 6: Save results
    summary_df.to_csv('crypto_volatility_forecasts.csv', index=False)
    print(f"\nResults saved to 'crypto_volatility_forecasts.csv'")

    # Additional analysis: Show top risky coins
    print("\n" + "=" * 80)
    print("TOP 10 RISKIEST COINS (Highest Crisis Probability)")
    print("=" * 80)

    risky_coins = summary_df.nlargest(10, 'Crisis_Prob')[['Coin', 'Forecast_Mean', 'Crisis_Prob']]
    print(risky_coins.to_string(index=False))

    return predictor, summary_df


def analyze_specific_coins(predictor: CryptoVolatilityPredictor,
                           coin_list: List[str]):
    """
    Analyze specific coins in detail
    """
    print(f"\nDetailed Analysis for Selected Coins: {coin_list}")

    for coin_symbol in coin_list:
        full_symbol = f"{coin_symbol}/USDT"

        if full_symbol not in predictor.calibrated_models:
            print(f"  No model found for {coin_symbol}")
            continue

        model = predictor.calibrated_models[full_symbol]
        forecast = predictor.volatility_forecasts.get(full_symbol, {})

        print(f"\n--- {coin_symbol} ---")
        print(f"  Current Volatility: {model.total_volatility():.1%}")
        print(f"  Roughness (H): {model.H:.3f}")
        print(f"  Long-term Vol: {np.sqrt(model.theta1):.1%}")
        print(f"  30-Day Forecast: {forecast.get('mean', 0):.1%}")
        print(f"  90% Confidence: [{forecast.get('q5', 0):.1%}, {forecast.get('q95', 0):.1%}]")
        print(f"  Crisis Probability: {forecast.get('probability_crisis', 0):.1%}")


if __name__ == "__main__":
    # Run complete analysis
    predictor = CryptoVolatilityPredictor()
    predictor.load_existing_data()
    predictor.calibrate_models_from_data()
    res = predictor.generate_volatility_forecasts(horizon_days=1, n_simulations=1000)
    print(res)
    # Analyze specific major coins
    # major_coins = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'DOGE']
    # analyze_specific_coins(predictor, major_coins)
