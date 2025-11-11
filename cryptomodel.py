# Written by @TTurbold

import numpy as np
import warnings

warnings.filterwarnings('ignore')


class CryptoFourTwoRSVolatilityModel:
    """
    Enhanced 4/2 Rough and Smooth Model for Cryptocurrency Volatility
    """

    def __init__(self,
                 # Rough component parameters (adjusted for crypto)
                 H=0.15, k1=2.0, theta1=0.10, sigma1=1.8, rho1=-0.3,
                 # Smooth component parameters
                 k2=8.0, theta2=0.08, sigma2=3.0, rho2=-0.2,
                 # Crypto-specific extensions
                 mu=0.7, jump_intensity=0.15, jump_size_mean=0.08,
                 jump_size_std=0.15, liquidity_threshold=1e6):

        # Initialize all parameters...
        self.H = H
        self.k1 = k1
        self.theta1 = theta1
        self.sigma1 = sigma1
        self.rho1 = rho1
        self.k2 = k2
        self.theta2 = theta2
        self.sigma2 = sigma2
        self.rho2 = rho2
        self.mu = mu
        self.jump_intensity = jump_intensity
        self.jump_size_mean = jump_size_mean
        self.jump_size_std = jump_size_std
        self.liquidity_threshold = liquidity_threshold

        # Regime multipliers
        self.regime_vol_multipliers = {
            'bull_calm': 0.7, 'bull_volatile': 1.2,
            'bear_calm': 1.5, 'bear_volatile': 2.5, 'crisis': 4.0
        }

        # State variables
        self.V1 = theta1
        self.V2 = theta2
        self.current_regime = 'bull_calm'

    def total_variance(self):
        return self.mu * self.V1 + (1 - self.mu) * self.V2

    def total_volatility(self):
        return np.sqrt(self.total_variance())

    def crypto_jump_component(self, dt=1 / 365):
        jump_prob = 1 - np.exp(-self.jump_intensity * dt)
        if np.random.random() < jump_prob:
            jump_size = np.random.normal(self.jump_size_mean, self.jump_size_std)
            if np.random.random() < 0.05:
                jump_size *= np.random.lognormal(0.5, 0.8)
            return jump_size
        return 0.0

    def liquidity_adjusted_volatility(self, volume, base_volatility):
        if volume < self.liquidity_threshold:
            liquidity_multiplier = 1 + (self.liquidity_threshold - volume) / self.liquidity_threshold
            return base_volatility * min(liquidity_multiplier, 3.0)
        return base_volatility

    def detect_market_regime(self, price_data, volume_data, window=20):
        # Simplified regime detection
        if len(price_data) < 5:
            return 'bull_calm'

        returns = np.diff(price_data) / price_data[:-1]
        volatility = np.std(returns) * np.sqrt(365) if len(returns) > 0 else 0.15

        if volatility > 0.8:
            return 'crisis'
        elif volatility > 0.4:
            return 'bull_volatile' if np.mean(returns) > 0 else 'bear_volatile'
        else:
            return 'bull_calm' if np.mean(returns) > 0 else 'bear_calm'

    def simulate_crypto_volatility_path(self, days=90, dt=1 / 365, volumes=None, prices=None):
        """Add this method as shown above"""
        n_steps = int(days / dt)
        t = np.arange(0, days, dt)

        V1_path = np.zeros(n_steps)
        V2_path = np.zeros(n_steps)
        total_vol_path = np.zeros(n_steps)
        jump_component_path = np.zeros(n_steps)
        regime_path = []

        V1_path[0] = self.V1
        V2_path[0] = self.V2

        for i in range(1, n_steps):
            dW1 = np.random.normal(0, np.sqrt(dt))
            dW2 = np.random.normal(0, np.sqrt(dt))

            # Rough component
            V1_drift = self.k1 * (self.theta1 - V1_path[i - 1]) * dt
            V1_diffusion = self.sigma1 * np.sqrt(max(V1_path[i - 1], 1e-6)) * dW1
            V1_path[i] = max(V1_path[i - 1] + V1_drift + V1_diffusion, 1e-6)

            # Smooth component
            V2_drift = self.k2 * V2_path[i - 1] * (self.theta2 - V2_path[i - 1]) * dt
            V2_diffusion = self.sigma2 * (max(V2_path[i - 1], 1e-6) ** 1.5) * dW2
            V2_path[i] = max(V2_path[i - 1] + V2_drift + V2_diffusion, 1e-6)

            # Jumps and adjustments
            jump_component_path[i] = self.crypto_jump_component(dt)
            total_variance = self.mu * V1_path[i] + (1 - self.mu) * V2_path[i]
            base_volatility = np.sqrt(total_variance)

            if volumes is not None and i < len(volumes):
                current_volume = volumes[min(i, len(volumes) - 1)]
                base_volatility = self.liquidity_adjusted_volatility(current_volume, base_volatility)

            # Simple regime assignment for now
            current_vol = base_volatility
            if current_vol > 0.8:
                regime = 'crisis'
            elif current_vol > 0.4:
                regime = 'bull_volatile'
            else:
                regime = 'bull_calm'
            regime_path.append(regime)

            total_vol_path[i] = base_volatility

        return t, V1_path, V2_path, total_vol_path, jump_component_path, regime_path

    def forecast_crypto_volatility(self, horizon=30, n_simulations=1000):
        """Add this method as shown above"""
        forecasts = []
        regime_counts = {}

        for sim in range(n_simulations):
            t, V1, V2, vol_path, jumps, regimes = self.simulate_crypto_volatility_path(days=horizon)
            final_vol = vol_path[-1]
            forecasts.append(final_vol)

            final_regime = regimes[-1] if regimes else 'unknown'
            regime_counts[final_regime] = regime_counts.get(final_regime, 0) + 1

        forecasts = np.array(forecasts)

        return {
            'mean': np.mean(forecasts),
            'median': np.median(forecasts),
            'std': np.std(forecasts),
            'q5': np.percentile(forecasts, 5),
            'q95': np.percentile(forecasts, 95),
            'regime_distribution': regime_counts,
            'probability_high_vol': (forecasts > 0.5).mean(),
            'probability_crisis': (forecasts > 1.0).mean(),
            'distribution': forecasts
        }
