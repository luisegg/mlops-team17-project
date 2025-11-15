
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataDriftGenerator:
    """
    Robust Data Drift Generator

    Features:
    - Safe sampling (reset_index) to avoid pandas reindexing errors
    - Gradual, sudden, seasonal, multivariate, categorical drift generators
    - Add realistic stochastic noise and amplitude variation
    - combine_drifts method to synthesize datasets mixing multiple drift types
    - Input validation and optional random_state for reproducibility
    """

    def __init__(self, original_data: pd.DataFrame, numeric_cols: Optional[List[str]] = None,
                 categorical_cols: Optional[List[str]] = None, random_state: Optional[int] = None):
        self.original_data = original_data.copy()
        self.random_state = random_state

        # Allow user override of columns; otherwise keep defaults if present
        default_numeric = [
            'Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)',
            'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM'
        ]
        default_categorical = ['Day_of_week', 'WeekStatus', 'Load_Type']

        if numeric_cols is None:
            # keep only those default columns that exist in the dataframe
            self.numeric_cols = [c for c in default_numeric if c in self.original_data.columns]
        else:
            self.numeric_cols = [c for c in numeric_cols if c in self.original_data.columns]

        if categorical_cols is None:
            self.categorical_cols = [c for c in default_categorical if c in self.original_data.columns]
        else:
            self.categorical_cols = [c for c in categorical_cols if c in self.original_data.columns]

        # compute statistics
        self._calculate_statistics()

    def _calculate_statistics(self):
        """Compute per-feature summary stats and correlation matrix."""
        self.numeric_stats = {}
        for col in self.numeric_cols:
            series = self.original_data[col].dropna()
            if len(series) == 0:
                continue
            self.numeric_stats[col] = {
                'mean': float(series.mean()),
                'std': float(series.std(ddof=0)),
                'min': float(series.min()),
                'max': float(series.max())
            }

        # correlation matrix for the numeric columns actually present
        if len(self.numeric_cols) >= 2:
            numeric_data = self.original_data[self.numeric_cols].dropna()
            if not numeric_data.empty:
                self.correlation_matrix = numeric_data.corr()
            else:
                self.correlation_matrix = None
        else:
            self.correlation_matrix = None

        # categorical distributions
        self.categorical_stats = {}
        for col in self.categorical_cols:
            if col in self.original_data.columns:
                self.categorical_stats[col] = self.original_data[col].value_counts(normalize=True).to_dict()

    def _sample_reset(self, n: int) -> pd.DataFrame:
        """Sample from original data with replacement and reset index (safe)."""
        return self.original_data.sample(n=n, replace=True, random_state=self.random_state).reset_index(drop=True).copy()

    def _clip_series(self, arr: np.ndarray, col: str) -> np.ndarray:
        stats = self.numeric_stats.get(col, None)
        if stats is None:
            return arr
        lower = stats['min']
        upper = stats['max'] * 1.2
        return np.clip(arr, lower, upper)

    def generate_gradual_drift(self, n_samples: int = 2000, drift_intensity: float = 0.3,
                               drift_features: Optional[List[str]] = None, add_noise: bool = True) -> pd.DataFrame:
        """Gradually change the mean and variance over time.

        - drift_intensity in [0,1]
        - add_noise: whether to add small stochastic noise to make drift realistic
        """
        if drift_features is None:
            drift_features = self.numeric_cols

        drifted = self._sample_reset(n_samples)
        t = np.linspace(0, 1, n_samples)

        for col in drift_features:
            if col not in self.numeric_stats or col not in drifted.columns:
                continue

            std = self.numeric_stats[col]['std']
            mean = self.numeric_stats[col]['mean']

            # mean drifts linearly over time (fraction of std)
            mean_shift = std * drift_intensity * t
            arr = drifted[col].values.astype(float) + mean_shift

            # gradually increase variance
            variance_factor = 1 + (drift_intensity * 0.5 * t)
            arr = (arr - mean) * variance_factor + mean

            # optional small noise
            if add_noise:
                noise = np.random.RandomState(self.random_state).normal(0, std * 0.02, size=n_samples)
                arr += noise

            arr = self._clip_series(arr, col)
            drifted[col] = arr

        return drifted

    def generate_sudden_drift(self, n_samples: int = 2000, changepoint: float = 0.5,
                              drift_intensity: float = 0.5) -> pd.DataFrame:
        """Introduce a sudden shift at changepoint (fraction 0-1).
        Implementation avoids pandas alignment by operating on numpy arrays.
        """
        drifted = self._sample_reset(n_samples)
        change_idx = int(n_samples * changepoint)

        for col in self.numeric_cols:
            if col not in self.numeric_stats or col not in drifted.columns:
                continue

            std = self.numeric_stats[col]['std']
            shift = std * drift_intensity

            # apply shift to post-change segment
            arr = drifted[col].values.astype(float)
            arr[change_idx:] += shift

            # change variance for post-change
            post = arr[change_idx:]
            if len(post) > 1:
                mean_val = post.mean()
                new_post = (post - mean_val) * (1 + drift_intensity * 0.3) + mean_val
                arr[change_idx:] = new_post

            arr = self._clip_series(arr, col)
            drifted[col] = arr

        return drifted

    def generate_seasonal_drift(self, n_samples: int = 2000, amplitude: float = 0.3,
                                drift_features: Optional[List[str]] = None,
                                noise_scale: float = 0.05, periods: Dict[str, int] = None) -> pd.DataFrame:
        """Apply multiple seasonal components (daily, weekly, yearly by default).

        - noise_scale: fraction of std to add as stochastic noise to seasonal effect
        - periods: a dict to override default readings per cycle, e.g. {'daily':96}
        """
        if drift_features is None:
            drift_features = self.numeric_cols

        if periods is None:
            periods = {'daily': 96, 'weekly': 672, 'yearly': 35040}

        drifted = self._sample_reset(n_samples)
        n = len(drifted)
        t = np.arange(n)

        daily = np.sin(2 * np.pi * t / periods['daily']) if periods.get('daily') else 0
        weekly = np.sin(2 * np.pi * t / periods['weekly']) if periods.get('weekly') else 0
        yearly = np.sin(2 * np.pi * t / periods['yearly']) if periods.get('yearly') else 0

        combined = 0.5 * daily + 0.3 * weekly + 0.2 * yearly

        rng = np.random.RandomState(self.random_state)

        for col in drift_features:
            if col not in self.numeric_stats or col not in drifted.columns:
                continue

            std = self.numeric_stats[col]['std']
            seasonal_effect = std * amplitude * combined

            # add multiplicative variance to seasonal to make it less deterministic
            modulation = 1 + rng.uniform(-0.15, 0.15, size=n)
            arr = drifted[col].values.astype(float) + seasonal_effect * modulation

            # small gaussian noise
            arr += rng.normal(0, std * noise_scale, size=n)

            arr = self._clip_series(arr, col)
            drifted[col] = arr

        return drifted

    def generate_multivariate_drift(self, n_samples: int = 1000, drift_intensity: float = 0.4) -> pd.DataFrame:
        """Generate drift that respects feature correlations by sampling and adding correlated noise."""
        # sample base rows
        available_cols = [c for c in self.numeric_cols if c in self.original_data.columns]
        base_samples = self.original_data[available_cols].sample(n=n_samples, replace=True, random_state=self.random_state)

        # compute covariance on base_samples
        cov_matrix = base_samples.cov().values
        mean_vector = base_samples.mean().values

        rng = np.random.RandomState(self.random_state)
        drift_mean = mean_vector * drift_intensity
        noise = rng.multivariate_normal(drift_mean, cov_matrix * drift_intensity, size=n_samples)

        drifted_numeric = base_samples.values.astype(float) + noise
        drifted = pd.DataFrame(drifted_numeric, columns=available_cols)

        # attach categorical columns by sampling
        for col in self.categorical_cols:
            if col in self.original_data.columns:
                drifted[col] = self.original_data[col].sample(n=n_samples, replace=True, random_state=self.random_state).values

        return drifted

    def generate_categorical_drift(self, n_samples: int = 1000, shift_probabilities: Dict[str, float] = None) -> pd.DataFrame:
        drifted = self._sample_reset(n_samples)
        if shift_probabilities is None:
            shift_probabilities = {col: 0.3 for col in self.categorical_cols}

        rng = np.random.RandomState(self.random_state)

        for col, shift in shift_probabilities.items():
            if col not in self.categorical_stats:
                continue

            original_probs = self.categorical_stats[col]
            categories = list(original_probs.keys())
            probs = np.array(list(original_probs.values()))

            # random perturbation around original distribution
            perturb = rng.uniform(-shift, shift, size=len(categories))
            new_probs = probs + perturb
            new_probs = np.maximum(new_probs, 1e-3)
            new_probs = new_probs / new_probs.sum()

            drifted[col] = rng.choice(categories, size=n_samples, p=new_probs)

        return drifted

    def augment_with_noise(self, data: pd.DataFrame, noise_level: float = 0.05) -> pd.DataFrame:
        augmented = data.copy()
        rng = np.random.RandomState(self.random_state)
        for col in self.numeric_cols:
            if col in augmented.columns and col in self.numeric_stats:
                std = self.numeric_stats[col]['std']
                noise = rng.normal(0, std * noise_level, size=len(augmented))
                augmented[col] = augmented[col].values.astype(float) + noise
                augmented[col] = augmented[col].clip(lower=self.numeric_stats[col]['min'], upper=self.numeric_stats[col]['max'])
        return augmented

    def augment_with_mixup(self, data: pd.DataFrame, alpha: float = 0.2, n_augmented: int = None) -> pd.DataFrame:
        if n_augmented is None:
            n_augmented = len(data)
        rng = np.random.RandomState(self.random_state)
        augmented_list = []
        for _ in range(n_augmented):
            idx1, idx2 = rng.choice(len(data), 2, replace=False)
            row1 = data.iloc[idx1]
            row2 = data.iloc[idx2]
            lam = rng.beta(alpha, alpha)
            mixed = {}
            for col in self.numeric_cols:
                if col in data.columns:
                    mixed[col] = lam * row1[col] + (1 - lam) * row2[col]
            for col in self.categorical_cols:
                if col in data.columns:
                    mixed[col] = row1[col] if rng.random() < lam else row2[col]
            augmented_list.append(mixed)
        return pd.DataFrame(augmented_list)

    def combine_drifts(self, n_samples: int = 2000, mix: Dict[str, float] = None, **kwargs) -> pd.DataFrame:
        """Create a combined dataset mixing different drift types.

        mix: dict mapping method name -> relative weight. Supported keys:
            'gradual', 'sudden', 'seasonal', 'multivariate', 'categorical'

        Example: mix={'gradual':0.5, 'seasonal':0.3, 'sudden':0.2}
        """
        supported = {
            'gradual': self.generate_gradual_drift,
            'sudden': self.generate_sudden_drift,
            'seasonal': self.generate_seasonal_drift,
            'multivariate': self.generate_multivariate_drift,
            'categorical': self.generate_categorical_drift
        }

        if mix is None:
            mix = {'gradual': 0.5, 'seasonal': 0.3, 'sudden': 0.2}

        # normalize weights
        total = sum(mix.values())
        if total <= 0:
            raise ValueError('mix weights must sum to > 0')
        norm = {k: v / total for k, v in mix.items()}

        parts = []
        allocated = 0
        for key, weight in norm.items():
            if key not in supported:
                continue
            cnt = int(round(n_samples * weight))
            allocated += cnt
            # call generator with appropriate n
            func = supported[key]
            # pass-through kwargs so user can tune e.g., drift_intensity
            part = func(n_samples=cnt, **kwargs) if cnt > 0 else None
            if part is not None:
                parts.append(part)

        # if rounding caused shortage/overflow, adjust
        if allocated < n_samples:
            # sample remaining from original (safe sample)
            rem = n_samples - allocated
            parts.append(self._sample_reset(rem))
        elif allocated > n_samples:
            # concat then trim
            df = pd.concat(parts, ignore_index=True).sample(n=n_samples, random_state=self.random_state).reset_index(drop=True)
            return df

        df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        return df

    def visualize_drift(self, original: pd.DataFrame, drifted: pd.DataFrame, features: List[str] = None, save_path: str = None):
        if features is None:
            features = self.numeric_cols[:4]
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        for idx, col in enumerate(features):
            if idx >= 4:
                break
            if col in original.columns and col in drifted.columns:
                axes[idx].hist(original[col].dropna(), bins=30, alpha=0.5, label='Original', density=True)
                axes[idx].hist(drifted[col].dropna(), bins=30, alpha=0.5, label='Drifted', density=True)
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Density')
                axes[idx].legend()
                axes[idx].set_title(f'Distribution Shift: {col}')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
