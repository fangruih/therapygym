# Rate Limiting Configuration (ported)

class RateLimitConfig:
    # Solution 1: Retry Logic
    USE_RETRY_LOGIC = True
    MAX_RETRIES = 5
    BASE_RETRY_DELAY = 1.0  # Base delay in seconds for exponential backoff

    # Solution 2: API Call Delays
    USE_API_DELAYS = True
    DELAY_BETWEEN_API_CALLS = 0.5  # Delay between patient and therapist calls (seconds)
    DELAY_BETWEEN_TURNS = 0.5  # Delay between conversation turns (seconds)

    # Solution 3: Execution Mode
    RUN_MODE = "SEQUENTIAL"  # Options: "PARALLEL", "SEQUENTIAL", "STAGGERED"
    STAGGER_DELAY = 30  # Delay between starting parallel processes (seconds)
    SEQUENTIAL_DELAY = 10  # Delay between sequential runs (seconds)

    # Advanced settings
    VERBOSE_RATE_LIMITING = True  # Print rate limiting messages

    @classmethod
    def get_retry_settings(cls):
        return {"enabled": cls.USE_RETRY_LOGIC, "max_retries": cls.MAX_RETRIES, "base_delay": cls.BASE_RETRY_DELAY}

    @classmethod
    def get_delay_settings(cls):
        return {"enabled": cls.USE_API_DELAYS, "api_delay": cls.DELAY_BETWEEN_API_CALLS, "turn_delay": cls.DELAY_BETWEEN_TURNS}

    @classmethod
    def get_execution_settings(cls):
        return {"mode": cls.RUN_MODE, "stagger_delay": cls.STAGGER_DELAY, "sequential_delay": cls.SEQUENTIAL_DELAY}


class Presets:
    FAST_PARALLEL = {"USE_RETRY_LOGIC": False, "USE_API_DELAYS": False, "RUN_MODE": "PARALLEL"}
    SEQUENTIAL_FULL_PROTECTION = {"USE_RETRY_LOGIC": True, "USE_API_DELAYS": True, "RUN_MODE": "SEQUENTIAL"}
    STAGGERED_WITH_RETRY = {"USE_RETRY_LOGIC": True, "USE_API_DELAYS": False, "RUN_MODE": "STAGGERED"}

    # Legacy aliases
    AGGRESSIVE = FAST_PARALLEL
    CONSERVATIVE = SEQUENTIAL_FULL_PROTECTION
    BALANCED = STAGGERED_WITH_RETRY

    @classmethod
    def apply_preset(cls, preset_name: str):
        preset = getattr(cls, preset_name.upper(), None)
        if not preset:
            raise ValueError(f"Unknown preset: {preset_name}")
        for key, value in preset.items():
            setattr(RateLimitConfig, key, value)


