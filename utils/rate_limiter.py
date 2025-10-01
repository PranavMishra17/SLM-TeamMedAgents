"""
Rate limiter with exponential backoff retry logic for SLM TeamMedAgents.
Handles API rate limits, request tracking, and automatic retries.
"""

import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional
from collections import deque
import threading


class RateLimiter:
    """Thread-safe rate limiter with exponential backoff retry logic."""
    
    def __init__(self, model_name: str, rate_limits: Dict[str, Any], retry_config: Dict[str, Any]):
        self.model_name = model_name
        self.rate_limits = rate_limits
        self.retry_config = retry_config
        
        # Thread-safe tracking
        self.lock = threading.Lock()
        self.request_times = deque()  # Track request timestamps
        self.token_usage = deque()    # Track token usage with timestamps
        self.daily_requests = 0
        self.last_reset_date = datetime.now().date()
        
        logging.info(f"RateLimiter initialized for {model_name}: RPM={rate_limits.get('rpm')}, TPM={rate_limits.get('tpm')}, RPD={rate_limits.get('rpd')}")
    
    def _reset_daily_counters(self):
        """Reset daily counters if it's a new day."""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_requests = 0
            self.last_reset_date = today
            # Clear old token usage (older than 1 day)
            cutoff = datetime.now() - timedelta(days=1)
            while self.token_usage and self.token_usage[0][0] < cutoff:
                self.token_usage.popleft()
    
    def _clean_old_requests(self):
        """Remove request timestamps older than 1 minute."""
        cutoff = datetime.now() - timedelta(minutes=1)
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()
    
    def _clean_old_tokens(self):
        """Remove token usage older than 1 minute."""
        cutoff = datetime.now() - timedelta(minutes=1)
        while self.token_usage and self.token_usage[0][0] < cutoff:
            self.token_usage.popleft()
    
    def _get_current_usage(self):
        """Get current usage statistics."""
        now = datetime.now()
        
        # Clean old data
        self._clean_old_requests()
        self._clean_old_tokens()
        self._reset_daily_counters()
        
        # Calculate current usage
        rpm_count = len(self.request_times)
        tpm_count = sum(usage[1] for usage in self.token_usage)  # Sum tokens from last minute
        rpd_count = self.daily_requests
        
        return rpm_count, tpm_count, rpd_count
    
    def can_make_request(self, estimated_tokens: int = 0) -> tuple[bool, Optional[float]]:
        """Check if a request can be made without hitting rate limits."""
        with self.lock:
            rpm_count, tpm_count, rpd_count = self._get_current_usage()
            
            # Check limits
            rpm_limit = self.rate_limits.get('rpm', float('inf'))
            tpm_limit = self.rate_limits.get('tpm', float('inf'))
            rpd_limit = self.rate_limits.get('rpd', float('inf'))
            
            # Check if we'd exceed limits
            if rpm_count >= rpm_limit:
                return False, 60.0  # Wait 1 minute for RPM reset
            
            if tpm_count + estimated_tokens > tpm_limit:
                return False, 60.0  # Wait 1 minute for TPM reset
            
            if rpd_count >= rpd_limit:
                # Calculate seconds until next day
                tomorrow = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                wait_time = (tomorrow - datetime.now()).total_seconds()
                return False, wait_time
            
            return True, None
    
    def record_request(self, token_count: int = 0):
        """Record a successful request."""
        with self.lock:
            now = datetime.now()
            self.request_times.append(now)
            if token_count > 0:
                self.token_usage.append((now, token_count))
            self.daily_requests += 1
    
    def wait_if_needed(self, estimated_tokens: int = 0) -> float:
        """Wait if necessary to respect rate limits."""
        can_proceed, wait_time = self.can_make_request(estimated_tokens)
        
        if not can_proceed and wait_time:
            logging.info(f"Rate limit reached for {self.model_name}, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            return wait_time
        
        return 0.0
    
    def exponential_backoff_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry logic."""
        max_retries = self.retry_config.get('max_retries', 5)
        initial_delay = self.retry_config.get('initial_delay', 1.0)
        max_delay = self.retry_config.get('max_delay', 60.0)
        exponential_base = self.retry_config.get('exponential_base', 2.0)
        jitter = self.retry_config.get('jitter', True)
        rate_limit_delay = self.retry_config.get('rate_limit_delay', 65.0)
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Wait for rate limits before attempting request
                estimated_tokens = kwargs.get('estimated_tokens', 0)
                wait_time = self.wait_if_needed(estimated_tokens)
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Record successful request if we have token count
                if hasattr(result, '__dict__') and 'usage_metadata' in str(result):
                    # Try to extract token count from result
                    try:
                        if hasattr(result, 'usage_metadata') and result.usage_metadata:
                            token_count = getattr(result.usage_metadata, 'total_token_count', estimated_tokens)
                            self.record_request(token_count)
                        else:
                            self.record_request(estimated_tokens)
                    except:
                        self.record_request(estimated_tokens)
                else:
                    self.record_request(estimated_tokens)
                
                if attempt > 0:
                    logging.info(f"Request succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                if attempt == max_retries:
                    logging.error(f"Max retries ({max_retries}) exceeded for {self.model_name}")
                    break
                
                # Determine delay based on error type
                if any(term in error_str for term in ['rate limit', 'quota', 'too many requests', '429']):
                    delay = rate_limit_delay
                    logging.warning(f"Rate limit error on attempt {attempt + 1}, waiting {delay}s: {e}")
                elif any(term in error_str for term in ['timeout', 'connection', 'network']):
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                    logging.warning(f"Network error on attempt {attempt + 1}, waiting {delay:.1f}s: {e}")
                else:
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                    logging.warning(f"Error on attempt {attempt + 1}, waiting {delay:.1f}s: {e}")
                
                # Add jitter to prevent thundering herd
                if jitter:
                    jitter_factor = random.uniform(0.1, 0.3)
                    delay *= (1 + jitter_factor)
                
                time.sleep(delay)
        
        # If we get here, all retries failed
        logging.error(f"All retry attempts failed for {self.model_name}: {last_exception}")
        raise last_exception
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiter statistics."""
        with self.lock:
            rpm_count, tpm_count, rpd_count = self._get_current_usage()
            
            return {
                "model": self.model_name,
                "current_rpm": rpm_count,
                "current_tpm": tpm_count,
                "current_rpd": rpd_count,
                "limits": self.rate_limits,
                "rpm_utilization": rpm_count / self.rate_limits.get('rpm', 1),
                "tpm_utilization": tpm_count / self.rate_limits.get('tpm', 1),
                "rpd_utilization": rpd_count / self.rate_limits.get('rpd', 1)
            }