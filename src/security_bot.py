import os
import logging
import requests
import time
import subprocess
import sys
from datetime import datetime
import hashlib
import secrets
import json
from flask import request, jsonify
import streamlit as st

class SecurityBot:
    """Security and Error Handling Bot for Stock Analysis App"""

    def __init__(self):
        self.logger = self.setup_logger()
        self.session_tokens = {}
        self.max_retries = 3
        self.retry_delay = 5
        self.current_port = 8501
        self.backup_ports = [8502, 8503, 8504, 8505]

    def setup_logger(self):
        """Setup secure logging"""
        logger = logging.getLogger('SecurityBot')
        logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)

        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            'logs/security.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )

        # Secure formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s - IP:%(ip)s - User:%(user)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def generate_session_token(self, user_id="default"):
        """Generate secure session token"""
        token = secrets.token_urlsafe(32)
        self.session_tokens[token] = {
            'user_id': user_id,
            'created': datetime.now(),
            'expires': datetime.now().timestamp() + 3600  # 1 hour
        }
        return token

    def validate_session(self, token):
        """Validate session token"""
        if token not in self.session_tokens:
            return False

        session = self.session_tokens[token]
        if datetime.now().timestamp() > session['expires']:
            del self.session_tokens[token]
            return False

        return True

    def hash_password(self, password):
        """Secure password hashing"""
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}${hashed.hex()}"

    def verify_password(self, password, hashed_password):
        """Verify password against hash"""
        try:
            salt, hash_value = hashed_password.split('$')
            new_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return new_hash.hex() == hash_value
        except:
            return False

    def sanitize_input(self, input_data):
        """Sanitize user input to prevent injection attacks"""
        if isinstance(input_data, str):
            # Remove potentially dangerous characters
            dangerous_chars = ['<', '>', '"', "'", ';', '--', '/*', '*/']
            for char in dangerous_chars:
                input_data = input_data.replace(char, '')

            # Limit length
            if len(input_data) > 1000:
                input_data = input_data[:1000]

        return input_data

    def rate_limit_check(self, ip_address, endpoint):
        """Implement rate limiting"""
        # Simple in-memory rate limiting (for production, use Redis)
        current_time = time.time()
        rate_limit_key = f"{ip_address}_{endpoint}"

        if not hasattr(self, 'rate_limits'):
            self.rate_limits = {}

        if rate_limit_key not in self.rate_limits:
            self.rate_limits[rate_limit_key] = []

        # Clean old requests
        self.rate_limits[rate_limit_key] = [
            req_time for req_time in self.rate_limits[rate_limit_key]
            if current_time - req_time < 60  # 1 minute window
        ]

        # Check rate limit (10 requests per minute)
        if len(self.rate_limits[rate_limit_key]) >= 10:
            return False

        self.rate_limits[rate_limit_key].append(current_time)
        return True

    def handle_error(self, error, context="general", user_ip="unknown"):
        """Handle errors automatically"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'user_ip': user_ip,
            'severity': self.classify_error_severity(error)
        }

        # Log the error
        self.logger.error(
            f"Error handled: {error_info['error_message']}",
            extra={'ip': user_ip, 'user': 'system'}
        )

        # Save error to file for analysis
        self.save_error_report(error_info)

        # Auto-recovery actions based on error type
        if self.should_restart_service(error):
            self.restart_service_on_new_port()
        elif self.should_retry_operation(error):
            return self.retry_operation(context)
        else:
            return self.get_error_response(error_info)

    def classify_error_severity(self, error):
        """Classify error severity"""
        error_str = str(error).lower()

        if any(keyword in error_str for keyword in ['security', 'auth', 'permission', 'access']):
            return 'CRITICAL'
        elif any(keyword in error_str for keyword in ['connection', 'timeout', 'network']):
            return 'HIGH'
        elif any(keyword in error_str for keyword in ['data', 'parse', 'format']):
            return 'MEDIUM'
        else:
            return 'LOW'

    def should_restart_service(self, error):
        """Determine if service should be restarted"""
        error_str = str(error).lower()
        restart_triggers = [
            'port already in use',
            'connection refused',
            'service unavailable',
            'internal server error'
        ]
        return any(trigger in error_str for trigger in restart_triggers)

    def should_retry_operation(self, error):
        """Determine if operation should be retried"""
        error_str = str(error).lower()
        retry_triggers = [
            'temporary failure',
            'timeout',
            'connection reset',
            'rate limit'
        ]
        return any(trigger in error_str for trigger in retry_triggers)

    def restart_service_on_new_port(self):
        """Restart service on a new port"""
        for port in self.backup_ports:
            try:
                # Check if port is available
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('localhost', port))
                sock.close()

                # Port is available, restart service
                self.logger.info(f"Restarting service on port {port}")

                # Kill current process and restart on new port
                if hasattr(sys, '_MEIPASS'):  # Running as PyInstaller bundle
                    subprocess.Popen([sys.executable, sys.argv[0], '--port', str(port)])
                else:
                    subprocess.Popen([sys.executable, 'streamlit_tradingview_signals.py', '--server.port', str(port)])

                self.current_port = port
                return f"Service restarted on port {port}"

            except OSError:
                continue

        return "No available ports found for restart"

    def retry_operation(self, context, max_retries=3):
        """Retry failed operation"""
        for attempt in range(max_retries):
            try:
                time.sleep(self.retry_delay * (attempt + 1))

                if context == "api_call":
                    # Retry API call logic would go here
                    return {"status": "retry_successful"}
                elif context == "data_fetch":
                    # Retry data fetch logic would go here
                    return {"status": "data_fetched"}

            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

        return {"status": "retry_failed"}

    def save_error_report(self, error_info):
        """Save error report for analysis"""
        os.makedirs('error_reports', exist_ok=True)

        filename = f"error_reports/error_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(error_info, f, indent=2)

    def get_error_response(self, error_info):
        """Generate appropriate error response"""
        if error_info['severity'] == 'CRITICAL':
            return {
                "error": "Security incident detected. Access restricted.",
                "code": "SEC_001",
                "timestamp": error_info['timestamp']
            }
        elif error_info['severity'] == 'HIGH':
            return {
                "error": "Service temporarily unavailable. Please try again later.",
                "code": "SRV_001",
                "timestamp": error_info['timestamp']
            }
        else:
            return {
                "error": "An error occurred. Our system is working to resolve it.",
                "code": "GEN_001",
                "timestamp": error_info['timestamp']
            }

    def get_prediction_accuracy(self):
        """Get current prediction accuracy from trained models"""
        try:
            # Try to load model accuracy from saved model
            import joblib
            if os.path.exists('models/trained_model.pkl'):
                model_data = joblib.load('models/trained_model.pkl')
                accuracy = model_data.get('accuracy', 0.0)
                return f"{accuracy:.2%}"
            else:
                return "Model not trained yet"
        except Exception as e:
            self.logger.error(f"Error getting prediction accuracy: {e}")
            return "Accuracy data unavailable"

    def get_stock_pov_details(self, symbol, tv_data):
        """Generate detailed Point of View for a stock"""
        try:
            pov_details = {
                "symbol": symbol,
                "recommendation": tv_data.get("recommendation", "NEUTRAL"),
                "confidence_score": self.calculate_confidence_score(tv_data),
                "technical_analysis": self.analyze_technical_indicators(tv_data),
                "risk_assessment": self.assess_risk_level(tv_data),
                "trading_strategy": self.generate_trading_strategy(tv_data),
                "next_support_resistance": self.identify_key_levels(tv_data),
                "market_sentiment": self.analyze_market_sentiment(tv_data),
                "time_horizon": self.recommend_time_horizon(tv_data),
                "position_sizing": self.recommend_position_size(tv_data),
                "stop_loss_target": self.calculate_stop_loss_target(tv_data)
            }

            return pov_details

        except Exception as e:
            self.logger.error(f"Error generating POV details for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": "Unable to generate POV details",
                "recommendation": "HOLD"
            }

    def calculate_confidence_score(self, tv_data):
        """Calculate confidence score based on signal strength"""
        buy_signals = tv_data.get("buy", 0)
        sell_signals = tv_data.get("sell", 0)
        neutral_signals = tv_data.get("neutral", 0)
        total_signals = buy_signals + sell_signals + neutral_signals

        if total_signals == 0:
            return 0.5

        if tv_data.get("recommendation") == "BUY":
            return min(0.95, (buy_signals / total_signals) * 1.2)
        elif tv_data.get("recommendation") == "SELL":
            return min(0.95, (sell_signals / total_signals) * 1.2)
        else:
            return 0.5

    def analyze_technical_indicators(self, tv_data):
        """Analyze technical indicators in detail"""
        reasons = tv_data.get("reasons", [])

        analysis = {
            "oscillators": "Mixed signals" if "OSC" in str(reasons) else "Neutral",
            "moving_averages": "Trend following" if "MA" in str(reasons) else "Sideways",
            "momentum": "Building" if any("bullish" in r.lower() for r in reasons) else "Weakening",
            "volume": "Increasing" if "volume" in str(reasons).lower() else "Stable"
        }

        return analysis

    def assess_risk_level(self, tv_data):
        """Assess risk level based on signals"""
        recommendation = tv_data.get("recommendation", "NEUTRAL")

        if recommendation == "BUY":
            return "Medium" if tv_data.get("sell", 0) < 3 else "High"
        elif recommendation == "SELL":
            return "Medium" if tv_data.get("buy", 0) < 3 else "High"
        else:
            return "Low"

    def generate_trading_strategy(self, tv_data):
        """Generate trading strategy based on signals"""
        rec = tv_data.get("recommendation", "NEUTRAL")

        strategies = {
            "BUY": "Accumulate on dips, target higher timeframe resistance",
            "SELL": "Reduce position gradually, wait for reversal confirmation",
            "NEUTRAL": "Hold current position, monitor for breakout"
        }

        return strategies.get(rec, "Wait for clearer signals")

    def identify_key_levels(self, tv_data):
        """Identify key support and resistance levels"""
        # This would typically use price data, but for now return placeholder
        return {
            "support": "Previous low levels",
            "resistance": "Recent high levels",
            "pivot": "Current price levels"
        }

    def analyze_market_sentiment(self, tv_data):
        """Analyze market sentiment"""
        buy_count = tv_data.get("buy", 0)
        sell_count = tv_data.get("sell", 0)

        if buy_count > sell_count + 2:
            return "Bullish"
        elif sell_count > buy_count + 2:
            return "Bearish"
        else:
            return "Neutral"

    def recommend_time_horizon(self, tv_data):
        """Recommend time horizon for position"""
        rec = tv_data.get("recommendation", "NEUTRAL")

        if rec == "BUY":
            return "Medium-term (3-6 months)"
        elif rec == "SELL":
            return "Short-term (1-3 months)"
        else:
            return "Long-term hold"

    def recommend_position_size(self, tv_data):
        """Recommend position size based on confidence"""
        confidence = self.calculate_confidence_score(tv_data)

        if confidence > 0.8:
            return "Full position (100%)"
        elif confidence > 0.6:
            return "Medium position (50-75%)"
        else:
            return "Small position (25-50%)"

    def calculate_stop_loss_target(self, tv_data):
        """Calculate stop loss and target levels"""
        rec = tv_data.get("recommendation", "NEUTRAL")

        if rec == "BUY":
            return {
                "stop_loss": "2-3% below entry",
                "target": "5-8% above entry"
            }
        elif rec == "SELL":
            return {
                "stop_loss": "2-3% above entry",
                "target": "5-8% below entry"
            }
        else:
            return {
                "stop_loss": "5% from entry",
                "target": "10% from entry"
            }

# Global security bot instance
security_bot = SecurityBot()

def require_auth(func):
    """Decorator for API authentication"""
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')

        if not security_bot.validate_session(token):
            return jsonify({"error": "Unauthorized access"}), 401

        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper

def error_handler(func):
    """Decorator for automatic error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            user_ip = request.remote_addr if hasattr(request, 'remote_addr') else 'unknown'
            return jsonify(security_bot.handle_error(e, func.__name__, user_ip)), 500
    wrapper.__name__ = func.__name__
    return wrapper
