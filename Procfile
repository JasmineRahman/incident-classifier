# Gunicorn configuration with:
# - timeout of 120 seconds to handle larger images
# - 2 worker processes for better request handling
# - access to logging
# - thread configuration for better performance
web: gunicorn app:app --timeout 120 --workers 2 --threads 2 --worker-class gthread --log-level info --access-logfile - --error-logfile -