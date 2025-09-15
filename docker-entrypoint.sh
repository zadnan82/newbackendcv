#!/bin/bash
set -e

echo "Starting CV Privacy Platform..."

# Wait for database
until python -c "
from app.database import check_database_connection
import sys
if check_database_connection():
    print('Database connected')
    sys.exit(0)
else:
    print('Waiting for database...')
    sys.exit(1)
"; do
    sleep 2
done

# Create tables
python -c "
from app.database import create_tables, initialize_pricing_tiers
create_tables()
initialize_pricing_tiers()
print('Database setup complete')
"

echo "Starting application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload