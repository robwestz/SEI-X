#!/bin/bash
# Generate secure secrets for SIE-X production deployment
#
# Usage:
#   ./scripts/generate_secrets.sh >> .env.production
#   # Then edit .env.production to set other values

set -e

echo "# Generated secrets - $(date)"
echo "#"
echo "# SECURITY: Keep these secrets secure! Add .env.production to .gitignore"
echo ""

# JWT Secret (256-bit = 32 bytes = 43 chars base64)
JWT_SECRET=$(openssl rand -base64 48)
echo "SIE_X_JWT_SECRET=$JWT_SECRET"

# Database Password
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
echo "DB_PASSWORD=$DB_PASSWORD"

# Redis Password
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
echo "REDIS_PASSWORD=$REDIS_PASSWORD"

echo ""
echo "# Next steps:"
echo "# 1. Set ALLOWED_ORIGINS to your frontend domain(s)"
echo "# 2. Set SIE_X_DB_URL with the generated DB_PASSWORD"
echo "# 3. Set SIE_X_REDIS_URL with the generated REDIS_PASSWORD"
echo "# 4. Review all settings in .env.production"
echo "# 5. chmod 600 .env.production (restrict permissions)"
