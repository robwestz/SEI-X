#!/bin/bash
# SIE-X Production Deployment Script
#
# This script deploys SIE-X to production with all security checks
#
# Usage:
#   ./scripts/deploy.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================="
echo "SIE-X Production Deployment"
echo "================================================================="
echo ""

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================

echo "Running pre-flight checks..."
echo ""

# Check .env.production exists
if [ ! -f .env.production ]; then
    echo -e "${RED}❌ .env.production not found${NC}"
    echo "   Run: cp .env.production.example .env.production"
    echo "   Then: ./scripts/generate_secrets.sh >> .env.production"
    exit 1
fi
echo -e "${GREEN}✅ .env.production found${NC}"

# Load environment
set -a
source .env.production
set +a

# Check JWT secret length
JWT_LENGTH=${#SIE_X_JWT_SECRET}
if [ "$JWT_LENGTH" -lt 32 ]; then
    echo -e "${RED}❌ JWT_SECRET too short ($JWT_LENGTH chars, need 32+)${NC}"
    echo "   Run: ./scripts/generate_secrets.sh"
    exit 1
fi
echo -e "${GREEN}✅ JWT_SECRET is strong ($JWT_LENGTH chars)${NC}"

# Check CORS configuration
if [[ "$SIE_X_ALLOWED_ORIGINS" == *"*"* ]] || [[ "$SIE_X_ALLOWED_ORIGINS" == *"localhost"* ]]; then
    echo -e "${RED}❌ CORS configured insecurely${NC}"
    echo "   ALLOWED_ORIGINS: $SIE_X_ALLOWED_ORIGINS"
    echo "   Set to your actual frontend domain(s)"
    exit 1
fi
echo -e "${GREEN}✅ CORS configured: $SIE_X_ALLOWED_ORIGINS${NC}"

# Check HTTPS enforcement
if [ "$SIE_X_HTTPS_ONLY" != "true" ]; then
    echo -e "${YELLOW}⚠️  HTTPS enforcement disabled${NC}"
    echo "   Set SIE_X_HTTPS_ONLY=true in production"
fi

# Check database URL
if [[ "$SIE_X_DB_URL" == *"CHANGE_ME"* ]]; then
    echo -e "${RED}❌ Database URL not configured${NC}"
    echo "   Update SIE_X_DB_URL in .env.production"
    exit 1
fi
echo -e "${GREEN}✅ Database URL configured${NC}"

# Check Redis URL
if [[ "$SIE_X_REDIS_URL" == *"localhost"* ]]; then
    echo -e "${YELLOW}⚠️  Redis URL points to localhost${NC}"
    echo "   Update to production Redis server"
fi

echo ""
echo -e "${GREEN}✅ All pre-flight checks passed${NC}"
echo ""

# =============================================================================
# DATABASE MIGRATION
# =============================================================================

echo "Running database migration..."
if [ -f scripts/migrate.sh ]; then
    ./scripts/migrate.sh
    echo -e "${GREEN}✅ Database migrated${NC}"
else
    echo -e "${YELLOW}⚠️  No migration script found${NC}"
    echo "   Manually run: psql -f database/schema.sql"
fi
echo ""

# =============================================================================
# BUILD & DEPLOY
# =============================================================================

echo "Building Docker images..."
docker-compose -f docker-compose.production.yml build
echo -e "${GREEN}✅ Docker images built${NC}"
echo ""

echo "Starting services..."
docker-compose -f docker-compose.production.yml up -d
echo -e "${GREEN}✅ Services started${NC}"
echo ""

# =============================================================================
# HEALTH CHECKS
# =============================================================================

echo "Waiting for services to be healthy..."
sleep 10

# Check PostgreSQL
echo -n "PostgreSQL... "
if docker-compose -f docker-compose.production.yml exec -T postgres pg_isready -U siex_app > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    exit 1
fi

# Check Redis
echo -n "Redis... "
if docker-compose -f docker-compose.production.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC}"
else
    echo -e "${RED}❌${NC}"
    exit 1
fi

# Check API
echo -n "API... "
for i in {1..30}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Timeout${NC}"
        echo "Check logs: docker-compose -f docker-compose.production.yml logs siex-api"
        exit 1
    fi
    sleep 2
done

echo ""
echo "================================================================="
echo -e "${GREEN}✅ SIE-X deployed successfully!${NC}"
echo "================================================================="
echo ""
echo "Endpoints:"
echo "  - API: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "  - Health: http://localhost:8000/health"
echo ""
echo "Next steps:"
echo "  1. Test authentication: curl http://localhost:8000/auth/register-tenant"
echo "  2. Monitor logs: docker-compose -f docker-compose.production.yml logs -f"
echo "  3. Set up reverse proxy (nginx/caddy) with HTTPS"
echo "  4. Configure monitoring & alerts"
echo ""
