#!/bin/bash
set -e

echo "Creating initial user accounts..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create admin user (admin / adminpw)
    -- Bcrypt hash generated with cost factor 10
    INSERT INTO users (username, email, password_hash, role, is_active, created_at, updated_at)
    VALUES (
      'admin',
      'admin@example.com',
      '\$2b\$10\$ZaRCQmZB.JMJaulz67ZiteUHNRnxCaawMh8KOlJ30GI3A4bCDQFky',
      'admin',
      true,
      now(),
      now()
    );

    -- Create regular user (user / userpw)
    -- Bcrypt hash generated with cost factor 10
    INSERT INTO users (username, email, password_hash, role, is_active, created_at, updated_at)
    VALUES (
      'user',
      'user@example.com',
      '\$2b\$10\$sZRoQ6bYIsv8xQX28MiFKuyn5WPfFhlBwyyKnhkLHxtAcynQoBwaK',
      'user',
      true,
      now(),
      now()
    );

    -- Display created users (without password hash)
    SELECT
      id,
      username,
      email,
      role,
      is_active,
      created_at
    FROM users
    WHERE deleted_at IS NULL
    ORDER BY role DESC;
EOSQL

echo ""
echo "✅ Initial user accounts created successfully!"
echo ""
echo "   📋 Login Credentials:"
echo "   ┌─────────────────────────────────┐"
echo "   │ Admin Account                   │"
echo "   │   Username: admin               │"
echo "   │   Password: adminpw             │"
echo "   ├─────────────────────────────────┤"
echo "   │ User Account                    │"
echo "   │   Username: user                │"
echo "   │   Password: userpw              │"
echo "   └─────────────────────────────────┘"
echo ""
echo "   ⚠️  WARNING: Change these passwords before deploying to production!"
echo ""
