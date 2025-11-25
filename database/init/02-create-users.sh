#!/bin/bash
set -e

echo "Creating initial user accounts..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create admin user (admin / admin4080!)
    -- Bcrypt hash generated with cost factor 10
    -- Security fields: failed_login_attempts=0, no lock, no session
    INSERT INTO users (
      username,
      email,
      password_hash,
      role,
      is_active,
      failed_login_attempts,
      locked_until,
      current_session_id,
      last_login_at,
      created_at,
      updated_at
    )
    VALUES (
      'admin',
      'admin@example.com',
      '\$2b\$10\$51V3ovU6G7L8C0B.ymEdZukoYLYQOc7FbHX3NdK.ERb0jhAPZVdyq',
      'admin',
      true,
      0,
      NULL,
      NULL,
      NULL,
      now(),
      now()
    );

    -- Create regular user (user / user4080!)
    -- Bcrypt hash generated with cost factor 10
    -- Security fields: failed_login_attempts=0, no lock, no session
    INSERT INTO users (
      username,
      email,
      password_hash,
      role,
      is_active,
      failed_login_attempts,
      locked_until,
      current_session_id,
      last_login_at,
      created_at,
      updated_at
    )
    VALUES (
      'user',
      'user@example.com',
      '\$2b\$10\$EzvxqLHQ5wXJaVmN0CK.eOPx3Fu1mSXk1blpGYOWF6cmNR53wj1pq',
      'user',
      true,
      0,
      NULL,
      NULL,
      NULL,
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
      failed_login_attempts,
      locked_until,
      last_login_at,
      created_at
    FROM users
    WHERE deleted_at IS NULL
    ORDER BY role DESC;
EOSQL

echo ""
echo "✅ Initial user accounts created successfully!"
echo ""
echo "   📋 Login Credentials:"
echo "   ┌─────────────────────────────────────────────────────┐"
echo "   │ Admin Account                                       │"
echo "   │   Username: admin                                   │"
echo "   │   Password: admin4080!                              │"
echo "   ├─────────────────────────────────────────────────────┤"
echo "   │ User Account                                        │"
echo "   │   Username: user                                    │"
echo "   │   Password: user4080!                               │"
echo "   └─────────────────────────────────────────────────────┘"
echo ""
echo "   🔒 Security Features Enabled:"
echo "   • Login attempt tracking (5 failed attempts = 30 min lockout)"
echo "   • Single session enforcement (one login per user)"
echo "   • Password policy validation (9+ chars, mixed types)"
echo ""
echo "   ✅ Default passwords now meet security policy requirements!"
echo "   ⚠️  WARNING: Change these passwords before deploying to production!"
echo ""
