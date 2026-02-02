"""Database setup script for PostgreSQL with pgvector extension."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg
from psycopg import sql

from app.config import get_settings


def setup_database():
    """Set up the PostgreSQL database with pgvector extension."""
    settings = get_settings()

    # Parse the connection URL to get database name
    # Format: postgresql://user:pass@host:port/dbname
    db_url = settings.database_url
    
    # Extract components from the URL
    # Remove the postgresql:// prefix
    url_without_prefix = db_url.replace("postgresql://", "")
    
    # Split user:pass@host:port/dbname
    auth_and_rest = url_without_prefix.split("@")
    if len(auth_and_rest) == 2:
        user_pass = auth_and_rest[0]
        host_port_db = auth_and_rest[1]
    else:
        user_pass = ""
        host_port_db = auth_and_rest[0]
    
    # Parse user:pass
    if ":" in user_pass:
        user, password = user_pass.split(":", 1)
    else:
        user = user_pass
        password = ""
    
    # Parse host:port/dbname
    if "/" in host_port_db:
        host_port, dbname = host_port_db.rsplit("/", 1)
    else:
        host_port = host_port_db
        dbname = "rag_db"
    
    if ":" in host_port:
        host, port = host_port.split(":")
        port = int(port)
    else:
        host = host_port
        port = 5432

    print(f"Database setup script")
    print(f"=" * 50)
    print(f"Host: {host}:{port}")
    print(f"Database: {dbname}")
    print(f"User: {user}")
    print()

    # Connect to the default 'postgres' database first
    try:
        print("Connecting to PostgreSQL server...")
        with psycopg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname="postgres",
            autocommit=True,
        ) as conn:
            with conn.cursor() as cur:
                # Check if database exists
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (dbname,)
                )
                exists = cur.fetchone()

                if not exists:
                    print(f"Creating database '{dbname}'...")
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname))
                    )
                    print(f"Database '{dbname}' created successfully!")
                else:
                    print(f"Database '{dbname}' already exists.")

    except psycopg.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        print("\nMake sure PostgreSQL is running and the credentials are correct.")
        return False

    # Connect to the target database and enable pgvector
    try:
        print(f"\nConnecting to database '{dbname}'...")
        with psycopg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
        ) as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                print("Enabling pgvector extension...")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                print("pgvector extension enabled successfully!")

                # Verify the extension
                cur.execute(
                    "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
                )
                result = cur.fetchone()
                if result:
                    print(f"pgvector version: {result[1]}")
                else:
                    print("Warning: pgvector extension may not be properly installed.")

    except psycopg.Error as e:
        print(f"Error setting up pgvector: {e}")
        print("\nMake sure pgvector extension is installed on your PostgreSQL server.")
        print("Install instructions: https://github.com/pgvector/pgvector")
        return False

    print()
    print("=" * 50)
    print("Database setup complete!")
    print(f"Connection string: {settings.database_url}")
    return True


if __name__ == "__main__":
    success = setup_database()
    sys.exit(0 if success else 1)
